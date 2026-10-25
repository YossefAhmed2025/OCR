"""Optional image preprocessing.

Design rule: the original image is never written to and never mutated
in place. Every function takes an array and returns a new one, and the
caller keeps the untouched original for comparison.

Second rule: nothing is automatic. Arabic tashkeel occupy only a few
pixels at typical book-scan resolutions. Denoising, binarisation and
aggressive sharpening routinely erase them while leaving the consonants
looking cleaner -- which makes the page *appear* easier to read while
destroying exactly the signal this project exists to measure. So each
operation is off unless switched on, and :func:`apply_preprocessing`
reports which ones ran.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .config import PreprocessConfig
from .utils import get_logger

LOGGER = get_logger(__name__)

#: Images below this short side are unlikely to carry legible diacritics.
MIN_RECOMMENDED_SHORT_SIDE = 600

#: Above this long side, detection downscales internally anyway.
MAX_RECOMMENDED_LONG_SIDE = 4000


class PreprocessingError(RuntimeError):
    """Raised when an image cannot be loaded or transformed."""


def load_image(path) -> np.ndarray:
    """Load an image as BGR.

    Uses ``cv2.imdecode`` on bytes read in Python rather than
    ``cv2.imread``, because ``cv2.imread`` silently returns ``None`` for
    paths containing non-ASCII characters on Windows -- which is exactly
    what an Arabic-named sample file will do.
    """
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        raise PreprocessingError(f"Image not found: {path}")
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise PreprocessingError(f"Could not read {path}: {exc}") from exc
    if buffer.size == 0:
        raise PreprocessingError(f"Image file is empty: {path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError(
            f"Could not decode {path}. Is it a valid image file?"
        )
    return image


def save_image(path, image: np.ndarray) -> None:
    """Write an image, handling non-ASCII paths on Windows."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise PreprocessingError(f"Could not encode image for {path}")
    encoded.tofile(str(path))


def inspect_image(image: np.ndarray) -> Dict[str, Any]:
    """Dimension and quality observations, with advisory warnings.

    The warnings are advisory only. Nothing acts on them automatically.
    """
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    grey = to_grayscale(image)

    # Variance of Laplacian: a conventional, rough sharpness proxy. It is
    # scale- and content-dependent, so it is reported, not thresholded
    # into a pass/fail verdict.
    blur_score = float(cv2.Laplacian(grey, cv2.CV_64F).var())

    warnings = []
    if min(height, width) < MIN_RECOMMENDED_SHORT_SIDE:
        warnings.append(
            f"Short side is {min(height, width)}px (< {MIN_RECOMMENDED_SHORT_SIDE}px). "
            "Arabic diacritics may be too small to resolve."
        )
    if max(height, width) > MAX_RECOMMENDED_LONG_SIDE:
        warnings.append(
            f"Long side is {max(height, width)}px (> {MAX_RECOMMENDED_LONG_SIDE}px). "
            "Detection will downscale internally."
        )
    if blur_score < 100:
        warnings.append(
            f"Low Laplacian variance ({blur_score:.1f}); the image may be blurred."
        )

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "mean_intensity": float(grey.mean()),
        "std_intensity": float(grey.std()),
        "laplacian_variance": blur_score,
        "warnings": warnings,
    }


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation.

    Applied to the L channel in LAB space for colour images so that hue
    is left alone. CLAHE is the gentlest of the contrast operations here
    and is usually safe for tashkeel, but the clip limit still matters:
    a high limit amplifies paper grain into mark-sized specks.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray, strength: int = 3) -> np.ndarray:
    """Non-local means denoising.

    WARNING: this is the single most likely operation to remove Arabic
    diacritics. A fatha on a 300dpi scan can be a 4x6 pixel blob, which
    non-local means treats as noise. Compare against an un-denoised run
    before trusting any result produced with this enabled.
    """
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, float(strength), 7, 21)
    return cv2.fastNlMeansDenoisingColored(image, None, float(strength), float(strength), 7, 21)


def estimate_skew_angle(image: np.ndarray, max_degrees: float = 15.0) -> float:
    """Estimate page skew in degrees via the minimum-area rectangle.

    Positive means the content is rotated clockwise. Returns 0.0 when no
    confident estimate is available or the estimate exceeds
    ``max_degrees`` -- a large estimate on a text page almost always
    means the heuristic latched onto a figure or a border, and rotating
    by it would make things worse.
    """
    grey = to_grayscale(image)
    # Invert so text is foreground, then threshold with Otsu.
    _, binary = cv2.threshold(
        grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        LOGGER.debug("Skew estimation: too few foreground pixels")
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV returns the angle in (0, 90]; map to a small signed rotation.
    if angle > 45:
        angle -= 90
    if abs(angle) > max_degrees:
        LOGGER.debug(
            "Skew estimate %.2f deg exceeds limit %.2f deg; not rotating",
            angle,
            max_degrees,
        )
        return 0.0
    return float(angle)


def rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate about the centre, expanding the canvas to avoid cropping.

    Uses cubic interpolation and replicates the border so that rotation
    does not introduce black wedges that the detector would read as
    content.
    """
    if angle_degrees == 0.0:
        return image.copy()
    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle_degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += (new_width / 2.0) - centre[0]
    matrix[1, 2] += (new_height / 2.0) - centre[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def upscale(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize by ``factor`` with cubic interpolation.

    Upscaling was expected to help diacritics by giving the recogniser
    more pixels per mark. **Measured, it did not.** On the clean
    synthetic set it cost 0.20 of diacritic recall at 2x and still hurt
    at 3x (``docs/camera_experiments.md`` section 2). The plausible
    reading is that clean input has nothing to gain and only acquires
    resampling artefacts; whether it helps genuinely low-resolution
    camera images is untested and remains an open question.

    Coordinates come back in the upscaled space and must be divided by
    ``factor`` to return to the original image space --
    :mod:`phonics_ocr.engine` does this and records the factor in the
    output.
    """
    if factor == 1.0:
        return image.copy()
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def four_point_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-correct a quadrilateral to a rectangle.

    This is the hook for page dewarping once the camera rig exists.
    ``corners`` must be four ``(x, y)`` points. They are ordered here as
    top-left, top-right, bottom-right, bottom-left before the transform,
    so the caller does not have to pre-sort them.

    It is **not** wired into :func:`apply_preprocessing`: automatic page
    corner detection is not implemented, and guessing corners wrongly
    warps text far more destructively than leaving the page alone.
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    # Order by the classic sum/difference trick.
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left

    (tl, tr, br, bl) = ordered
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 1 or height < 1:
        raise PreprocessingError("Degenerate quadrilateral for perspective transform")

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


def apply_preprocessing(
    image: np.ndarray, config: PreprocessConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the enabled operations in a fixed order.

    Returns ``(processed_image, report)``. The input array is not
    modified. ``report["applied"]`` lists the operations that actually
    ran and ``report["scale_factor"]`` is the factor needed to map
    coordinates from the processed image back to the original.
    """
    report: Dict[str, Any] = {
        "applied": [],
        "scale_factor": 1.0,
        "deskew_angle_degrees": 0.0,
        "config": config.to_dict(),
    }

    if config.is_identity:
        report["applied"] = []
        return image.copy(), report

    out = image.copy()

    # Order matters: geometry first (so contrast operations see the final
    # pixel grid), then contrast, then denoising last.
    if config.deskew:
        angle = estimate_skew_angle(out, config.max_deskew_degrees)
        if angle != 0.0:
            out = rotate_image(out, angle)
            report["applied"].append("deskew")
            report["deskew_angle_degrees"] = angle

    if config.upscale != 1.0:
        out = upscale(out, config.upscale)
        report["applied"].append(f"upscale x{config.upscale}")
        report["scale_factor"] = config.upscale

    if config.grayscale:
        out = to_grayscale(out)
        # Paddle expects 3 channels; convert back while keeping the
        # luminance-only content.
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        report["applied"].append("grayscale")

    if config.clahe:
        out = apply_clahe(out, config.clahe_clip_limit, config.clahe_tile_grid)
        report["applied"].append("clahe")

    if config.denoise:
        out = denoise(out, config.denoise_strength)
        report["applied"].append("denoise")

    return out, report

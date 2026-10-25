"""Word-level localisation from CTC alignment.

Why this module exists
----------------------
PP-OCRv5 detection returns **line** polygons. The Phonics Prism
projector needs to highlight **individual words**, so something has to
turn a line into words.

The obvious approach -- split the recognised string on spaces and cut
the line box into proportional slices -- is wrong, and specifically
wrong for Arabic. Arabic is cursive: glyph widths vary enormously,
letters join, diacritics stack above and below without consuming
horizontal space, and the proportion of a line a word occupies has no
fixed relationship to its character count. Slicing by character count
would produce boxes that look plausible and land on the wrong word.

So this module does not guess. It reads the recognition model's own
**CTC output matrix**, which assigns each decoded character to a
timestep, and each timestep corresponds to a horizontal strip of the
line image. That gives a measured, model-derived horizontal position
per character, which is then grouped into words at the space class.

What this is NOT
----------------
CTC alignment is an *approximation of* position, not a bounding box.
Three honest limitations, all of which are measured rather than
asserted (see ``scripts/evaluate_localization.py``):

1. **CTC peaks are spiky and can be offset.** The network emits a
   character at whichever timestep maximises the likelihood, which is
   not guaranteed to be the column where that glyph's ink sits. Peaks
   typically lag or lead the true position slightly.
2. **No vertical information.** A timestep is a full-height column, so
   every word in a line inherits the line's vertical extent. Word boxes
   are therefore *x-slices of the line polygon*, which is correct for
   horizontal text and wrong for anything else.
3. **A missed space merges two words.** Word boundaries come entirely
   from the model emitting the space class. If it does not, the two
   words become one span, and that span is still returned -- flagged,
   not hidden.

Every span produced here carries a ``confidence`` and a ``flags`` list,
and anything uncertain is marked rather than silently dropped.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import geometry as geo
from . import postprocessing as pp
from .utils import get_logger

LOGGER = get_logger(__name__)

#: Height the recognition model expects, from the model's own
#: ``RecResizeImg.image_shape`` ([3, 48, 320]).
REC_INPUT_HEIGHT = 48

#: Horizontal downsampling factor from model input to CTC timestep.
#:
#: VERIFIED EMPIRICALLY, not assumed: feeding widths from 16 to 800 px
#: at height 48 gives ``T == floor(W / 8)`` in every case. A
#: single-glyph position sweep then confirmed the mapping is linear
#: with this stride -- least-squares slope 7.98-8.01 with ~2px
#: residual for confidently-recognised glyphs.
#:
#: Consequence: the model never sees the final ``W mod 8`` columns of
#: the resized image. A character there is lost or folded into the
#: last timestep.
REC_WIDTH_STRIDE = 8

#: CTC blank occupies index 0; the charset follows; the trailing index
#: is the space class that PaddleOCR appends when ``use_space_char`` is
#: on. Verified against this model: 747 charset entries, 749 outputs.
BLANK_INDEX = 0

GRANULARITY_WORD = "word_ctc_aligned"


class WordSegmentationError(RuntimeError):
    """Raised when word segmentation cannot run as requested."""


@dataclasses.dataclass
class CharAlignment:
    """One decoded character and the image column range it came from."""

    char: str
    timestep: int
    confidence: float
    #: Horizontal extent within the *rectified line crop*, in pixels.
    x_start: float
    x_end: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class WordSpan:
    """A word, its text, and where it sits in the original image."""

    index: int
    text: str
    confidence: float
    #: Polygon in ORIGINAL image coordinates, 4 vertices. This is the
    #: box to use; it is vertically tightened when that succeeded.
    polygon: geo.Polygon
    bbox: Tuple[float, float, float, float]
    #: Extent inside the rectified line crop, kept for debugging.
    crop_x_start: float
    crop_x_end: float
    n_chars: int
    n_tashkeel: int
    flags: List[str]
    #: The ORIGINAL estimate before vertical tightening -- the CTC
    #: column range at full line height. Kept separately so that the
    #: effect of tightening stays auditable and a regression in the
    #: ink measurement cannot destroy the underlying geometry.
    polygon_untightened: geo.Polygon = dataclasses.field(default_factory=list)
    bbox_untightened: Optional[Tuple[float, float, float, float]] = None
    #: Outcome of the ink measurement: ok / no_ink / suspect_polarity /
    #: empty_window / disabled.
    vertical_source: str = "line_height"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "confidence": self.confidence,
            "polygon": self.polygon,
            "bbox": list(self.bbox),
            "polygon_untightened": self.polygon_untightened,
            "bbox_untightened": (
                list(self.bbox_untightened) if self.bbox_untightened else None
            ),
            "vertical_source": self.vertical_source,
            "granularity": GRANULARITY_WORD,
            "crop_x_start": self.crop_x_start,
            "crop_x_end": self.crop_x_end,
            "n_chars": self.n_chars,
            "n_tashkeel": self.n_tashkeel,
            "flags": self.flags,
        }


# ---------------------------------------------------------------------
# Line rectification
# ---------------------------------------------------------------------


def order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise WordSegmentationError(
            f"Line polygon must have 4 vertices to rectify, got {pts.shape[0]}"
        )
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def rectify_line(
    image: np.ndarray, polygon: Sequence[Sequence[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a line polygon to an axis-aligned crop.

    Returns ``(crop, inverse_matrix)``. ``inverse_matrix`` maps points
    from crop coordinates back to original image coordinates, which is
    what lets a word's column range become a polygon on the page even
    when the line was rotated or perspective-distorted.
    """
    quad = order_quad(polygon)
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if width < 2 or height < 2:
        raise WordSegmentationError(
            f"Degenerate line polygon: rectified size would be {width}x{height}"
        )
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, destination)
    crop = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, np.linalg.inv(matrix)


#: Above this fraction of "ink" pixels, the window is assumed to be
#: inverted (light text on a dark page) rather than densely printed.
#:
#: Measured: a white-on-black crop returned the FULL crop height with
#: no warning, because the dark background fell below the threshold
#: everywhere. That is a silently wrong box, which is worse than no
#: box. Normal dark-on-light printed text covers well under half the
#: pixels of its own bounding window, so 0.6 separates the two cases
#: with margin.
INK_POLARITY_LIMIT = 0.6


def ink_rows(
    crop: np.ndarray,
    x_start: float,
    x_end: float,
    threshold: int = 200,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Vertical extent of dark pixels within a column range of the crop.

    Returns ``(extent, status)``. ``extent`` is ``None`` whenever the
    measurement is not trustworthy, and ``status`` says why:

    ``"ok"``
        Ink found and the polarity looks like dark-on-light.
    ``"no_ink"``
        Nothing below the threshold. Correct answer for a blank slice,
        and also what very light print produces -- measured, print at
        grey >= 200 (contrast <= 55) falls back here.
    ``"suspect_polarity"``
        More than :data:`INK_POLARITY_LIMIT` of the window is below the
        threshold, which means the *background* is being measured, not
        the text. Returns ``None`` rather than a full-height box.
    ``"empty_window"``
        The column range is degenerate.

    CTC gives no vertical information, so without this a word box spans
    the whole line height. This measures where the ink actually sits
    using the image itself rather than an assumption -- but only when
    the measurement is credible, and it says so when it is not.

    Verified to capture marks **above** (fatha) and **below** (kasra)
    the glyph body, plus English ascenders, descenders and punctuation,
    with zero clipping. See ``tests/test_wordseg.py``.
    """
    height, width = crop.shape[:2]
    x0 = max(0, min(int(np.floor(x_start)), width - 1))
    x1 = max(x0 + 1, min(int(np.ceil(x_end)), width))
    window = crop[:, x0:x1]
    if window.size == 0:
        return None, "empty_window"
    grey = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY) if window.ndim == 3 else window
    mask = grey < threshold
    if not mask.any():
        return None, "no_ink"
    if mask.mean() > INK_POLARITY_LIMIT:
        return None, "suspect_polarity"
    rows = np.where(mask.any(axis=1))[0]
    return (float(rows.min()), float(rows.max() + 1)), "ok"


def crop_rect_to_image_polygon(
    x_start: float,
    x_end: float,
    height: int,
    inverse: np.ndarray,
    y_start: float = 0.0,
    y_end: Optional[float] = None,
) -> geo.Polygon:
    """Map a rectangle of the crop back to a polygon on the page."""
    if y_end is None:
        y_end = float(height - 1)
    corners = np.array(
        [
            [x_start, y_start],
            [x_end, y_start],
            [x_end, y_end],
            [x_start, y_end],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, inverse).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in mapped]


# ---------------------------------------------------------------------
# CTC recogniser
# ---------------------------------------------------------------------


class CTCAligner:
    """Runs the recognition model directly to recover per-character columns.

    PaddleOCR's own ``TextRecognition`` API returns only ``rec_text``
    and ``rec_score`` -- no positions -- so this loads the same
    ``inference.json``/``inference.pdiparams`` through
    ``paddle.inference`` and reads the raw CTC matrix instead.

    It is the *same weights* the normal pipeline uses. Nothing here is
    a different or substituted model.
    """

    def __init__(self, model_dir: Path, device: str = "gpu:0") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.labels = self._load_labels()
        self._predictor = self._build(device)

    # -- setup ---------------------------------------------------

    def _load_labels(self) -> List[str]:
        import yaml

        config_path = self.model_dir / "inference.yml"
        if not config_path.is_file():
            raise WordSegmentationError(
                f"Missing {config_path}. Run scripts/download_models.py first."
            )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        charset = _find_charset(config)
        if not charset:
            raise WordSegmentationError(
                f"Could not find a character dictionary in {config_path}"
            )
        # index 0 = CTC blank, 1..N = charset, N+1 = space.
        return ["\x00"] + list(charset) + [" "]

    def _build(self, device: str):
        from paddle import inference

        model_file = self.model_dir / "inference.json"
        params_file = self.model_dir / "inference.pdiparams"
        for path in (model_file, params_file):
            if not path.is_file():
                raise WordSegmentationError(f"Missing model file: {path}")

        config = inference.Config(str(model_file), str(params_file))
        config.disable_glog_info()
        if device.startswith("gpu"):
            index = int(device.split(":")[1]) if ":" in device else 0
            config.enable_use_gpu(512, index)
        else:
            config.disable_gpu()
        predictor = inference.create_predictor(config)
        self._input_name = predictor.get_input_names()[0]
        self._output_name = predictor.get_output_names()[0]
        return predictor

    # -- inference -----------------------------------------------

    def _preprocess(
        self, crop: np.ndarray, stretch_to: Optional[int]
    ) -> Tuple[np.ndarray, int]:
        """Resize to the model's input height, normalise, to NCHW.

        Returns ``(batch, resized_width)``. The resized width is needed
        by the caller to convert timesteps into crop columns, and
        deriving it from the output tensor instead would lose the
        0-7px remainder (see :data:`REC_WIDTH_STRIDE`).

        No right-padding. Padding to a fixed 320px width breaks the
        timestep-to-column mapping: the model emits characters inside
        the padded region, so a timestep no longer corresponds to a
        known column of the real image. Dynamic width keeps the mapping
        linear, which is the whole basis of this module.
        """
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        height, width = crop.shape[:2]
        if height < 2 or width < 2:
            raise WordSegmentationError(f"Crop too small: {width}x{height}")
        target_width = (
            int(stretch_to)
            if stretch_to
            else max(2, int(np.ceil(REC_INPUT_HEIGHT * width / float(height))))
        )
        resized = cv2.resize(
            crop, (target_width, REC_INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC
        ).astype("float32")
        normalised = (resized / 255.0 - 0.5) / 0.5
        return normalised.transpose(2, 0, 1)[None], target_width

    def logits(
        self, crop: np.ndarray, stretch_to: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Raw CTC matrix ``(timesteps, n_classes)`` and the resized width."""
        batch, resized_width = self._preprocess(crop, stretch_to)
        handle = self._predictor.get_input_handle(self._input_name)
        handle.reshape(batch.shape)
        handle.copy_from_cpu(batch)
        self._predictor.run()
        matrix = self._predictor.get_output_handle(self._output_name).copy_to_cpu()[0]
        return matrix, resized_width

    def align(
        self, crop: np.ndarray, stretch_to: Optional[int] = None
    ) -> List[CharAlignment]:
        """Greedy CTC decode, keeping each character's column range.

        The decoded characters come out in **image order (left to
        right)**, which for Arabic is the reverse of logical reading
        order. That is deliberate: this function reports geometry, and
        geometry is left-to-right regardless of script. Logical
        ordering is applied later, per word, by
        :func:`segment_words`.
        """
        matrix, resized_width = self.logits(crop, stretch_to)
        crop_width = crop.shape[1]

        # Timestep t covers resized columns [t*8, (t+1)*8). The width
        # stride is 8 exactly, verified by sweeping input widths:
        # T == floor(W / 8) for every width tested from 16 to 800.
        #
        # Using crop_width / T instead would be wrong whenever the
        # resized width is not a multiple of 8, because the model
        # simply does not see the final 0-7 columns. Measured on a
        # glyph-position sweep across non-multiple-of-8 widths, the
        # stride-8 mapping held a constant +0.50px error (sd 0.00)
        # while crop_width/T drifted to +3.6px and grew with width.
        scale = crop_width / float(resized_width)

        indices = matrix.argmax(axis=1)
        scores = matrix.max(axis=1)

        alignments: List[CharAlignment] = []
        previous = -1
        for step, index in enumerate(indices):
            if index != BLANK_INDEX and index != previous:
                label = (
                    self.labels[index] if index < len(self.labels) else "�"
                )
                alignments.append(
                    CharAlignment(
                        char=label,
                        timestep=int(step),
                        confidence=float(scores[step]),
                        x_start=step * REC_WIDTH_STRIDE * scale,
                        x_end=(step + 1) * REC_WIDTH_STRIDE * scale,
                    )
                )
            previous = index
        return alignments


def _find_charset(node: Any) -> Optional[List[str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            if "character" in str(key).lower() and isinstance(value, list):
                return value
            found = _find_charset(value)
            if found:
                return found
    return None


# ---------------------------------------------------------------------
# Word grouping
# ---------------------------------------------------------------------


def segment_words(
    alignments: Sequence[CharAlignment],
    *,
    rtl: bool,
    min_confidence: float = 0.0,
) -> List[Tuple[str, float, float, float, List[str], int]]:
    """Group aligned characters into words at the space class.

    Returns tuples of
    ``(text, confidence, x_start, x_end, flags, n_tashkeel)`` with
    ``text`` in **logical** order (reversed for RTL, since the
    alignments arrive in image order).

    Diacritics do not start or end a word: they are combining marks
    that sit on the preceding base letter and occupy the same columns.
    """
    words: List[Tuple[str, float, float, float, List[str], int]] = []
    current: List[CharAlignment] = []

    def flush() -> None:
        if not current:
            return
        chars = [a.char for a in current]
        text = "".join(reversed(chars)) if rtl else "".join(chars)
        confidences = [a.confidence for a in current]
        confidence = float(min(confidences))
        flags: List[str] = []
        if confidence < 0.5:
            flags.append("low_confidence")
        if len(current) == 1 and pp.is_tashkeel(current[0].char):
            flags.append("orphan_diacritic")
        words.append(
            (
                text,
                confidence,
                min(a.x_start for a in current),
                max(a.x_end for a in current),
                flags,
                sum(1 for c in chars if pp.is_tashkeel(c)),
            )
        )
        current.clear()

    for alignment in alignments:
        if alignment.char == " ":
            flush()
            continue
        if alignment.confidence < min_confidence:
            continue
        current.append(alignment)
    flush()
    return words


def locate_words_in_line(
    image: np.ndarray,
    line_polygon: Sequence[Sequence[float]],
    aligner: CTCAligner,
    *,
    rtl: bool,
    start_index: int = 0,
    stretch_to: Optional[int] = None,
    tighten_vertical: bool = True,
) -> List[WordSpan]:
    """Full pipeline: rectify a line, align, group, map back to the page."""
    crop, inverse = rectify_line(image, line_polygon)
    alignments = aligner.align(crop, stretch_to=stretch_to)
    if not alignments:
        return []

    height = crop.shape[0]
    image_h, image_w = image.shape[:2]
    spans: List[WordSpan] = []

    grouped = segment_words(alignments, rtl=rtl)
    for offset, (text, confidence, x0, x1, flags, n_marks) in enumerate(grouped):
        if x1 <= x0:
            flags = flags + ["degenerate_extent"]
            x1 = x0 + 1.0

        # The untightened estimate is always computed and kept, so the
        # effect of tightening stays auditable.
        raw_polygon = geo.clip_polygon(
            crop_rect_to_image_polygon(x0, x1, height, inverse),
            image_w, image_h,
        )
        raw_bbox = geo.clip_bbox(geo.polygon_to_bbox(raw_polygon), image_w, image_h)

        y0, y1 = 0.0, float(height - 1)
        vertical_source = "disabled"
        if tighten_vertical:
            rows, status = ink_rows(crop, x0, x1)
            vertical_source = status
            if rows is None:
                flags = flags + [f"vertical_{status}"]
            else:
                y0, y1 = rows
                flags = flags + ["vertically_tightened"]

        polygon = crop_rect_to_image_polygon(x0, x1, height, inverse, y0, y1)
        polygon = geo.clip_polygon(polygon, image_w, image_h)
        bbox = geo.clip_bbox(geo.polygon_to_bbox(polygon), image_w, image_h)
        if len(grouped) == 1 and " " not in text:
            # One span for the whole line can mean the model never
            # emitted a space. Flag it rather than presenting a
            # line-wide box as a word box.
            flags = flags + ["single_span_line"]
        spans.append(
            WordSpan(
                index=start_index + offset,
                text=text,
                confidence=confidence,
                polygon=polygon,
                bbox=bbox,
                polygon_untightened=raw_polygon,
                bbox_untightened=raw_bbox,
                vertical_source=vertical_source,
                crop_x_start=float(x0),
                crop_x_end=float(x1),
                n_chars=len(text),
                n_tashkeel=n_marks,
                flags=flags,
            )
        )
    return spans


# ---------------------------------------------------------------------
# Alternative baseline: projection-profile segmentation
# ---------------------------------------------------------------------


def segment_by_projection(
    crop: np.ndarray,
    *,
    threshold: int = 200,
    min_gap_px: Optional[int] = None,
    min_word_px: int = 4,
) -> List[Tuple[float, float]]:
    """Split a line crop into column runs separated by whitespace gaps.

    A deliberately simple alternative to CTC alignment, implemented so
    the comparison in ``docs/word_localization.md`` rests on a
    measurement instead of an argument. It uses no model at all: it
    projects ink onto the x-axis and cuts wherever the blank run is
    wide enough.

    The known objection for Arabic is that the script is cursive but
    not fully connected -- ``ا د ذ ر ز و`` do not join to the following
    letter, so word-sized blank gaps occur *inside* words. This
    function cannot tell those from real word boundaries, which is
    exactly what the benchmark measures.

    ``min_gap_px`` defaults to 40% of the crop height, which scales
    with the print size rather than assuming a fixed pixel count.

    Returns ``(x_start, x_end)`` pairs in crop coordinates.
    """
    height, width = crop.shape[:2]
    if height < 2 or width < 2:
        return []
    if min_gap_px is None:
        min_gap_px = max(2, int(round(height * 0.4)))

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mask = grey < threshold
    if mask.mean() > INK_POLARITY_LIMIT:
        # Same polarity guard as ink_rows: measuring the background
        # would return one run covering everything.
        return []
    ink_columns = mask.any(axis=0)
    if not ink_columns.any():
        return []

    runs: List[Tuple[float, float]] = []
    start: Optional[int] = None
    blank = 0
    for x in range(width):
        if ink_columns[x]:
            if start is None:
                start = x
            blank = 0
        else:
            if start is not None:
                blank += 1
                if blank >= min_gap_px:
                    end = x - blank + 1
                    if end - start >= min_word_px:
                        runs.append((float(start), float(end)))
                    start = None
                    blank = 0
    if start is not None and width - start >= min_word_px:
        runs.append((float(start), float(width)))
    return runs


def locate_words_by_projection(
    image: np.ndarray,
    line_polygon: Sequence[Sequence[float]],
    *,
    start_index: int = 0,
    tighten_vertical: bool = True,
) -> List[WordSpan]:
    """Word spans from projection profiles instead of CTC alignment.

    Produces no text -- it is a geometry-only baseline, so ``text`` is
    empty and every span is flagged ``projection_baseline``. Used by
    ``evaluate_localization.py --method projection``.
    """
    crop, inverse = rectify_line(image, line_polygon)
    height = crop.shape[0]
    image_h, image_w = image.shape[:2]

    spans: List[WordSpan] = []
    for offset, (x0, x1) in enumerate(segment_by_projection(crop)):
        raw_polygon = geo.clip_polygon(
            crop_rect_to_image_polygon(x0, x1, height, inverse), image_w, image_h
        )
        raw_bbox = geo.clip_bbox(geo.polygon_to_bbox(raw_polygon), image_w, image_h)

        y0, y1 = 0.0, float(height - 1)
        vertical_source = "disabled"
        flags = ["projection_baseline"]
        if tighten_vertical:
            rows, status = ink_rows(crop, x0, x1)
            vertical_source = status
            if rows is None:
                flags.append(f"vertical_{status}")
            else:
                y0, y1 = rows
                flags.append("vertically_tightened")

        polygon = geo.clip_polygon(
            crop_rect_to_image_polygon(x0, x1, height, inverse, y0, y1),
            image_w, image_h,
        )
        bbox = geo.clip_bbox(geo.polygon_to_bbox(polygon), image_w, image_h)
        spans.append(
            WordSpan(
                index=start_index + offset,
                text="",
                confidence=0.0,
                polygon=polygon,
                bbox=bbox,
                crop_x_start=float(x0),
                crop_x_end=float(x1),
                n_chars=0,
                n_tashkeel=0,
                flags=flags,
                polygon_untightened=raw_polygon,
                bbox_untightened=raw_bbox,
                vertical_source=vertical_source,
            )
        )
    return spans

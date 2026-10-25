"""Draw OCR regions over a copy of the input image.

The visualisation is a debugging aid. It is produced from the same
polygons that go into the JSON, but the JSON is the authoritative
record: labels here are truncated, reshaped for display and may be
repositioned to stay on canvas, so they must never be read back as
data.

The original image file is never modified -- drawing happens on a copy.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .utils import get_logger

LOGGER = get_logger(__name__)

#: Fonts shipped with Windows that cover Arabic. Checked in order.
_WINDOWS_ARABIC_FONTS = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\times.ttf",
)

_LINUX_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)

_COLOUR_BY_SCRIPT = {
    "arabic": (0, 165, 255),   # orange (BGR)
    "latin": (0, 200, 0),      # green
    "unknown": (160, 160, 160),
}


def find_arabic_font() -> Optional[str]:
    """Path to a system font that can render Arabic, or ``None``."""
    candidates = _WINDOWS_ARABIC_FONTS if platform.system() == "Windows" else _LINUX_FONTS
    for path in candidates:
        if Path(path).is_file():
            return path
    return None


def shape_for_display(text: str) -> str:
    """Reshape and bidi-reorder Arabic text for *rendering only*.

    PIL draws glyphs left to right in string order and does not apply
    the Arabic joining or bidirectional algorithms, so logical-order
    Arabic renders as disconnected, reversed letters. This function
    produces a display-ordered string purely for the image overlay.

    The returned string must never be stored or evaluated: it is in
    visual order with presentation-form glyphs, which is the wrong
    representation for text comparison.

    ``delete_harakat=False`` is essential. arabic-reshaper **deletes
    every diacritic by default**, so the library's out-of-the-box
    behaviour silently removes exactly what this project exists to
    measure. That default has already caused one wrong conclusion here:
    a vowelled test image rendered through the default shaper came out
    with no tashkeel on it at all, which made the recognition model
    look as though it were dropping diacritics it had never been shown.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshaper = arabic_reshaper.ArabicReshaper(
            configuration={"delete_harakat": False, "support_ligatures": True}
        )
        return get_display(reshaper.reshape(text))
    except ImportError:
        return text
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("Arabic shaping failed (%s); drawing raw string", exc)
        return text


def draw_regions(
    image: np.ndarray,
    regions: Sequence[Dict[str, Any]],
    *,
    draw_labels: bool = True,
    font_size: int = 18,
    max_label_chars: int = 28,
) -> np.ndarray:
    """Return a copy of ``image`` with region polygons and labels drawn."""
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for region in regions:
        polygon = np.asarray(region["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        colour = _COLOUR_BY_SCRIPT.get(
            region.get("dominant_script", "unknown"), _COLOUR_BY_SCRIPT["unknown"]
        )
        cv2.polylines(canvas, [polygon], isClosed=True, color=colour, thickness=2)
        x_min, y_min = int(region["bbox"][0]), int(region["bbox"][1])
        cv2.circle(canvas, (x_min, y_min), 4, colour, -1)

    if not draw_labels:
        return canvas

    return _draw_labels(canvas, regions, font_size, max_label_chars)


def _draw_labels(
    canvas: np.ndarray,
    regions: Sequence[Dict[str, Any]],
    font_size: int,
    max_label_chars: int,
) -> np.ndarray:
    """Overlay text labels using PIL so that Arabic glyphs render."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        LOGGER.warning("Pillow is not installed; drawing boxes without text labels.")
        return canvas

    font_path = find_arabic_font()
    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default()
        )
    except Exception as exc:  # pragma: no cover - font loading is env dependent
        LOGGER.warning("Could not load font %s (%s); using PIL default.", font_path, exc)
        font = ImageFont.load_default()

    if font_path is None:
        LOGGER.warning(
            "No Arabic-capable system font found; Arabic labels will not render "
            "correctly in the visualisation. The JSON output is unaffected."
        )

    pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image, "RGBA")
    height, width = canvas.shape[:2]

    for region in regions:
        text = region.get("text_raw", "")
        if not text:
            continue
        label = text[:max_label_chars] + ("..." if len(text) > max_label_chars else "")
        label = f"{region['index']}:{label} ({region['rec_score']:.2f})"
        display = shape_for_display(label)

        x_min, y_min = float(region["bbox"][0]), float(region["bbox"][1])
        try:
            box = draw.textbbox((0, 0), display, font=font)
            text_w, text_h = box[2] - box[0], box[3] - box[1]
        except Exception:
            text_w, text_h = len(display) * font_size // 2, font_size

        # Prefer above the region; drop below if that would clip the top.
        ty = y_min - text_h - 4
        if ty < 0:
            ty = min(y_min + 2, height - text_h - 2)
        tx = min(max(x_min, 0), max(width - text_w - 2, 0))

        colour = _COLOUR_BY_SCRIPT.get(
            region.get("dominant_script", "unknown"), _COLOUR_BY_SCRIPT["unknown"]
        )
        rgb = (colour[2], colour[1], colour[0])
        draw.rectangle(
            [tx - 2, ty - 2, tx + text_w + 4, ty + text_h + 4], fill=(0, 0, 0, 170)
        )
        draw.text((tx, ty), display, font=font, fill=rgb)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def legend_lines() -> List[str]:
    """Human-readable description of the visualisation colours."""
    return [
        "orange outline = region whose recognised text is mostly Arabic script",
        "green outline  = region whose recognised text is mostly Latin script",
        "grey outline   = region with no script-bearing characters",
        "filled dot     = the (x_min, y_min) corner of the axis-aligned bbox",
        "label          = index:text (recognition score), truncated for display",
    ]


#: Word spans are DERIVED geometry, not detector output. They are drawn
#: in a different colour and with a dashed outline so that a glance at
#: an overlay cannot confuse them with the detector's line polygons.
_WORD_COLOUR = (255, 120, 0)        # blue-ish (BGR)
_WORD_LOW_CONF_COLOUR = (0, 0, 255)  # red
_WORD_FLAGGED_COLOUR = (0, 215, 255)  # amber


def _dashed_polyline(canvas, points, colour, thickness=2, dash=9, gap=6) -> None:
    """Draw a dashed closed polygon, to mark geometry as derived."""
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    for i in range(len(pts)):
        start = pts[i].astype(float)
        end = pts[(i + 1) % len(pts)].astype(float)
        span = end - start
        length = float(np.hypot(*span))
        if length < 1e-6:
            continue
        step = span / length
        position = 0.0
        while position < length:
            a = start + step * position
            b = start + step * min(position + dash, length)
            cv2.line(
                canvas,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                colour,
                thickness,
            )
            position += dash + gap


def draw_word_spans(
    image: np.ndarray,
    words: Sequence[Dict[str, Any]],
    *,
    low_confidence: float = 0.5,
    draw_labels: bool = True,
    font_size: int = 16,
) -> np.ndarray:
    """Overlay CTC-derived word spans on a copy of ``image``.

    Drawn **dashed**, and in a different colour from
    :func:`draw_regions`, because these are not detector-produced word
    boxes -- they are inferred from recognition timesteps. The legend
    in :func:`word_legend_lines` states this, and the caller is
    expected to show it.

    Colour encodes trust: amber when the span carries any flag, red
    when its confidence is below ``low_confidence``, blue otherwise.
    """
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for word in words:
        confidence = float(word.get("confidence", 0.0))
        flags = word.get("flags") or []
        if confidence < low_confidence:
            colour = _WORD_LOW_CONF_COLOUR
        elif flags and flags != ["vertically_tightened"]:
            colour = _WORD_FLAGGED_COLOUR
        else:
            colour = _WORD_COLOUR
        _dashed_polyline(canvas, word["polygon"], colour)

    if not draw_labels:
        return canvas

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return canvas

    font_path = find_arabic_font()
    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default()
        )
    except Exception:  # pragma: no cover - font loading is env dependent
        font = ImageFont.load_default()

    pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image, "RGBA")
    height, width = canvas.shape[:2]

    for word in words:
        text = (word.get("text") or "").strip()
        if not text:
            continue
        confidence = float(word.get("confidence", 0.0))
        label = f"{shape_for_display(text)} {confidence:.2f}"
        x_min, y_max = float(word["bbox"][0]), float(word["bbox"][3])
        try:
            box = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = box[2] - box[0], box[3] - box[1]
        except Exception:
            text_w, text_h = len(label) * font_size // 2, font_size

        ty = min(y_max + 3, height - text_h - 2)
        tx = min(max(x_min, 0), max(width - text_w - 2, 0))
        colour = (
            _WORD_LOW_CONF_COLOUR if confidence < low_confidence else _WORD_COLOUR
        )
        draw.rectangle(
            [tx - 2, ty - 2, tx + text_w + 4, ty + text_h + 4], fill=(0, 0, 0, 180)
        )
        draw.text((tx, ty), label, font=font, fill=(colour[2], colour[1], colour[0]))

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def word_legend_lines() -> List[str]:
    """Wording that must accompany any word-span overlay."""
    return [
        "DASHED boxes are EXPERIMENTAL word spans DERIVED from CTC",
        "recognition timesteps. They are NOT detector-produced word",
        "boxes, and they are not validated on photographed pages.",
        "  blue  = ok",
        "  amber = carries a flag (see 'flags' in the JSON)",
        "  red   = confidence below threshold",
        "Solid boxes are the detector's own text-line regions.",
    ]

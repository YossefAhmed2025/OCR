"""Typographically correct Arabic rendering, via HarfBuzz.

Why this module exists
----------------------

The generator used to draw Arabic by running text through
``arabic_reshaper`` -- which rewrites letters into the legacy
presentation-form block U+FE70-U+FEFF -- and handing the result to
Pillow one codepoint at a time.

That approach cannot place a harakah. A presentation form carries the
letter's joined *shape*, but nothing tells the renderer that a fatha
belongs centred above the letter it follows: Pillow advances the pen
and draws the mark beside the letter instead. Rendered side by side,
only Tahoma and Arial survived it; Amiri, Traditional Arabic, Andalus
and Sakkal Majalla all produced marks detached and in the wrong place
(docs/baseline_findings.md section 4b).

Since the whole project is about reading tashkeel, a training set
drawn that way teaches the model the wrong thing -- and the faces that
misrender are exactly the Naskh ones printed books use.

HarfBuzz does what a typesetter does: it reads the font's own GSUB and
GPOS tables, substitutes the correct contextual glyphs, and returns an
offset for every mark relative to its base letter. FreeType then
rasterises those glyph ids at those offsets. No presentation forms, no
reshaper, and marks land where the type designer put them.

    from phonics_ocr.shaping import render_line
    image = render_line("كَتَبَ الْوَلَدُ", font_path, 64)

The text passed in is logical order. Direction is set explicitly
rather than guessed, so this does not implement bidi: a line mixing
scripts must be split by the caller.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

#: 26.6 fixed point: HarfBuzz and FreeType report positions in units
#: of 1/64 px.
F26_6 = 64.0


class ShapingError(RuntimeError):
    """Raised when a font cannot be loaded or shaped with."""


@functools.lru_cache(maxsize=32)
def _hb_font(font_path: str, size: int):
    import uharfbuzz as hb

    try:
        blob = hb.Blob.from_file_path(font_path)
        face = hb.Face(blob)
        font = hb.Font(face)
    except Exception as exc:  # pragma: no cover - depends on the file
        raise ShapingError(f"could not load {font_path}: {exc}") from exc
    font.scale = (size * int(F26_6), size * int(F26_6))
    return font


@functools.lru_cache(maxsize=32)
def _ft_face(font_path: str, size: int):
    import freetype

    try:
        face = freetype.Face(font_path)
    except Exception as exc:  # pragma: no cover - depends on the file
        raise ShapingError(f"could not load {font_path}: {exc}") from exc
    face.set_char_size(size * int(F26_6))
    return face


def shape(text: str, font_path: Path, size: int, rtl: bool = True):
    """Glyph ids and pen positions for one run of text.

    Returns a list of ``(glyph_id, x_offset, y_offset, x_advance)`` in
    pixels, in visual order -- HarfBuzz reverses an RTL run itself, so
    the caller passes logical order and does not pre-reverse.
    """
    import uharfbuzz as hb

    if not text:
        # An empty buffer leaves glyph_infos as None rather than an
        # empty list, which fails far from the cause.
        raise ShapingError("nothing to shape: the text is empty")

    font = _hb_font(str(font_path), size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = "rtl" if rtl else "ltr"
    buf.script = "Arab" if rtl else "Latn"
    buf.language = "ar" if rtl else "en"
    hb.shape(font, buf)

    out = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        out.append(
            (
                info.codepoint,            # glyph id, not a character
                pos.x_offset / F26_6,
                pos.y_offset / F26_6,
                pos.x_advance / F26_6,
            )
        )
    return out


def measure(text: str, font_path: Path, size: int, rtl: bool = True) -> Tuple[int, int, int]:
    """Advance width, ascent and descent in pixels for a shaped run."""
    face = _ft_face(str(font_path), size)
    width = sum(g[3] for g in shape(text, font_path, size, rtl))
    ascent = face.size.ascender / F26_6
    descent = -face.size.descender / F26_6
    return int(round(width)), int(round(ascent)), int(round(descent))


def render_line(
    text: str,
    font_path: Path,
    size: int,
    padding: int = 12,
    fg: int = 20,
    bg: int = 255,
    rtl: bool = True,
    min_height: Optional[int] = None,
):
    """Render one line and return it as a PIL grayscale image.

    The canvas is sized from the font's own ascent and descent plus
    padding, then cropped to the ink, because an Arabic line with
    tashkeel routinely overshoots the nominal ascent and a fixed box
    clips the marks off the top -- which is the failure this module
    exists to avoid.
    """
    import freetype
    from PIL import Image

    face = _ft_face(str(font_path), size)
    glyphs = shape(text, font_path, size, rtl)
    if not glyphs:
        raise ShapingError(f"nothing shaped for {text!r}")

    width = sum(g[3] for g in glyphs)
    ascent = face.size.ascender / F26_6
    descent = -face.size.descender / F26_6

    # Generous margins: marks sit above the ascent, and the crop at the
    # end removes whatever is not used.
    slack = int(size * 1.2)
    W = int(width) + padding * 2 + slack
    H = int(ascent + descent) + padding * 2 + slack * 2
    canvas = np.full((H, W), bg, dtype=np.int32)

    pen_x = float(padding + slack / 2)
    base_y = float(padding + slack + ascent)

    for gid, dx, dy, adv in glyphs:
        face.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        bitmap = face.glyph.bitmap
        gw, gh = bitmap.width, bitmap.rows
        if gw and gh:
            buf = np.array(bitmap.buffer, dtype=np.uint8).reshape(gh, bitmap.pitch)[:, :gw]
            x0 = int(round(pen_x + dx + face.glyph.bitmap_left))
            y0 = int(round(base_y - dy - face.glyph.bitmap_top))
            x1, y1 = x0 + gw, y0 + gh
            if x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H:
                # Alpha-composite so overlapping marks darken rather
                # than erase the letter underneath.
                alpha = buf.astype(np.int32)
                region = canvas[y0:y1, x0:x1]
                canvas[y0:y1, x0:x1] = (region * (255 - alpha) + fg * alpha) // 255
        pen_x += adv

    arr = np.clip(canvas, 0, 255).astype(np.uint8)
    ink = arr < (bg - 30)
    if ink.any():
        ys, xs = np.where(ink)
        y0 = max(0, ys.min() - padding)
        y1 = min(arr.shape[0], ys.max() + padding + 1)
        x0 = max(0, xs.min() - padding)
        x1 = min(arr.shape[1], xs.max() + padding + 1)
        arr = arr[y0:y1, x0:x1]

    if min_height and arr.shape[0] < min_height:
        pad = min_height - arr.shape[0]
        top = pad // 2
        arr = np.pad(arr, ((top, pad - top), (0, 0)), constant_values=bg)

    return Image.fromarray(arr, mode="L")


def font_shapes_marks(font_path: Path, size: int = 64) -> bool:
    """Whether a font positions a harakah above its base letter.

    Shapes a single vowelled letter and checks that the mark glyph
    carries a non-zero vertical offset or sits in the upper half of the
    ink. A font whose GPOS cannot place marks fails here, which is the
    check the old base-codepoint test could not make.
    """
    try:
        glyphs = shape("بَ", font_path, size)   # baa + fatha
    except ShapingError:
        return False
    if len(glyphs) < 2:
        # One glyph means the mark was dropped or merged away.
        return False
    # A correctly placed harakah is a combining glyph: it advances the
    # pen by nothing and is positioned by GPOS relative to its base.
    # Order is visual, and RTL puts the base last, so look for any
    # zero-advance glyph rather than assuming a position.
    return any(advance == 0 for _, _, _, advance in glyphs)

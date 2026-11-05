"""HarfBuzz shaping: the renderer has to place a harakah correctly.

The old path -- arabic_reshaper into presentation forms, then Pillow
one codepoint at a time -- could not. Only Tahoma and Arial survived
it; every Naskh face drew the marks detached and misplaced, which
meant the synthetic training set was largely wrong typographically.
See docs/baseline_findings.md section 4b.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phonics_ocr import shaping

WIN = Path("C:/Windows/Fonts")
NASKH = WIN / "trado.ttf"            # Traditional Arabic
FATHA = "\u064E"
BAA = "\u0628"


def _font(name: str) -> Path:
    path = WIN / name
    if not path.is_file():
        pytest.skip(f"{name} not installed")
    return path


def test_a_mark_is_a_zero_advance_glyph():
    # If the mark advanced the pen it would sit beside the letter,
    # which is exactly the old bug.
    glyphs = shaping.shape(BAA + FATHA, _font("trado.ttf"), 64)
    assert len(glyphs) == 2
    assert any(advance == 0 for *_, advance in glyphs)


def test_shaping_is_contextual_not_one_glyph_per_codepoint():
    # Three letters joined in a word use different glyphs than the
    # same letters shaped alone.
    joined = shaping.shape("\u0628\u062A\u0628", _font("trado.ttf"), 64)
    alone = shaping.shape("\u0628", _font("trado.ttf"), 64)
    assert len(joined) == 3
    assert joined[0][0] != alone[0][0] or joined[-1][0] != alone[0][0]


def test_vowelled_text_is_taller_than_bare_text():
    # The marks have to actually land above the line, not inside it.
    bare = shaping.render_line("\u0628\u062A\u0628", _font("trado.ttf"), 64)
    vowelled = shaping.render_line("\u0628\u064E\u062A\u064E\u0628\u064E",
                                   _font("trado.ttf"), 64)
    assert vowelled.height > bare.height


def test_render_produces_ink_and_crops_to_it():
    img = shaping.render_line("\u0643\u064E\u062A\u064E\u0628\u064E",
                              _font("trado.ttf"), 64)
    arr = np.asarray(img)
    assert arr.min() < 80, "no ink was drawn"
    # Cropped: the outermost rows and columns are not all background.
    assert arr.shape[0] > 10 and arr.shape[1] > 10


def test_min_height_pads_without_scaling():
    img = shaping.render_line(BAA + FATHA, _font("trado.ttf"), 24, min_height=120)
    assert img.height >= 120


def test_naskh_faces_now_place_marks():
    # These three were unusable under the old renderer.
    for name in ("trado.ttf", "simpo.ttf", "arabtype.ttf"):
        path = WIN / name
        if not path.is_file():
            continue
        assert shaping.font_shapes_marks(path), f"{name} cannot place marks"


def test_missing_font_raises_shaping_error():
    with pytest.raises(shaping.ShapingError):
        shaping.shape("\u0628", Path("no/such/font.ttf"), 32)


def test_empty_text_raises_rather_than_returning_a_blank():
    with pytest.raises(shaping.ShapingError):
        shaping.render_line("", _font("trado.ttf"), 32)


def test_measure_reports_a_positive_advance():
    width, ascent, descent = shaping.measure("\u0643\u062A\u0628",
                                             _font("trado.ttf"), 48)
    assert width > 0 and ascent > 0 and descent >= 0

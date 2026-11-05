"""Font weighting: printed Arabic is Naskh, so the training set must be.

The first fine-tune sampled every discovered font uniformly and then
read Naskh 5.5x worse than the screen faces it had seen most
(docs/baseline_findings.md section 4b). These tests pin the fix.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phonics_ocr import synth


def test_monospaced_and_non_arabic_faces_are_dropped():
    fonts = [Path("C:/Windows/Fonts")/n for n in
             ("cour.ttf", "DejaVuSansMono.ttf", "UrdType.ttf", "trado.ttf")]
    kept, weights = synth.weight_fonts(fonts)
    assert [p.name for p in kept] == ["trado.ttf"]
    assert weights == [synth.PRINT_NASKH_WEIGHTS["trado.ttf"]]


def test_naskh_outweighs_screen_faces():
    fonts = [Path("f")/n for n in ("trado.ttf", "arial.ttf", "segoeui.ttf")]
    kept, weights = synth.weight_fonts(fonts)
    by_name = dict(zip((p.name for p in kept), weights))
    assert by_name["trado.ttf"] > by_name["arial.ttf"]
    assert by_name["arial.ttf"] == synth.DEFAULT_FONT_WEIGHT


def test_unknown_face_keeps_the_default_weight():
    kept, weights = synth.weight_fonts([Path("x/SomeNewFont.ttf")])
    assert weights == [synth.DEFAULT_FONT_WEIGHT]


def test_everything_excluded_raises_rather_than_returning_empty():
    # An empty pool would make rng.choices raise far from the cause.
    with pytest.raises(synth.SynthError):
        synth.weight_fonts([Path("a/cour.ttf"), Path("a/DejaVuSansMono.ttf")])


def test_naskh_share_of_a_realistic_pool_is_a_majority():
    pool = [Path("f")/n for n in (
        "trado.ttf", "tradbdo.ttf", "andlso.ttf", "majalla.ttf",
        "Amiri-Regular.ttf", "Amiri-Bold.ttf",
        "arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf",
        "tahoma.ttf", "tahomabd.ttf", "times.ttf", "calibri.ttf",
    )]
    kept, weights = synth.weight_fonts(pool)
    naskh = sum(w for p, w in zip(kept, weights)
                if p.name in synth.PRINT_NASKH_WEIGHTS)
    assert naskh / sum(weights) > 0.5


def test_weight_tables_do_not_overlap():
    # A face in both tables would be silently dropped despite a weight.
    assert not (set(synth.PRINT_NASKH_WEIGHTS) & synth.NON_PRINT_FONTS)


def test_shipped_naskh_fonts_render_the_shaped_forms():
    shipped = sorted(synth.PROJECT_FONT_DIR.glob("*.ttf"))
    if not shipped:
        pytest.skip("no project fonts installed")
    for path in shipped:
        assert synth.font_covers_shaped(path), f"{path.name} renders tofu"
        assert path.name in synth.PRINT_NASKH_WEIGHTS, (
            f"{path.name} is shipped but carries no weight"
        )

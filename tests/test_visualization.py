"""Visualisation tests.

The overlay itself is cosmetic, but one property is not: the display
shaper must not destroy diacritics. arabic-reshaper deletes every
harakat by default, and that default has already produced a wrong
experimental conclusion in this project -- a "vowelled" test image
rendered through it carried no tashkeel at all, which made the
recognition model look as though it were dropping marks it had never
been shown.
"""

import pytest

from phonics_ocr import postprocessing as pp
from phonics_ocr.visualization import find_arabic_font, legend_lines, shape_for_display

KATABA_VOWELLED = "\u0643\u064e\u062a\u064e\u0628\u064e"
WALAD_VOWELLED = "\u0627\u0644\u0648\u064e\u0644\u064e\u062f\u064f"

reshaper = pytest.importorskip("arabic_reshaper")


class TestDisplayShaping:
    def test_shaping_preserves_every_diacritic(self):
        shaped = shape_for_display(KATABA_VOWELLED)
        assert pp.count_tashkeel(shaped) == pp.count_tashkeel(KATABA_VOWELLED) == 3

    def test_shaping_preserves_diacritics_in_a_longer_phrase(self):
        phrase = f"{KATABA_VOWELLED} {WALAD_VOWELLED}"
        assert pp.count_tashkeel(shape_for_display(phrase)) == pp.count_tashkeel(phrase)

    def test_shaping_does_not_add_diacritics(self):
        plain = "\u0643\u062a\u0628"
        assert pp.count_tashkeel(shape_for_display(plain)) == 0

    def test_shaping_leaves_latin_unchanged(self):
        assert shape_for_display("Hello world") == "Hello world"

    def test_shaping_changes_arabic_order_for_display(self):
        # It must actually do something -- a no-op would mean the
        # overlay renders unjoined, reversed glyphs.
        assert shape_for_display(KATABA_VOWELLED) != KATABA_VOWELLED


class TestLegend:
    def test_legend_mentions_both_scripts(self):
        text = " ".join(legend_lines()).lower()
        assert "arabic" in text and "latin" in text


class TestFontLookup:
    def test_font_lookup_returns_a_path_or_none(self):
        font = find_arabic_font()
        assert font is None or font.endswith(".ttf")

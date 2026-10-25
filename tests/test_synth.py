"""Synthetic training-data generator tests.

Three conventions are load-bearing and all three failed silently
during development, so each has an asserting test:

1. Labels are **visual order** (bidi), matching the raw CTC sequence.
2. Labels use **base Arabic characters**, never presentation forms --
   labelling with reshaped glyphs added ~80 phantom classes to the
   dictionary.
3. Rendering **preserves every diacritic** -- arabic-reshaper deletes
   them by default.
"""

import random

import pytest

from phonics_ocr import postprocessing as pp
from phonics_ocr import synth

KATABA = "\u0643\u064e\u062a\u064e\u0628\u064e"
ZAHRA = "\u0632\u064e\u0647\u0652\u0631\u064e\u0629"


def is_presentation_form(ch: str) -> bool:
    code = ord(ch)
    return 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFF


class TestVisualLabel:
    def test_reverses_logical_order(self):
        label = synth.visual_label(ZAHRA)
        assert label != ZAHRA
        assert pp.strip_tashkeel(label)[0] == pp.strip_tashkeel(ZAHRA)[-1]

    def test_uses_base_characters_not_presentation_forms(self):
        # The bug this guards: reshaped glyphs are display codes the
        # recogniser never predicts, and they inflate the dictionary.
        for text in (KATABA, ZAHRA, "\u0645\u064f\u0639\u064e\u0644\u0651\u0650\u0645"):
            assert not any(is_presentation_form(c) for c in synth.visual_label(text))

    def test_preserves_every_diacritic(self):
        for text in (KATABA, ZAHRA):
            assert pp.count_tashkeel(synth.visual_label(text)) == pp.count_tashkeel(text)

    def test_preserves_character_count(self):
        assert len(synth.visual_label(KATABA)) == len(KATABA)

    def test_latin_is_left_alone(self):
        assert synth.visual_label("cat") == "cat"

    def test_empty_string(self):
        assert synth.visual_label("") == ""


class TestVowelize:
    def test_adds_diacritics(self):
        rng = random.Random(1)
        out = synth.vowelize("\u0643\u062a\u0628", rng)
        assert pp.count_tashkeel(out) > 0

    def test_keeps_the_consonant_skeleton(self):
        rng = random.Random(2)
        word = "\u0643\u062a\u0628"
        assert pp.strip_tashkeel(synth.vowelize(word, rng)) == word

    def test_tanween_only_ever_ends_a_word(self):
        rng = random.Random(3)
        for _ in range(60):
            out = synth.vowelize("\u0643\u062a\u0628\u0645", rng)
            marks = pp.extract_tashkeel_positions(out)
            base_count = len(pp.strip_tashkeel(out))
            for index, mark in marks:
                if mark in synth.TANWEEN:
                    assert index == base_count - 1

    def test_sukun_never_on_the_first_letter(self):
        rng = random.Random(4)
        for _ in range(60):
            out = synth.vowelize("\u0643\u062a\u0628\u0645", rng)
            for index, mark in pp.extract_tashkeel_positions(out):
                if mark == synth.SUKUN:
                    assert index > 0

    def test_long_alif_carries_no_vowel(self):
        rng = random.Random(5)
        for _ in range(40):
            out = synth.vowelize("\u0628\u0627\u0628", rng)
            marks = dict(
                (i, m) for i, m in pp.extract_tashkeel_positions(out)
            )
            assert 1 not in marks  # the alif at index 1

    def test_is_deterministic_for_a_seed(self):
        a = synth.vowelize("\u0643\u062a\u0628", random.Random(9))
        b = synth.vowelize("\u0643\u062a\u0628", random.Random(9))
        assert a == b


class TestLineGeneration:
    def test_respects_max_words(self):
        rng = random.Random(6)
        for _ in range(30):
            line = synth.make_line(rng, max_words=3)
            assert 1 <= len(line.split()) <= 3

    def test_lines_carry_diacritics(self):
        rng = random.Random(7)
        lines = [synth.make_line(rng) for _ in range(40)]
        assert all(pp.count_tashkeel(line) > 0 for line in lines)

    def test_random_words_are_arabic(self):
        rng = random.Random(8)
        word = synth.random_word(rng, length=4)
        assert all(pp.is_arabic_char(c) for c in word)

    def test_seed_words_all_vowelled(self):
        # A curated word with no marks would quietly dilute the data.
        for word in synth.SEED_WORDS:
            assert pp.count_tashkeel(word) > 0, word

    def test_seed_words_use_base_characters(self):
        for word in synth.SEED_WORDS:
            assert not any(is_presentation_form(c) for c in word)


class TestDictionary:
    def test_covers_every_label_character(self):
        labels = ["\u0643\u062a\u0628", "\u0645\u0646"]
        chars = set(synth.build_dictionary(labels))
        assert set("".join(labels)) <= chars

    def test_warm_start_keeps_base_order(self):
        base = ["a", "b", "c"]
        out = synth.build_dictionary(["cd"], base=base)
        assert out[:3] == base
        assert "d" in out

    def test_no_duplicates(self):
        out = synth.build_dictionary(["aab", "bb"], base=["a"])
        assert len(out) == len(set(out))

    def test_labels_within_base_add_nothing(self):
        base = list("\u0643\u062a\u0628")
        out = synth.build_dictionary(["\u0643\u062a\u0628"], base=base)
        assert out == base


class TestRenderConfigValidation:
    def test_rejects_tiny_height(self):
        with pytest.raises(synth.SynthError, match="height"):
            synth.RenderConfig(height=4)

    def test_rejects_bad_font_range(self):
        with pytest.raises(synth.SynthError, match="font_size_range"):
            synth.RenderConfig(font_size_range=(40, 10))

    def test_defaults_match_model_input_height(self):
        assert synth.RenderConfig().height == 48


class TestFontCoverage:
    """Base-codepoint coverage is not enough; shaped forms are drawn."""

    def test_probe_strings_are_vowelled(self):
        for probe in synth._COVERAGE_PROBES:
            assert pp.count_tashkeel(probe) > 0

    def test_probes_cover_the_tracked_confusables(self):
        joined = "".join(synth._COVERAGE_PROBES)
        for ch in ("\u0629", "\u0626", "\u0623", "\u0625", "\u0651"):
            assert ch in joined

    def test_rejects_a_latin_only_font(self):
        from pathlib import Path

        candidate = Path(r"C:\Windows\Fonts\BAUHS93.TTF")
        if not candidate.is_file():
            pytest.skip("font not present on this machine")
        assert synth.font_covers_shaped(candidate) is False

    def test_missing_font_is_rejected_not_raised(self):
        from pathlib import Path

        assert synth.font_covers_shaped(Path("no_such_font.ttf")) is False


@pytest.fixture(scope="module")
def font():
    fonts = synth.find_vowel_capable_fonts()
    if not fonts:
        pytest.skip("no Arabic-capable font on this machine")
    return fonts[0]


class TestRendering:

    def test_produces_an_image_at_configured_height(self, font):
        rng = random.Random(11)
        config = synth.RenderConfig(height=48)
        image, _ = synth.render_line(KATABA, font, config, rng)
        assert image.shape[0] == 48
        assert image.shape[1] > 8

    def test_label_is_visual_base_form(self, font):
        rng = random.Random(12)
        _, label = synth.render_line(ZAHRA, font, synth.RenderConfig(), rng)
        assert label == synth.visual_label(ZAHRA)
        assert not any(is_presentation_form(c) for c in label)

    def test_label_keeps_all_diacritics(self, font):
        rng = random.Random(13)
        _, label = synth.render_line(ZAHRA, font, synth.RenderConfig(), rng)
        assert pp.count_tashkeel(label) == pp.count_tashkeel(ZAHRA)

    def test_rendered_image_is_not_blank(self, font):
        rng = random.Random(14)
        config = synth.RenderConfig(
            blur_prob=0, noise_prob=0, jpeg_prob=0, gradient_prob=0
        )
        image, _ = synth.render_line(KATABA, font, config, rng)
        # Real ink must be present; a tofu-only render would still vary,
        # so also require a reasonable dark-pixel share.
        assert image.std() > 5
        assert (image < 128).mean() > 0.01

    def test_augmentation_keeps_the_image_usable(self, font):
        rng = random.Random(15)
        image, _ = synth.render_line(KATABA, font, synth.RenderConfig(), rng)
        assert image.dtype.name == "uint8"
        assert 0 < image.mean() < 255

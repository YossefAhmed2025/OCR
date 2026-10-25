"""Text normalisation, diacritic preservation and region-merging tests.

The central guarantee under test: nothing in the default path removes
Arabic diacritics, and nothing anywhere adds them.
"""

import pytest

from phonics_ocr import postprocessing as pp

FATHA = "َ"
DAMMA = "ُ"
KASRA = "ِ"
SHADDA = "ّ"
FATHATAN = "ً"
SUKUN = "ْ"
DAGGER_ALIF = "ٰ"
TATWEEL = "ـ"

KATABA_VOWELLED = "كَتَبَ"
KATABA_PLAIN = "كتب"
MADRASA = "مدرسة"  # madrasa, ends in taa marbuta


class TestCharacterClassification:
    @pytest.mark.parametrize("ch", ["ا", "ب", "ة", FATHA, "ٱ"])
    def test_arabic_characters_detected(self, ch):
        assert pp.is_arabic_char(ch)

    @pytest.mark.parametrize("ch", ["a", "Z", "1", " ", "."])
    def test_non_arabic_not_detected(self, ch):
        assert not pp.is_arabic_char(ch)

    @pytest.mark.parametrize(
        "mark", [FATHA, DAMMA, KASRA, SHADDA, FATHATAN, SUKUN, DAGGER_ALIF]
    )
    def test_tashkeel_recognised(self, mark):
        assert pp.is_tashkeel(mark)

    def test_quranic_marks_recognised(self):
        assert pp.is_tashkeel("ۖ")

    @pytest.mark.parametrize("ch", ["ك", "ا", TATWEEL, "a"])
    def test_base_letters_are_not_tashkeel(self, ch):
        assert not pp.is_tashkeel(ch)

    def test_tatweel_is_not_a_diacritic(self):
        # Tatweel is typographic stretching, not a vowel mark; treating
        # it as tashkeel would corrupt the diacritic counts.
        assert not pp.is_tashkeel(TATWEEL)


class TestDiacriticCounting:
    def test_count(self):
        assert pp.count_tashkeel(KATABA_VOWELLED) == 3

    def test_count_on_plain_text_is_zero(self):
        assert pp.count_tashkeel(KATABA_PLAIN) == 0

    def test_has_tashkeel(self):
        assert pp.has_tashkeel(KATABA_VOWELLED)
        assert not pp.has_tashkeel(KATABA_PLAIN)
        assert not pp.has_tashkeel("hello world")

    def test_positions_are_indexed_against_stripped_text(self):
        assert pp.extract_tashkeel_positions(KATABA_VOWELLED) == [
            (0, FATHA),
            (1, FATHA),
            (2, FATHA),
        ]

    def test_leading_mark_gets_index_minus_one(self):
        assert pp.extract_tashkeel_positions(FATHA + "ك") == [(-1, FATHA)]

    def test_stacked_marks_share_a_base_index(self):
        # shadda + fatha both sit on the same consonant.
        assert pp.extract_tashkeel_positions("م" + SHADDA + FATHA) == [
            (0, SHADDA),
            (0, FATHA),
        ]


class TestDiacriticPreservation:
    """The core safety property of this project."""

    def test_default_normalisation_keeps_every_diacritic(self):
        out = pp.normalise_text(KATABA_VOWELLED)
        assert pp.count_tashkeel(out) == 3
        assert out == KATABA_VOWELLED

    def test_nfc_does_not_strip_combining_vowel_marks(self):
        # Fatha and friends have no precomposed form, so NFC must leave
        # them in place. If this ever fails, every diacritic metric in
        # the project is silently wrong.
        assert pp.normalise_unicode(KATABA_VOWELLED, "NFC") == KATABA_VOWELLED
        assert pp.count_tashkeel(pp.normalise_unicode(KATABA_VOWELLED, "NFC")) == 3

    def test_nfd_then_nfc_round_trips(self):
        decomposed = pp.normalise_unicode(KATABA_VOWELLED, "NFD")
        assert pp.normalise_unicode(decomposed, "NFC") == KATABA_VOWELLED

    def test_nfc_composes_alif_plus_combining_hamza(self):
        # U+0627 + U+0654 is canonically equivalent to U+0623.
        assert pp.normalise_unicode("أ", "NFC") == "أ"

    def test_whitespace_collapse_keeps_diacritics(self):
        messy = f"  {KATABA_VOWELLED}   {KATABA_VOWELLED}  "
        out = pp.normalise_text(messy)
        assert pp.count_tashkeel(out) == 6
        assert out == f"{KATABA_VOWELLED} {KATABA_VOWELLED}"

    def test_tatweel_removal_keeps_diacritics(self):
        text = "ك" + TATWEEL + FATHA + "ت"
        out = pp.normalise_text(text, remove_tatweel=True)
        assert TATWEEL not in out
        assert pp.count_tashkeel(out) == 1

    def test_normalisation_never_adds_a_diacritic(self):
        for text in [KATABA_PLAIN, MADRASA, "كتاب", "hello"]:
            assert pp.count_tashkeel(pp.normalise_text(text)) == 0

    def test_stripping_is_opt_in_only(self):
        assert pp.count_tashkeel(pp.normalise_text(KATABA_VOWELLED)) == 3
        assert (
            pp.count_tashkeel(pp.normalise_text(KATABA_VOWELLED, remove_tashkeel=True))
            == 0
        )

    def test_strip_tashkeel_leaves_the_skeleton(self):
        assert pp.strip_tashkeel(KATABA_VOWELLED) == KATABA_PLAIN

    def test_strip_tashkeel_is_idempotent(self):
        once = pp.strip_tashkeel(KATABA_VOWELLED)
        assert pp.strip_tashkeel(once) == once


class TestOptionalFolds:
    def test_alif_folding_is_off_by_default(self):
        assert pp.normalise_text("أحمد") == "أحمد"

    def test_alif_folding_when_requested(self):
        assert pp.fold_alif("آأإٱ") == "ا" * 4

    def test_taa_marbuta_folding(self):
        assert pp.fold_taa_marbuta(MADRASA).endswith("ه")

    def test_alif_maqsura_folding(self):
        assert pp.fold_alif_maqsura("على") == "علي"

    def test_arabic_digits_map_to_ascii(self):
        assert pp.map_arabic_digits("١٢٣") == "123"

    def test_extended_arabic_digits_map_to_ascii(self):
        assert pp.map_arabic_digits("۴۵") == "45"

    def test_digit_mapping_is_off_by_default(self):
        assert pp.normalise_text("١٢") == "١٢"

    def test_nfkc_is_rejected_from_being_silent(self):
        # NFKC is allowed but must be asked for by name.
        assert pp.normalise_unicode("ﻟ", "NFKC") != "ﻟ"

    def test_unknown_normalisation_form_raises(self):
        with pytest.raises(ValueError, match="Unknown normalisation form"):
            pp.normalise_unicode("x", "NFZ")


class TestScriptDetection:
    def test_pure_arabic(self):
        assert pp.dominant_script(KATABA_VOWELLED) == "arabic"

    def test_pure_latin(self):
        assert pp.dominant_script("Hello world") == "latin"

    def test_digits_only_is_unknown(self):
        # Digits belong to neither script and must not cast a vote.
        assert pp.dominant_script("12345") == "unknown"

    def test_punctuation_only_is_unknown(self):
        assert pp.dominant_script("...!?") == "unknown"

    def test_empty_string_is_unknown(self):
        assert pp.dominant_script("") == "unknown"

    def test_majority_wins(self):
        assert pp.dominant_script(f"{KATABA_VOWELLED} ok") == "arabic"
        assert pp.dominant_script("ك understanding") == "latin"

    def test_profile_counts_each_class(self):
        profile = pp.script_profile("ab كتب 12 !")
        assert profile == {"arabic": 3, "latin": 2, "digit": 2, "other": 1}

    def test_profile_ignores_whitespace(self):
        assert pp.script_profile("   ") == {
            "arabic": 0,
            "latin": 0,
            "digit": 0,
            "other": 0,
        }


class TestMixedCandidateSelection:
    def test_prefers_the_pass_whose_script_matches_its_model(self):
        # The English model returned mojibake with a high score; the
        # Arabic model returned Arabic with a lower score. Script match
        # must win, because the two scores are not comparable.
        chosen = pp.choose_mixed_candidate(
            [("ar", KATABA_VOWELLED, 0.72), ("en", "ktb,", 0.95)]
        )
        assert chosen[0] == "ar"

    def test_prefers_english_pass_when_arabic_pass_read_latin(self):
        # The Arabic model's charset includes Latin, so it can return
        # Latin for a Latin region. When it does, the English pass wins.
        chosen = pp.choose_mixed_candidate(
            [("ar", "Hello", 0.60), ("en", "Hello", 0.88)]
        )
        assert chosen[0] == "en"

    def test_known_failure_hallucinated_arabic_beats_correct_english(self):
        # DOCUMENTED LIMITATION, asserted so a change in behaviour is
        # noticed: if the Arabic pass hallucinates Arabic script for an
        # English word, this rule takes it over the correct English
        # reading. Nothing in the recognition output distinguishes this
        # from a genuinely Arabic region.
        chosen = pp.choose_mixed_candidate(
            [("ar", "هلو", 0.60), ("en", "Hello", 0.88)]
        )
        assert chosen[0] == "ar"

    def test_falls_back_to_score_when_neither_matches(self):
        chosen = pp.choose_mixed_candidate([("ar", "123", 0.40), ("en", "456", 0.90)])
        assert chosen[2] == 0.90

    def test_single_candidate_is_returned(self):
        only = ("ar", KATABA_PLAIN, 0.5)
        assert pp.choose_mixed_candidate([only]) == only

    def test_empty_candidates_raise(self):
        with pytest.raises(ValueError, match="at least one candidate"):
            pp.choose_mixed_candidate([])

    def test_empty_recognition_loses_to_real_text(self):
        chosen = pp.choose_mixed_candidate([("ar", "", 0.99), ("en", "Hello", 0.50)])
        assert chosen[1] == "Hello"


class TestReadingOrder:
    def test_rtl_orders_right_to_left_within_a_line(self):
        # Three boxes on one line at x = 10, 110, 210.
        boxes = [(10, 10, 100, 40), (110, 10, 200, 40), (210, 10, 300, 40)]
        assert pp.sort_regions_for_reading(boxes, rtl=True) == [2, 1, 0]

    def test_ltr_orders_left_to_right_within_a_line(self):
        boxes = [(210, 10, 300, 40), (10, 10, 100, 40), (110, 10, 200, 40)]
        assert pp.sort_regions_for_reading(boxes, rtl=False) == [1, 2, 0]

    def test_lines_are_ordered_top_to_bottom_in_both_directions(self):
        boxes = [(0, 200, 100, 230), (0, 10, 100, 40)]
        assert pp.sort_regions_for_reading(boxes, rtl=True) == [1, 0]
        assert pp.sort_regions_for_reading(boxes, rtl=False) == [1, 0]

    def test_slight_vertical_jitter_stays_on_one_line(self):
        # Real detections on one line vary by a few pixels in y.
        boxes = [(10, 12, 100, 42), (110, 8, 200, 38), (210, 14, 300, 44)]
        assert pp.sort_regions_for_reading(boxes, rtl=True, line_tolerance=20) == [
            2,
            1,
            0,
        ]

    def test_jitter_across_a_grid_boundary_still_groups(self):
        # Regression: a fixed y/tolerance grid put y=8 and y=12 in
        # different buckets and split one text line in two.
        boxes = [(10, 8, 100, 38), (110, 12, 200, 42)]
        assert pp.group_regions_into_lines(boxes, 20.0) == [[0, 1]]

    def test_two_separated_lines_are_two_groups(self):
        boxes = [(0, 0, 50, 30), (60, 4, 110, 34), (0, 200, 50, 230)]
        lines = pp.group_regions_into_lines(boxes, 20.0)
        assert len(lines) == 2
        assert sorted(lines[0]) == [0, 1]
        assert lines[1] == [2]

    def test_multi_line_rtl_page_reads_line_by_line(self):
        # Two RTL lines, each with two regions.
        boxes = [
            (10, 10, 100, 40),    # 0: line 1 left
            (110, 10, 200, 40),   # 1: line 1 right
            (10, 100, 100, 130),  # 2: line 2 left
            (110, 100, 200, 130), # 3: line 2 right
        ]
        assert pp.sort_regions_for_reading(boxes, rtl=True) == [1, 0, 3, 2]

    def test_reading_order_does_not_touch_the_strings(self):
        # Sorting returns indices only. Arabic strings stay in logical
        # order; reversing them would corrupt the text.
        boxes = [(0, 0, 10, 10)]
        assert pp.sort_regions_for_reading(boxes, rtl=True) == [0]

    def test_invalid_tolerance_raises(self):
        with pytest.raises(ValueError, match="line_tolerance"):
            pp.sort_regions_for_reading([(0, 0, 1, 1)], rtl=True, line_tolerance=0)


class TestDiacriticInventory:
    def test_collects_distinct_marks(self):
        found = pp.diacritic_char_set([KATABA_VOWELLED, "م" + DAMMA + SHADDA])
        assert found == {FATHA, DAMMA, SHADDA}

    def test_empty_for_plain_text(self):
        assert pp.diacritic_char_set([KATABA_PLAIN, "hello"]) == set()

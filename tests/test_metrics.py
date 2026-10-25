"""CER / WER / diacritic metric tests.

All references and hypotheses here are hand-written fixtures. None of
them is model output, and nothing in this file measures OCR accuracy.
"""

import pytest

from phonics_ocr import evaluation as ev

# Arabic fixtures, written as escapes so the file stays ASCII-safe and
# the exact code points are unambiguous.
KATABA_VOWELLED = "كَتَبَ"  # kataba, fully vowelled
KATABA_PLAIN = "كتب"  # ktb, no diacritics
WALAD_VOWELLED = "الوَلَدُ"


class TestLevenshteinCounts:
    def test_identical_sequences(self):
        assert ev._levenshtein_counts("abc", "abc") == (0, 0, 0)

    def test_pure_substitution(self):
        assert ev._levenshtein_counts("abc", "axc") == (1, 0, 0)

    def test_pure_deletion(self):
        assert ev._levenshtein_counts("abc", "ac") == (0, 1, 0)

    def test_pure_insertion(self):
        assert ev._levenshtein_counts("ac", "abc") == (0, 0, 1)

    def test_empty_reference_counts_all_as_insertions(self):
        assert ev._levenshtein_counts("", "abcd") == (0, 0, 4)

    def test_empty_hypothesis_counts_all_as_deletions(self):
        assert ev._levenshtein_counts("abcd", "") == (0, 4, 0)

    def test_both_empty(self):
        assert ev._levenshtein_counts("", "") == (0, 0, 0)

    def test_total_matches_classic_edit_distance(self):
        # kitten -> sitting is the textbook distance-3 example.
        subs, dels, ins = ev._levenshtein_counts("kitten", "sitting")
        assert subs + dels + ins == 3


class TestCharacterErrorRate:
    def test_perfect_match_is_zero(self):
        rate = ev.character_error_rate("hello world", "hello world")
        assert rate.rate == 0.0
        assert rate.errors == 0
        assert rate.reference_length == 11

    def test_one_substitution_in_ten(self):
        rate = ev.character_error_rate("abcdefghij", "abcdefghiX")
        assert rate.rate == pytest.approx(0.1)

    def test_empty_reference_gives_undefined_rate_not_zero(self):
        # A zero denominator must not read as a perfect score.
        rate = ev.character_error_rate("", "spurious")
        assert rate.rate is None
        assert rate.errors == 8

    def test_rate_can_exceed_one(self):
        rate = ev.character_error_rate("ab", "abcdefgh")
        assert rate.rate > 1.0

    def test_denominator_is_always_exposed(self):
        rate = ev.character_error_rate("abcdef", "abc")
        assert rate.to_dict()["reference_length"] == 6

    def test_counts_sum_to_errors(self):
        rate = ev.character_error_rate("the quick brown", "teh quick browne")
        assert rate.errors == rate.substitutions + rate.deletions + rate.insertions

    def test_combining_marks_count_as_characters(self):
        # The vowelled form has 3 extra combining marks; dropping all of
        # them must register as 3 deletions out of 6 code points.
        rate = ev.character_error_rate(KATABA_VOWELLED, KATABA_PLAIN)
        assert rate.reference_length == 6
        assert rate.errors == 3
        assert rate.rate == pytest.approx(0.5)


class TestWordErrorRate:
    def test_perfect_match(self):
        rate = ev.word_error_rate("the cat sat", "the cat sat")
        assert rate.rate == 0.0
        assert rate.reference_length == 3

    def test_one_wrong_word_in_four(self):
        rate = ev.word_error_rate("a b c d", "a b X d")
        assert rate.rate == pytest.approx(0.25)

    def test_missing_word(self):
        rate = ev.word_error_rate("one two three", "one three")
        assert rate.deletions == 1
        assert rate.rate == pytest.approx(1 / 3)

    def test_whitespace_runs_do_not_create_empty_tokens(self):
        assert ev.word_error_rate("a  b\tc", "a b c").rate == 0.0

    def test_empty_reference_is_undefined(self):
        assert ev.word_error_rate("", "x y").rate is None

    def test_arabic_words_tokenise_on_whitespace(self):
        reference = f"{KATABA_VOWELLED} {WALAD_VOWELLED}"
        rate = ev.word_error_rate(reference, reference)
        assert rate.reference_length == 2
        assert rate.rate == 0.0


class TestErrorRateAggregation:
    def test_addition_sums_every_field(self):
        a = ev.ErrorRate(1, 2, 3, 10)
        b = ev.ErrorRate(4, 0, 1, 20)
        total = a + b
        assert (total.substitutions, total.deletions, total.insertions) == (5, 2, 4)
        assert total.reference_length == 30

    def test_micro_average_weights_by_length(self):
        # A 1-error/4-char sample and a 1-error/96-char sample.
        # Micro average = 2/100, NOT the macro mean of 0.25 and ~0.01.
        short = ev.ErrorRate(1, 0, 0, 4)
        long = ev.ErrorRate(1, 0, 0, 96)
        assert (short + long).rate == pytest.approx(0.02)


class TestDiacriticPRF:
    def test_perfect_diacritics(self):
        prf = ev.diacritic_prf(KATABA_VOWELLED, KATABA_VOWELLED)
        assert prf.true_positives == 3
        assert prf.false_positives == 0
        assert prf.false_negatives == 0
        assert prf.f1 == 1.0

    def test_no_diacritics_predicted_scores_zero_recall(self):
        # The failure mode that matters most: a model that simply never
        # outputs tashkeel must not look good here.
        prf = ev.diacritic_prf(KATABA_VOWELLED, KATABA_PLAIN)
        assert prf.true_positives == 0
        assert prf.false_negatives == 3
        assert prf.recall == 0.0
        assert prf.precision is None  # it made no predictions at all
        assert prf.f1 is None

    def test_spurious_diacritics_cost_precision(self):
        prf = ev.diacritic_prf(KATABA_PLAIN, KATABA_VOWELLED)
        assert prf.true_positives == 0
        assert prf.false_positives == 3
        assert prf.precision == 0.0

    def test_wrong_mark_on_right_position(self):
        # fatha on the first letter vs damma on the first letter.
        reference = "كَتب"
        hypothesis = "كُتب"
        prf = ev.diacritic_prf(reference, hypothesis)
        assert prf.true_positives == 0
        assert prf.false_positives == 1
        assert prf.false_negatives == 1

    def test_right_mark_on_wrong_position(self):
        reference = "كَتب"  # fatha on letter 0
        hypothesis = "كتَب"  # fatha on letter 1
        prf = ev.diacritic_prf(reference, hypothesis)
        assert prf.true_positives == 0
        assert prf.false_negatives == 1
        assert prf.false_positives == 1

    def test_partial_credit(self):
        reference = KATABA_VOWELLED  # 3 marks
        hypothesis = "كَتَب"  # first 2 marks only
        prf = ev.diacritic_prf(reference, hypothesis)
        assert prf.true_positives == 2
        assert prf.false_negatives == 1
        assert prf.precision == 1.0
        assert prf.recall == pytest.approx(2 / 3)

    def test_no_diacritics_on_either_side_is_undefined_not_perfect(self):
        prf = ev.diacritic_prf(KATABA_PLAIN, KATABA_PLAIN)
        assert (prf.precision, prf.recall, prf.f1) == (None, None, None)

    def test_prf_addition(self):
        total = ev.PRF(2, 1, 1) + ev.PRF(3, 0, 2)
        assert (total.true_positives, total.false_positives, total.false_negatives) == (
            5,
            1,
            3,
        )


class TestConfusableCounts:
    def test_identical_strings_have_no_mismatch(self):
        counts = ev.confusable_error_counts(WALAD_VOWELLED, WALAD_VOWELLED)
        assert all(v == 0 for v in counts.values())

    def test_alif_hamza_dropped_to_bare_alif(self):
        # U+0623 (alif with hamza above) read as U+0627 (bare alif).
        counts = ev.confusable_error_counts("أحمد", "احمد")
        assert counts["hamza"] == 1
        assert counts["alif_variants"] == 0  # both are in the alif class

    def test_taa_marbuta_read_as_haa(self):
        counts = ev.confusable_error_counts("مدرسة", "مدرسه")
        assert counts["taa_marbuta_haa"] == 0  # the class total is unchanged

    def test_missing_shadda_is_counted(self):
        counts = ev.confusable_error_counts("مُدّرس", "مُدرس")
        assert counts["shadda"] == 1

    def test_missing_tanween_is_counted(self):
        counts = ev.confusable_error_counts("كتابً", "كتاب")
        assert counts["tanween"] == 1

    def test_every_class_is_reported(self):
        counts = ev.confusable_error_counts("a", "b")
        assert set(counts) == {
            "hamza",
            "alif_variants",
            "taa_marbuta_haa",
            "alif_maqsura_yaa",
            "shadda",
            "tanween",
        }


class TestSampleScoringAndAggregation:
    def test_score_sample_applies_no_hidden_normalisation(self):
        # The caller controls normalisation; score_sample must not add
        # any, or the reported denominator would not match the inputs.
        score = ev.score_sample("s1", "arabic_tashkeel", KATABA_VOWELLED, KATABA_PLAIN)
        assert score.reference == KATABA_VOWELLED
        assert score.hypothesis == KATABA_PLAIN
        assert score.cer.reference_length == 6

    def test_exact_match_flag(self):
        assert ev.score_sample("s", "g", "abc", "abc").exact is True
        assert ev.score_sample("s", "g", "abc", "abd").exact is False

    def test_diacritics_can_be_skipped_for_english(self):
        score = ev.score_sample("s", "english", "cat", "cat", score_diacritics=False)
        assert score.diacritic is None

    def test_aggregate_of_empty_list_is_all_none(self):
        result = ev.aggregate([])
        assert result["n_samples"] == 0
        assert result["cer"] is None

    def test_aggregate_micro_averages(self):
        scores = [
            ev.score_sample("a", "g", "abcd", "abcX"),      # 1/4
            ev.score_sample("b", "g", "a" * 96, "a" * 96),  # 0/96
        ]
        result = ev.aggregate(scores)
        assert result["cer"]["reference_length"] == 100
        assert result["cer"]["rate"] == pytest.approx(0.01)
        assert result["exact_match_rate"] == pytest.approx(0.5)

    def test_aggregate_by_group_adds_overall(self):
        scores = [
            ev.score_sample("a", "english", "cat", "cat", score_diacritics=False),
            ev.score_sample("b", "arabic_plain", KATABA_PLAIN, KATABA_PLAIN),
        ]
        grouped = ev.aggregate_by_group(scores)
        assert set(grouped) == {"english", "arabic_plain", "overall"}
        assert grouped["overall"]["n_samples"] == 2
        assert grouped["english"]["n_samples"] == 1

    def test_group_denominators_are_independent(self):
        scores = [
            ev.score_sample("a", "g1", "abcd", "abcd"),
            ev.score_sample("b", "g2", "ab", "XY"),
        ]
        grouped = ev.aggregate_by_group(scores)
        assert grouped["g1"]["cer"]["rate"] == 0.0
        assert grouped["g2"]["cer"]["rate"] == 1.0
        assert grouped["overall"]["cer"]["rate"] == pytest.approx(2 / 6)


class TestWilsonInterval:
    """Confidence intervals for recall/precision at small n."""

    def test_no_trials_is_undefined(self):
        assert ev.wilson_interval(0, 0) == (None, None)

    def test_interval_brackets_the_point_estimate(self):
        lo, hi = ev.wilson_interval(18, 36)
        assert lo < 0.5 < hi

    def test_perfect_score_has_a_real_lower_bound(self):
        # The normal approximation gives width zero here, which is
        # false. Wilson must report genuine uncertainty.
        lo, hi = ev.wilson_interval(36, 36)
        assert hi == 1.0
        assert 0.85 < lo < 0.95

    def test_zero_score_has_a_real_upper_bound(self):
        lo, hi = ev.wilson_interval(0, 10)
        assert lo == 0.0
        assert 0.2 < hi < 0.35

    def test_interval_narrows_as_n_grows(self):
        small = ev.wilson_interval(9, 10)
        large = ev.wilson_interval(900, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_bounds_stay_within_zero_and_one(self):
        for successes, trials in [(0, 1), (1, 1), (1, 2), (37, 37), (2, 37)]:
            lo, hi = ev.wilson_interval(successes, trials)
            assert 0.0 <= lo <= hi <= 1.0

    def test_small_sample_interval_is_wide_enough_to_be_honest(self):
        # 36 boxes cannot support a tight claim; the interval must show it.
        lo, hi = ev.wilson_interval(29, 36)
        assert (hi - lo) > 0.2

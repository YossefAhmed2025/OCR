"""Word-segmentation geometry and grouping tests.

These cover the parts of :mod:`phonics_ocr.wordseg` that do not need
the recognition model: line rectification, crop-to-page mapping, ink
measurement and word grouping. The model-dependent part (CTC
alignment) is exercised by ``scripts/evaluate_localization.py`` against
annotated boxes, because asserting on model output in a unit test
would be asserting on the model, not on this code.
"""

import numpy as np
import pytest

from phonics_ocr import wordseg as ws
from phonics_ocr.wordseg import CharAlignment

FATHA = "َ"
DAMMA = "ُ"
KAF, TEH, BEH = "ك", "ت", "ب"


def align(char, step, conf=0.9, x0=None, x1=None):
    x0 = step * 10.0 if x0 is None else x0
    x1 = x0 + 10.0 if x1 is None else x1
    return CharAlignment(char=char, timestep=step, confidence=conf, x_start=x0, x_end=x1)


class TestQuadOrdering:
    def test_orders_axis_aligned_quad(self):
        out = ws.order_quad([[10, 60], [10, 10], [90, 60], [90, 10]])
        assert out.tolist() == [[10, 10], [90, 10], [90, 60], [10, 60]]

    def test_already_ordered_is_unchanged(self):
        pts = [[0, 0], [100, 0], [100, 40], [0, 40]]
        assert ws.order_quad(pts).tolist() == pts

    def test_rejects_wrong_vertex_count(self):
        with pytest.raises(ws.WordSegmentationError, match="4 vertices"):
            ws.order_quad([[0, 0], [1, 0], [1, 1]])


class TestRectifyLine:
    def _image(self, w=200, h=100):
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        img[40:60, 20:180] = 0
        return img

    def test_axis_aligned_crop_has_expected_size(self):
        image = self._image()
        crop, _ = ws.rectify_line(image, [[20, 30], [180, 30], [180, 70], [20, 70]])
        assert crop.shape[0] == pytest.approx(40, abs=1)
        assert crop.shape[1] == pytest.approx(160, abs=1)

    def test_inverse_maps_crop_origin_back_to_polygon_corner(self):
        image = self._image()
        polygon = [[20, 30], [180, 30], [180, 70], [20, 70]]
        crop, inverse = ws.rectify_line(image, polygon)
        mapped = ws.crop_rect_to_image_polygon(0, 1, crop.shape[0], inverse)
        assert mapped[0][0] == pytest.approx(20, abs=1)
        assert mapped[0][1] == pytest.approx(30, abs=1)

    def test_full_width_maps_back_to_full_polygon(self):
        image = self._image()
        polygon = [[20, 30], [180, 30], [180, 70], [20, 70]]
        crop, inverse = ws.rectify_line(image, polygon)
        mapped = ws.crop_rect_to_image_polygon(
            0, crop.shape[1] - 1, crop.shape[0], inverse
        )
        xs = [p[0] for p in mapped]
        assert min(xs) == pytest.approx(20, abs=2)
        assert max(xs) == pytest.approx(180, abs=2)

    def test_rotated_line_maps_back_to_rotated_polygon(self):
        # A sheared quad must round-trip: this is what keeps word boxes
        # correct on a photographed, non-horizontal line.
        image = self._image()
        polygon = [[20, 30], [180, 40], [180, 80], [20, 70]]
        crop, inverse = ws.rectify_line(image, polygon)
        mapped = ws.crop_rect_to_image_polygon(
            0, crop.shape[1] - 1, crop.shape[0], inverse
        )
        # Top edge should slope downward to the right, as the input does.
        assert mapped[1][1] > mapped[0][1]

    def test_degenerate_polygon_raises(self):
        image = self._image()
        with pytest.raises(ws.WordSegmentationError, match="Degenerate"):
            ws.rectify_line(image, [[10, 10], [10, 10], [10, 11], [10, 11]])


class TestInkRows:
    """Vertical tightening. Returns ``(extent, status)``."""

    def test_finds_ink_band(self):
        crop = np.full((50, 100, 3), 255, dtype=np.uint8)
        crop[10:30, 20:60] = 0
        assert ws.ink_rows(crop, 20, 60) == ((10.0, 30.0), "ok")

    def test_blank_columns_report_no_ink(self):
        crop = np.full((50, 100, 3), 255, dtype=np.uint8)
        crop[10:30, 0:10] = 0
        assert ws.ink_rows(crop, 50, 90) == (None, "no_ink")

    def test_clamps_out_of_range_columns(self):
        crop = np.full((20, 30, 3), 255, dtype=np.uint8)
        crop[5:15, :] = 0
        assert ws.ink_rows(crop, -50, 999)[0] == (5.0, 15.0)

    def test_degenerate_window(self):
        crop = np.full((0, 0, 3), 255, dtype=np.uint8)
        assert ws.ink_rows(crop, 0, 1) == (None, "empty_window")


class TestInkRowsDiacriticCoverage:
    """Regression: a box that clips tashkeel is wrong for this project.

    Verified against rendered text that marks above (fatha), below
    (kasra), English ascenders, descenders and punctuation are all
    captured with zero clipping.
    """

    def test_includes_diacritic_above_the_baseline(self):
        crop = np.full((60, 40, 3), 255, dtype=np.uint8)
        crop[30:50, 5:35] = 0   # body
        crop[8:14, 15:25] = 0   # mark above
        assert ws.ink_rows(crop, 0, 40)[0] == (8.0, 50.0)

    def test_includes_diacritic_below_the_baseline(self):
        # Kasra sits BELOW the letter; clipping it loses the vowel.
        crop = np.full((60, 40, 3), 255, dtype=np.uint8)
        crop[10:30, 5:35] = 0   # body
        crop[40:46, 15:25] = 0  # mark below
        assert ws.ink_rows(crop, 0, 40)[0] == (10.0, 46.0)

    def test_includes_marks_both_above_and_below(self):
        # shadda above + kasra below on the same letter.
        crop = np.full((70, 40, 3), 255, dtype=np.uint8)
        crop[6:12, 15:25] = 0    # above
        crop[25:45, 5:35] = 0    # body
        crop[52:58, 15:25] = 0   # below
        assert ws.ink_rows(crop, 0, 40)[0] == (6.0, 58.0)

    def test_includes_english_descender(self):
        crop = np.full((60, 40, 3), 255, dtype=np.uint8)
        crop[15:40, 5:35] = 0   # x-height body
        crop[40:50, 10:20] = 0  # descender of p/g/y
        assert ws.ink_rows(crop, 0, 40)[0] == (15.0, 50.0)

    def test_includes_punctuation_below_baseline(self):
        # A comma hangs below the baseline.
        crop = np.full((60, 60, 3), 255, dtype=np.uint8)
        crop[15:40, 5:35] = 0
        crop[38:46, 40:46] = 0
        assert ws.ink_rows(crop, 0, 60)[0] == (15.0, 46.0)

    def test_measures_only_the_requested_columns(self):
        # A neighbouring word's descender must not stretch this box.
        crop = np.full((60, 100, 3), 255, dtype=np.uint8)
        crop[20:30, 0:40] = 0    # this word
        crop[20:55, 60:100] = 0  # neighbour with a descender
        assert ws.ink_rows(crop, 0, 40)[0] == (20.0, 30.0)


class TestInkRowsFailureModes:
    def test_inverted_print_is_flagged_not_silently_wrong(self):
        # Light text on a dark page: the background falls below the
        # threshold everywhere, which previously produced a full-height
        # box with no warning. It must refuse instead.
        crop = np.full((50, 100, 3), 20, dtype=np.uint8)
        crop[10:30, 20:60] = 240
        extent, status = ws.ink_rows(crop, 0, 100)
        assert extent is None
        assert status == "suspect_polarity"

    def test_light_print_falls_back_rather_than_guessing(self):
        # Very low contrast: nothing crosses the threshold.
        crop = np.full((50, 100, 3), 255, dtype=np.uint8)
        crop[10:30, 20:60] = 230
        extent, status = ws.ink_rows(crop, 0, 100, threshold=200)
        assert extent is None and status == "no_ink"

    def test_lowering_the_threshold_recovers_light_print(self):
        crop = np.full((50, 100, 3), 255, dtype=np.uint8)
        crop[10:30, 20:60] = 230
        extent, status = ws.ink_rows(crop, 0, 100, threshold=240)
        assert status == "ok" and extent == (10.0, 30.0)

    def test_dense_dark_text_is_not_mistaken_for_inversion(self):
        # Half the window inked is still normal print.
        crop = np.full((50, 100, 3), 255, dtype=np.uint8)
        crop[10:35, :] = 0
        extent, status = ws.ink_rows(crop, 0, 100)
        assert status == "ok" and extent == (10.0, 35.0)

    def test_two_stacked_lines_produce_one_tall_box(self):
        # Documented limitation: if detection merges two lines, the
        # word box covers both. Asserted so a change is noticed.
        crop = np.full((170, 100, 3), 255, dtype=np.uint8)
        crop[16:60, 10:90] = 0
        crop[100:142, 10:90] = 0
        assert ws.ink_rows(crop, 0, 100)[0] == (16.0, 142.0)


class TestWordGrouping:
    def test_splits_on_space(self):
        words = ws.segment_words(
            [align("a", 0), align("b", 1), align(" ", 2), align("c", 3)], rtl=False
        )
        assert [w[0] for w in words] == ["ab", "c"]

    def test_no_space_yields_one_word(self):
        words = ws.segment_words([align("a", 0), align("b", 1)], rtl=False)
        assert len(words) == 1

    def test_empty_input_yields_nothing(self):
        assert ws.segment_words([], rtl=False) == []

    def test_rtl_reverses_to_logical_order(self):
        # Alignments arrive in image order (left to right). For Arabic
        # the logical string is the reverse of that.
        words = ws.segment_words(
            [align(BEH, 0), align(TEH, 1), align(KAF, 2)], rtl=True
        )
        assert words[0][0] == KAF + TEH + BEH

    def test_ltr_preserves_order(self):
        words = ws.segment_words(
            [align("c", 0), align("a", 1), align("t", 2)], rtl=False
        )
        assert words[0][0] == "cat"

    def test_extent_spans_the_group(self):
        words = ws.segment_words(
            [align("a", 0, x0=5, x1=15), align("b", 1, x0=15, x1=30)], rtl=False
        )
        assert words[0][2] == 5.0 and words[0][3] == 30.0

    def test_confidence_is_the_weakest_character(self):
        words = ws.segment_words(
            [align("a", 0, conf=0.9), align("b", 1, conf=0.4)], rtl=False
        )
        assert words[0][1] == pytest.approx(0.4)

    def test_low_confidence_is_flagged_not_dropped(self):
        words = ws.segment_words([align("a", 0, conf=0.2)], rtl=False)
        assert "low_confidence" in words[0][4]
        assert words[0][0] == "a"

    def test_diacritics_do_not_split_a_word(self):
        words = ws.segment_words(
            [align(KAF, 0), align(FATHA, 1), align(TEH, 2), align(FATHA, 3)],
            rtl=True,
        )
        assert len(words) == 1
        assert words[0][5] == 2  # n_tashkeel

    def test_tashkeel_counted_per_word(self):
        words = ws.segment_words(
            [align(KAF, 0), align(FATHA, 1), align(" ", 2), align(BEH, 3)],
            rtl=True,
        )
        assert [w[5] for w in words] == [1, 0]

    def test_lone_diacritic_is_flagged(self):
        words = ws.segment_words([align(DAMMA, 0)], rtl=True)
        assert "orphan_diacritic" in words[0][4]

    def test_min_confidence_filters_characters(self):
        words = ws.segment_words(
            [align("a", 0, conf=0.9), align("b", 1, conf=0.1)],
            rtl=False,
            min_confidence=0.5,
        )
        assert words[0][0] == "a"

    def test_consecutive_spaces_do_not_emit_empty_words(self):
        words = ws.segment_words(
            [align("a", 0), align(" ", 1), align(" ", 2), align("b", 3)], rtl=False
        )
        assert [w[0] for w in words] == ["a", "b"]

    def test_leading_and_trailing_spaces_ignored(self):
        words = ws.segment_words(
            [align(" ", 0), align("a", 1), align(" ", 2)], rtl=False
        )
        assert [w[0] for w in words] == ["a"]


class TestWordSpanSerialisation:
    def test_to_dict_is_json_ready(self):
        import json

        span = ws.WordSpan(
            index=0,
            text=KAF + FATHA,
            confidence=0.8,
            polygon=[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
            bbox=(0.0, 0.0, 10.0, 5.0),
            crop_x_start=0.0,
            crop_x_end=10.0,
            n_chars=2,
            n_tashkeel=1,
            flags=["vertically_tightened"],
        )
        payload = span.to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_granularity_is_never_labelled_as_detection_output(self):
        # The projector must be able to tell a model-detected line box
        # from a derived word box.
        assert ws.GRANULARITY_WORD == "word_ctc_aligned"
        assert ws.GRANULARITY_WORD != "text_region"


class TestStrideConstant:
    """The timestep->column mapping depends on this being exactly 8.

    Verified empirically by sweeping input widths 16..800 at height 48:
    T == floor(W / 8) in every case. A glyph-position sweep then gave a
    least-squares slope of 7.98-8.01.
    """

    def test_stride_is_eight(self):
        assert ws.REC_WIDTH_STRIDE == 8

    def test_input_height_matches_model_config(self):
        assert ws.REC_INPUT_HEIGHT == 48

    def test_timesteps_predicted_by_floor_division(self):
        # The relation the alignment maths relies on.
        for width in (16, 32, 100, 108, 320, 801):
            assert width // ws.REC_WIDTH_STRIDE == int(width / 8)


class TestProjectionBaseline:
    """Model-free comparison baseline, not the production path."""

    def _line(self, runs, width=300, height=50):
        crop = np.full((height, width, 3), 255, dtype=np.uint8)
        for x0, x1 in runs:
            crop[15:35, x0:x1] = 0
        return crop

    def test_splits_on_wide_gaps(self):
        crop = self._line([(10, 80), (160, 240)])
        runs = ws.segment_by_projection(crop)
        assert len(runs) == 2
        assert runs[0][0] == 10.0 and runs[1][1] == 240.0

    def test_merges_across_narrow_gaps(self):
        # A 5px gap is below the 40%-of-height threshold, so the two
        # blobs are treated as one word. This is the known weakness
        # that makes the method unsuitable for Arabic.
        crop = self._line([(10, 80), (85, 150)])
        assert len(ws.segment_by_projection(crop)) == 1

    def test_blank_crop_yields_nothing(self):
        assert ws.segment_by_projection(np.full((50, 200, 3), 255, np.uint8)) == []

    def test_inverted_crop_refuses_rather_than_returning_one_run(self):
        crop = np.full((50, 200, 3), 20, dtype=np.uint8)
        crop[15:35, 20:60] = 240
        assert ws.segment_by_projection(crop) == []

    def test_gap_threshold_scales_with_height(self):
        # The same pixel gap splits small text and not large text,
        # because the threshold is relative to the line height.
        small = self._line([(10, 60), (80, 130)], height=20)
        large = self._line([(10, 60), (80, 130)], height=100)
        assert len(ws.segment_by_projection(small)) == 2
        assert len(ws.segment_by_projection(large)) == 1

    def test_tiny_runs_are_dropped(self):
        crop = np.full((50, 300, 3), 255, dtype=np.uint8)
        crop[15:35, 10:12] = 0      # 2px speck
        crop[15:35, 150:220] = 0    # real word
        runs = ws.segment_by_projection(crop, min_word_px=5)
        assert len(runs) == 1 and runs[0][0] == 150.0


class TestUntightenedBoxPreserved:
    def test_span_carries_both_boxes(self):
        span = ws.WordSpan(
            index=0, text="x", confidence=0.9,
            polygon=[[0, 10], [10, 10], [10, 20], [0, 20]],
            bbox=(0.0, 10.0, 10.0, 20.0),
            crop_x_start=0.0, crop_x_end=10.0,
            n_chars=1, n_tashkeel=0, flags=["vertically_tightened"],
            polygon_untightened=[[0, 0], [10, 0], [10, 50], [0, 50]],
            bbox_untightened=(0.0, 0.0, 10.0, 50.0),
            vertical_source="ok",
        )
        payload = span.to_dict()
        assert payload["bbox"] == [0.0, 10.0, 10.0, 20.0]
        assert payload["bbox_untightened"] == [0.0, 0.0, 10.0, 50.0]
        assert payload["vertical_source"] == "ok"

    def test_defaults_are_json_safe(self):
        import json

        span = ws.WordSpan(
            index=0, text="x", confidence=0.5,
            polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
            bbox=(0.0, 0.0, 1.0, 1.0),
            crop_x_start=0.0, crop_x_end=1.0,
            n_chars=1, n_tashkeel=0, flags=[],
        )
        payload = span.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["bbox_untightened"] is None

"""Annotation schema validation tests.

The property that matters most here: a dataset of rendered images must
never be able to pass itself off as real-world evidence. Provenance is
validated, defaulted conservatively, and surfaced in the summary.
"""

import pytest

from phonics_ocr import dataset as ds

MINIMAL = {"image": "a.png", "text": "cat", "lang": "en"}


def record(**overrides):
    return {**MINIMAL, **overrides}


class TestRequiredFields:
    def test_minimal_record_is_valid(self):
        sample = ds.validate_record(record(), "t:1")
        assert sample.text == "cat"
        assert sample.lang == "en"

    @pytest.mark.parametrize("field", ["image", "text", "lang"])
    def test_missing_required_field_is_rejected(self, field):
        payload = record()
        del payload[field]
        with pytest.raises(ds.DatasetError, match=f"missing required field '{field}'"):
            ds.validate_record(payload, "t:1")

    def test_non_object_rejected(self):
        with pytest.raises(ds.DatasetError, match="must be a JSON object"):
            ds.validate_record([1, 2], "t:1")

    def test_error_message_names_the_location(self):
        with pytest.raises(ds.DatasetError, match="labels.jsonl:7"):
            ds.validate_record({"image": "a.png"}, "labels.jsonl:7")

    def test_id_defaults_to_image_stem(self):
        assert ds.validate_record(record(image="pages/p12.png"), "t:1").id == "p12"


class TestLanguageAndGroup:
    @pytest.mark.parametrize("lang", ["ar", "en", "mixed"])
    def test_valid_languages(self, lang):
        assert ds.validate_record(record(lang=lang), "t:1").lang == lang

    @pytest.mark.parametrize("lang", ["fr", "AR", "arabic", ""])
    def test_invalid_language_rejected(self, lang):
        with pytest.raises(ds.DatasetError, match="lang must be one of"):
            ds.validate_record(record(lang=lang), "t:1")

    def test_group_defaults_per_language(self):
        assert ds.validate_record(record(lang="ar"), "t:1").group == "ar_ungrouped"

    def test_explicit_group_kept(self):
        sample = ds.validate_record(record(group="arabic_tashkeel"), "t:1")
        assert sample.group == "arabic_tashkeel"


class TestCaptureProvenance:
    """Provenance decides whether a score may be called real-world."""

    @pytest.mark.parametrize("kind", ds.CAPTURE_KINDS)
    def test_all_declared_kinds_accepted(self, kind):
        assert ds.validate_record(record(capture=kind), "t:1").capture == kind

    def test_unknown_kind_rejected(self):
        with pytest.raises(ds.DatasetError, match="capture must be one of"):
            ds.validate_record(record(capture="camera"), "t:1")

    def test_missing_capture_defaults_to_unknown_not_photograph(self):
        # Defaulting to 'photograph' would let an unlabelled set claim
        # real-world evidence by omission.
        assert ds.validate_record(record(), "t:1").capture == "unknown"

    def test_only_photographs_count_as_real(self):
        assert ds.validate_record(record(capture="photograph"), "t:1").is_real_capture
        for kind in ("synthetic_render", "flatbed_scan", "screenshot", "unknown"):
            assert not ds.validate_record(record(capture=kind), "t:1").is_real_capture

    def test_synthetic_render_is_not_real(self):
        sample = ds.validate_record(record(capture="synthetic_render"), "t:1")
        assert sample.is_real_capture is False


class TestWordBoxes:
    def test_valid_word_boxes(self):
        sample = ds.validate_record(
            record(words=[{"text": "cat", "bbox": [0, 0, 10, 10]}]), "t:1"
        )
        assert sample.has_word_boxes
        assert sample.words[0].bbox == (0.0, 0.0, 10.0, 10.0)

    def test_word_without_bbox_is_rejected(self):
        with pytest.raises(ds.DatasetError, match="needs a 'bbox'"):
            ds.validate_record(record(words=[{"text": "cat"}]), "t:1")

    def test_word_without_text_is_rejected(self):
        with pytest.raises(ds.DatasetError, match="needs a 'text' field"):
            ds.validate_record(record(words=[{"bbox": [0, 0, 1, 1]}]), "t:1")

    @pytest.mark.parametrize("bbox", [[0, 0, 0, 10], [0, 0, 10, 0], [10, 0, 5, 10]])
    def test_degenerate_bbox_rejected(self, bbox):
        with pytest.raises(ds.DatasetError, match="degenerate bbox"):
            ds.validate_record(record(words=[{"text": "x", "bbox": bbox}]), "t:1")

    def test_wrong_length_bbox_rejected(self):
        with pytest.raises(ds.DatasetError, match="bbox must be"):
            ds.validate_record(record(words=[{"text": "x", "bbox": [0, 0, 1]}]), "t:1")

    def test_optional_polygon_accepted(self):
        sample = ds.validate_record(
            record(
                words=[
                    {
                        "text": "x",
                        "bbox": [0, 0, 10, 10],
                        "polygon": [[0, 0], [10, 1], [10, 11], [0, 10]],
                    }
                ]
            ),
            "t:1",
        )
        assert len(sample.words[0].polygon) == 4

    def test_no_words_key_is_valid(self):
        # Transcription-only annotation stays legal; word boxes are an
        # optional addition, not a new requirement.
        assert ds.validate_record(record(), "t:1").has_word_boxes is False

    def test_words_must_be_a_list(self):
        with pytest.raises(ds.DatasetError, match="'words' must be a list"):
            ds.validate_record(record(words={"text": "x"}), "t:1")


class TestLinesAndConditions:
    def test_line_polygons_accepted(self):
        sample = ds.validate_record(
            record(lines=[[[0, 0], [10, 0], [10, 5], [0, 5]]]), "t:1"
        )
        assert len(sample.lines) == 1

    def test_polygon_needs_three_vertices(self):
        with pytest.raises(ds.DatasetError, match="at least 3 vertices"):
            ds.validate_record(record(lines=[[[0, 0], [1, 1]]]), "t:1")

    def test_valid_conditions(self):
        sample = ds.validate_record(
            record(conditions={"lighting": "harsh", "blur": "slight"}), "t:1"
        )
        assert sample.conditions["lighting"] == "harsh"

    def test_unknown_condition_key_rejected(self):
        with pytest.raises(ds.DatasetError, match="unknown condition key"):
            ds.validate_record(record(conditions={"weather": "rain"}), "t:1")


class TestDimensions:
    def test_positive_dimensions_accepted(self):
        sample = ds.validate_record(record(width=800, height=600), "t:1")
        assert (sample.width, sample.height) == (800, 600)

    @pytest.mark.parametrize("value", [0, -5, "800", 1.5])
    def test_invalid_width_rejected(self, value):
        with pytest.raises(ds.DatasetError, match="width must be a positive integer"):
            ds.validate_record(record(width=value), "t:1")


class TestSummary:
    def _samples(self, captures):
        return [
            ds.validate_record(record(image=f"{i}.png", capture=c), f"t:{i}")
            for i, c in enumerate(captures)
        ]

    def test_counts_by_capture(self):
        summary = ds.summarise(self._samples(["photograph", "synthetic_render"]))
        assert summary["by_capture"] == {"photograph": 1, "synthetic_render": 1}
        assert summary["n_real_capture"] == 1

    def test_all_synthetic_does_not_support_real_claims(self):
        summary = ds.summarise(self._samples(["synthetic_render"] * 5))
        assert summary["real_world_claims_supported"] is False
        assert summary["n_real_capture"] == 0

    def test_one_photograph_supports_real_claims(self):
        summary = ds.summarise(self._samples(["synthetic_render", "photograph"]))
        assert summary["real_world_claims_supported"] is True

    def test_empty_set_supports_nothing(self):
        summary = ds.summarise([])
        assert summary["n_samples"] == 0
        assert summary["real_world_claims_supported"] is False

    def test_word_box_totals(self):
        samples = [
            ds.validate_record(
                record(
                    words=[
                        {"text": "a", "bbox": [0, 0, 1, 1]},
                        {"text": "b", "bbox": [2, 0, 3, 1]},
                    ]
                ),
                "t:1",
            ),
            ds.validate_record(record(image="b.png"), "t:2"),
        ]
        summary = ds.summarise(samples)
        assert summary["n_with_word_boxes"] == 1
        assert summary["n_word_boxes"] == 2


class TestSerialisation:
    def test_round_trips_through_json(self):
        import json

        sample = ds.validate_record(
            record(
                capture="photograph",
                width=100,
                height=50,
                words=[{"text": "cat", "bbox": [0, 0, 10, 10]}],
                conditions={"blur": "none"},
            ),
            "t:1",
        )
        payload = sample.to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_arabic_survives_serialisation(self):
        vowelled = "كَتَبَ"
        sample = ds.validate_record(record(text=vowelled, lang="ar"), "t:1")
        assert sample.to_dict()["text"] == vowelled

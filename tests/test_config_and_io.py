"""Configuration validation and JSON serialisation tests.

These never import paddle, so they run on any machine.
"""

import json

import pytest

from phonics_ocr import config as cfg
from phonics_ocr import utils


class TestLanguageAndDeviceValidation:
    @pytest.mark.parametrize("lang", ["ar", "en", "mixed"])
    def test_supported_languages_accepted(self, lang):
        assert cfg.OCRConfig(lang=lang).lang == lang

    @pytest.mark.parametrize("lang", ["fr", "arabic", "AR", "", "ar-EG"])
    def test_unsupported_language_rejected_with_a_useful_message(self, lang):
        with pytest.raises(cfg.ConfigError) as exc:
            cfg.OCRConfig(lang=lang)
        # The error must name what IS supported, not just what failed.
        assert "Supported:" in str(exc.value)

    @pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
    def test_supported_devices_accepted(self, device):
        assert cfg.OCRConfig(device=device).device == device

    @pytest.mark.parametrize("device", ["cuda", "gpu:0", "GPU", "tpu"])
    def test_unsupported_device_rejected(self, device):
        with pytest.raises(cfg.ConfigError, match="Unsupported device"):
            cfg.OCRConfig(device=device)


class TestThresholdValidation:
    @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0])
    def test_det_thresh_out_of_range(self, value):
        with pytest.raises(cfg.ConfigError, match="det_db_thresh"):
            cfg.OCRConfig(det_db_thresh=value)

    @pytest.mark.parametrize("value", [-0.01, 1.5])
    def test_box_thresh_out_of_range(self, value):
        with pytest.raises(cfg.ConfigError, match="det_db_box_thresh"):
            cfg.OCRConfig(det_db_box_thresh=value)

    @pytest.mark.parametrize("value", [-1.0, 1.01])
    def test_rec_score_thresh_out_of_range(self, value):
        with pytest.raises(cfg.ConfigError, match="rec_score_thresh"):
            cfg.OCRConfig(rec_score_thresh=value)

    @pytest.mark.parametrize("value", [0.0, -2.0])
    def test_unclip_ratio_must_be_positive(self, value):
        with pytest.raises(cfg.ConfigError, match="unclip_ratio"):
            cfg.OCRConfig(det_db_unclip_ratio=value)

    def test_det_side_len_lower_bound(self):
        with pytest.raises(cfg.ConfigError, match="det_limit_side_len"):
            cfg.OCRConfig(det_limit_side_len=16)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_boundary_values_are_accepted(self, value):
        assert cfg.OCRConfig(det_db_thresh=value).det_db_thresh == value


class TestPreprocessConfigValidation:
    def test_defaults_are_all_off(self):
        preprocess = cfg.PreprocessConfig()
        assert preprocess.is_identity
        assert not any(
            [preprocess.grayscale, preprocess.clahe, preprocess.denoise, preprocess.deskew]
        )
        assert preprocess.upscale == 1.0

    def test_any_enabled_option_breaks_identity(self):
        assert not cfg.PreprocessConfig(clahe=True).is_identity
        assert not cfg.PreprocessConfig(upscale=1.5).is_identity
        assert not cfg.PreprocessConfig(denoise=True).is_identity

    @pytest.mark.parametrize("value", [0.0, 0.05, 9.0])
    def test_upscale_bounds(self, value):
        with pytest.raises(cfg.ConfigError, match="upscale"):
            cfg.PreprocessConfig(upscale=value)

    def test_clip_limit_must_be_positive(self):
        with pytest.raises(cfg.ConfigError, match="clahe_clip_limit"):
            cfg.PreprocessConfig(clahe_clip_limit=0)

    def test_tile_grid_must_be_at_least_one(self):
        with pytest.raises(cfg.ConfigError, match="clahe_tile_grid"):
            cfg.PreprocessConfig(clahe_tile_grid=0)

    def test_denoise_strength_must_be_at_least_one(self):
        with pytest.raises(cfg.ConfigError, match="denoise_strength"):
            cfg.PreprocessConfig(denoise_strength=0)

    def test_negative_deskew_limit_rejected(self):
        with pytest.raises(cfg.ConfigError, match="max_deskew_degrees"):
            cfg.PreprocessConfig(max_deskew_degrees=-1)


class TestModelSelection:
    def test_single_language_runs_one_pass(self):
        assert cfg.OCRConfig(lang="ar").passes == ("ar",)
        assert cfg.OCRConfig(lang="en").passes == ("en",)

    def test_mixed_runs_both_passes(self):
        assert cfg.OCRConfig(lang="mixed").passes == ("ar", "en")

    def test_arabic_and_english_use_different_recognition_models(self):
        config = cfg.OCRConfig(lang="mixed")
        assert config.recognition_model("ar") != config.recognition_model("en")

    def test_model_names_are_explicit_ppocrv5_names(self):
        config = cfg.OCRConfig()
        assert config.recognition_model("ar") == "arabic_PP-OCRv5_mobile_rec"
        assert config.recognition_model("en") == "en_PP-OCRv5_mobile_rec"
        assert cfg.DETECTION_MODEL == "PP-OCRv5_mobile_det"

    def test_unknown_pass_language_raises_rather_than_substituting(self):
        with pytest.raises(cfg.ConfigError, match="No recognition model"):
            cfg.OCRConfig().recognition_model("fr")

    def test_required_models_for_each_language(self):
        assert cfg.required_models("ar") == (
            "PP-OCRv5_mobile_det",
            "arabic_PP-OCRv5_mobile_rec",
        )
        assert cfg.required_models("en") == (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
        )

    def test_mixed_requires_both_recognition_models(self):
        required = cfg.required_models("mixed")
        assert len(required) == 3
        assert "arabic_PP-OCRv5_mobile_rec" in required
        assert "en_PP-OCRv5_mobile_rec" in required

    def test_required_models_rejects_unknown_language(self):
        with pytest.raises(cfg.ConfigError, match="Unsupported language"):
            cfg.required_models("de")

    def test_all_required_models_has_no_duplicates(self):
        models = cfg.all_required_models()
        assert len(models) == len(set(models))

    def test_model_dir_is_coerced_to_path(self, tmp_path):
        config = cfg.OCRConfig(model_dir=str(tmp_path))
        assert config.model_dir == tmp_path
        assert config.model_dir_for("x") == tmp_path / "x"


class TestConfigSerialisation:
    def test_to_dict_is_json_serialisable(self):
        payload = cfg.OCRConfig(lang="mixed").to_dict()
        text = json.dumps(payload, ensure_ascii=False)
        assert json.loads(text) == payload

    def test_to_dict_records_the_models_actually_used(self):
        payload = cfg.OCRConfig(lang="ar").to_dict()
        assert payload["detection_model"] == "PP-OCRv5_mobile_det"
        assert payload["recognition_models"] == {"ar": "arabic_PP-OCRv5_mobile_rec"}
        assert payload["ocr_version"] == "PP-OCRv5"

    def test_to_dict_includes_preprocessing(self):
        payload = cfg.OCRConfig(preprocess=cfg.PreprocessConfig(clahe=True)).to_dict()
        assert payload["preprocess"]["clahe"] is True
        assert payload["preprocess"]["denoise"] is False

    def test_model_dir_serialises_as_a_string(self):
        assert isinstance(cfg.OCRConfig().to_dict()["model_dir"], str)

    def test_config_is_immutable(self):
        config = cfg.OCRConfig()
        with pytest.raises(Exception):
            config.lang = "en"


class TestJsonIO:
    def test_arabic_survives_a_write_read_round_trip(self, tmp_path):
        vowelled = "كَتَبَ"
        path = utils.write_json(tmp_path / "r.json", {"text": vowelled})
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["text"] == vowelled

    def test_diacritics_are_not_escaped_away_by_default(self, tmp_path):
        vowelled = "كَتَبَ"
        path = utils.write_json(tmp_path / "r.json", {"text": vowelled})
        raw = path.read_text(encoding="utf-8")
        assert vowelled in raw  # written as real characters, not \uXXXX

    def test_ensure_ascii_still_round_trips(self, tmp_path):
        vowelled = "كَتَبَ"
        path = utils.write_json(tmp_path / "a.json", {"t": vowelled}, ensure_ascii=True)
        assert json.loads(path.read_text(encoding="utf-8"))["t"] == vowelled

    def test_write_json_creates_parent_directories(self, tmp_path):
        path = utils.write_json(tmp_path / "a" / "b" / "c.json", {"ok": True})
        assert path.is_file()

    def test_write_text_uses_utf8(self, tmp_path):
        vowelled = "كَتَبَ"
        path = utils.write_text(tmp_path / "t.txt", vowelled)
        assert path.read_text(encoding="utf-8") == vowelled


class TestJsonlReading:
    def test_reads_records(self, tmp_path):
        path = tmp_path / "gt.jsonl"
        path.write_text(
            '{"image": "a.png", "text": "كتب", "lang": "ar"}\n'
            '{"image": "b.png", "text": "cat", "lang": "en"}\n',
            encoding="utf-8",
        )
        records = utils.read_jsonl(path)
        assert len(records) == 2
        assert records[0]["text"] == "كتب"

    def test_blank_and_comment_lines_are_skipped(self, tmp_path):
        path = tmp_path / "gt.jsonl"
        path.write_text(
            '// a comment\n\n{"image": "a.png", "text": "x", "lang": "en"}\n\n',
            encoding="utf-8",
        )
        assert len(utils.read_jsonl(path)) == 1

    def test_bad_json_reports_the_line_number(self, tmp_path):
        path = tmp_path / "gt.jsonl"
        path.write_text(
            '{"image": "a.png", "text": "x", "lang": "en"}\n{not json}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            utils.read_jsonl(path)

    def test_non_object_line_is_rejected(self, tmp_path):
        path = tmp_path / "gt.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            utils.read_jsonl(path)


class TestPathHelpers:
    def test_unique_path_returns_the_original_when_free(self, tmp_path):
        target = tmp_path / "out.json"
        assert utils.unique_path(target) == target

    def test_unique_path_avoids_overwriting(self, tmp_path):
        target = tmp_path / "out.json"
        target.write_text("first", encoding="utf-8")
        assert utils.unique_path(target).name == "out_2.json"

    def test_unique_path_keeps_counting(self, tmp_path):
        (tmp_path / "out.json").write_text("1", encoding="utf-8")
        (tmp_path / "out_2.json").write_text("2", encoding="utf-8")
        assert utils.unique_path(tmp_path / "out.json").name == "out_3.json"

    def test_find_images_filters_by_extension(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        assert [p.name for p in utils.find_images(tmp_path)] == ["a.png", "b.jpg"]

    def test_find_images_rejects_a_non_image_file(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="not a supported image type"):
            utils.find_images(path)

    def test_find_images_missing_target(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            utils.find_images(tmp_path / "nope")

    def test_run_ids_are_distinct_enough_to_not_collide(self):
        assert utils.run_id("ocr").startswith("ocr_")
        assert utils.run_id("eval").startswith("eval_")

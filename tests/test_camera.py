"""Camera workflow tests.

A live camera cannot run in CI, so these drive the workflow through
:class:`ReplayFrameSource` and a stub engine. That covers everything
except the device handle itself: frame assessment, the capture loop,
saving and the `--no-save` guarantee, word-span attachment, Arabic
preservation and the session record.

The parts that genuinely need hardware -- opening a device, switching
cameras, the preview window -- are exercised by hand; see
`docs/camera_testing.md`.
"""

import json

import numpy as np
import pytest

from phonics_ocr import camera as cam


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def page(width=400, height=120, ink=True):
    """A white frame with a dark bar standing in for a text line."""
    frame = np.full((height, width, 3), 250, dtype=np.uint8)
    if ink:
        frame[40:80, 30:width - 30] = 20
    return frame


KATABA = "كَتَبَ"


class StubEngine:
    """Minimal stand-in for OCREngine: no model, no GPU."""

    def __init__(self, texts=(KATABA, "world"), scores=(0.92, 0.41), rtl=True):
        self._texts = list(texts)
        self._scores = list(scores)
        self._rtl = rtl
        self.device = "cpu"
        self.calls = []

        class _Config:
            def to_dict(self):
                return {"lang": "ar", "device": "cpu"}

        self.config = _Config()

    def process_image(self, path):
        self.calls.append(str(path))
        regions = []
        for i, (text, score) in enumerate(zip(self._texts, self._scores)):
            x0 = 30 + i * 120
            regions.append(
                {
                    "index": i,
                    "text_raw": text,
                    "text_normalised": text,
                    "rec_score": score,
                    "polygon": [[x0, 40], [x0 + 100, 40], [x0 + 100, 80], [x0, 80]],
                    "bbox": [x0, 40, x0 + 100, 80],
                    "dominant_script": "arabic" if i == 0 else "latin",
                    "has_tashkeel": "َ" in text,
                    "tashkeel_count": sum(1 for c in text if c == "َ"),
                    "geometry_granularity": "text_region",
                    "source_pass": "ar",
                    "recognition_model": "stub",
                }
            )
        return {
            "schema_version": "1.0",
            "image": {"path": str(path), "name": "x.png", "width": 400, "height": 120,
                      "inspection": {"warnings": []}},
            "regions": regions,
            "n_regions": len(regions),
            "reading_order": {"rtl": self._rtl,
                              "region_indices": list(range(len(regions)))},
            "transcription_raw": "\n".join(self._texts),
            "timing": {"inference_seconds": 0.01},
            "device_used": self.device,
        }


# ---------------------------------------------------------------------
# frame sources
# ---------------------------------------------------------------------


class TestReplayFrameSource:
    def test_yields_the_supplied_frames(self):
        a, b = page(), page(width=500)
        source = cam.ReplayFrameSource([a, b])
        assert source.read().shape == a.shape
        assert source.read().shape == b.shape

    def test_loops_by_default(self):
        source = cam.ReplayFrameSource([page()])
        for _ in range(3):
            assert source.read() is not None

    def test_can_stop_instead_of_looping(self):
        source = cam.ReplayFrameSource([page()], loop=False)
        source.read()
        with pytest.raises(cam.CameraError, match="exhausted"):
            source.read()

    def test_returns_a_copy_so_callers_cannot_corrupt_the_source(self):
        original = page()
        source = cam.ReplayFrameSource([original])
        frame = source.read()
        frame[:] = 0
        assert source.read().max() > 0

    def test_rejects_an_empty_list(self):
        with pytest.raises(cam.CameraError, match="at least one image"):
            cam.ReplayFrameSource([])

    def test_describe_reports_dimensions_and_kind(self):
        described = cam.ReplayFrameSource([page(320, 240)]).describe()
        assert described["kind"] == "ReplayFrameSource"
        assert (described["width"], described["height"]) == (320, 240)

    def test_context_manager_releases(self):
        with cam.ReplayFrameSource([page()]) as source:
            assert source.read() is not None

    def test_loads_from_disk(self, tmp_path):
        from phonics_ocr.preprocessing import save_image

        path = tmp_path / "f.png"
        save_image(path, page())
        source = cam.ReplayFrameSource([path])
        assert source.read().shape[:2] == (120, 400)
        assert source.current_path == str(path)


class TestOpenFrameSource:
    def test_replay_takes_precedence_over_index(self):
        source = cam.open_frame_source(index=0, replay=[page()])
        assert isinstance(source, cam.ReplayFrameSource)

    def test_camera_error_is_raised_not_swallowed(self):
        # Index 9999 will not exist on any test machine; the failure
        # must be explicit rather than a silent fallback.
        with pytest.raises(cam.CameraError, match="Could not open camera"):
            cam.open_frame_source(index=9999)


# ---------------------------------------------------------------------
# frame assessment
# ---------------------------------------------------------------------


class TestAssessFrame:
    def test_flags_a_dark_frame(self):
        frame = np.full((800, 800, 3), 10, dtype=np.uint8)
        assert any("dark" in h for h in cam.assess_frame(frame)["hints"])

    def test_flags_an_overexposed_frame(self):
        frame = np.full((800, 800, 3), 250, dtype=np.uint8)
        assert any("overexposed" in h for h in cam.assess_frame(frame)["hints"])

    def test_flags_low_resolution(self):
        hints = cam.assess_frame(page(400, 120))["hints"]
        assert any("low resolution" in h for h in hints)

    def test_flags_blur(self):
        frame = np.full((800, 800, 3), 128, dtype=np.uint8)
        assert any("blur" in h for h in cam.assess_frame(frame)["hints"])

    def test_reports_numbers_not_just_hints(self):
        report = cam.assess_frame(page())
        assert "laplacian_variance" in report
        assert "mean_intensity" in report

    def test_ready_is_false_when_any_hint_fires(self):
        assert cam.assess_frame(page(400, 120))["ready"] is False

    def test_assessment_never_raises_on_grayscale(self):
        grey = np.full((700, 700), 128, dtype=np.uint8)
        assert isinstance(cam.assess_frame(grey)["hints"], list)


# ---------------------------------------------------------------------
# capture session
# ---------------------------------------------------------------------


class TestCaptureSessionSaving:
    def test_writes_image_json_text_and_overlay(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=True)
        capture = session.capture(page())
        for key in ("image", "json", "text", "overlay"):
            assert key in capture.saved
            assert (tmp_path / Path_name(capture.saved[key])).exists()

    def test_no_save_writes_absolutely_nothing(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        capture = session.capture(page())
        assert capture.saved == {}
        assert list(tmp_path.iterdir()) == []

    def test_no_save_still_returns_the_ocr_result(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        assert session.capture(page()).result["n_regions"] == 2

    def test_no_save_marks_the_path_as_unsaved(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        capture = session.capture(page())
        assert capture.result["image"]["path"] == "<not saved>"

    def test_no_save_leaves_no_temporary_file_behind(self, tmp_path):
        import tempfile

        before = set(Path_iter(tempfile.gettempdir()))
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        session.capture(page())
        after = set(Path_iter(tempfile.gettempdir()))
        leaked = [p for p in after - before if p.endswith(".png")]
        assert leaked == []

    def test_saving_requires_an_output_dir(self):
        with pytest.raises(ValueError, match="requires an output_dir"):
            cam.CaptureSession(StubEngine(), None, save=True)

    def test_captures_are_numbered_and_do_not_overwrite(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=True)
        session.capture(page())
        session.capture(page())
        names = sorted(
            p.name for p in tmp_path.glob("capture_*.png")
            if not p.name.endswith("_overlay.png")
        )
        assert names == ["capture_001.png", "capture_002.png"]
        # And each capture keeps its own JSON.
        assert sorted(p.name for p in tmp_path.glob("capture_*.json")) == [
            "capture_001.json",
            "capture_002.json",
        ]


class TestCaptureSessionContent:
    def test_arabic_diacritics_survive_to_disk(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=True)
        capture = session.capture(page())
        written = (tmp_path / "capture_001.txt").read_text(encoding="utf-8")
        assert KATABA in written
        assert written.count("َ") == 3

    def test_json_round_trips_arabic(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=True)
        session.capture(page())
        data = json.loads((tmp_path / "capture_001.json").read_text(encoding="utf-8"))
        assert data["regions"][0]["text_raw"] == KATABA

    def test_rtl_flag_is_preserved(self, tmp_path):
        session = cam.CaptureSession(StubEngine(rtl=True), tmp_path, save=False)
        assert session.capture(page()).result["reading_order"]["rtl"] is True

    def test_capture_metadata_records_the_source(self, tmp_path):
        source = cam.ReplayFrameSource([page()])
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        capture = session.capture(source.read(), source)
        assert capture.result["capture"]["source"]["kind"] == "ReplayFrameSource"

    def test_capture_kind_is_recorded(self, tmp_path):
        session = cam.CaptureSession(
            StubEngine(), tmp_path, save=False, capture_kind="screenshot"
        )
        assert session.capture(page()).result["capture"]["kind"] == "screenshot"

    def test_replay_runs_are_not_labelled_as_photographs(self, tmp_path):
        # A replay run must never be mistakable for a real camera test.
        session = cam.CaptureSession(
            StubEngine(), tmp_path, save=False, capture_kind="screenshot"
        )
        assert session.capture(page()).result["capture"]["kind"] != "photograph"

    def test_frame_assessment_is_attached(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        capture = session.capture(page())
        assert "hints" in capture.result["capture"]["frame_assessment"]


class TestConfidenceReporting:
    def test_mean_and_min_confidence(self, tmp_path):
        session = cam.CaptureSession(
            StubEngine(scores=(0.9, 0.5)), tmp_path, save=False
        )
        capture = session.capture(page())
        assert capture.mean_confidence == pytest.approx(0.7)
        assert capture.min_confidence == pytest.approx(0.5)

    def test_confidence_is_none_when_nothing_was_read(self, tmp_path):
        session = cam.CaptureSession(
            StubEngine(texts=(), scores=()), tmp_path, save=False
        )
        capture = session.capture(page())
        assert capture.mean_confidence is None
        assert capture.min_confidence is None

    def test_transcription_is_exposed(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        assert KATABA in session.capture(page()).transcription


class TestWordSpanAttachment:
    class StubAligner:
        pass

    def test_no_words_key_when_no_aligner(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        assert "words" not in session.capture(page()).result

    def test_word_note_marks_spans_as_derived_and_experimental(self, tmp_path, monkeypatch):
        session = cam.CaptureSession(
            StubEngine(), tmp_path, save=False, aligner=self.StubAligner()
        )
        monkeypatch.setattr(session, "_locate_words", lambda frame, result: [])
        note = session.capture(page()).result["word_geometry_note"]
        assert "EXPERIMENTAL" in note
        assert "NOT detector-produced" in note

    def test_a_failing_line_does_not_abort_the_capture(self, tmp_path, monkeypatch):
        import phonics_ocr.wordseg as ws

        def boom(*a, **k):
            raise RuntimeError("bad line")

        monkeypatch.setattr(ws, "locate_words_in_line", boom)
        session = cam.CaptureSession(
            StubEngine(), tmp_path, save=False, aligner=self.StubAligner()
        )
        capture = session.capture(page())
        assert capture.result["words"] == []
        assert capture.result["n_regions"] == 2


class TestSessionSummary:
    def test_summary_counts_captures(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        session.capture(page())
        session.capture(page())
        assert session.summary()["n_captures"] == 2

    def test_summary_records_whether_saving_was_on(self, tmp_path):
        assert cam.CaptureSession(StubEngine(), tmp_path, save=False).summary()[
            "saving_enabled"
        ] is False

    def test_summary_is_json_serialisable(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        session.capture(page())
        payload = session.summary()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_write_summary_returns_none_when_not_saving(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=False)
        session.capture(page())
        assert session.write_summary() is None

    def test_write_summary_writes_when_saving(self, tmp_path):
        session = cam.CaptureSession(StubEngine(), tmp_path, save=True)
        session.capture(page())
        path = session.write_summary()
        assert path is not None and path.exists()

    def test_summary_mean_confidence_over_captures(self, tmp_path):
        session = cam.CaptureSession(
            StubEngine(scores=(1.0, 0.0)), tmp_path, save=False
        )
        session.capture(page())
        assert session.summary()["mean_confidence_over_captures"] == pytest.approx(0.5)


class TestCameraInfo:
    def test_label_mentions_index_and_backend(self):
        info = cam.CameraInfo(1, "CAP_DSHOW", 1920, 1080)
        assert "1" in info.label and "CAP_DSHOW" in info.label

    def test_is_serialisable(self):
        payload = cam.CameraInfo(0, "CAP_ANY", 640, 480).to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_probe_returns_a_list_without_raising(self):
        # Must not raise on a machine with no camera -- the workflow
        # reports "none found" rather than crashing.
        assert isinstance(cam.probe_cameras(max_index=1), list)


# -- small path helpers used above ------------------------------------


def Path_name(path_str):
    from pathlib import Path

    return Path(path_str).name


def Path_iter(directory):
    from pathlib import Path

    try:
        return [p.name for p in Path(directory).iterdir()]
    except OSError:
        return []

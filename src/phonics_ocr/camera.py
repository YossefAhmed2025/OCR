"""Webcam frame sources for real-world OCR testing.

The OCR pipeline is already offline and local; this module only adds a
way to get frames into it from a camera.

Why a ``FrameSource`` abstraction
---------------------------------
A live camera cannot be exercised in an automated test, and a workflow
that is only ever run by hand is a workflow whose bugs are found by
hand. Everything here therefore talks to a :class:`FrameSource`, and
:class:`ReplayFrameSource` supplies frames from images on disk, so the
capture-to-OCR path can be tested end to end with no hardware.

Nothing in this module reaches the network.
"""

from __future__ import annotations

import dataclasses
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .utils import get_logger

LOGGER = get_logger(__name__)

#: Backends tried in order on Windows.
#:
#: DirectShow first: MSMF is the OpenCV default on Windows but can take
#: several seconds to fail on a missing device, and it is the usual
#: cause of "the preview takes 10 seconds to start".
_WINDOWS_BACKENDS = ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY")
_OTHER_BACKENDS = ("CAP_ANY",)

#: Frames discarded when opening, before judging the stream.
#:
#: Webcams routinely emit black or garbage for the first frames while
#: exposure and white balance settle. Judging frame 1 rejects working
#: cameras.
OPEN_WARMUP_FRAMES = 8

#: Max pixel value below which a frame counts as blank.
BLANK_FRAME_LEVEL = 8


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or read."""


@dataclasses.dataclass(frozen=True)
class CameraInfo:
    """A camera index that was probed successfully."""

    index: int
    backend: str
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"camera {self.index} ({self.backend}, {self.width}x{self.height})"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _backend_names() -> Sequence[str]:
    return _WINDOWS_BACKENDS if platform.system() == "Windows" else _OTHER_BACKENDS


def _backend_value(name: str) -> int:
    import cv2

    return int(getattr(cv2, name))


def probe_cameras(max_index: int = 4, warmup_frames: int = 2) -> List[CameraInfo]:
    """Return the camera indices that actually deliver a frame.

    A device can open and still never produce an image -- a disabled
    integrated camera, or one held by another application, both do
    this. Only indices that return a real frame are reported, because
    an index that opens but never reads is worse than no index at all.

    Several cameras need a frame or two before the sensor settles, so
    ``warmup_frames`` reads are discarded before the verdict.
    """
    import cv2

    found: List[CameraInfo] = []
    for index in range(max_index):
        for backend_name in _backend_names():
            capture = None
            try:
                capture = cv2.VideoCapture(index, _backend_value(backend_name))
                if not capture.isOpened():
                    continue
                frame = None
                for _ in range(max(1, warmup_frames)):
                    ok, candidate = capture.read()
                    if ok and candidate is not None and candidate.size:
                        frame = candidate
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                found.append(CameraInfo(index, backend_name, int(width), int(height)))
                break  # this index works; do not try slower backends
            except Exception as exc:  # pragma: no cover - driver dependent
                LOGGER.debug("probe %s on index %s failed: %s", backend_name, index, exc)
            finally:
                if capture is not None:
                    capture.release()
    return found


class FrameSource:
    """Something that yields BGR frames."""

    name: str = "frame source"

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()

    def describe(self) -> Dict[str, Any]:
        return {"kind": type(self).__name__, "name": self.name}


class CameraFrameSource(FrameSource):
    """Live frames from an attached camera."""

    def __init__(
        self,
        index: int = 0,
        backend: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        import cv2

        self.index = index
        self._cv2 = cv2
        self._capture = None
        last_error = ""

        for backend_name in ([backend] if backend else list(_backend_names())):
            try:
                capture = cv2.VideoCapture(index, _backend_value(backend_name))
            except Exception as exc:  # pragma: no cover - driver dependent
                last_error = f"{backend_name}: {exc}"
                continue
            if not capture.isOpened():
                capture.release()
                last_error = f"{backend_name}: device did not open"
                continue
            if width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            # Warm up before judging anything. The first frames off a
            # webcam are routinely black or garbage while exposure and
            # white balance settle, so a verdict based on frame 1 is a
            # false negative on a perfectly good camera.
            frame = None
            brightest = 0.0
            for _ in range(OPEN_WARMUP_FRAMES):
                ok, candidate = capture.read()
                if ok and candidate is not None and candidate.size:
                    frame = candidate
                    brightest = max(brightest, float(candidate.max()))

            if frame is None:
                capture.release()
                last_error = f"{backend_name}: opened but returned no frame"
                continue
            # A device held by another process commonly opens and then
            # streams solid black rather than failing, which looks like
            # a working camera and silently wastes a whole session. If
            # EVERY warm-up frame was black, say so.
            if brightest < BLANK_FRAME_LEVEL:
                capture.release()
                last_error = (
                    f"{backend_name}: opened but streamed {OPEN_WARMUP_FRAMES} "
                    "blank (black) frames -- the camera is held by another "
                    "application, or a privacy shutter is closed"
                )
                continue
            self._capture = capture
            self.backend = backend_name
            self.name = f"camera {index} ({backend_name})"
            break

        if self._capture is None:
            raise CameraError(
                f"Could not open camera index {index}.\n"
                f"  Last error: {last_error or 'no backend succeeded'}\n"
                "  Run 'python scripts/camera_test.py --list' to see which\n"
                "  indices work, and see docs/camera_testing.md."
            )

    @property
    def resolution(self) -> tuple:
        if self._capture is None:
            return (0, 0)
        return (
            int(self._capture.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Camera is already released")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError(
                f"Camera {self.index} stopped returning frames. It may have "
                "been unplugged or claimed by another application."
            )
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def describe(self) -> Dict[str, Any]:
        width, height = self.resolution
        return {
            "kind": "CameraFrameSource",
            "name": self.name,
            "index": self.index,
            "backend": getattr(self, "backend", "unknown"),
            "width": width,
            "height": height,
        }


class ReplayFrameSource(FrameSource):
    """Frames from image files, standing in for a camera.

    This exists so the capture-to-OCR path is testable and so the
    workflow can be demonstrated on a machine with no camera. Results
    produced through it are recorded with ``capture: "screenshot"`` or
    whatever the caller declares -- never ``photograph`` -- so a replay
    run can never be mistaken for a real camera test.
    """

    def __init__(self, images: Sequence[Any], loop: bool = True) -> None:
        from .preprocessing import load_image

        self._frames: List[np.ndarray] = []
        self._paths: List[str] = []
        for item in images:
            if isinstance(item, np.ndarray):
                self._frames.append(item)
                self._paths.append("<array>")
            else:
                self._frames.append(load_image(item))
                self._paths.append(str(item))
        if not self._frames:
            raise CameraError("ReplayFrameSource needs at least one image")
        self._position = 0
        self._loop = loop
        self.name = f"replay ({len(self._frames)} image(s))"

    def read(self) -> np.ndarray:
        if self._position >= len(self._frames):
            if not self._loop:
                raise CameraError("Replay source exhausted")
            self._position = 0
        frame = self._frames[self._position]
        self._position += 1
        return frame.copy()

    @property
    def current_path(self) -> str:
        index = (self._position - 1) % len(self._paths)
        return self._paths[index]

    def describe(self) -> Dict[str, Any]:
        first = self._frames[0]
        return {
            "kind": "ReplayFrameSource",
            "name": self.name,
            "n_images": len(self._frames),
            "width": int(first.shape[1]),
            "height": int(first.shape[0]),
            "paths": self._paths,
        }


def open_frame_source(
    index: Optional[int] = None,
    *,
    replay: Optional[Iterable[Any]] = None,
    backend: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> FrameSource:
    """Open a replay source when given one, otherwise a live camera."""
    if replay is not None:
        return ReplayFrameSource(list(replay))
    return CameraFrameSource(
        index=0 if index is None else index,
        backend=backend,
        width=width,
        height=height,
    )


def assess_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Quick quality read on a frame, with advisory warnings only.

    Nothing acts on these automatically. They are shown live so the
    person holding the page can fix focus or lighting *before*
    capturing, which is far cheaper than discovering it afterwards.
    """
    from .preprocessing import inspect_image

    report = inspect_image(frame)
    sharpness = report["laplacian_variance"]
    mean = report["mean_intensity"]

    hints: List[str] = []
    if sharpness < 60:
        hints.append("very blurred - hold still or move closer")
    elif sharpness < 120:
        hints.append("soft focus")
    if mean < 60:
        hints.append("too dark")
    elif mean > 215:
        hints.append("overexposed / glare")
    if min(frame.shape[:2]) < 600:
        hints.append("low resolution - diacritics may be unreadable")

    report["hints"] = hints
    report["ready"] = not hints
    return report


@dataclasses.dataclass
class CaptureResult:
    """One capture, its OCR output, and where it was written."""

    index: int
    result: Dict[str, Any]
    saved: Dict[str, str]

    @property
    def transcription(self) -> str:
        return self.result.get("transcription_raw", "")

    @property
    def mean_confidence(self) -> Optional[float]:
        scores = [r["rec_score"] for r in self.result.get("regions", [])]
        return sum(scores) / len(scores) if scores else None

    @property
    def min_confidence(self) -> Optional[float]:
        scores = [r["rec_score"] for r in self.result.get("regions", [])]
        return min(scores) if scores else None


class CaptureSession:
    """Capture frames, run the local OCR pipeline, write results.

    Holds one loaded engine (and optionally one CTC aligner) across
    captures, because loading them costs seconds and the point of this
    workflow is rapid iteration in front of a real page.

    Saving is on by default and is turned off entirely with
    ``save=False``; nothing is written to disk in that case.
    """

    def __init__(
        self,
        engine: Any,
        output_dir: Optional[Path] = None,
        *,
        save: bool = True,
        aligner: Any = None,
        capture_kind: str = "photograph",
        tighten_vertical: bool = True,
    ) -> None:
        self.engine = engine
        self.aligner = aligner
        self.save = save
        self.capture_kind = capture_kind
        self.tighten_vertical = tighten_vertical
        self.output_dir = Path(output_dir) if output_dir else None
        self.captures: List[CaptureResult] = []
        if self.save and self.output_dir is None:
            raise ValueError("save=True requires an output_dir")
        if self.save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self, frame: np.ndarray, source: Optional[FrameSource] = None
    ) -> "CaptureResult":
        """Run OCR on one frame and optionally persist everything."""
        import tempfile

        from .preprocessing import save_image
        from .utils import write_json, write_text
        from .visualization import draw_regions, draw_word_spans

        index = len(self.captures) + 1
        stem = f"capture_{index:03d}"

        # The engine reads from a path, and the captured frame must be
        # preserved exactly as it came off the sensor, so it is written
        # first and the saved file is what gets processed. With saving
        # disabled a temporary file is used and removed afterwards.
        temp_path: Optional[Path] = None
        if self.save:
            image_path = self.output_dir / f"{stem}.png"
            save_image(image_path, frame)
        else:
            handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            handle.close()
            temp_path = Path(handle.name)
            save_image(temp_path, frame)
            image_path = temp_path

        saved: Dict[str, str] = {}
        try:
            result = self.engine.process_image(image_path)
            result["capture"] = {
                "index": index,
                "kind": self.capture_kind,
                "source": source.describe() if source else None,
                "frame_assessment": assess_frame(frame),
            }

            if self.aligner is not None:
                result["words"] = self._locate_words(frame, result)
                result["word_geometry_note"] = (
                    "EXPERIMENTAL. DERIVED from CTC recognition timesteps, "
                    "NOT detector-produced word boxes. Unvalidated on "
                    "photographed pages. See docs/word_localization.md."
                )

            if self.save:
                overlay = draw_regions(frame, result["regions"])
                if result.get("words"):
                    overlay = draw_word_spans(overlay, result["words"])
                overlay_path = self.output_dir / f"{stem}_overlay.png"
                save_image(overlay_path, overlay)
                saved = {
                    "image": str(image_path),
                    "json": str(
                        write_json(self.output_dir / f"{stem}.json", result)
                    ),
                    "text": str(
                        write_text(
                            self.output_dir / f"{stem}.txt",
                            result.get("transcription_raw", "") + "\n",
                        )
                    ),
                    "overlay": str(overlay_path),
                }
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:  # pragma: no cover - Windows file locking
                    pass

        if not self.save:
            # That path pointed at a temporary file that is now gone.
            result["image"]["path"] = "<not saved>"

        capture = CaptureResult(index=index, result=result, saved=saved)
        self.captures.append(capture)
        return capture

    def _locate_words(
        self, frame: np.ndarray, result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        from .wordseg import locate_words_in_line

        rtl = result["reading_order"]["rtl"]
        spans: List[Dict[str, Any]] = []
        for region in result["regions"]:
            try:
                spans.extend(
                    span.to_dict()
                    for span in locate_words_in_line(
                        frame,
                        region["polygon"],
                        self.aligner,
                        rtl=rtl,
                        start_index=len(spans),
                        tighten_vertical=self.tighten_vertical,
                    )
                )
            except Exception as exc:
                LOGGER.warning("word localisation skipped a line: %s", exc)
        return spans

    def summary(self) -> Dict[str, Any]:
        from .utils import iso_now

        confidences = [
            c.mean_confidence for c in self.captures if c.mean_confidence is not None
        ]
        return {
            "generated_at": iso_now(),
            "n_captures": len(self.captures),
            "saving_enabled": self.save,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "capture_kind": self.capture_kind,
            "word_localisation": self.aligner is not None,
            "config": self.engine.config.to_dict(),
            "device_used": self.engine.device,
            "mean_confidence_over_captures": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "captures": [
                {
                    "index": c.index,
                    "n_regions": c.result.get("n_regions", 0),
                    "n_words": len(c.result.get("words", []) or []),
                    "mean_confidence": c.mean_confidence,
                    "min_confidence": c.min_confidence,
                    "inference_seconds": c.result.get("timing", {}).get(
                        "inference_seconds"
                    ),
                    "transcription": c.transcription,
                    "saved": c.saved,
                }
                for c in self.captures
            ],
        }

    def write_summary(self) -> Optional[Path]:
        if not self.save or self.output_dir is None:
            return None
        from .utils import write_json

        return write_json(self.output_dir / "session.json", self.summary())

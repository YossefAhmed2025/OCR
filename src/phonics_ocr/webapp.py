"""Local web UI for camera OCR testing.

A small HTTP server on localhost. The browser supplies the camera
through ``getUserMedia`` and posts a captured frame; the server runs
the same offline pipeline the CLI uses and returns JSON.

Why a browser rather than the OpenCV window: the browser's camera
stack handles device selection, autofocus and resolution far better
than ``cv2.VideoCapture``, and it works when OpenCV is built headless.

Nothing leaves the machine. The server binds to 127.0.0.1, serves its
own HTML, and loads no external scripts or fonts.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .utils import get_logger

LOGGER = get_logger(__name__)

INDEX_HTML = (Path(__file__).parent / "webui" / "index.html")


class OCRService:
    """Holds the loaded engines and runs captures. Thread-safe."""

    def __init__(
        self,
        device: str = "auto",
        offline: bool = True,
        output_dir: Optional[Path] = None,
        save: bool = True,
        words: bool = False,
    ) -> None:
        from .config import DEFAULT_MODEL_DIR, OCRConfig

        self.device = device
        self.offline = offline
        self.save = save
        self.words = words
        self.output_dir = Path(output_dir) if output_dir else None
        self.model_dir = DEFAULT_MODEL_DIR
        self._engines: Dict[str, Any] = {}
        self._aligners: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._config_cls = OCRConfig
        self.counter = 0
        if self.save and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def engine(self, lang: str):
        """Load one engine per language, once."""
        with self._lock:
            if lang not in self._engines:
                from .engine import OCREngine

                LOGGER.info("loading engine for %s", lang)
                self._engines[lang] = OCREngine(
                    self._config_cls(
                        lang=lang,
                        device=self.device,
                        model_dir=self.model_dir,
                        offline=self.offline,
                    )
                )
            return self._engines[lang]

    def aligner(self, lang: str, device: str):
        if not self.words or lang == "mixed":
            return None
        with self._lock:
            if lang not in self._aligners:
                from .config import RECOGNITION_MODELS
                from .wordseg import CTCAligner, WordSegmentationError

                try:
                    config = self._config_cls(lang=lang, model_dir=self.model_dir)
                    self._aligners[lang] = CTCAligner(
                        config.recognition_weights_dir(lang), device=device
                    )
                except WordSegmentationError as exc:
                    LOGGER.warning("word localisation unavailable: %s", exc)
                    self._aligners[lang] = None
            return self._aligners[lang]

    def info(self) -> Dict[str, Any]:
        from .config import FINETUNED_MODELS, OCRConfig

        rows = []
        for lang in ("ar", "en"):
            config = OCRConfig(lang=lang, model_dir=self.model_dir)
            rows.append(
                {
                    "lang": lang,
                    "weights": config.recognition_weights_dir(lang).name,
                    "finetuned": config.uses_finetuned(lang),
                }
            )
        return {"models": rows, "device": self.device, "offline": self.offline}

    def run(self, image_bgr: np.ndarray, lang: str) -> Dict[str, Any]:
        """Run OCR on one frame and build the response payload."""
        import tempfile

        from .camera import assess_frame
        from .preprocessing import save_image
        from .utils import write_json
        from .visualization import draw_regions, draw_word_spans

        engine = self.engine(lang)
        self.counter += 1
        index = self.counter
        stem = f"capture_{index:03d}"

        if self.save and self.output_dir:
            image_path = self.output_dir / f"{stem}.png"
            save_image(image_path, image_bgr)
            temp_path = None
        else:
            handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            handle.close()
            temp_path = Path(handle.name)
            save_image(temp_path, image_bgr)
            image_path = temp_path

        try:
            result = engine.process_image(image_path)
            result["capture"] = {
                "index": index,
                "kind": "photograph",
                "frame_assessment": assess_frame(image_bgr),
            }

            aligner = self.aligner(lang, engine.device)
            if aligner is not None:
                from .wordseg import locate_words_in_line

                spans = []
                rtl = result["reading_order"]["rtl"]
                for region in result["regions"]:
                    try:
                        spans.extend(
                            s.to_dict()
                            for s in locate_words_in_line(
                                image_bgr, region["polygon"], aligner,
                                rtl=rtl, start_index=len(spans),
                            )
                        )
                    except Exception as exc:
                        LOGGER.warning("word span skipped: %s", exc)
                result["words"] = spans

            overlay = draw_regions(image_bgr, result["regions"])
            if result.get("words"):
                overlay = draw_word_spans(overlay, result["words"])

            if self.save and self.output_dir:
                save_image(self.output_dir / f"{stem}_overlay.png", overlay)
                write_json(self.output_dir / f"{stem}.json", result)
                result["saved_to"] = str(self.output_dir)
            else:
                result["saved_to"] = None
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

        import cv2

        ok, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        result["overlay_jpeg_b64"] = (
            base64.b64encode(buffer.tobytes()).decode("ascii") if ok else None
        )
        config = engine.config
        result["weights_used"] = config.recognition_weights_dir(
            "ar" if lang == "mixed" else lang
        ).name
        return result


class Handler(BaseHTTPRequestHandler):
    service: OCRService = None  # set by serve()

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.debug(fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        self._send(
            code,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            try:
                body = INDEX_HTML.read_bytes()
            except OSError as exc:
                self._json(500, {"error": f"UI missing: {exc}"})
                return
            self._send(200, body, "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._json(200, self.service.info())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ocr":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            lang = payload.get("lang", "ar")
            if lang not in ("ar", "en", "mixed"):
                self._json(400, {"error": f"bad lang {lang!r}"})
                return
            data = payload.get("image", "")
            if "," in data:
                data = data.split(",", 1)[1]
            raw = base64.b64decode(data)
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return

        import cv2

        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._json(400, {"error": "could not decode the image"})
            return

        try:
            self._json(200, self.service.run(frame, lang))
        except Exception as exc:
            LOGGER.exception("OCR failed")
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def serve(service: OCRService, host: str = "127.0.0.1", port: int = 8000):
    """Start the server. Returns it; caller runs serve_forever()."""
    Handler.service = service
    return ThreadingHTTPServer((host, port), Handler)

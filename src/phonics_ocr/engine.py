"""The OCR engine wrapper around PaddleOCR.

What this module guarantees
---------------------------
* The model actually loaded is the model that was asked for. If a
  requested model directory is missing in offline mode, it raises
  rather than falling back to a download or a different model.
* The device actually used is the device that was asked for. ``--device
  gpu`` fails loudly if the GPU is unusable; it never quietly runs on
  CPU. ``auto`` picks GPU when available and says which it chose.
* Detection geometry is reported as the pipeline returned it, labelled
  with the granularity the pipeline actually produces (see
  :data:`GEOMETRY_GRANULARITY`).
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as geo
from . import postprocessing as pp
from .config import (
    DETECTION_MODEL,
    OCR_VERSION,
    ConfigError,
    OCRConfig,
)
from .preprocessing import apply_preprocessing, inspect_image, load_image
from .utils import get_logger, iso_now

LOGGER = get_logger(__name__)

#: What the PP-OCRv5 detection model actually returns.
#:
#: PP-OCRv5 text detection is a *line/text-region* detector. A polygon
#: typically encloses a whole text line, not a single word. Calling
#: these "word boxes" would be false, so every record is labelled
#: ``"text_region"`` and the projector-facing work of splitting a line
#: into words is left to a later, separately validated stage.
GEOMETRY_GRANULARITY = "text_region"

#: Schema version for the JSON written by :meth:`OCREngine.process_image`.
RESULT_SCHEMA_VERSION = "1.0"


class EngineError(RuntimeError):
    """Raised when the OCR engine cannot be built or run as requested."""


@dataclasses.dataclass
class TextRegion:
    """One detected and recognised region of an image."""

    index: int
    text_raw: str
    text_normalised: str
    rec_score: float
    polygon: geo.Polygon
    bbox: Tuple[float, float, float, float]
    granularity: str
    source_pass: str
    recognition_model: str
    dominant_script: str
    has_tashkeel: bool
    tashkeel_count: int
    candidates: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "index": self.index,
            # The model output, untouched. Evaluation of diacritics
            # depends on this never being normalised in place.
            "text_raw": self.text_raw,
            # A separate, conservatively normalised view (NFC, tatweel
            # removed, whitespace collapsed). Tashkeel are preserved.
            "text_normalised": self.text_normalised,
            "rec_score": self.rec_score,
            "polygon": self.polygon,
            "bbox": list(self.bbox),
            "geometry_granularity": self.granularity,
            "source_pass": self.source_pass,
            "recognition_model": self.recognition_model,
            "dominant_script": self.dominant_script,
            "has_tashkeel": self.has_tashkeel,
            "tashkeel_count": self.tashkeel_count,
        }
        if self.candidates is not None:
            record["mixed_candidates"] = self.candidates
        return record


def detect_gpu() -> Dict[str, Any]:
    """Report whether a usable Paddle GPU device is present.

    Never raises: the caller decides whether absence is fatal.
    """
    info: Dict[str, Any] = {
        "paddle_importable": False,
        "compiled_with_cuda": False,
        "device_count": 0,
        "usable": False,
        "detail": "",
    }
    try:
        import paddle  # noqa: PLC0415  (imported lazily and optionally)
    except Exception as exc:  # pragma: no cover - environment dependent
        info["detail"] = f"paddle import failed: {exc}"
        return info

    info["paddle_importable"] = True
    info["paddle_version"] = getattr(paddle, "__version__", "unknown")
    try:
        info["compiled_with_cuda"] = bool(paddle.is_compiled_with_cuda())
        if info["compiled_with_cuda"]:
            info["device_count"] = int(paddle.device.cuda.device_count())
    except Exception as exc:  # pragma: no cover - environment dependent
        info["detail"] = f"CUDA query failed: {exc}"
        return info

    if not info["compiled_with_cuda"]:
        info["detail"] = "This paddlepaddle build is CPU-only (not compiled with CUDA)."
        return info
    if info["device_count"] == 0:
        info["detail"] = "Paddle has CUDA support but reports 0 visible GPU devices."
        return info

    # Compiling-with-CUDA and seeing a device is not the same as being
    # able to run a kernel: a driver/runtime mismatch shows up only on
    # the first real allocation. So actually allocate.
    try:
        paddle.set_device("gpu:0")
        probe = paddle.zeros([8, 8], dtype="float32")
        _ = float((probe + 1.0).sum())
        info["usable"] = True
        info["detail"] = f"GPU allocation probe succeeded on {info['device_count']} device(s)."
        try:
            props = paddle.device.cuda.get_device_properties(0)
            info["device_name"] = getattr(props, "name", "unknown")
            info["compute_capability"] = (
                f"{getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')}"
            )
            total = getattr(props, "total_memory", None)
            if total:
                info["total_memory_mb"] = int(total) // (1024 * 1024)
        except Exception:
            pass
    except Exception as exc:
        info["detail"] = f"GPU allocation probe failed: {exc}"
    finally:
        try:
            paddle.set_device("cpu")
        except Exception:
            pass
    return info


def resolve_device(requested: str) -> Tuple[str, Dict[str, Any]]:
    """Turn a requested device into a concrete Paddle device string.

    Returns ``(device_string, gpu_report)``. Raises :class:`EngineError`
    when ``gpu`` was demanded but is unusable -- silently downgrading to
    CPU would make a benchmark meaningless.
    """
    report = detect_gpu()
    if requested == "cpu":
        return "cpu", report
    if requested == "gpu":
        if not report["usable"]:
            raise EngineError(
                "GPU was requested with --device gpu but is not usable.\n"
                f"  Reason: {report['detail']}\n"
                "  Run 'python scripts/check_environment.py' for the full "
                "diagnosis, or re-run with --device cpu."
            )
        return "gpu:0", report
    if requested == "auto":
        return ("gpu:0" if report["usable"] else "cpu"), report
    raise ConfigError(f"Unknown device request {requested!r}")


class OCREngine:
    """A configured, reusable OCR pipeline.

    One :class:`OCREngine` owns one PaddleOCR instance per recognition
    pass. Building a PaddleOCR object is slow (model load), so an engine
    is meant to be constructed once and reused across a batch.
    """

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self.device, self.gpu_report = resolve_device(config.device)
        self._pipelines: Dict[str, Any] = {}
        self._build()

    # -- construction ------------------------------------------------

    def _local_model_dir(self, model_name: str) -> Optional[str]:
        """Path to a locally stored model, if it is present and complete.

        Returns ``None`` when the directory is absent, which lets
        PaddleOCR use its own cache/download path -- but only when we
        are not in offline mode.
        """
        directory = self.config.model_dir_for(model_name)
        if directory.is_dir() and any(directory.iterdir()):
            return str(directory)
        return None

    def _build(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise EngineError(
                "paddleocr is not installed in this environment.\n"
                "  Install it with: pip install -r requirements.txt"
            ) from exc

        for pass_lang in self.config.passes:
            rec_model = self.config.recognition_model(pass_lang)
            det_dir = self._local_model_dir(DETECTION_MODEL)
            rec_dir = self._local_model_dir(rec_model)

            if self.config.offline:
                missing = []
                if det_dir is None:
                    missing.append(DETECTION_MODEL)
                if rec_dir is None:
                    missing.append(rec_model)
                if missing:
                    raise EngineError(
                        "Offline mode is on but these models are not present "
                        f"in {self.config.model_dir}:\n"
                        + "".join(f"    - {m}\n" for m in missing)
                        + "  Run 'python scripts/download_models.py' once with "
                        "network access."
                    )

            # NOTE: `lang` and `ocr_version` are deliberately NOT passed.
            # PaddleOCR ignores both when explicit model names are given
            # ("`lang` and `ocr_version` will be ignored when model names
            # or model directories are not `None`") and warns about it.
            # Explicit model names are the stronger guarantee anyway --
            # they pin the exact PP-OCRv5 weights rather than asking
            # PaddleOCR to resolve a language code to whatever its
            # default version happens to be, which changes between
            # releases (newer ones default to PP-OCRv6).
            kwargs: Dict[str, Any] = {
                "device": self.device,
                "text_detection_model_name": DETECTION_MODEL,
                "text_recognition_model_name": rec_model,
                # These three sub-models are off by default: each adds
                # load time and a failure mode, and none is needed for
                # flat, upright book pages.
                "use_doc_orientation_classify": self.config.use_doc_orientation_classify,
                "use_doc_unwarping": self.config.use_doc_unwarping,
                "use_textline_orientation": self.config.use_textline_orientation,
                "text_det_limit_side_len": self.config.det_limit_side_len,
                "text_det_thresh": self.config.det_db_thresh,
                "text_det_box_thresh": self.config.det_db_box_thresh,
                "text_det_unclip_ratio": self.config.det_db_unclip_ratio,
                "text_rec_score_thresh": self.config.rec_score_thresh,
                # See OCRConfig.enable_mkldnn: paddle 3.3.1's oneDNN
                # path errors out on these models, so it is off by
                # default and CPU inference works as a result.
                "enable_mkldnn": self.config.enable_mkldnn,
                "cpu_threads": self.config.cpu_threads,
            }
            if det_dir:
                kwargs["text_detection_model_dir"] = det_dir
            if rec_dir:
                kwargs["text_recognition_model_dir"] = rec_dir

            LOGGER.info(
                "Loading pipeline [%s]: det=%s rec=%s device=%s",
                pass_lang,
                DETECTION_MODEL,
                rec_model,
                self.device,
            )
            try:
                self._pipelines[pass_lang] = PaddleOCR(**kwargs)
            except Exception as exc:
                raise EngineError(
                    f"Failed to build the PaddleOCR pipeline for '{pass_lang}'.\n"
                    f"  detection model:   {DETECTION_MODEL}\n"
                    f"  recognition model: {rec_model}\n"
                    f"  device:            {self.device}\n"
                    f"  model dir:         {self.config.model_dir}\n"
                    f"  underlying error:  {type(exc).__name__}: {exc}"
                ) from exc

    # -- inference ---------------------------------------------------

    @staticmethod
    def _extract(result: Any) -> Dict[str, Any]:
        """Pull the result dict out of a PaddleOCR 3.x result object."""
        if isinstance(result, dict):
            return result
        for attribute in ("json", "res", "_json"):
            value = getattr(result, attribute, None)
            if isinstance(value, dict):
                # ``.json`` wraps the payload under a "res" key.
                return value.get("res", value) if "res" in value else value
        raise EngineError(
            f"Unrecognised PaddleOCR result object of type {type(result).__name__}. "
            "The installed paddleocr version may differ from the one this code "
            "was written against (3.x)."
        )

    def _run_pass(
        self, pass_lang: str, image: np.ndarray
    ) -> List[Tuple[str, float, geo.Polygon]]:
        """Run one recognition pass, returning ``(text, score, polygon)``."""
        pipeline = self._pipelines[pass_lang]
        try:
            results = pipeline.predict(image)
        except Exception as exc:
            raise EngineError(
                f"Inference failed on pass '{pass_lang}' "
                f"(device={self.device}): {type(exc).__name__}: {exc}"
            ) from exc

        regions: List[Tuple[str, float, geo.Polygon]] = []
        for result in results or []:
            payload = self._extract(result)
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            polys = payload.get("rec_polys")
            if polys is None or len(polys) == 0:
                polys = payload.get("dt_polys") or []

            if not (len(texts) == len(scores) == len(polys)):
                LOGGER.warning(
                    "Pass %s returned mismatched lengths "
                    "(texts=%d scores=%d polys=%d); truncating to the shortest.",
                    pass_lang,
                    len(texts),
                    len(scores),
                    len(polys),
                )
            for text, score, poly in zip(texts, scores, polys):
                regions.append(
                    (str(text), float(score), geo.normalise_polygon(np.asarray(poly)))
                )
        return regions

    @staticmethod
    def _merge_passes(
        per_pass: Dict[str, List[Tuple[str, float, geo.Polygon]]],
        iou_threshold: float = 0.5,
    ) -> List[Tuple[str, str, float, geo.Polygon, List[Dict[str, Any]]]]:
        """Merge per-language passes over the same image into one region list.

        Both passes share the same detection model and the same
        detection parameters, so their polygons are expected to be
        near-identical; regions are matched by IoU and the winning
        recognition is chosen by
        :func:`postprocessing.choose_mixed_candidate`.

        Returns ``(pass_lang, text, score, polygon, candidates)`` tuples.
        """
        pass_names = list(per_pass.keys())
        if len(pass_names) == 1:
            only = pass_names[0]
            return [(only, t, s, p, []) for t, s, p in per_pass[only]]

        base_name = pass_names[0]
        base = per_pass[base_name]
        others = {n: list(per_pass[n]) for n in pass_names[1:]}

        merged: List[Tuple[str, str, float, geo.Polygon, List[Dict[str, Any]]]] = []
        for text, score, poly in base:
            bbox = geo.polygon_to_bbox(poly)
            candidates = [(base_name, text, score)]
            records = [
                {"pass": base_name, "text": text, "rec_score": score, "matched": True}
            ]
            for other_name, pool in others.items():
                best_index, best_iou = -1, iou_threshold
                for i, (_, _, other_poly) in enumerate(pool):
                    iou = geo.bbox_iou(bbox, geo.polygon_to_bbox(other_poly))
                    if iou >= best_iou:
                        best_index, best_iou = i, iou
                if best_index >= 0:
                    o_text, o_score, _ = pool.pop(best_index)
                    candidates.append((other_name, o_text, o_score))
                    records.append(
                        {
                            "pass": other_name,
                            "text": o_text,
                            "rec_score": o_score,
                            "matched": True,
                            "iou": round(best_iou, 4),
                        }
                    )
            winner_pass, winner_text, winner_score = pp.choose_mixed_candidate(
                candidates
            )
            merged.append((winner_pass, winner_text, winner_score, poly, records))

        # Regions only the non-base passes found are kept, not discarded.
        for other_name, pool in others.items():
            for text, score, poly in pool:
                merged.append(
                    (
                        other_name,
                        text,
                        score,
                        poly,
                        [
                            {
                                "pass": other_name,
                                "text": text,
                                "rec_score": score,
                                "matched": False,
                            }
                        ],
                    )
                )
        return merged

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Run the configured pipeline on one image and build the record."""
        image_path = Path(image_path)
        original = load_image(image_path)
        original_h, original_w = original.shape[:2]
        image_info = inspect_image(original)

        processed, prep_report = apply_preprocessing(original, self.config.preprocess)
        proc_h, proc_w = processed.shape[:2]
        scale_back = 1.0 / prep_report["scale_factor"]

        start = time.perf_counter()
        per_pass = {
            pass_lang: self._run_pass(pass_lang, processed)
            for pass_lang in self.config.passes
        }
        elapsed = time.perf_counter() - start

        merged = self._merge_passes(per_pass)

        regions: List[TextRegion] = []
        for index, (pass_lang, text, score, poly, candidates) in enumerate(merged):
            # Map coordinates back into the ORIGINAL image space so that
            # every consumer sees one coordinate system regardless of
            # which preprocessing was used.
            poly_original = (
                geo.scale_polygon(poly, scale_back) if scale_back != 1.0 else poly
            )
            poly_original = geo.clip_polygon(poly_original, original_w, original_h)
            bbox = geo.clip_bbox(
                geo.polygon_to_bbox(poly_original), original_w, original_h
            )
            regions.append(
                TextRegion(
                    index=index,
                    text_raw=text,
                    text_normalised=pp.normalise_text(text),
                    rec_score=score,
                    polygon=poly_original,
                    bbox=bbox,
                    granularity=GEOMETRY_GRANULARITY,
                    source_pass=pass_lang,
                    recognition_model=self.config.recognition_model(pass_lang),
                    dominant_script=pp.dominant_script(text),
                    has_tashkeel=pp.has_tashkeel(text),
                    tashkeel_count=pp.count_tashkeel(text),
                    candidates=candidates or None,
                )
            )

        # Page reading order. RTL when the page is predominantly Arabic.
        arabic_chars = sum(
            pp.script_profile(r.text_raw)["arabic"] for r in regions
        )
        latin_chars = sum(pp.script_profile(r.text_raw)["latin"] for r in regions)
        rtl = arabic_chars > latin_chars
        order = pp.sort_regions_for_reading([r.bbox for r in regions], rtl=rtl)

        transcription = "\n".join(regions[i].text_raw for i in order)

        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": iso_now(),
            "image": {
                "path": str(image_path),
                "name": image_path.name,
                "width": original_w,
                "height": original_h,
                "processed_width": proc_w,
                "processed_height": proc_h,
                "inspection": image_info,
            },
            "coordinate_space": {
                "origin": "top-left",
                "x_axis": "rightward",
                "y_axis": "downward",
                "units": "pixels",
                "reference_image": "original input image (pre-preprocessing)",
                "note": (
                    "Image coordinates only. These are not projector "
                    "coordinates; camera-to-page and page-to-projector "
                    "calibration is a separate, unimplemented stage."
                ),
            },
            "config": self.config.to_dict(),
            "device_used": self.device,
            "preprocessing": prep_report,
            "timing": {
                "inference_seconds": round(elapsed, 4),
                "passes_run": list(self.config.passes),
            },
            "geometry_granularity": GEOMETRY_GRANULARITY,
            "geometry_granularity_note": (
                "PP-OCRv5 detection returns text-line / text-region polygons. "
                "These are NOT word-level boxes. Word-level localisation for "
                "the projector is not implemented or verified."
            ),
            "reading_order": {"rtl": rtl, "region_indices": order},
            "n_regions": len(regions),
            "regions": [r.to_dict() for r in regions],
            "transcription_raw": transcription,
        }

    def process_batch(self, image_paths: Sequence[Path]) -> List[Dict[str, Any]]:
        """Process several images with one loaded engine."""
        results: List[Dict[str, Any]] = []
        for path in image_paths:
            LOGGER.info("Processing %s", path.name)
            results.append(self.process_image(Path(path)))
        return results

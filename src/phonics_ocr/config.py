"""Configuration objects and validation for the Phonics Prism OCR baseline.

Every model choice is explicit. Nothing here silently substitutes a
different model or device than the one that was asked for: if a request
cannot be honoured, construction raises :class:`ConfigError`.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Final, Literal, Tuple

Language = Literal["ar", "en", "mixed"]
Device = Literal["cpu", "gpu", "auto"]


class ConfigError(ValueError):
    """Raised when a requested configuration cannot be honoured."""


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "data" / "results"

#: PaddleOCR pipeline version these model names belong to.
OCR_VERSION: Final[str] = "PP-OCRv5"

#: Detection model shared by every language pipeline below.
#:
#: PP-OCRv5 detection localises text regions without recognising them,
#: so the same detector serves Arabic and English. Recognition is what
#: differs per language.
DETECTION_MODEL: Final[str] = "PP-OCRv5_mobile_det"

#: Explicit per-language recognition models.
#:
#: arabic_PP-OCRv5_mobile_rec is an Arabic-SCRIPT model: its charset also
#: covers Persian, Urdu, Uyghur, Pashto, Kurdish, Sindhi and Balochi,
#: plus Latin digits. It is not Arabic-language-exclusive.
RECOGNITION_MODELS: Final[Dict[str, str]] = {
    "ar": "arabic_PP-OCRv5_mobile_rec",
    "en": "en_PP-OCRv5_mobile_rec",
}

#: PaddleOCR's own ``lang`` code for each of our language keys.
PADDLE_LANG_CODES: Final[Dict[str, str]] = {"ar": "ar", "en": "en"}

SUPPORTED_LANGUAGES: Final[Tuple[str, ...]] = ("ar", "en", "mixed")

#: The ``mixed`` mode is an explicitly experimental two-pass experiment,
#: not a single bilingual model. See ``docs/architecture.md``.
MIXED_PASSES: Final[Tuple[str, ...]] = ("ar", "en")


@dataclasses.dataclass(frozen=True)
class PreprocessConfig:
    """Optional, individually switchable image preprocessing.

    Every option defaults to *off*. Arabic tashkeel are small, low-contrast
    marks that denoising and aggressive thresholding routinely erase, so
    nothing is applied unless it is asked for explicitly.
    """

    grayscale: bool = False
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    denoise: bool = False
    denoise_strength: int = 3
    deskew: bool = False
    max_deskew_degrees: float = 15.0
    upscale: float = 1.0

    def __post_init__(self) -> None:
        if self.clahe_clip_limit <= 0:
            raise ConfigError("clahe_clip_limit must be > 0")
        if self.clahe_tile_grid < 1:
            raise ConfigError("clahe_tile_grid must be >= 1")
        if self.denoise_strength < 1:
            raise ConfigError("denoise_strength must be >= 1")
        if not 0.1 <= self.upscale <= 8.0:
            raise ConfigError(f"upscale must be in [0.1, 8.0], got {self.upscale}")
        if self.max_deskew_degrees < 0:
            raise ConfigError("max_deskew_degrees must be >= 0")

    @property
    def is_identity(self) -> bool:
        """True when no transformation at all would be applied."""
        return not (
            self.grayscale
            or self.clahe
            or self.denoise
            or self.deskew
            or self.upscale != 1.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class OCRConfig:
    """A fully resolved, explicit OCR pipeline configuration."""

    lang: Language = "ar"
    device: Device = "auto"
    model_dir: Path = DEFAULT_MODEL_DIR
    offline: bool = False
    det_limit_side_len: int = 960
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6
    det_db_unclip_ratio: float = 1.5
    rec_score_thresh: float = 0.0
    use_textline_orientation: bool = False
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    #: oneDNN (MKL-DNN) CPU acceleration.
    #:
    #: Default OFF. PaddleOCR defaults it ON, but paddlepaddle 3.3.1's
    #: oneDNN path fails on these PP-OCRv5 models with
    #: "ConvertPirAttribute2RuntimeAttribute not support
    #: [pir::ArrayAttribute<pir::DoubleAttribute>]", which makes every
    #: CPU run error out. Disabling it makes CPU inference work; the
    #: cost is slower CPU inference. Ignored on GPU.
    enable_mkldnn: bool = False
    cpu_threads: int = 10
    preprocess: PreprocessConfig = dataclasses.field(default_factory=PreprocessConfig)

    def __post_init__(self) -> None:
        if self.lang not in SUPPORTED_LANGUAGES:
            raise ConfigError(
                f"Unsupported language {self.lang!r}. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        if self.device not in ("cpu", "gpu", "auto"):
            raise ConfigError(
                f"Unsupported device {self.device!r}. Use 'cpu', 'gpu' or 'auto'."
            )
        if not 0.0 <= self.det_db_thresh <= 1.0:
            raise ConfigError("det_db_thresh must be in [0, 1]")
        if not 0.0 <= self.det_db_box_thresh <= 1.0:
            raise ConfigError("det_db_box_thresh must be in [0, 1]")
        if not 0.0 <= self.rec_score_thresh <= 1.0:
            raise ConfigError("rec_score_thresh must be in [0, 1]")
        if self.det_db_unclip_ratio <= 0:
            raise ConfigError("det_db_unclip_ratio must be > 0")
        if self.det_limit_side_len < 32:
            raise ConfigError("det_limit_side_len must be >= 32")
        if self.cpu_threads < 1:
            raise ConfigError("cpu_threads must be >= 1")
        object.__setattr__(self, "model_dir", Path(self.model_dir))

    # -- derived model selection ------------------------------------

    @property
    def passes(self) -> Tuple[str, ...]:
        """Recognition passes this configuration will run.

        ``mixed`` runs one pass per script and merges by region; see
        :mod:`phonics_ocr.postprocessing`.
        """
        return MIXED_PASSES if self.lang == "mixed" else (self.lang,)

    def recognition_model(self, pass_lang: str) -> str:
        try:
            return RECOGNITION_MODELS[pass_lang]
        except KeyError:
            raise ConfigError(
                f"No recognition model registered for {pass_lang!r}"
            ) from None

    def paddle_lang(self, pass_lang: str) -> str:
        try:
            return PADDLE_LANG_CODES[pass_lang]
        except KeyError:
            raise ConfigError(f"No PaddleOCR lang code for {pass_lang!r}") from None

    def model_dir_for(self, model_name: str) -> Path:
        return self.model_dir / model_name

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable record of this configuration."""
        return {
            "lang": self.lang,
            "device": self.device,
            "ocr_version": OCR_VERSION,
            "detection_model": DETECTION_MODEL,
            "recognition_models": {p: self.recognition_model(p) for p in self.passes},
            "model_dir": str(self.model_dir),
            "offline": self.offline,
            "det_limit_side_len": self.det_limit_side_len,
            "det_db_thresh": self.det_db_thresh,
            "det_db_box_thresh": self.det_db_box_thresh,
            "det_db_unclip_ratio": self.det_db_unclip_ratio,
            "rec_score_thresh": self.rec_score_thresh,
            "use_textline_orientation": self.use_textline_orientation,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "enable_mkldnn": self.enable_mkldnn,
            "cpu_threads": self.cpu_threads,
            "preprocess": self.preprocess.to_dict(),
        }


def required_models(lang: str) -> Tuple[str, ...]:
    """Model names that must be present locally to run ``lang`` offline."""
    if lang not in SUPPORTED_LANGUAGES:
        raise ConfigError(f"Unsupported language {lang!r}")
    passes = MIXED_PASSES if lang == "mixed" else (lang,)
    return (DETECTION_MODEL,) + tuple(RECOGNITION_MODELS[p] for p in passes)


def all_required_models() -> Tuple[str, ...]:
    """Every model name used by any supported language."""
    return (DETECTION_MODEL,) + tuple(RECOGNITION_MODELS[k] for k in ("ar", "en"))


def offline_env_active() -> bool:
    """Whether the PaddleX offline switch is set in this process."""
    return os.environ.get("PADDLE_PDX_OFFLINE_MODE", "") == "1"

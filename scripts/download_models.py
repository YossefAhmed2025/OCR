"""Download and record the OCR models this project uses.

This is the one step that needs network access. After it succeeds,
inference runs entirely offline against ``models/``.

    python scripts/download_models.py              # all languages
    python scripts/download_models.py --lang ar    # Arabic only
    python scripts/download_models.py --verify     # check, do not download

How it works: PaddleOCR resolves and caches official model files on
first use. This script triggers that resolution, then copies the
resolved files into the project-local ``models/<model_name>/``
directory and writes ``models/manifest.json`` with the file list, byte
sizes and SHA-256 checksums. Later runs point PaddleOCR at those local
directories explicitly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phonics_ocr.config import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DETECTION_MODEL,
    OCR_VERSION,
    RECOGNITION_MODELS,
    SUPPORTED_LANGUAGES,
    required_models,
)
from phonics_ocr.utils import (  # noqa: E402
    configure_logging,
    human_bytes,
    iso_now,
    sha256_dir,
    write_json,
)

LOGGER = configure_logging()

MANIFEST_PATH = DEFAULT_MODEL_DIR / "manifest.json"

#: Where PaddleX caches the official model files it downloads.
_CACHE_CANDIDATES = (
    Path.home() / ".paddlex" / "official_models",
    Path.home() / ".paddleocr" / "official_models",
)


def find_cached_model(model_name: str) -> Optional[Path]:
    """Locate a model in the PaddleX/PaddleOCR cache."""
    for cache_root in _CACHE_CANDIDATES:
        candidate = cache_root / model_name
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def model_is_local(model_name: str, model_dir: Path) -> bool:
    directory = model_dir / model_name
    return directory.is_dir() and any(directory.iterdir())


def trigger_download(lang: str, device: str = "cpu") -> None:
    """Instantiate a PaddleOCR pipeline so it resolves its model files.

    Built on CPU regardless of the eventual inference device: the model
    files are identical and CPU construction avoids needing a working
    GPU just to fetch weights.
    """
    from paddleocr import PaddleOCR

    rec_model = RECOGNITION_MODELS[lang]
    LOGGER.info(
        "Resolving models for '%s' (det=%s, rec=%s) ...",
        lang,
        DETECTION_MODEL,
        rec_model,
    )
    # lang/ocr_version omitted on purpose: PaddleOCR ignores them when
    # explicit model names are supplied, and the names already pin the
    # exact PP-OCRv5 weights. See the note in engine.py.
    PaddleOCR(
        device=device,
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def copy_into_project(model_name: str, model_dir: Path) -> Optional[Path]:
    """Copy a cached model into ``models/<model_name>/``."""
    source = find_cached_model(model_name)
    if source is None:
        return None
    destination = model_dir / model_name
    if destination.is_dir() and any(destination.iterdir()):
        LOGGER.info("  %s already present in %s", model_name, destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    LOGGER.info("  copied %s -> %s", source, destination)
    return destination


def describe_model(model_name: str, model_dir: Path) -> Dict[str, Any]:
    """Build the manifest entry for one model."""
    directory = model_dir / model_name
    if not directory.is_dir():
        return {"name": model_name, "present": False}
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    checksums = sha256_dir(directory)
    total = sum(p.stat().st_size for p in files)
    return {
        "name": model_name,
        "present": True,
        "local_dir": str(directory),
        "ocr_version": OCR_VERSION,
        "source": (
            "PaddleOCR official model cache, resolved by the installed "
            "paddleocr/paddlex release. See "
            "https://www.paddleocr.ai/main/en/version3.x/"
        ),
        "n_files": len(files),
        "total_bytes": total,
        "total_human": human_bytes(total),
        "files": [
            {
                "name": p.relative_to(directory).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": checksums.get(p.relative_to(directory).as_posix(), ""),
            }
            for p in files
        ],
    }


def write_manifest(model_dir: Path, models: List[str]) -> Path:
    import paddleocr

    payload = {
        "generated_at": iso_now(),
        "ocr_version": OCR_VERSION,
        "paddleocr_version": getattr(paddleocr, "__version__", "unknown"),
        "detection_model": DETECTION_MODEL,
        "recognition_models": RECOGNITION_MODELS,
        "note": (
            "Checksums are of the files as downloaded into this project. "
            "They pin what this machine is running; they are not official "
            "upstream checksums, which PaddleOCR does not publish per file."
        ),
        "models": [describe_model(name, model_dir) for name in models],
    }
    return write_json(MANIFEST_PATH, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang",
        choices=[lang for lang in SUPPORTED_LANGUAGES],
        action="append",
        help="language(s) to fetch; repeatable. Default: ar and en.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"where to store models (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="report what is present locally; download nothing",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-copy even if already present"
    )
    args = parser.parse_args()

    languages = args.lang or ["ar", "en"]
    wanted: List[str] = []
    for lang in languages:
        for name in required_models(lang):
            if name not in wanted:
                wanted.append(name)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {args.model_dir}")
    print(f"Required models: {', '.join(wanted)}\n")

    if args.verify:
        missing = [n for n in wanted if not model_is_local(n, args.model_dir)]
        for name in wanted:
            present = model_is_local(name, args.model_dir)
            print(f"  [{'OK' if present else '--'}] {name}")
        if missing:
            print(f"\nMissing {len(missing)} model(s). Run without --verify.")
            return 1
        print("\nAll required models are present locally.")
        return 0

    if args.force:
        for name in wanted:
            target = args.model_dir / name
            if target.is_dir():
                shutil.rmtree(target)
                LOGGER.info("Removed %s for re-download", target)

    try:
        import paddleocr  # noqa: F401
    except ImportError:
        print("ERROR: paddleocr is not installed. Run scripts/setup_windows.bat first.")
        return 1

    failures: List[str] = []
    for lang in [lang for lang in languages if lang in RECOGNITION_MODELS]:
        try:
            trigger_download(lang)
        except Exception as exc:
            LOGGER.error("Failed to resolve models for '%s': %s", lang, exc)
            failures.append(f"{lang}: {exc}")

    print("\nCopying resolved models into the project ...")
    not_found: List[str] = []
    for name in wanted:
        if copy_into_project(name, args.model_dir) is None:
            not_found.append(name)

    if not_found:
        print("\nERROR: these models were not found in the PaddleOCR cache:")
        for name in not_found:
            print(f"  - {name}")
        print("\nChecked cache locations:")
        for path in _CACHE_CANDIDATES:
            print(f"  - {path}")
        print(
            "\nThe installed paddleocr release may use different model names. "
            "Do not substitute a different model silently -- check\n"
            "  https://www.paddleocr.ai/main/en/version3.x/"
            "algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html\n"
            "and update RECOGNITION_MODELS in src/phonics_ocr/config.py."
        )
        return 1

    manifest = write_manifest(args.model_dir, wanted)
    print(f"\nManifest written to {manifest}")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    total = sum(m.get("total_bytes", 0) for m in data["models"])
    print(f"Total model size: {human_bytes(total)}")
    for model in data["models"]:
        if model.get("present"):
            print(f"  {model['name']:<32} {model['total_human']:>10}")

    if failures:
        print("\nWarnings during resolution:")
        for failure in failures:
            print(f"  - {failure}")

    print("\nModels are local. Inference can now run offline:")
    print("  python scripts/verify_offline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

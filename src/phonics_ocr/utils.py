"""Shared helpers: logging, JSON writing, run identifiers, image discovery."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LOGGER_NAME = "phonics_ocr"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Attach a single stderr handler to the package logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def utc_timestamp() -> str:
    """UTC timestamp, filesystem-safe, second resolution."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def run_id(prefix: str = "run") -> str:
    """A per-run identifier used to keep result files from colliding."""
    return f"{prefix}_{utc_timestamp()}"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> Dict[str, str]:
    """Checksums of every file in a directory, keyed by relative path."""
    out: Dict[str, str] = {}
    if not path.is_dir():
        return out
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        out[file_path.relative_to(path).as_posix()] = sha256_file(file_path)
    return out


def write_json(path: Path, payload: Any, *, ensure_ascii: bool = False) -> Path:
    """Write UTF-8 JSON, creating parent directories.

    ``ensure_ascii=False`` is the default so that Arabic text stays
    readable in the result files. The bytes are UTF-8 either way; this
    only affects whether the characters are escaped.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=ensure_ascii, indent=2)
        handle.write("\n")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, reporting the line number of any bad record."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON -- {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_no}: each line must be a JSON object, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def find_images(target: Path) -> List[Path]:
    """Image files at ``target``, which may be a file or a directory."""
    if target.is_file():
        if target.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"{target} is not a supported image type "
                f"({', '.join(IMAGE_EXTENSIONS)})"
            )
        return [target]
    if target.is_dir():
        return sorted(
            p
            for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"No such file or directory: {target}")


def unique_path(path: Path) -> Path:
    """A path that does not exist yet, suffixing ``_2``, ``_3``, ... if needed."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def set_offline_env() -> Dict[str, str]:
    """Set the environment switches that forbid runtime model downloads.

    Returns the variables that were set, for logging. This must run
    *before* paddlex or paddleocr is imported to take full effect.
    """
    variables = {
        # PaddleX honours this to skip its model-hub lookups.
        "PADDLE_PDX_OFFLINE_MODE": "1",
        # Belt and braces for any transformers/hub code paths pulled in
        # by optional dependencies.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    os.environ.update(variables)
    return variables


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def summarise_paths(paths: Iterable[Path], root: Optional[Path] = None) -> List[str]:
    """Render paths relative to ``root`` where possible, for logging."""
    out: List[str] = []
    for p in paths:
        if root is not None:
            try:
                out.append(p.relative_to(root).as_posix())
                continue
            except ValueError:
                pass
        out.append(str(p))
    return out

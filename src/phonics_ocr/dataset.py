"""Dataset annotation schema (v2) and validation.

Schema v1 (``image``, ``text``, ``lang``, ``group``) is still accepted:
it covers transcription-only evaluation and nothing about it was wrong.
v2 adds what the projector work needs -- image dimensions, capture
provenance, and optional line/word polygons -- without making any of it
mandatory, so an existing v1 file keeps working untouched.

The one field v2 really adds is ``capture``. Whether a page was
*rendered*, *scanned* or *photographed* changes what a score from it
means, and that distinction is too important to leave implied by a
filename. Validation treats a missing ``capture`` as ``"unknown"`` and
reporting never pools unknown provenance with photographed pages.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "2.0"

#: How the image was produced. This drives whether a result may be
#: described as real-world performance.
CAPTURE_KINDS: Tuple[str, ...] = (
    "photograph",        # a camera photographing a physical page
    "flatbed_scan",      # a scanner: clean, flat, evenly lit
    "screenshot",        # a digital page captured from a screen
    "synthetic_render",  # glyphs rendered by code -- NOT a real page
    "unknown",
)

#: Capture kinds that may be described as real-world evidence.
REAL_CAPTURE_KINDS: Tuple[str, ...] = ("photograph",)

VALID_GROUPS: Tuple[str, ...] = (
    "arabic_tashkeel",
    "arabic_plain",
    "english",
    "mixed",
    "clean",
    "degraded",
    "clean_synthetic",
    "rotated_synthetic",
)

VALID_LANGS: Tuple[str, ...] = ("ar", "en", "mixed")

#: Conditions worth recording per image, for the camera experiments.
CONDITION_KEYS: Tuple[str, ...] = (
    "lighting",      # even | dim | harsh | shadowed | glare
    "blur",          # none | slight | moderate | severe
    "perspective",   # flat | slight | moderate | severe
    "curvature",     # flat | slight | moderate | severe
    "print_size",    # large | medium | small
)


class DatasetError(ValueError):
    """Raised for a malformed annotation record."""


@dataclasses.dataclass
class WordBox:
    """One annotated word."""

    text: str
    bbox: Tuple[float, float, float, float]
    polygon: Optional[List[List[float]]] = None
    line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text": self.text, "bbox": list(self.bbox)}
        if self.polygon is not None:
            out["polygon"] = self.polygon
        if self.line is not None:
            out["line"] = self.line
        return out


@dataclasses.dataclass
class Sample:
    """One validated annotation record."""

    id: str
    image: Path
    text: str
    lang: str
    group: str
    capture: str
    width: Optional[int] = None
    height: Optional[int] = None
    words: Optional[List[WordBox]] = None
    lines: Optional[List[List[List[float]]]] = None
    conditions: Optional[Dict[str, str]] = None
    box_source: str = ""
    notes: str = ""

    @property
    def is_real_capture(self) -> bool:
        """Whether a score from this sample describes real-world use."""
        return self.capture in REAL_CAPTURE_KINDS

    @property
    def has_word_boxes(self) -> bool:
        return bool(self.words)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "image": str(self.image),
            "lang": self.lang,
            "group": self.group,
            "capture": self.capture,
            "text": self.text,
        }
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        if self.words:
            out["words"] = [w.to_dict() for w in self.words]
        if self.lines:
            out["lines"] = self.lines
        if self.conditions:
            out["conditions"] = self.conditions
        if self.box_source:
            out["box_source"] = self.box_source
        if self.notes:
            out["notes"] = self.notes
        return out


def _validate_bbox(raw: Any, where: str) -> Tuple[float, float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise DatasetError(f"{where}: bbox must be [x_min, y_min, x_max, y_max]")
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError):
        raise DatasetError(f"{where}: bbox values must be numbers") from None
    if x1 <= x0 or y1 <= y0:
        raise DatasetError(
            f"{where}: degenerate bbox {raw} -- max must exceed min"
        )
    return (x0, y0, x1, y1)


def _validate_polygon(raw: Any, where: str) -> List[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise DatasetError(f"{where}: polygon needs at least 3 vertices")
    out: List[List[float]] = []
    for vertex in raw:
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            raise DatasetError(f"{where}: each polygon vertex must be [x, y]")
        out.append([float(vertex[0]), float(vertex[1])])
    return out


def validate_record(
    record: Dict[str, Any], where: str, root: Optional[Path] = None
) -> Sample:
    """Validate one annotation record and resolve its image path.

    ``where`` is used verbatim in error messages, so pass something
    locatable such as ``"labels.jsonl:12"``.
    """
    if not isinstance(record, dict):
        raise DatasetError(f"{where}: record must be a JSON object")

    for field in ("image", "text", "lang"):
        if field not in record:
            raise DatasetError(f"{where}: missing required field {field!r}")

    lang = record["lang"]
    if lang not in VALID_LANGS:
        raise DatasetError(
            f"{where}: lang must be one of {', '.join(VALID_LANGS)}, got {lang!r}"
        )

    image_path = Path(record["image"])
    if not image_path.is_absolute() and root is not None:
        for base in (root, root.parent):
            candidate = base / image_path
            if candidate.is_file():
                image_path = candidate.resolve()
                break

    capture = record.get("capture", "unknown")
    if capture not in CAPTURE_KINDS:
        raise DatasetError(
            f"{where}: capture must be one of {', '.join(CAPTURE_KINDS)}, "
            f"got {capture!r}"
        )

    group = record.get("group") or f"{lang}_ungrouped"

    words: Optional[List[WordBox]] = None
    if "words" in record:
        if not isinstance(record["words"], list):
            raise DatasetError(f"{where}: 'words' must be a list")
        words = []
        for i, entry in enumerate(record["words"]):
            place = f"{where} word[{i}]"
            if not isinstance(entry, dict) or "text" not in entry:
                raise DatasetError(f"{place}: each word needs a 'text' field")
            if "bbox" not in entry:
                raise DatasetError(
                    f"{place}: each word needs a 'bbox'. Omit the whole "
                    "'words' list if boxes have not been annotated."
                )
            words.append(
                WordBox(
                    text=entry["text"],
                    bbox=_validate_bbox(entry["bbox"], place),
                    polygon=(
                        _validate_polygon(entry["polygon"], place)
                        if "polygon" in entry
                        else None
                    ),
                    line=entry.get("line"),
                )
            )

    lines: Optional[List[List[List[float]]]] = None
    if "lines" in record:
        if not isinstance(record["lines"], list):
            raise DatasetError(f"{where}: 'lines' must be a list of polygons")
        lines = [
            _validate_polygon(p, f"{where} line[{i}]")
            for i, p in enumerate(record["lines"])
        ]

    conditions = record.get("conditions")
    if conditions is not None:
        if not isinstance(conditions, dict):
            raise DatasetError(f"{where}: 'conditions' must be an object")
        unknown = set(conditions) - set(CONDITION_KEYS)
        if unknown:
            raise DatasetError(
                f"{where}: unknown condition key(s) {sorted(unknown)}. "
                f"Valid: {', '.join(CONDITION_KEYS)}"
            )

    width = record.get("width")
    height = record.get("height")
    for name, value in (("width", width), ("height", height)):
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise DatasetError(f"{where}: {name} must be a positive integer")

    return Sample(
        id=record.get("id") or image_path.stem,
        image=image_path,
        text=record["text"],
        lang=lang,
        group=group,
        capture=capture,
        width=width,
        height=height,
        words=words,
        lines=lines,
        conditions=conditions,
        box_source=record.get("box_source", ""),
        notes=record.get("notes", ""),
    )


def check_against_image(sample: Sample) -> List[str]:
    """Cross-check a record against the actual image file.

    Returns a list of human-readable problems; empty means consistent.
    Boxes outside the image, or declared dimensions that disagree with
    the file, silently corrupt every IoU score computed from them, so
    they are worth catching before a run rather than after.
    """
    problems: List[str] = []
    if not sample.image.is_file():
        return [f"image not found: {sample.image}"]

    try:
        from .preprocessing import load_image

        image = load_image(sample.image)
    except Exception as exc:
        return [f"could not read image: {exc}"]

    height, width = image.shape[:2]
    if sample.width is not None and sample.width != width:
        problems.append(
            f"declared width {sample.width} but the file is {width}px wide"
        )
    if sample.height is not None and sample.height != height:
        problems.append(
            f"declared height {sample.height} but the file is {height}px tall"
        )

    for i, word in enumerate(sample.words or []):
        x0, y0, x1, y1 = word.bbox
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            problems.append(
                f"word[{i}] {word.text!r} bbox {list(word.bbox)} lies outside "
                f"the {width}x{height} image"
            )

    if sample.words:
        from . import postprocessing as pp

        joined = pp.normalise_text(" ".join(w.text for w in sample.words))
        full = pp.normalise_text(sample.text)
        if joined != full:
            problems.append(
                "the concatenated word annotations do not match 'text' "
                "(whitespace-normalised). Word boxes and the transcription "
                "must describe the same content."
            )
    return problems


def summarise(samples: Sequence[Sample]) -> Dict[str, Any]:
    """Counts a reader needs before trusting any score from this set."""
    by_group: Dict[str, int] = {}
    by_capture: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    for sample in samples:
        by_group[sample.group] = by_group.get(sample.group, 0) + 1
        by_capture[sample.capture] = by_capture.get(sample.capture, 0) + 1
        by_lang[sample.lang] = by_lang.get(sample.lang, 0) + 1

    real = [s for s in samples if s.is_real_capture]
    with_words = [s for s in samples if s.has_word_boxes]
    return {
        "schema_version": SCHEMA_VERSION,
        "n_samples": len(samples),
        "n_real_capture": len(real),
        "n_with_word_boxes": len(with_words),
        "n_word_boxes": sum(len(s.words or []) for s in with_words),
        "by_group": dict(sorted(by_group.items())),
        "by_capture": dict(sorted(by_capture.items())),
        "by_lang": dict(sorted(by_lang.items())),
        "real_world_claims_supported": bool(real),
    }

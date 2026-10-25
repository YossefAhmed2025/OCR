"""Evaluate OCR output against ground truth.

    python scripts/evaluate.py --ground-truth data/ground_truth/samples.jsonl
    python scripts/evaluate.py --ground-truth data/ground_truth/samples.jsonl \
        --device cpu --strip-tashkeel-view

Ground truth is JSONL, one object per line. See
``data/ground_truth/README.md`` and
``data/ground_truth/example.jsonl`` for the schema.

This script runs inference itself so that a score is always traceable
to a specific configuration, and writes a report containing both the
metrics and the configuration that produced them.

It will not invent ground truth and it will not report a score without
the counts behind it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

VALID_GROUPS = (
    "arabic_tashkeel",
    "arabic_plain",
    "english",
    "mixed",
    "clean",
    "degraded",
)


def validate_record(record: Dict[str, Any], line_no: int, root: Path) -> Dict[str, Any]:
    """Check one ground-truth record and resolve its image path."""
    for field in ("image", "text", "lang"):
        if field not in record:
            raise ValueError(f"record {line_no}: missing required field '{field}'")

    image_path = Path(record["image"])
    if not image_path.is_absolute():
        for base in (root, root.parent, PROJECT_ROOT):
            candidate = base / image_path
            if candidate.is_file():
                image_path = candidate.resolve()
                break
    if not image_path.is_file():
        raise ValueError(f"record {line_no}: image not found: {record['image']}")

    if record["lang"] not in ("ar", "en", "mixed"):
        raise ValueError(
            f"record {line_no}: lang must be ar, en or mixed, got {record['lang']!r}"
        )

    group = record.get("group", "")
    if group and group not in VALID_GROUPS:
        print(
            f"  [warning] record {line_no}: group {group!r} is not one of "
            f"{', '.join(VALID_GROUPS)}; it will still be reported."
        )

    return {
        "id": record.get("id", image_path.stem),
        "image": image_path,
        "text": record["text"],
        "lang": record["lang"],
        "group": group or f"{record['lang']}_ungrouped",
        "notes": record.get("notes", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="a .jsonl file, or a directory containing .jsonl files",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--strip-tashkeel-view",
        action="store_true",
        help=(
            "ALSO report metrics on tashkeel-stripped text, as a separate "
            "secondary view. The vowelled metrics are always reported too."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from phonics_ocr import evaluation as ev
    from phonics_ocr import postprocessing as pp
    from phonics_ocr.config import DEFAULT_MODEL_DIR, OCRConfig
    from phonics_ocr.engine import EngineError, OCREngine
    from phonics_ocr.utils import (
        configure_logging,
        iso_now,
        read_jsonl,
        run_id,
        set_offline_env,
        write_json,
    )

    configure_logging(args.verbose)
    if args.offline:
        set_offline_env()

    # -- load ground truth ------------------------------------------
    gt_path = args.ground_truth
    if gt_path.is_dir():
        files = sorted(gt_path.glob("*.jsonl"))
        if not files:
            print(f"ERROR: no .jsonl files in {gt_path}", file=sys.stderr)
            print("       See data/ground_truth/README.md for the format.")
            return 1
    elif gt_path.is_file():
        files = [gt_path]
    else:
        print(f"ERROR: ground truth not found: {gt_path}", file=sys.stderr)
        return 1

    samples: List[Dict[str, Any]] = []
    try:
        for file_path in files:
            print(f"Reading {file_path}")
            for line_no, record in enumerate(read_jsonl(file_path), start=1):
                samples.append(validate_record(record, line_no, file_path.parent))
    except ValueError as exc:
        print(f"ERROR in ground truth: {exc}", file=sys.stderr)
        return 1

    if not samples:
        print("ERROR: ground truth contains no records.", file=sys.stderr)
        print("       Label some images first -- see data/ground_truth/README.md.")
        return 1

    print(f"Loaded {len(samples)} labelled sample(s)\n")

    # -- run inference, one engine per language ---------------------
    by_lang: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        by_lang.setdefault(sample["lang"], []).append(sample)

    predictions: Dict[str, Dict[str, Any]] = {}
    engine_reports: Dict[str, Any] = {}
    for lang, group in by_lang.items():
        print(f"Running '{lang}' pipeline over {len(group)} image(s) ...")
        try:
            config = OCRConfig(
                lang=lang,
                device=args.device,
                model_dir=args.model_dir or DEFAULT_MODEL_DIR,
                offline=args.offline,
            )
            engine = OCREngine(config)
        except (EngineError, ValueError) as exc:
            print(f"ERROR: could not build the '{lang}' engine -- {exc}", file=sys.stderr)
            return 1

        engine_reports[lang] = {
            "config": config.to_dict(),
            "device_used": engine.device,
        }
        for sample in group:
            try:
                result = engine.process_image(sample["image"])
            except Exception as exc:
                print(f"  FAILED {sample['id']}: {exc}", file=sys.stderr)
                continue
            predictions[sample["id"]] = result
            print(
                f"  {sample['id']:<28} {result['n_regions']:>3} regions "
                f"{result['timing']['inference_seconds']:>7.3f}s"
            )

    if not predictions:
        print("\nERROR: inference produced no results; nothing to score.", file=sys.stderr)
        return 1

    # -- score -------------------------------------------------------
    # Both sides go through the SAME normalisation so the comparison is
    # fair, and the exact settings are recorded in the report.
    normalisation = {
        "unicode_form": "NFC",
        "collapse_whitespace": True,
        "remove_tatweel": True,
        "remove_tashkeel": False,
        "description": (
            "NFC composition, tatweel removed, whitespace collapsed. "
            "Tashkeel are PRESERVED: a missing diacritic counts as an error."
        ),
    }

    def prepare(text: str, strip: bool) -> str:
        return pp.normalise_text(
            text,
            unicode_form="NFC",
            collapse_whitespace=True,
            remove_tatweel=True,
            remove_tashkeel=strip,
        )

    primary: List[ev.SampleScore] = []
    stripped: List[ev.SampleScore] = []
    per_sample: List[Dict[str, Any]] = []

    for sample in samples:
        result = predictions.get(sample["id"])
        if result is None:
            continue
        hypothesis_raw = result["transcription_raw"]

        score = ev.score_sample(
            sample["id"],
            sample["group"],
            prepare(sample["text"], False),
            prepare(hypothesis_raw, False),
            score_diacritics=sample["lang"] in ("ar", "mixed"),
        )
        primary.append(score)

        entry: Dict[str, Any] = {
            **score.to_dict(),
            "lang": sample["lang"],
            "image": str(sample["image"]),
            "notes": sample["notes"],
            "n_regions": result["n_regions"],
            "inference_seconds": result["timing"]["inference_seconds"],
            "reference_has_tashkeel": pp.has_tashkeel(sample["text"]),
            "hypothesis_has_tashkeel": pp.has_tashkeel(hypothesis_raw),
            "reference_tashkeel_count": pp.count_tashkeel(sample["text"]),
            "hypothesis_tashkeel_count": pp.count_tashkeel(hypothesis_raw),
        }

        if args.strip_tashkeel_view:
            s2 = ev.score_sample(
                sample["id"],
                sample["group"],
                prepare(sample["text"], True),
                prepare(hypothesis_raw, True),
                score_diacritics=False,
            )
            stripped.append(s2)
            entry["consonants_only"] = {
                "cer": s2.cer.to_dict(),
                "wer": s2.wer.to_dict(),
                "exact_match": s2.exact,
            }
        per_sample.append(entry)

    times = [
        predictions[s["id"]]["timing"]["inference_seconds"]
        for s in samples
        if s["id"] in predictions
    ]

    report: Dict[str, Any] = {
        "generated_at": iso_now(),
        "ground_truth_files": [str(f) for f in files],
        "n_labelled": len(samples),
        "n_scored": len(primary),
        "n_inference_failures": len(samples) - len(primary),
        "normalisation": normalisation,
        "tokenisation": "whitespace split; punctuation is not separated",
        "cer_unit": "Unicode code point (a combining mark counts as one unit)",
        "averaging": "micro -- edit distances and reference lengths are summed",
        "engines": engine_reports,
        "mean_inference_seconds": round(sum(times) / len(times), 4) if times else None,
        "metrics_with_tashkeel": ev.aggregate_by_group(primary),
        "per_sample": per_sample,
    }
    if args.strip_tashkeel_view:
        report["metrics_consonants_only"] = ev.aggregate_by_group(stripped)
        report["metrics_consonants_only_note"] = (
            "SECONDARY VIEW. Tashkeel removed from BOTH sides before scoring. "
            "This measures the consonantal skeleton only and says nothing "
            "about whether the vowelling was read correctly."
        )

    output_root = args.output_dir or (PROJECT_ROOT / "data" / "results")
    out_path = output_root / f"{run_id('eval')}.json"
    write_json(out_path, report)

    # -- print -------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("EVALUATION  (tashkeel preserved -- a missing diacritic is an error)")
    print(f"{'=' * 72}")
    # Diacritic RECALL is shown rather than F1: when a model predicts no
    # diacritics at all, precision and F1 are undefined while recall is
    # a meaningful 0.0, and that is exactly the case worth seeing.
    print(
        f"{'group':<22} {'n':>3} {'CER':>8} {'WER':>8} {'exact':>7} "
        f"{'dia-R':>7} {'dia-F1':>7}"
    )
    print(f"{'-' * 72}")
    for name, metrics in report["metrics_with_tashkeel"].items():
        if metrics["n_samples"] == 0:
            continue
        cer = metrics["cer"]["rate"]
        wer = metrics["wer"]["rate"]
        exact = metrics["exact_match_rate"]
        dia = metrics["diacritic"] or {}
        recall, f1 = dia.get("recall"), dia.get("f1")
        print(
            f"{name:<22} {metrics['n_samples']:>3} "
            f"{('n/a' if cer is None else f'{cer:.4f}'):>8} "
            f"{('n/a' if wer is None else f'{wer:.4f}'):>8} "
            f"{('n/a' if exact is None else f'{exact:.2f}'):>7} "
            f"{('n/a' if recall is None else f'{recall:.4f}'):>7} "
            f"{('n/a' if f1 is None else f'{f1:.4f}'):>7}"
        )

    overall = report["metrics_with_tashkeel"]["overall"]
    print(f"\nDenominators: {overall['cer']['reference_length']} reference characters, "
          f"{overall['wer']['reference_length']} reference words")
    dia = overall.get("diacritic")
    if dia:
        print(
            f"Diacritics  : TP={dia['true_positives']} FP={dia['false_positives']} "
            f"FN={dia['false_negatives']}"
        )
        if dia["precision"] is None:
            print("              precision undefined: the model predicted NO diacritics.")

    if args.strip_tashkeel_view:
        print(f"\n{'-' * 72}")
        print("SECONDARY VIEW: consonants only (tashkeel stripped from both sides)")
        c_overall = report["metrics_consonants_only"]["overall"]
        c_cer, c_wer = c_overall["cer"]["rate"], c_overall["wer"]["rate"]
        print(
            f"  CER {('n/a' if c_cer is None else f'{c_cer:.4f}')}   "
            f"WER {('n/a' if c_wer is None else f'{c_wer:.4f}')}"
        )
        print("  This does NOT measure diacritic accuracy.")

    if report["n_inference_failures"]:
        print(f"\nWARNING: {report['n_inference_failures']} image(s) failed inference "
              "and are excluded from every figure above.")

    print(f"\nFull report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

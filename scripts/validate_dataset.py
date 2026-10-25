"""Validate an annotation file before anyone evaluates against it.

    python scripts/validate_dataset.py --ground-truth data/ground_truth/my.jsonl
    python scripts/validate_dataset.py --ground-truth data/localization/boxes.jsonl

Checks every record against the v2 schema, cross-checks declared
dimensions and word boxes against the actual image files, and prints a
provenance summary.

The provenance summary is the point. It states plainly how many
samples are real photographs, because that is what decides whether a
score from this set may be described as real-world performance. A set
of rendered images can produce a perfectly valid 0.98 and still say
nothing about a camera.

Exit codes: 0 clean, 1 errors found, 2 file problems.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--skip-image-checks",
        action="store_true",
        help="validate the schema only; do not open the image files",
    )
    parser.add_argument("--json", type=Path, help="write the summary here")
    args = parser.parse_args()

    from phonics_ocr.dataset import (
        DatasetError,
        REAL_CAPTURE_KINDS,
        check_against_image,
        summarise,
        validate_record,
    )
    from phonics_ocr.utils import read_jsonl, write_json

    paths: List[Path]
    if args.ground_truth.is_dir():
        paths = sorted(args.ground_truth.glob("*.jsonl"))
        if not paths:
            print(f"ERROR: no .jsonl files in {args.ground_truth}", file=sys.stderr)
            return 2
    elif args.ground_truth.is_file():
        paths = [args.ground_truth]
    else:
        print(f"ERROR: not found: {args.ground_truth}", file=sys.stderr)
        return 2

    samples = []
    errors: List[str] = []
    warnings: List[str] = []

    for path in paths:
        print(f"Reading {path}")
        try:
            records = read_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        for line_no, record in enumerate(records, start=1):
            where = f"{path.name}:{line_no}"
            try:
                sample = validate_record(record, where, root=path.parent)
            except DatasetError as exc:
                errors.append(str(exc))
                continue

            if "TEMPLATE ROW" in sample.notes.upper():
                warnings.append(f"{where}: template row -- not real data")

            if not args.skip_image_checks:
                for problem in check_against_image(sample):
                    errors.append(f"{where}: {problem}")
            samples.append(sample)

    print(f"\nParsed {len(samples)} record(s)")

    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for item in errors:
            print(f"  - {item}")
    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for item in warnings:
            print(f"  - {item}")

    if not samples:
        print("\nNothing valid to summarise.")
        return 1 if errors else 2

    summary = summarise(samples)
    print(f"\n{'=' * 68}")
    print("DATASET SUMMARY")
    print(f"{'=' * 68}")
    print(f"  samples            : {summary['n_samples']}")
    print(f"  with word boxes    : {summary['n_with_word_boxes']} "
          f"({summary['n_word_boxes']} boxes)")
    print(f"  by language        : {summary['by_lang']}")
    print(f"  by group           : {summary['by_group']}")
    print(f"  by capture         : {summary['by_capture']}")

    print(f"\n{'-' * 68}")
    if summary["real_world_claims_supported"]:
        print(f"  {summary['n_real_capture']} of {summary['n_samples']} sample(s) "
              f"are real photographs.")
        print("  Scores from THOSE samples may be described as real-world.")
        if summary["n_real_capture"] < 50:
            print(f"  But {summary['n_real_capture']} is a small sample -- see")
            print("  docs/evaluation_protocol.md section 6 on sample size.")
    else:
        print("  NO sample in this set is a real photograph")
        print(f"  (capture kinds present: {sorted(summary['by_capture'])};")
        print(f"   real kinds are: {', '.join(REAL_CAPTURE_KINDS)}).")
        print("\n  Any score from this set is an UPPER BOUND. It must not be")
        print("  reported as Phonics Prism accuracy on real pages.")
    print(f"{'-' * 68}")

    if args.json:
        write_json(args.json, {**summary, "errors": errors, "warnings": warnings})
        print(f"\nSummary written to {args.json}")

    if errors:
        print(f"\nFAILED: {len(errors)} error(s). Fix these before evaluating.")
        return 1
    print("\nOK: schema and image cross-checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

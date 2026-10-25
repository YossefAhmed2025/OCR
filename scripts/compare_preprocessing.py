"""Measure whether preprocessing actually helps. Do not assume it does.

    python scripts/compare_preprocessing.py --ground-truth data/ground_truth/my.jsonl
    python scripts/compare_preprocessing.py --ground-truth data/ground_truth/my.jsonl \
        --variants original clahe upscale2 denoise

Runs the same labelled images through several preprocessing variants
and reports CER, WER and **diacritic recall** per variant, against the
same ground truth.

The diacritic column is the one to watch. Denoising and aggressive
contrast routinely make consonants cleaner while erasing tashkeel, so
a variant can improve CER on unvowelled text and destroy the signal
this project exists to measure. A single pooled accuracy number hides
that; this table does not.

Variants are compared against `original` and the delta is shown with
its sign, so a harmful variant is visibly harmful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

#: name -> PreprocessConfig kwargs
VARIANTS: Dict[str, Dict[str, Any]] = {
    "original": {},
    "grayscale": {"grayscale": True},
    "clahe": {"clahe": True},
    "clahe_strong": {"clahe": True, "clahe_clip_limit": 4.0},
    "denoise": {"denoise": True},
    "deskew": {"deskew": True},
    "upscale15": {"upscale": 1.5},
    "upscale2": {"upscale": 2.0},
    "upscale3": {"upscale": 3.0},
    "clahe_upscale2": {"clahe": True, "upscale": 2.0},
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", default=["original", "clahe", "upscale2", "denoise"],
        help=f"any of: {', '.join(VARIANTS)}",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from phonics_ocr import evaluation as ev
    from phonics_ocr import postprocessing as pp
    from phonics_ocr.config import DEFAULT_MODEL_DIR, OCRConfig, PreprocessConfig
    from phonics_ocr.dataset import DatasetError, validate_record
    from phonics_ocr.engine import EngineError, OCREngine
    from phonics_ocr.utils import configure_logging, iso_now, read_jsonl, run_id, write_json

    configure_logging(args.verbose)

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        print(f"ERROR: unknown variant(s) {unknown}", file=sys.stderr)
        print(f"       available: {', '.join(VARIANTS)}", file=sys.stderr)
        return 1

    if not args.ground_truth.is_file():
        print(f"ERROR: not found: {args.ground_truth}", file=sys.stderr)
        return 1

    samples = []
    for line_no, record in enumerate(read_jsonl(args.ground_truth), start=1):
        try:
            samples.append(
                validate_record(
                    record, f"{args.ground_truth.name}:{line_no}",
                    root=args.ground_truth.parent,
                )
            )
        except DatasetError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if not samples:
        print("ERROR: no samples.", file=sys.stderr)
        return 1

    captures = {s.capture for s in samples}
    real = sum(1 for s in samples if s.is_real_capture)
    print(f"{len(samples)} sample(s); {real} real photograph(s)\n")

    def prepare(text: str) -> str:
        return pp.normalise_text(text, unicode_form="NFC", collapse_whitespace=True,
                                 remove_tatweel=True, remove_tashkeel=False)

    rows: List[Dict[str, Any]] = []
    for variant in args.variants:
        kwargs = VARIANTS[variant]
        print(f"--- {variant} {kwargs or '(no preprocessing)'} ---")
        scores: List[ev.SampleScore] = []
        dia_total = ev.PRF(0, 0, 0)
        times: List[float] = []
        engines: Dict[str, Any] = {}

        for sample in samples:
            lang = sample.lang if sample.lang != "mixed" else "ar"
            if lang not in engines:
                try:
                    engines[lang] = OCREngine(
                        OCRConfig(
                            lang=lang, device=args.device,
                            model_dir=DEFAULT_MODEL_DIR,
                            preprocess=PreprocessConfig(**kwargs),
                        )
                    )
                except (EngineError, ValueError) as exc:
                    print(f"  ERROR: {exc}", file=sys.stderr)
                    return 1
            try:
                result = engines[lang].process_image(sample.image)
            except Exception as exc:
                print(f"  [fail] {sample.id}: {exc}")
                continue

            hypothesis = result["transcription_raw"]
            score = ev.score_sample(
                sample.id, sample.group, prepare(sample.text), prepare(hypothesis),
                score_diacritics=sample.lang in ("ar", "mixed"),
            )
            scores.append(score)
            if score.diacritic:
                dia_total = dia_total + score.diacritic
            times.append(result["timing"]["inference_seconds"])

        if not scores:
            print("  no results\n")
            continue
        agg = ev.aggregate(scores)
        rows.append(
            {
                "variant": variant,
                "config": kwargs,
                "n_samples": agg["n_samples"],
                "cer": agg["cer"]["rate"],
                "wer": agg["wer"]["rate"],
                "exact_match_rate": agg["exact_match_rate"],
                "reference_characters": agg["cer"]["reference_length"],
                "diacritic": dia_total.to_dict(),
                "mean_inference_seconds": round(sum(times) / len(times), 4) if times else None,
            }
        )
        print(f"  CER {agg['cer']['rate']}  dia-recall {dia_total.recall}\n")

    baseline = next((r for r in rows if r["variant"] == "original"), None)

    print("=" * 86)
    print("PREPROCESSING COMPARISON  (tashkeel preserved)")
    print("=" * 86)
    print(f"{'variant':<16}{'CER':>8}{'dCER':>8}{'WER':>8}"
          f"{'dia-R':>8}{'d-dia':>8}{'sec':>8}")
    print("-" * 86)
    for row in rows:
        cer = row["cer"]
        dia = row["diacritic"]["recall"]

        def delta(value, base_value, invert=False):
            if value is None or base_value is None or row is baseline:
                return "   --   "
            diff = value - base_value
            if invert:
                diff = -diff
            return f"{diff:+8.4f}"

        wer = row["wer"]
        secs = row["mean_inference_seconds"]
        base_cer = baseline["cer"] if baseline else None
        base_dia = baseline["diacritic"]["recall"] if baseline else None

        cer_s = f"{cer:.4f}" if cer is not None else "n/a"
        wer_s = f"{wer:.4f}" if wer is not None else "n/a"
        dia_s = f"{dia:.4f}" if dia is not None else "n/a"
        sec_s = f"{secs:.3f}" if secs else "n/a"

        print(
            f"{row['variant']:<16}{cer_s:>8}{delta(cer, base_cer):>8}"
            f"{wer_s:>8}{dia_s:>8}{delta(dia, base_dia):>8}{sec_s:>8}"
        )

    print("\ndCER: negative is better (fewer errors).")
    print("d-dia: positive is better (more diacritics recovered).")
    print("A variant that improves CER while reducing dia-R is removing")
    print("tashkeel, not reading better. Check both columns.")
    if real == 0:
        print(f"\nNOTE: no real photographs in this set (capture: "
              f"{sorted(captures)}).\n      These deltas may not hold on "
              f"camera images.")

    report = {
        "generated_at": iso_now(),
        "ground_truth": str(args.ground_truth),
        "n_samples": len(samples),
        "n_real_capture": real,
        "capture_kinds": sorted(captures),
        "variants": rows,
        "caveat": (
            "Preprocessing effects are image-dependent. Deltas measured "
            "on one set do not transfer to differently-degraded images."
        ),
    }
    out = args.output or (PROJECT_ROOT / "data" / "results" / f"{run_id('preprocessing')}.json")
    write_json(out, report)
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

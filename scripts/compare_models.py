"""Compare two recognition models on the same labelled images.

    python scripts/compare_models.py --ground-truth data/ground_truth/book_page.jsonl
    python scripts/compare_models.py --ground-truth data/ground_truth/book_page.jsonl \
        --candidate output/arabic_vowelled/best_accuracy/inference

Exists because a fine-tune is only worth keeping if it beats the model
it started from, on the *same* images, under the *same* metrics. The
training loop's own accuracy number is exact-string-match on a
synthetic validation split from the same generator -- it says the
model fitted the data, not that it reads pages better.

Reports CER, WER and diacritic recall for each model side by side,
with the delta. Diacritic recall is the column that matters: a model
that improves CER while losing diacritics has learned to skip them,
which is the exact failure this project exists to avoid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def score_with_model(
    samples: List[Any],
    model_dir: Optional[Path],
    model_name: str,
    device: str,
    label: str,
) -> Dict[str, Any]:
    """Run one recognition model over the samples and score it."""
    from phonics_ocr import evaluation as ev
    from phonics_ocr import postprocessing as pp
    from phonics_ocr.config import DEFAULT_MODEL_DIR, DETECTION_MODEL, OCRConfig
    from phonics_ocr.engine import EngineError, OCREngine

    from paddleocr import PaddleOCR

    kwargs: Dict[str, Any] = {
        "device": device,
        "text_detection_model_name": DETECTION_MODEL,
        "text_detection_model_dir": str(DEFAULT_MODEL_DIR / DETECTION_MODEL),
        "text_recognition_model_name": model_name,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
    }
    if model_dir is not None:
        kwargs["text_recognition_model_dir"] = str(model_dir)

    pipeline = PaddleOCR(**kwargs)

    cer_total = ev.ErrorRate(0, 0, 0, 0)
    wer_total = ev.ErrorRate(0, 0, 0, 0)
    dia_total = ev.PRF(0, 0, 0)
    rows: List[Dict[str, Any]] = []

    for sample in samples:
        results = pipeline.predict(str(sample.image))
        texts: List[str] = []
        for result in results or []:
            payload = result.json.get("res", result.json)
            texts.extend(str(t) for t in (payload.get("rec_texts") or []))
        hypothesis = " ".join(texts)

        reference = pp.normalise_text(sample.text)
        got = pp.normalise_text(hypothesis)
        cer = ev.character_error_rate(reference, got)
        wer = ev.word_error_rate(reference, got)
        dia = ev.diacritic_prf(reference, got)
        cer_total = cer_total + cer
        wer_total = wer_total + wer
        dia_total = dia_total + dia
        rows.append(
            {
                "id": sample.id,
                "cer": cer.rate,
                "wer": wer.rate,
                "dia_tp": dia.true_positives,
                "dia_fn": dia.false_negatives,
                "hypothesis": hypothesis,
            }
        )

    return {
        "label": label,
        "model_name": model_name,
        "model_dir": str(model_dir) if model_dir else "(bundled)",
        "cer": cer_total.rate,
        "wer": wer_total.rate,
        "diacritic_recall": dia_total.recall,
        "diacritic_precision": dia_total.precision,
        "dia_tp": dia_total.true_positives,
        "dia_fp": dia_total.false_positives,
        "dia_fn": dia_total.false_negatives,
        "reference_characters": cer_total.reference_length,
        "per_sample": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "models" / "arabic_PP-OCRv5_mobile_rec",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=PROJECT_ROOT / "output" / "arabic_vowelled" / "best_accuracy" / "inference",
        help="the fine-tuned inference model directory",
    )
    parser.add_argument("--model-name", default="arabic_PP-OCRv5_mobile_rec")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from phonics_ocr.dataset import DatasetError, validate_record
    from phonics_ocr.utils import configure_logging, iso_now, read_jsonl, write_json

    configure_logging(False)

    if not args.ground_truth.is_file():
        print(f"ERROR: not found: {args.ground_truth}", file=sys.stderr)
        return 1

    samples = []
    for line_no, record in enumerate(read_jsonl(args.ground_truth), start=1):
        try:
            samples.append(
                validate_record(
                    record,
                    f"{args.ground_truth.name}:{line_no}",
                    root=args.ground_truth.parent,
                )
            )
        except DatasetError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if not samples:
        print("ERROR: no samples.", file=sys.stderr)
        return 1

    real = sum(1 for s in samples if s.is_real_capture)
    print(f"{len(samples)} sample(s), {real} real photograph(s)\n")

    runs: List[Dict[str, Any]] = []

    print(f"--- baseline: {args.baseline.name} ---")
    runs.append(
        score_with_model(
            samples, args.baseline, args.model_name, args.device, "baseline"
        )
    )

    if args.candidate.is_dir():
        print(f"--- candidate: {args.candidate} ---")
        runs.append(
            score_with_model(
                samples, args.candidate, args.model_name, args.device, "fine-tuned"
            )
        )
    else:
        print(f"[skip] candidate not found: {args.candidate}")
        print("       Train it first -- see docs/finetuning.md")

    def fmt(value: Optional[float], width: int = 9) -> str:
        return f"{value:.4f}".rjust(width) if value is not None else "n/a".rjust(width)

    print(f"\n{'=' * 74}")
    print(f"{'model':<14}{'CER':>9}{'WER':>9}{'dia-recall':>12}{'dia TP/FN':>12}")
    print("-" * 74)
    for run in runs:
        print(
            f"{run['label']:<14}{fmt(run['cer'])}{fmt(run['wer'])}"
            f"{fmt(run['diacritic_recall'], 12)}"
            f"{(str(run['dia_tp']) + '/' + str(run['dia_fn'])):>12}"
        )

    if len(runs) == 2:
        base, cand = runs
        print("-" * 74)

        def delta(new: Optional[float], old: Optional[float], lower_is_better: bool) -> str:
            if new is None or old is None:
                return "n/a"
            diff = new - old
            good = (diff < 0) if lower_is_better else (diff > 0)
            arrow = "better" if good else ("same" if diff == 0 else "WORSE")
            return f"{diff:+.4f}  {arrow}"

        print(f"  CER            {delta(cand['cer'], base['cer'], True)}")
        print(f"  WER            {delta(cand['wer'], base['wer'], True)}")
        print(
            f"  diacritic rec. "
            f"{delta(cand['diacritic_recall'], base['diacritic_recall'], False)}"
        )
        print(
            "\n  Keep the fine-tune only if diacritic recall improved."
            "\n  A CER gain with a diacritic loss means it learned to skip marks."
        )

    print(f"\nDenominator: {runs[0]['reference_characters']} reference characters")
    if real < 20:
        print(f"NOTE: {real} real photograph(s) is a small sample. See")
        print("      docs/evaluation_protocol.md section 6.")

    out = args.output or (
        PROJECT_ROOT / "data" / "results" / "model_comparison.json"
    )
    write_json(
        out,
        {
            "generated_at": iso_now(),
            "ground_truth": str(args.ground_truth),
            "n_samples": len(samples),
            "n_real_photographs": real,
            "runs": runs,
        },
    )
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

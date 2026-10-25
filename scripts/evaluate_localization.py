"""Evaluate word-level localisation against annotated reference boxes.

    python scripts/evaluate_localization.py
    python scripts/evaluate_localization.py --ground-truth data/localization/boxes.jsonl
    python scripts/evaluate_localization.py --iou-thresholds 0.3 0.5 0.7

Protocol
--------
For each image the OCR engine produces line polygons; each line is
rectified and its words are localised by CTC alignment
(:mod:`phonics_ocr.wordseg`). Predicted word boxes are then matched to
reference boxes **greedily by descending IoU**, one-to-one, at each
threshold. From that:

* **Recall**    = matched references / total references
* **Precision** = matched predictions / total predictions
* **F1**        = harmonic mean
* **mean IoU**  = over matched pairs only (stated, because it ignores
  the misses and so always looks better than the whole picture)

Text is reported separately from geometry. A box can be in the right
place with the wrong text, or vice versa, and conflating them hides
which half is broken.

Reference boxes must come from an annotation file -- this script never
invents them, and it refuses to run on images with no `words` entry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_GT = PROJECT_ROOT / "data" / "localization" / "boxes.jsonl"


def horizontal_iou(
    a: Sequence[float], b: Sequence[float]
) -> float:
    """1-D IoU over the x-axis only.

    CTC alignment estimates **horizontal** position; it has no vertical
    information at all, so every word box inherits its line's vertical
    extent. Judging it with 2-D IoU therefore mixes a real measurement
    (where along the line the word sits) with a known constant (the
    line height), and the constant dominates the score.

    This metric isolates the part the method actually predicts. It is
    reported alongside 2-D IoU, not instead of it: the projector needs
    a real rectangle, so the 2-D number is the one that matters for
    the product, and the 1-D number is the one that tells you whether
    the alignment itself works.
    """
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    intersection = min(ax1, bx1) - max(ax0, bx0)
    if intersection <= 0:
        return 0.0
    union = (ax1 - ax0) + (bx1 - bx0) - intersection
    return intersection / union if union > 0 else 0.0


def _rate(image_row: Dict[str, Any], key: str) -> str:
    """Format one per-image recall for the progress line."""
    value = image_row["thresholds"].get(key, {}).get("recall")
    return "n/a" if value is None else f"{value:.2f}"


def greedy_match(
    predictions: Sequence[Tuple[float, float, float, float]],
    references: Sequence[Tuple[float, float, float, float]],
    threshold: float,
    axis: str = "2d",
) -> List[Tuple[int, int, float]]:
    """One-to-one greedy IoU matching above ``threshold``."""
    from phonics_ocr.evaluation import wilson_interval
    from phonics_ocr.geometry import bbox_iou

    score = bbox_iou if axis == "2d" else horizontal_iou

    pairs: List[Tuple[float, int, int]] = []
    for p_index, prediction in enumerate(predictions):
        for r_index, reference in enumerate(references):
            iou = score(prediction, reference)
            if iou >= threshold:
                pairs.append((iou, p_index, r_index))
    pairs.sort(reverse=True)

    used_p: set = set()
    used_r: set = set()
    matched: List[Tuple[int, int, float]] = []
    for iou, p_index, r_index in pairs:
        if p_index in used_p or r_index in used_r:
            continue
        used_p.add(p_index)
        used_r.add(r_index)
        matched.append((p_index, r_index, iou))
    return matched


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7]
    )
    parser.add_argument(
        "--stretch",
        type=int,
        default=None,
        help="stretch each line crop to this width before recognition",
    )
    parser.add_argument(
        "--no-tighten-vertical",
        action="store_true",
        help=(
            "disable ink-based vertical tightening, so word boxes span "
            "the full line height (the raw CTC behaviour). Use to A/B "
            "whether tightening actually helps."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["ctc", "projection"],
        default="ctc",
        help=(
            "ctc = CTC timestep alignment (baseline). projection = "
            "whitespace projection profile, a model-free comparison "
            "that produces geometry but no text."
        ),
    )
    parser.add_argument(
        "--oracle-lines",
        action="store_true",
        help=(
            "skip detection and take the line extent from the union of "
            "the reference word boxes. Isolates CTC alignment quality "
            "from detection quality -- the two failure sources are "
            "otherwise impossible to tell apart."
        ),
    )
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from phonics_ocr import postprocessing as pp
    from phonics_ocr.config import DEFAULT_MODEL_DIR, OCRConfig
    from phonics_ocr.engine import EngineError, OCREngine
    from phonics_ocr.evaluation import wilson_interval
    from phonics_ocr.geometry import bbox_iou
    from phonics_ocr.preprocessing import load_image, save_image
    from phonics_ocr.utils import configure_logging, iso_now, read_jsonl, run_id, write_json
    from phonics_ocr.wordseg import (
        CTCAligner,
        locate_words_by_projection,
        locate_words_in_line,
    )

    configure_logging(args.verbose)

    if not args.ground_truth.is_file():
        print(f"ERROR: annotation file not found: {args.ground_truth}", file=sys.stderr)
        print("       Generate fixtures with:")
        print("         python scripts/make_localization_fixtures.py")
        return 1

    records = read_jsonl(args.ground_truth)
    root = args.ground_truth.parent
    samples = []
    for line_no, record in enumerate(records, start=1):
        words = record.get("words")
        if not words:
            print(f"  [skip] record {line_no} ({record.get('id')}) has no 'words'")
            continue
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = (root / image_path).resolve()
        if not image_path.is_file():
            print(f"ERROR: image not found: {image_path}", file=sys.stderr)
            return 1
        samples.append({**record, "_path": image_path, "_words": words})

    if not samples:
        print("ERROR: no annotated samples with word boxes.", file=sys.stderr)
        return 1

    print(f"Loaded {len(samples)} annotated image(s), "
          f"{sum(len(s['_words']) for s in samples)} reference word boxes\n")

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    engines: Dict[str, Any] = {}
    aligners: Dict[str, CTCAligner] = {}

    def get_engine(lang: str):
        if lang not in engines:
            engines[lang] = OCREngine(
                OCRConfig(lang=lang, device=args.device, model_dir=model_dir)
            )
        return engines[lang]

    def get_aligner(lang: str, device: str) -> CTCAligner:
        if lang not in aligners:
            from phonics_ocr.config import RECOGNITION_MODELS

            aligners[lang] = CTCAligner(
                model_dir / RECOGNITION_MODELS[lang], device=device
            )
        return aligners[lang]

    per_image: List[Dict[str, Any]] = []
    all_pred: List[Tuple] = []
    all_ref: List[Tuple] = []
    # Keep global index offsets so matching stays per-image.
    results_by_threshold: Dict[Tuple[str, float], Dict[str, Any]] = {
        (axis, t): {"matched": 0, "n_pred": 0, "n_ref": 0, "iou_sum": 0.0}
        for axis in ("2d", "x")
        for t in args.iou_thresholds
    }
    text_correct = 0
    text_total = 0

    for sample in samples:
        lang = sample.get("lang", "ar")
        if lang == "mixed":
            lang = "ar"
        try:
            engine = get_engine(lang)
        except (EngineError, ValueError) as exc:
            print(f"ERROR building engine for {lang}: {exc}", file=sys.stderr)
            return 1
        aligner = get_aligner(lang, engine.device)

        image = load_image(sample["_path"])
        result = engine.process_image(sample["_path"])

        rtl = result["reading_order"]["rtl"]

        if args.oracle_lines:
            # Group reference words into lines by their 'line' field
            # (falling back to a single line), then use the union of
            # each group's boxes as the line polygon.
            groups: Dict[Any, List[Any]] = {}
            for word in sample["_words"]:
                groups.setdefault(word.get("line", 0), []).append(word["bbox"])
            regions_iter = []
            for key in sorted(groups):
                boxes = groups[key]
                x0 = min(b[0] for b in boxes)
                y0 = min(b[1] for b in boxes)
                x1 = max(b[2] for b in boxes)
                y1 = max(b[3] for b in boxes)
                regions_iter.append(
                    {"polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}
                )
        else:
            regions_iter = result["regions"]

        predictions: List[Dict[str, Any]] = []
        for region in regions_iter:
            try:
                if args.method == "projection":
                    spans = locate_words_by_projection(
                        image,
                        region["polygon"],
                        start_index=len(predictions),
                        tighten_vertical=not args.no_tighten_vertical,
                    )
                else:
                    spans = locate_words_in_line(
                        image,
                        region["polygon"],
                        aligner,
                        rtl=rtl,
                        start_index=len(predictions),
                        stretch_to=args.stretch,
                        tighten_vertical=not args.no_tighten_vertical,
                    )
            except Exception as exc:
                print(f"  [warn] {sample['id']}: line skipped ({exc})")
                continue
            predictions.extend(s.to_dict() for s in spans)

        refs = [tuple(w["bbox"]) for w in sample["_words"]]
        preds = [tuple(p["bbox"]) for p in predictions]

        image_row: Dict[str, Any] = {
            "id": sample["id"],
            "image": str(sample["_path"]),
            "lang": sample.get("lang"),
            "group": sample.get("group", ""),
            "capture": sample.get("capture", "unknown"),
            "n_reference_words": len(refs),
            "n_predicted_words": len(preds),
            "n_lines_detected": result["n_regions"],
            "n_lines_used": len(regions_iter),
            "oracle_lines": args.oracle_lines,
            "thresholds": {},
            "predictions": predictions,
        }

        for axis in ("2d", "x"):
            for threshold in args.iou_thresholds:
                matched = greedy_match(preds, refs, threshold, axis=axis)
                bucket = results_by_threshold[(axis, threshold)]
                bucket["matched"] += len(matched)
                bucket["n_pred"] += len(preds)
                bucket["n_ref"] += len(refs)
                bucket["iou_sum"] += sum(m[2] for m in matched)
                image_row["thresholds"][f"{axis}@{threshold}"] = {
                    "matched": len(matched),
                    "recall": len(matched) / len(refs) if refs else None,
                    "precision": len(matched) / len(preds) if preds else None,
                    "mean_iou_matched": (
                        sum(m[2] for m in matched) / len(matched) if matched else None
                    ),
                }

        # Text agreement on pairs matched at the loosest threshold.
        loose = min(args.iou_thresholds)
        for p_index, r_index, _ in greedy_match(preds, refs, loose):
            text_total += 1
            got = pp.normalise_text(predictions[p_index]["text"])
            want = pp.normalise_text(sample["_words"][r_index]["text"])
            if got == want:
                text_correct += 1

        per_image.append(image_row)
        print(
            f"  {sample['id']:<26} lines={result['n_regions']:>2} "
            f"ref={len(refs):>2} pred={len(preds):>2} "
            f"Rx@0.5={_rate(image_row, 'x@0.5')}  "
            f"R2d@0.5={_rate(image_row, '2d@0.5')}"
        )

        if args.save_visualizations:
            import cv2
            import numpy as np

            canvas = image.copy()
            for w in sample["_words"]:
                x0, y0, x1, y1 = [int(v) for v in w["bbox"]]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 180, 0), 2)
            for p in predictions:
                cv2.polylines(
                    canvas,
                    [np.array(p["polygon"], np.int32)],
                    True,
                    (0, 0, 230),
                    2,
                )
            out_dir = (args.output_dir or PROJECT_ROOT / "data" / "results") / "localization_viz"
            save_image(out_dir / f"{sample['id']}_loc.png", canvas)

    summary: Dict[str, Any] = {}
    for (axis, threshold), bucket in results_by_threshold.items():
        recall = bucket["matched"] / bucket["n_ref"] if bucket["n_ref"] else None
        precision = bucket["matched"] / bucket["n_pred"] if bucket["n_pred"] else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else None
        )
        r_lo, r_hi = wilson_interval(bucket["matched"], bucket["n_ref"])
        p_lo, p_hi = wilson_interval(bucket["matched"], bucket["n_pred"])
        summary[f"{axis}@{threshold}"] = {
            "axis": axis,
            "iou_threshold": threshold,
            "recall_ci95": [r_lo, r_hi],
            "precision_ci95": [p_lo, p_hi],
            "matched": bucket["matched"],
            "n_reference": bucket["n_ref"],
            "n_predicted": bucket["n_pred"],
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "mean_iou_matched": (
                bucket["iou_sum"] / bucket["matched"] if bucket["matched"] else None
            ),
        }

    report = {
        "generated_at": iso_now(),
        "ground_truth_file": str(args.ground_truth),
        "n_images": len(samples),
        "n_reference_words": sum(len(s["_words"]) for s in samples),
        "device": {lang: e.device for lang, e in engines.items()},
        "stretch_to": args.stretch,
        "oracle_lines": args.oracle_lines,
        "tighten_vertical": not args.no_tighten_vertical,
        "method_name": args.method,
        "method": (
            (
                "CTC timestep alignment (phonics_ocr.wordseg)"
                if args.method == "ctc"
                else "projection-profile segmentation (geometry only, no text)"
            )
            + (" with ORACLE line extents (detection bypassed)"
               if args.oracle_lines else " on detected lines")
        ),
        "matching": "greedy one-to-one by descending IoU, per image",
        "summary_by_iou": summary,
        "text_agreement_on_matched": {
            "correct": text_correct,
            "total": text_total,
            "rate": text_correct / text_total if text_total else None,
            "note": (
                "Text equality on geometrically matched pairs only. "
                "Separate from localisation: a correctly placed box may "
                "still carry wrong text."
            ),
        },
        "statistics_note": (
            "recall_ci95 / precision_ci95 are Wilson score intervals at "
            "95%. They quantify SAMPLING error for this set only. They "
            "do not say the sample is representative: a tight interval "
            "on rendered images is still an interval about rendered "
            "images."
        ),
        "caveat": (
            "Scores reflect the annotated set only. If that set is "
            "synthetic (capture='synthetic_render'), these are an UPPER "
            "BOUND and do not predict camera performance."
        ),
        "per_image": per_image,
    }

    out_root = args.output_dir or (PROJECT_ROOT / "data" / "results")
    out_path = out_root / f"{run_id('localization')}.json"
    write_json(out_path, report)

    captures = {s.get("capture", "unknown") for s in samples}
    print(f"\n{'=' * 72}")
    print(f"WORD LOCALISATION  (method: {args.method})")
    print(f"{'=' * 72}")
    def fmt(v, w=8):
        return f"{v:.4f}".rjust(w) if v is not None else "n/a".rjust(w)

    for axis, label in (
        ("x", "HORIZONTAL IoU (what CTC alignment actually estimates)"),
        ("2d", "2-D IoU (what the projector ultimately needs)"),
    ):
        print(f"\n{label}")
        print(f"{'IoU':>6} {'recall':>8} {'recall 95% CI':>18} "
              f"{'prec':>8} {'F1':>7} {'matched':>9} {'meanIoU':>8}")
        print(f"{'-' * 84}")
        for threshold in sorted(args.iou_thresholds):
            row = summary[f"{axis}@{threshold}"]
            lo, hi = row["recall_ci95"]
            ci = (
                f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "n/a"
            )
            print(
                f"{threshold:>6.2f} {fmt(row['recall'])} {ci:>18} "
                f"{fmt(row['precision'],8)} {fmt(row['f1'],7)} "
                f"{str(row['matched']) + '/' + str(row['n_reference']):>9} "
                f"{fmt(row['mean_iou_matched'])}"
            )
    print(f"\nReference words: {report['n_reference_words']} across "
          f"{report['n_images']} image(s)")
    ta = report["text_agreement_on_matched"]
    if ta["total"]:
        print(f"Text agreement on matched boxes: {ta['correct']}/{ta['total']} "
              f"({ta['rate']:.2%})")
    if captures == {"synthetic_render"}:
        print(
            "\nNOTE: every annotated image is a SYNTHETIC render. These "
            "numbers are an\n      upper bound and say nothing about "
            "photographed pages."
        )
    print(f"\nFull report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Translate word-box error into projector coverage failure.

    python scripts/simulate_projector.py
    python scripts/simulate_projector.py --ground-truth data/localization/boxes.jsonl

IoU is not what the projector cares about. A highlight either covers
the word a child is meant to read, or it clips it, or it spills onto
the neighbouring word. This script converts boxes into those three
product-level quantities and sweeps controlled perturbations to find
how much geometric error is survivable.

Metrics, per highlighted word
-----------------------------
* **coverage**  -- fraction of the TRUE word box lit by the highlight.
  Below 1.0 means part of the word is dark.
* **spill**     -- fraction of the highlight landing on a DIFFERENT
  word's box. Above 0 means a neighbouring word is partly lit too.
* **background**-- fraction of the highlight on neither (harmless, but
  it is wasted light and looks sloppy).

A highlight is counted a **success** when coverage >= `--min-coverage`
and spill <= `--max-spill`. Both thresholds are product decisions, not
measurements; the defaults are stated and swept so the sensitivity is
visible rather than baked in.

Perturbations simulate residual calibration error AFTER camera-to-page
and page-to-projector mapping. They are applied in image pixels and
reported as a fraction of mean word height, which is the scale-free way
to state them.

This does NOT model the calibration itself. See
`docs/projector_calibration.md` for what is still missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

Box = Tuple[float, float, float, float]


def intersect_area(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def perturb(box: Box, dx: float, dy: float, scale: float) -> Box:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * scale, (box[3] - box[1]) / 2 * scale
    return (cx + dx - hw, cy + dy - hh, cx + dx + hw, cy + dy + hh)


def score_highlight(
    highlight: Box, target: Box, others: Sequence[Box]
) -> Dict[str, float]:
    target_area = area(target)
    light_area = area(highlight)
    covered = intersect_area(highlight, target)
    spilled = sum(intersect_area(highlight, other) for other in others)
    return {
        "coverage": covered / target_area if target_area > 0 else 0.0,
        "spill": spilled / light_area if light_area > 0 else 0.0,
        "background": (
            max(0.0, light_area - covered - spilled) / light_area
            if light_area > 0 else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ground-truth", type=Path,
        default=PROJECT_ROOT / "data" / "localization" / "boxes.jsonl",
    )
    parser.add_argument(
        "--predictions", type=Path,
        help="a localisation report JSON; if given, uses REAL predicted boxes "
             "as the unperturbed starting point instead of perfect boxes",
    )
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-spill", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from phonics_ocr.evaluation import wilson_interval
    from phonics_ocr.utils import iso_now, read_jsonl, write_json

    if not args.ground_truth.is_file():
        print(f"ERROR: not found: {args.ground_truth}", file=sys.stderr)
        return 1

    pages: List[Dict[str, Any]] = []
    for record in read_jsonl(args.ground_truth):
        words = record.get("words")
        if words:
            pages.append(
                {
                    "id": record["id"],
                    "boxes": [tuple(float(v) for v in w["bbox"]) for w in words],
                    "capture": record.get("capture", "unknown"),
                }
            )
    if not pages:
        print("ERROR: no annotated word boxes.", file=sys.stderr)
        return 1

    predicted: Dict[str, List[Box]] = {}
    if args.predictions:
        report = json.loads(args.predictions.read_text(encoding="utf-8"))
        for row in report.get("per_image", []):
            predicted[row["id"]] = [
                tuple(float(v) for v in p["bbox"]) for p in row.get("predictions", [])
            ]

    heights = [b[3] - b[1] for p in pages for b in p["boxes"]]
    mean_h = sum(heights) / len(heights)
    total_boxes = sum(len(p["boxes"]) for p in pages)

    print("=" * 84)
    print("PROJECTOR COVERAGE SIMULATION")
    print("=" * 84)
    print(f"pages {len(pages)}  words {total_boxes}  mean word height {mean_h:.1f}px")
    print(f"success = coverage >= {args.min_coverage:.2f} AND "
          f"spill <= {args.max_spill:.2f}")
    print(f"capture kinds: {sorted({p['capture'] for p in pages})}")

    def run(dx: float, dy: float, scale: float) -> Dict[str, Any]:
        ok = 0
        n = 0
        cov_sum = 0.0
        spill_sum = 0.0
        any_spill = 0
        for page in pages:
            boxes = page["boxes"]
            for i, target in enumerate(boxes):
                others = [b for j, b in enumerate(boxes) if j != i]
                # When real predictions are supplied, the highlight is
                # driven by the PREDICTED box for this word. Predictions
                # are matched to the target by best IoU, not by index:
                # prediction order need not follow annotation order, and
                # a model may emit a different number of boxes.
                base = target
                if args.predictions:
                    candidates = predicted.get(page["id"], [])
                    if candidates:
                        best = max(
                            candidates,
                            key=lambda c: intersect_area(c, target)
                            / max(1e-9, area(c) + area(target) - intersect_area(c, target)),
                        )
                        overlap = intersect_area(best, target)
                        union = area(best) + area(target) - overlap
                        base = best if union > 0 and overlap / union > 0 else target
                highlight = perturb(base, dx, dy, scale)
                s = score_highlight(highlight, target, others)
                n += 1
                cov_sum += s["coverage"]
                spill_sum += s["spill"]
                if s["spill"] > 0:
                    any_spill += 1
                if s["coverage"] >= args.min_coverage and s["spill"] <= args.max_spill:
                    ok += 1
        lo, hi = wilson_interval(ok, n)
        return {
            "dx": dx, "dy": dy, "scale": scale,
            "n": n, "success": ok,
            "success_rate": ok / n if n else None,
            "success_ci95": [lo, hi],
            "mean_coverage": cov_sum / n if n else None,
            "mean_spill": spill_sum / n if n else None,
            "words_with_any_spill": any_spill,
        }

    sweeps: Dict[str, List[Dict[str, Any]]] = {}

    print(f"\n--- horizontal offset (dx), as fraction of word height ---")
    print(f"{'dx px':>8}{'dx/h':>8}{'success':>10}{'95% CI':>18}"
          f"{'coverage':>10}{'spill':>8}")
    print("-" * 64)
    rows = []
    for frac in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        dx = frac * mean_h
        r = run(dx, 0.0, 1.0)
        rows.append(r)
        lo, hi = r["success_ci95"]
        print(f"{dx:>8.1f}{frac:>8.2f}{r['success_rate']:>10.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{r['mean_coverage']:>10.3f}{r['mean_spill']:>8.3f}")
    sweeps["dx"] = rows

    print(f"\n--- vertical offset (dy) ---")
    print(f"{'dy px':>8}{'dy/h':>8}{'success':>10}{'coverage':>10}{'spill':>8}")
    print("-" * 46)
    rows = []
    for frac in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        dy = frac * mean_h
        r = run(0.0, dy, 1.0)
        rows.append(r)
        print(f"{dy:>8.1f}{frac:>8.2f}{r['success_rate']:>10.3f}"
              f"{r['mean_coverage']:>10.3f}{r['mean_spill']:>8.3f}")
    sweeps["dy"] = rows

    print(f"\n--- scale error ---")
    print(f"{'scale':>8}{'success':>10}{'coverage':>10}{'spill':>8}")
    print("-" * 38)
    rows = []
    for scale in (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50):
        r = run(0.0, 0.0, scale)
        rows.append(r)
        print(f"{scale:>8.2f}{r['success_rate']:>10.3f}"
              f"{r['mean_coverage']:>10.3f}{r['mean_spill']:>8.3f}")
    sweeps["scale"] = rows

    # Tolerance: largest dx keeping success >= 0.95
    tolerance = None
    for r in sweeps["dx"]:
        if r["success_rate"] is not None and r["success_rate"] >= 0.95:
            tolerance = r
    print("\n" + "=" * 84)
    if tolerance:
        frac = tolerance["dx"] / mean_h
        print(f"TOLERANCE: horizontal error up to {tolerance['dx']:.1f}px "
              f"({frac:.2f} x word height) keeps >=95% of highlights acceptable.")
    else:
        print("TOLERANCE: even zero offset does not reach 95% success at these "
              "thresholds.")
    print("This is a budget for TOTAL residual error: OCR box error plus")
    print("camera-to-page plus page-to-projector calibration error combined.")
    print("=" * 84)

    report = {
        "generated_at": iso_now(),
        "ground_truth": str(args.ground_truth),
        "predictions_used": str(args.predictions) if args.predictions else None,
        "n_pages": len(pages),
        "n_words": total_boxes,
        "mean_word_height_px": round(mean_h, 2),
        "success_criteria": {
            "min_coverage": args.min_coverage,
            "max_spill": args.max_spill,
            "note": "Product thresholds, not measurements. Swept above.",
        },
        "sweeps": sweeps,
        "tolerance_dx_px": tolerance["dx"] if tolerance else None,
        "caveats": [
            "Perturbations are rigid translations and uniform scale. Real "
            "calibration error includes rotation, keystone and lens "
            "distortion, which are not modelled here.",
            "Boxes are axis-aligned; a projector highlight on a rotated "
            "line would need the polygon, not the bbox.",
            "If the annotation set is synthetic, these tolerances inherit "
            "that limitation.",
            "Coverage assumes the highlight is a filled rectangle. An "
            "underline or outline highlight has different failure modes.",
        ],
    }
    out = args.output or (PROJECT_ROOT / "data" / "results" / "projector_simulation.json")
    write_json(out, report)
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

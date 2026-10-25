"""Measure VRAM, RAM, latency and model storage for each configuration.

    python scripts/measure_resources.py
    python scripts/measure_resources.py --image data/input/page.jpg --runs 10
    python scripts/measure_resources.py --configs ar:gpu ar:cpu mixed:gpu

Phonics Prism is meant to run on modest local hardware, so the
question "does this fit on the target device?" needs numbers rather
than assumptions. This reports, per configuration:

* **model storage** on disk,
* **process RAM** before and after loading,
* **GPU memory** before and after loading, and at peak during
  inference (via ``nvidia-smi``; see the caveat below),
* **latency**, as median and spread over repeated runs, with warm-up
  runs excluded.

Latency caveat: the first inference includes lazy kernel setup and is
far slower than the rest, so ``--warmup`` runs are discarded. The
median is reported rather than the mean because one scheduling hiccup
skews a small sample badly.

GPU memory caveat: ``nvidia-smi`` reports memory for the whole device,
including other processes and the driver's own overhead. It is an
upper bound on what this process uses, not an exact attribution.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def gpu_memory_mb() -> Optional[int]:
    """Used GPU memory in MiB, device-wide, or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
        return None


def process_rss_mb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def dir_size_mb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["ar:gpu", "en:gpu", "mixed:gpu", "ar:cpu"],
        help="lang:device pairs to measure",
    )
    parser.add_argument("--runs", type=int, default=5, help="timed runs per config")
    parser.add_argument("--warmup", type=int, default=2, help="discarded runs")
    parser.add_argument("--words", action="store_true", help="also time word localisation")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from phonics_ocr.config import DEFAULT_MODEL_DIR, OCRConfig, RECOGNITION_MODELS, all_required_models
    from phonics_ocr.engine import EngineError, OCREngine
    from phonics_ocr.preprocessing import load_image
    from phonics_ocr.utils import configure_logging, iso_now, write_json

    configure_logging(False)

    image_path = args.image
    if image_path is None:
        for candidate in (
            PROJECT_ROOT / "data" / "localization" / "images" / "loc_ar_tashkeel.png",
            PROJECT_ROOT / "data" / "input" / "test_arabic_tashkeel.png",
        ):
            if candidate.is_file():
                image_path = candidate
                break
    if image_path is None or not image_path.is_file():
        print("ERROR: no test image. Pass --image, or run:", file=sys.stderr)
        print("  python scripts/make_localization_fixtures.py", file=sys.stderr)
        return 1

    print("=" * 72)
    print("RESOURCE MEASUREMENT")
    print("=" * 72)
    print(f"image : {image_path.name}")
    print(f"runs  : {args.runs} timed, {args.warmup} warm-up discarded\n")

    storage = {
        name: round(dir_size_mb(DEFAULT_MODEL_DIR / name), 2)
        for name in all_required_models()
    }
    total_storage = round(sum(storage.values()), 2)
    print("Model storage on disk:")
    for name, size in storage.items():
        print(f"  {name:<34} {size:>8.2f} MB")
    print(f"  {'TOTAL':<34} {total_storage:>8.2f} MB\n")

    baseline_gpu = gpu_memory_mb()
    baseline_rss = process_rss_mb()
    print(f"Baseline: GPU {baseline_gpu} MiB, process RSS "
          f"{baseline_rss:.0f} MB\n" if baseline_rss else "")

    results: List[Dict[str, Any]] = []
    image = load_image(image_path)

    for spec in args.configs:
        if ":" not in spec:
            print(f"  [skip] bad config {spec!r}; expected lang:device")
            continue
        lang, device = spec.split(":", 1)
        print(f"--- {lang} on {device} ---")

        gpu_before = gpu_memory_mb()
        rss_before = process_rss_mb()
        load_start = time.perf_counter()
        try:
            engine = OCREngine(
                OCRConfig(lang=lang, device=device, model_dir=DEFAULT_MODEL_DIR)
            )
        except (EngineError, ValueError) as exc:
            print(f"  FAILED: {exc}\n")
            results.append({"lang": lang, "device": device, "error": str(exc)})
            continue
        load_seconds = time.perf_counter() - load_start

        aligner = None
        if args.words and lang != "mixed":
            from phonics_ocr.wordseg import CTCAligner

            aligner = CTCAligner(
                DEFAULT_MODEL_DIR / RECOGNITION_MODELS[lang], device=engine.device
            )

        gpu_loaded = gpu_memory_mb()
        rss_loaded = process_rss_mb()

        for _ in range(args.warmup):
            engine.process_image(image_path)

        timings: List[float] = []
        peak_gpu = gpu_loaded or 0
        for _ in range(args.runs):
            start = time.perf_counter()
            result = engine.process_image(image_path)
            if aligner is not None:
                from phonics_ocr.wordseg import locate_words_in_line

                for region in result["regions"]:
                    try:
                        locate_words_in_line(
                            image, region["polygon"], aligner,
                            rtl=result["reading_order"]["rtl"],
                        )
                    except Exception:
                        pass
            timings.append(time.perf_counter() - start)
            current = gpu_memory_mb()
            if current and current > peak_gpu:
                peak_gpu = current

        row = {
            "lang": lang,
            "device_requested": device,
            "device_used": engine.device,
            "word_localisation": aligner is not None,
            "model_load_seconds": round(load_seconds, 3),
            "latency_seconds": {
                "median": round(statistics.median(timings), 4),
                "min": round(min(timings), 4),
                "max": round(max(timings), 4),
                "stdev": round(statistics.stdev(timings), 4) if len(timings) > 1 else 0.0,
                "n_runs": len(timings),
            },
            "gpu_mib": {
                "before_load": gpu_before,
                "after_load": gpu_loaded,
                "peak_during_inference": peak_gpu or None,
                "delta_load": (
                    gpu_loaded - gpu_before
                    if gpu_before is not None and gpu_loaded is not None
                    else None
                ),
            },
            "process_rss_mb": {
                "before_load": round(rss_before, 1) if rss_before else None,
                "after_load": round(rss_loaded, 1) if rss_loaded else None,
            },
            "n_regions": result["n_regions"],
        }
        results.append(row)
        lat = row["latency_seconds"]
        print(f"  load      : {row['model_load_seconds']:.2f} s")
        print(f"  latency   : median {lat['median']:.3f} s "
              f"(min {lat['min']:.3f}, max {lat['max']:.3f})")
        if row["gpu_mib"]["delta_load"] is not None:
            print(f"  GPU delta : +{row['gpu_mib']['delta_load']} MiB on load, "
                  f"peak {row['gpu_mib']['peak_during_inference']} MiB")
        print()

        del engine, aligner

    report = {
        "generated_at": iso_now(),
        "image": str(image_path),
        "runs": args.runs,
        "warmup_discarded": args.warmup,
        "model_storage_mb": {**storage, "total": total_storage},
        "baseline_gpu_mib": baseline_gpu,
        "configurations": results,
        "caveats": [
            "GPU memory is read from nvidia-smi and covers the whole "
            "device, including other processes. It is an upper bound on "
            "this process's usage, not an exact attribution.",
            "Latency excludes warm-up runs; the first inference is much "
            "slower because of lazy kernel initialisation.",
            "Median is reported rather than mean: a single scheduling "
            "hiccup skews a small sample.",
            "Measured on one machine and one image size. Latency scales "
            "with image dimensions and detected region count.",
        ],
    }

    out = args.output or (PROJECT_ROOT / "data" / "results" / "resource_measurement.json")
    write_json(out, report)

    print("=" * 72)
    print(f"{'config':<16}{'device':<9}{'median s':>10}{'GPU +MiB':>10}{'peak MiB':>10}")
    print("-" * 72)
    for row in results:
        if "error" in row:
            print(f"{row['lang'] + ':' + row['device_requested']:<16}{'FAILED':<9}")
            continue
        gpu = row["gpu_mib"]
        print(
            f"{row['lang'] + ':' + row['device_requested']:<16}"
            f"{row['device_used']:<9}"
            f"{row['latency_seconds']['median']:>10.3f}"
            f"{(gpu['delta_load'] if gpu['delta_load'] is not None else 0):>10}"
            f"{(gpu['peak_during_inference'] or 0):>10}"
        )
    print(f"\nModel storage total: {total_storage:.1f} MB")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

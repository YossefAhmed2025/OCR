"""Run OCR over an image or a directory of images.

    python scripts/run_ocr.py --image data/input/test_arabic.jpg --lang ar
    python scripts/run_ocr.py --image data/input/test_english.jpg --lang en
    python scripts/run_ocr.py --image data/input/test_mixed.jpg --lang mixed
    python scripts/run_ocr.py --input-dir data/input --lang ar --device gpu

Outputs land under ``data/results/<run_id>/`` so that no run overwrites
another: a JSON record, a visualisation and a plain-text transcription
per image, plus a ``run_metadata.json`` for the batch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="a single image file")
    source.add_argument("--input-dir", type=Path, help="a directory of images")

    parser.add_argument(
        "--lang",
        choices=["ar", "en", "mixed"],
        required=True,
        help=(
            "ar = Arabic script model; en = English model; "
            "mixed = EXPERIMENTAL two-pass, see docs/architecture.md"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="'gpu' fails loudly if the GPU is unusable; it never falls back silently",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="refuse to run unless every model is already present locally",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None, help="override the model directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="override data/results"
    )

    group = parser.add_argument_group(
        "preprocessing (all OFF by default -- each can damage Arabic diacritics)"
    )
    group.add_argument("--grayscale", action="store_true")
    group.add_argument("--clahe", action="store_true", help="adaptive contrast")
    group.add_argument("--denoise", action="store_true", help="WARNING: erases tashkeel")
    group.add_argument("--deskew", action="store_true")
    group.add_argument(
        "--upscale",
        type=float,
        default=1.0,
        help="resize factor before detection; often helps small diacritics",
    )

    tuning = parser.add_argument_group("detection / recognition tuning")
    tuning.add_argument("--det-limit-side-len", type=int, default=960)
    tuning.add_argument("--det-thresh", type=float, default=0.3)
    tuning.add_argument("--det-box-thresh", type=float, default=0.6)
    tuning.add_argument("--det-unclip-ratio", type=float, default=1.5)
    tuning.add_argument("--rec-score-thresh", type=float, default=0.0)
    tuning.add_argument(
        "--textline-orientation",
        action="store_true",
        help="enable the 180-degree text line orientation classifier",
    )
    tuning.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help=(
            "enable oneDNN CPU acceleration. OFF by default because "
            "paddlepaddle 3.3.1's oneDNN path crashes on these models; "
            "try it only if you know your build is unaffected"
        ),
    )
    tuning.add_argument(
        "--cpu-threads", type=int, default=10, help="CPU inference threads"
    )

    words = parser.add_argument_group("word-level localisation (EXPERIMENTAL)")
    words.add_argument(
        "--words",
        action="store_true",
        help=(
            "also derive word boxes from CTC alignment. These are "
            "DERIVED geometry, labelled word_ctc_aligned, not detection "
            "output. See docs/word_localization.md"
        ),
    )
    words.add_argument(
        "--no-tighten-vertical",
        action="store_true",
        help="keep full line height for word boxes instead of ink extent",
    )
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="suppress per-region output")
    parser.add_argument("--verbose", action="store_true")
    return parser


def print_regions(result: dict, quiet: bool) -> None:
    """Human-readable console summary.

    Arabic is printed in logical order exactly as the model returned it.
    Terminals apply the Unicode bidirectional algorithm themselves; the
    string is never reversed here.
    """
    print(f"\n  image      : {result['image']['name']}")
    print(f"  size       : {result['image']['width']}x{result['image']['height']} px")
    print(f"  device     : {result['device_used']}")
    print(f"  regions    : {result['n_regions']} ({result['geometry_granularity']})")
    print(f"  inference  : {result['timing']['inference_seconds']:.3f} s")
    print(f"  reading    : {'right-to-left' if result['reading_order']['rtl'] else 'left-to-right'}")

    for warning in result["image"]["inspection"]["warnings"]:
        print(f"  [image warning] {warning}")

    if quiet or not result["regions"]:
        return

    print(f"\n  {'#':>3} {'score':>6} {'script':>7} {'tashkeel':>8}  text")
    print(f"  {'-' * 68}")
    for index in result["reading_order"]["region_indices"]:
        region = result["regions"][index]
        print(
            f"  {region['index']:>3} {region['rec_score']:>6.3f} "
            f"{region['dominant_script']:>7} {region['tashkeel_count']:>8}  "
            f"{region['text_raw']}"
        )


def main() -> int:
    args = build_parser().parse_args()

    from phonics_ocr.config import DEFAULT_MODEL_DIR, OCRConfig, PreprocessConfig
    from phonics_ocr.engine import EngineError, OCREngine
    from phonics_ocr.preprocessing import PreprocessingError, load_image, save_image
    from phonics_ocr.utils import (
        configure_logging,
        find_images,
        run_id,
        set_offline_env,
        write_json,
        write_text,
    )
    from phonics_ocr.visualization import draw_regions

    configure_logging(args.verbose)

    if args.offline:
        variables = set_offline_env()
        print(f"Offline mode: set {', '.join(variables)}")

    target = args.image if args.image else args.input_dir
    try:
        images: List[Path] = find_images(target)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not images:
        print(f"ERROR: no supported image files found in {target}", file=sys.stderr)
        return 1

    if args.lang == "mixed":
        print(
            "\n*** EXPERIMENTAL: --lang mixed runs the Arabic and English\n"
            "*** pipelines separately and merges them per detected region.\n"
            "*** A single line containing BOTH scripts will still be\n"
            "*** recognised by one model only. See docs/architecture.md.\n"
        )

    try:
        config = OCRConfig(
            lang=args.lang,
            device=args.device,
            model_dir=args.model_dir or DEFAULT_MODEL_DIR,
            offline=args.offline,
            det_limit_side_len=args.det_limit_side_len,
            det_db_thresh=args.det_thresh,
            det_db_box_thresh=args.det_box_thresh,
            det_db_unclip_ratio=args.det_unclip_ratio,
            rec_score_thresh=args.rec_score_thresh,
            use_textline_orientation=args.textline_orientation,
            enable_mkldnn=args.enable_mkldnn,
            cpu_threads=args.cpu_threads,
            preprocess=PreprocessConfig(
                grayscale=args.grayscale,
                clahe=args.clahe,
                denoise=args.denoise,
                deskew=args.deskew,
                upscale=args.upscale,
            ),
        )
    except ValueError as exc:
        print(f"ERROR: invalid configuration -- {exc}", file=sys.stderr)
        return 1

    print(f"Loading engine (lang={config.lang}, device={config.device}) ...")
    try:
        engine = OCREngine(config)
    except EngineError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    output_root = args.output_dir or (PROJECT_ROOT / "data" / "results")
    run_directory = output_root / run_id(f"ocr_{config.lang}")
    run_directory.mkdir(parents=True, exist_ok=True)
    print(f"Engine ready on '{engine.device}'. Results -> {run_directory}")

    aligner = None
    if args.words:
        from phonics_ocr.config import RECOGNITION_MODELS
        from phonics_ocr.wordseg import CTCAligner, WordSegmentationError

        if config.lang == "mixed":
            print(
                "ERROR: --words is not supported with --lang mixed. Each line "
                "would need a script decision before alignment; run --lang ar "
                "or --lang en instead.",
                file=sys.stderr,
            )
            return 1
        try:
            aligner = CTCAligner(
                config.model_dir_for(RECOGNITION_MODELS[config.lang]),
                device=engine.device,
            )
        except WordSegmentationError as exc:
            print(f"ERROR: could not load the CTC aligner -- {exc}", file=sys.stderr)
            return 1
        print(
            "Word localisation ON (experimental, derived geometry -- "
            "see docs/word_localization.md)"
        )

    succeeded, failed = [], []
    batch_start = time.perf_counter()

    for image_path in images:
        try:
            result = engine.process_image(image_path)
        except (EngineError, PreprocessingError) as exc:
            print(f"\nFAILED {image_path.name}: {exc}", file=sys.stderr)
            failed.append({"image": str(image_path), "error": str(exc)})
            continue

        if aligner is not None:
            from phonics_ocr.preprocessing import load_image as _load
            from phonics_ocr.wordseg import locate_words_in_line

            page = _load(image_path)
            rtl = result["reading_order"]["rtl"]
            spans = []
            for region in result["regions"]:
                try:
                    spans.extend(
                        s.to_dict()
                        for s in locate_words_in_line(
                            page,
                            region["polygon"],
                            aligner,
                            rtl=rtl,
                            start_index=len(spans),
                            tighten_vertical=not args.no_tighten_vertical,
                        )
                    )
                except Exception as exc:
                    print(f"  [warning] word localisation skipped a line: {exc}")
            result["words"] = spans
            result["word_geometry_note"] = (
                "DERIVED from CTC timestep alignment, not detection output. "
                "Horizontal extent is model-derived; vertical extent comes "
                "from ink measurement within the line crop. Accuracy is "
                "characterised in docs/word_localization.md."
            )
            print(f"  words      : {len(spans)} (derived, experimental)")

        stem = image_path.stem
        write_json(run_directory / f"{stem}.json", result)
        write_text(run_directory / f"{stem}.txt", result["transcription_raw"] + "\n")

        if not args.no_visualization:
            try:
                original = load_image(image_path)
                save_image(
                    run_directory / f"{stem}_viz.png",
                    draw_regions(original, result["regions"]),
                )
            except Exception as exc:
                print(f"  [warning] visualisation failed: {exc}", file=sys.stderr)

        print_regions(result, args.quiet)
        succeeded.append(
            {
                "image": str(image_path),
                "n_regions": result["n_regions"],
                "inference_seconds": result["timing"]["inference_seconds"],
            }
        )

    batch_elapsed = time.perf_counter() - batch_start
    times = [s["inference_seconds"] for s in succeeded]
    metadata = {
        "run_id": run_directory.name,
        "config": config.to_dict(),
        "device_used": engine.device,
        "gpu_report": engine.gpu_report,
        "n_images_attempted": len(images),
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "batch_wall_seconds": round(batch_elapsed, 3),
        "mean_inference_seconds": round(sum(times) / len(times), 4) if times else None,
        "succeeded": succeeded,
        "failed": failed,
    }
    write_json(run_directory / "run_metadata.json", metadata)

    print(f"\n{'=' * 70}")
    print(f"  processed : {len(succeeded)}/{len(images)} image(s)")
    if times:
        print(f"  mean time : {metadata['mean_inference_seconds']:.3f} s/image")
    print(f"  results   : {run_directory}")
    if failed:
        print(f"  FAILURES  : {len(failed)} -- see run_metadata.json")
    print(f"{'=' * 70}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

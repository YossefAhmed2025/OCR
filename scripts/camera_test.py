"""Live webcam OCR testing against real printed pages.

    python scripts/camera_test.py --list
    python scripts/camera_test.py --lang ar
    python scripts/camera_test.py --lang ar --words
    python scripts/camera_test.py --lang en --camera 1 --no-save
    python scripts/camera_test.py --lang ar --replay data/input   # no camera

Controls (preview window focused)
---------------------------------
  SPACE / c   capture the current frame and run OCR
  n           next camera (when more than one was found)
  s           toggle saving on/off
  w           toggle experimental word localisation
  q / ESC     stop the camera and exit

Everything runs locally against the models already in ``models/``.
No network access, no new downloads, no cloud OCR.

What this does NOT do
---------------------
There is **no automatic page detection and no perspective correction**.
``preprocessing.four_point_transform()` exists but nothing finds page
corners, so a photographed page is fed to OCR as-is. ``--deskew``
corrects small in-plane rotation only. Hold the camera roughly
parallel to the page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BANNER_H = 112


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list", action="store_true", help="probe cameras and exit"
    )
    parser.add_argument(
        "--lang", choices=["ar", "en", "mixed"], default="ar",
        help="ar = Arabic script, en = English, mixed = experimental two-pass",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--camera", type=int, default=None, help="camera index")
    parser.add_argument("--backend", default=None, help="force CAP_DSHOW / CAP_MSMF")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--replay", type=Path, default=None,
        help="use image files instead of a camera (for testing without hardware)",
    )
    parser.add_argument(
        "--words", action="store_true",
        help="also derive EXPERIMENTAL word spans from CTC alignment",
    )
    parser.add_argument("--deskew", action="store_true", help="correct small rotation")
    parser.add_argument("--clahe", action="store_true", help="adaptive contrast")
    parser.add_argument(
        "--no-save", action="store_true", help="run without writing anything to disk"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-preview", action="store_true",
        help="capture once immediately with no window (headless)",
    )
    parser.add_argument("--offline", action="store_true", help="require local models")
    parser.add_argument("--verbose", action="store_true")
    return parser


def print_capture(capture, lang: str) -> None:
    """Terminal report for one capture."""
    result = capture.result
    print(f"\n{'=' * 70}")
    print(f"CAPTURE {capture.index}   lang={lang}   device={result['device_used']}")
    print(f"{'=' * 70}")
    assessment = result["capture"]["frame_assessment"]
    print(f"  frame      : {result['image']['width']}x{result['image']['height']}  "
          f"sharpness {assessment['laplacian_variance']:.0f}  "
          f"brightness {assessment['mean_intensity']:.0f}")
    for hint in assessment["hints"]:
        print(f"  [frame]      {hint}")
    print(f"  regions    : {result['n_regions']} (detector text lines)")
    print(f"  inference  : {result['timing']['inference_seconds']:.3f} s")

    mean = capture.mean_confidence
    low = capture.min_confidence
    if mean is not None:
        print(f"  confidence : mean {mean:.3f}   lowest {low:.3f}")
        if low < 0.5:
            print("  [warning]    at least one region is below 0.50 -- treat its")
            print("               text as unreliable.")
    else:
        print("  confidence : n/a (no text detected)")

    if result["regions"]:
        print(f"\n  {'#':>3} {'score':>6} {'script':>7} {'tashkeel':>8}  text")
        print(f"  {'-' * 66}")
        for i in result["reading_order"]["region_indices"]:
            region = result["regions"][i]
            print(
                f"  {region['index']:>3} {region['rec_score']:>6.3f} "
                f"{region['dominant_script']:>7} {region['tashkeel_count']:>8}  "
                f"{region['text_raw']}"
            )
    else:
        print("\n  No text detected. Check focus, lighting and distance.")

    words = result.get("words")
    if words is not None:
        flagged = [w for w in words if w["flags"] and w["flags"] != ["vertically_tightened"]]
        lowconf = [w for w in words if w["confidence"] < 0.5]
        print(f"\n  {len(words)} EXPERIMENTAL word span(s) -- derived from CTC")
        print("  timesteps, NOT detector word boxes, NOT validated on photos.")
        if lowconf:
            print(f"  [warning]  {len(lowconf)} span(s) below confidence 0.50")
        if flagged:
            print(f"  [warning]  {len(flagged)} span(s) carry flags:")
            for word in flagged[:5]:
                print(f"               {word['text']!r}: {', '.join(word['flags'])}")

    if capture.saved:
        print(f"\n  saved      : {Path(capture.saved['image']).parent}")
    else:
        print("\n  saved      : nothing (saving disabled)")


#: Below this Laplacian variance, diacritics are unlikely to survive.
#:
#: Measured across 13 real camera captures of the same vowelled page:
#: frames under ~150 averaged 2.5 recovered marks, frames at or above
#: averaged 4.4, and the two blurriest (104, 107) recovered none at
#: all under ANY detection or preprocessing setting that was tried.
#: It is a guide, not a guarantee -- the variance between captures is
#: large and this is counted output, not verified-correct output.
TASHKEEL_SHARPNESS_FLOOR = 150

PANEL_W = 460


def render_results_panel(height, capture, lang):
    """Side panel showing the last capture's text, Arabic shaped for display."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from phonics_ocr.visualization import find_arabic_font, shape_for_display

    panel = np.full((height, PANEL_W, 3), 24, dtype=np.uint8)
    pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    font_path = find_arabic_font()
    try:
        big = ImageFont.truetype(font_path, 26) if font_path else ImageFont.load_default()
        mid = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
        small = ImageFont.truetype(font_path, 15) if font_path else ImageFont.load_default()
    except Exception:
        big = mid = small = ImageFont.load_default()

    y = 12
    if capture is None:
        draw.text((14, y), "Press SPACE to capture", font=mid, fill=(170, 170, 170))
        draw.text((14, y + 30), "Results will appear here.", font=small,
                  fill=(130, 130, 130))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    result = capture.result
    regions = result["regions"]
    marks = sum(r["tashkeel_count"] for r in regions)
    mean = capture.mean_confidence

    draw.text((14, y), f"CAPTURE {capture.index}", font=big, fill=(240, 240, 240))
    y += 36
    conf_text = f"{mean:.2f}" if mean is not None else "n/a"
    conf_colour = (
        (120, 230, 120) if mean and mean >= 0.8
        else (250, 210, 100) if mean and mean >= 0.6
        else (250, 120, 120)
    )
    draw.text((14, y), f"{len(regions)} lines   conf {conf_text}", font=mid,
              fill=conf_colour)
    y += 28
    mark_colour = (120, 230, 120) if marks else (250, 120, 120)
    draw.text((14, y), f"tashkeel found: {marks}", font=mid, fill=mark_colour)
    y += 34
    draw.line([(14, y), (PANEL_W - 14, y)], fill=(70, 70, 70), width=1)
    y += 12

    if not regions:
        draw.text((14, y), "No text detected.", font=mid, fill=(250, 120, 120))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    for i in result["reading_order"]["region_indices"]:
        if y > height - 34:
            draw.text((14, y), "...", font=small, fill=(150, 150, 150))
            break
        region = regions[i]
        score = region["rec_score"]
        colour = (
            (120, 230, 120) if score >= 0.8
            else (250, 210, 100) if score >= 0.6
            else (250, 120, 120)
        )
        draw.text((14, y + 4), f"{score:.2f}", font=small, fill=colour)
        # Arabic needs joining + bidi to be readable in a PIL-drawn panel.
        shown = shape_for_display(region["text_raw"])
        tag = f" +{region['tashkeel_count']}" if region["tashkeel_count"] else ""
        draw.text((62, y), shown, font=mid, fill=(240, 240, 240))
        if tag:
            draw.text((PANEL_W - 44, y + 4), tag, font=small, fill=(120, 230, 120))
        y += 30

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def compose_banner(width: int, lines: List[str], colour=(235, 235, 235)):
    import cv2
    import numpy as np

    banner = np.full((BANNER_H, width, 3), 28, dtype=np.uint8)
    for i, line in enumerate(lines[:5]):
        cv2.putText(
            banner, line, (10, 22 + i * 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
        )
    return banner


def run_preview(args, session, source, cameras, open_source) -> int:
    """Live preview loop. Returns an exit code."""
    import cv2
    import numpy as np

    from phonics_ocr.camera import CameraError, assess_frame

    window = "Phonics Prism - camera test"
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        print(f"ERROR: cannot open a preview window ({exc}).", file=sys.stderr)
        print("       Use --no-preview to capture without a window.", file=sys.stderr)
        return 1

    camera_position = 0
    last_capture = None
    status = "ready"
    print("\nPreview running. SPACE=capture  n=next camera  s=save toggle  "
          "w=words  q=quit")

    while True:
        try:
            frame = source.read()
        except CameraError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1

        preview = frame.copy()
        scale = min(1.0, 980 / preview.shape[1], 600 / preview.shape[0])
        if scale < 1.0:
            preview = cv2.resize(
                preview, (int(preview.shape[1] * scale), int(preview.shape[0] * scale))
            )

        assessment = assess_frame(frame)
        sharpness = assessment["laplacian_variance"]
        hint = "; ".join(assessment["hints"]) if assessment["hints"] else "looks usable"
        colour = (120, 200, 120) if assessment["ready"] else (120, 200, 250)

        # Sharpness gate. Blur is the one failure no setting recovers
        # from: on the blurriest real captures, every detection and
        # preprocessing variant tried returned zero diacritics.
        if sharpness < TASHKEEL_SHARPNESS_FLOOR:
            gate = (f"SHARPNESS {sharpness:.0f} -- TOO LOW FOR TASHKEEL "
                    f"(want {TASHKEEL_SHARPNESS_FLOOR}+). Move closer / hold still.")
            colour = (120, 120, 250)
        else:
            gate = f"SHARPNESS {sharpness:.0f} -- ok for tashkeel"
            colour = (120, 230, 120)

        banner = compose_banner(
            preview.shape[1],
            [
                f"{source.name}   lang={args.lang}   device={session.engine.device}",
                gate,
                f"frame: {frame.shape[1]}x{frame.shape[0]}  brightness "
                f"{assessment['mean_intensity']:.0f}  -  {hint}",
                f"captures: {len(session.captures)}   "
                f"saving: {'ON' if session.save else 'OFF'}   "
                f"words: {'ON (experimental)' if session.aligner else 'off'}   "
                "| SPACE=capture s=saving w=words q=quit",
                status,
            ],
            colour,
        )
        stage = np.vstack([banner, preview])
        panel = render_results_panel(stage.shape[0], last_capture, args.lang)
        cv2.imshow(window, np.hstack([stage, panel]))

        key = cv2.waitKey(25) & 0xFF

        if key in (ord("q"), 27):
            break

        if key in (ord(" "), ord("c")):
            status = "running OCR..."
            cv2.imshow(window, np.vstack([compose_banner(preview.shape[1], [status]), preview]))
            cv2.waitKey(1)
            try:
                capture = session.capture(frame, source)
            except Exception as exc:
                print(f"\nCapture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                status = f"capture failed: {exc}"
                continue
            print_capture(capture, args.lang)
            mean = capture.mean_confidence
            status = (
                f"capture {capture.index}: {capture.result['n_regions']} regions, "
                f"mean conf {mean:.2f}" if mean is not None
                else f"capture {capture.index}: no text found"
            )
            last_capture = capture
            from phonics_ocr.visualization import draw_regions, draw_word_spans

            overlay = draw_regions(frame, capture.result["regions"])
            if capture.result.get("words"):
                overlay = draw_word_spans(overlay, capture.result["words"])
            s2 = min(1.0, 900 / overlay.shape[1], 560 / overlay.shape[0])
            if s2 < 1.0:
                overlay = cv2.resize(
                    overlay, (int(overlay.shape[1] * s2), int(overlay.shape[0] * s2))
                )
            cv2.imshow("Phonics Prism - last capture (boxes)", overlay)

        elif key == ord("s"):
            if session.save:
                session.save = False
                status = "saving OFF"
            elif session.output_dir is not None:
                session.output_dir.mkdir(parents=True, exist_ok=True)
                session.save = True
                status = f"saving ON -> {session.output_dir}"
            else:
                status = "cannot enable saving: no output dir"
            print(f"  {status}")

        elif key == ord("w"):
            if session.aligner is not None:
                session.aligner = None
                status = "word localisation off"
            else:
                aligner = build_aligner(args, session.engine)
                if aligner is None:
                    status = "word localisation unavailable for this config"
                else:
                    session.aligner = aligner
                    status = "word localisation ON (experimental)"
            print(f"  {status}")

        elif key == ord("n") and len(cameras) > 1:
            camera_position = (camera_position + 1) % len(cameras)
            info = cameras[camera_position]
            source.release()
            try:
                source = open_source(info.index, info.backend)
                status = f"switched to {info.label}"
            except CameraError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            print(f"  {status}")

    source.release()
    cv2.destroyAllWindows()
    return 0


def build_aligner(args, engine):
    """Load the CTC aligner, or return None when unavailable."""
    from phonics_ocr.config import DEFAULT_MODEL_DIR, RECOGNITION_MODELS
    from phonics_ocr.wordseg import CTCAligner, WordSegmentationError

    if args.lang == "mixed":
        print("  word localisation is not supported with --lang mixed")
        return None
    try:
        model_dir = (args.output_dir and None) or DEFAULT_MODEL_DIR
        return CTCAligner(
            Path(model_dir) / RECOGNITION_MODELS[args.lang], device=engine.device
        )
    except WordSegmentationError as exc:
        print(f"  could not load the CTC aligner: {exc}", file=sys.stderr)
        return None


def main() -> int:
    args = build_parser().parse_args()

    from phonics_ocr.camera import (
        CameraError,
        CameraFrameSource,
        ReplayFrameSource,
        probe_cameras,
    )
    from phonics_ocr.config import DEFAULT_MODEL_DIR, OCRConfig, PreprocessConfig
    from phonics_ocr.engine import EngineError, OCREngine
    from phonics_ocr.utils import configure_logging, find_images, run_id, set_offline_env
    from phonics_ocr.visualization import word_legend_lines

    configure_logging(args.verbose)

    if args.list:
        print("Probing camera indices (this can take a few seconds)...")
        cameras = probe_cameras()
        if not cameras:
            print("\nNo working camera found.")
            print("See docs/camera_testing.md -- in particular Windows camera")
            print("privacy settings, and closing other apps that hold the camera.")
            return 1
        print(f"\n{len(cameras)} working camera(s):")
        for info in cameras:
            print(f"  --camera {info.index}   {info.label}")
        return 0

    if args.offline:
        set_offline_env()

    # -- frame source -------------------------------------------
    cameras: List[Any] = []
    replay_images: Optional[List[Path]] = None
    if args.replay is not None:
        try:
            replay_images = find_images(args.replay)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if not replay_images:
            print(f"ERROR: no images in {args.replay}", file=sys.stderr)
            return 1
        print(f"REPLAY MODE: {len(replay_images)} image(s) from {args.replay}")
        print("  This is NOT a camera test. Results are recorded with")
        print("  capture kind 'screenshot' so they cannot be mistaken for one.")
    else:
        if args.camera is None:
            print("Probing cameras...")
            cameras = probe_cameras()
            if not cameras:
                print("\nERROR: no working camera found.", file=sys.stderr)
                print("  Try:  python scripts/camera_test.py --list", file=sys.stderr)
                print("  Or test the workflow without hardware:", file=sys.stderr)
                print("        python scripts/camera_test.py --replay data/input",
                      file=sys.stderr)
                print("  See docs/camera_testing.md.", file=sys.stderr)
                return 1
            print(f"  found {len(cameras)}: " + ", ".join(c.label for c in cameras))

    def open_source(index: int, backend: Optional[str] = None):
        return CameraFrameSource(
            index=index, backend=backend, width=args.width, height=args.height
        )

    try:
        if replay_images is not None:
            source = ReplayFrameSource(replay_images)
        elif args.camera is not None:
            source = open_source(args.camera, args.backend)
        else:
            source = open_source(cameras[0].index, cameras[0].backend)
    except CameraError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    # -- engine --------------------------------------------------
    print(f"Loading OCR engine (lang={args.lang}, device={args.device})...")
    try:
        config = OCRConfig(
            lang=args.lang,
            device=args.device,
            model_dir=DEFAULT_MODEL_DIR,
            offline=args.offline,
            preprocess=PreprocessConfig(deskew=args.deskew, clahe=args.clahe),
        )
        engine = OCREngine(config)
    except (EngineError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        source.release()
        return 1
    print(f"  ready on {engine.device}")

    aligner = build_aligner(args, engine) if args.words else None
    if args.words:
        print("\n" + "\n".join("  " + line for line in word_legend_lines()))

    output_dir = args.output_dir or (
        PROJECT_ROOT / "data" / "results" / run_id(f"camera_{args.lang}")
    )
    from phonics_ocr.camera import CaptureSession

    session = CaptureSession(
        engine,
        output_dir=output_dir,
        save=not args.no_save,
        aligner=aligner,
        capture_kind="screenshot" if replay_images is not None else "photograph",
    )
    if session.save:
        print(f"  saving to {output_dir}")
    else:
        print("  saving DISABLED (--no-save): nothing will be written")

    # -- run -----------------------------------------------------
    if args.no_preview:
        frame = source.read()
        capture = session.capture(frame, source)
        print_capture(capture, args.lang)
        exit_code = 0
        source.release()
    else:
        exit_code = run_preview(args, session, source, cameras, open_source)

    summary_path = session.write_summary()
    print(f"\n{len(session.captures)} capture(s) this session")
    if summary_path:
        print(f"Session summary: {summary_path}")
    elif session.captures:
        print("Nothing was saved (--no-save).")

    if session.captures:
        print("\nTo evaluate these against ground truth:")
        print("  1. label them   -> docs/dataset_preparation.md")
        print("  2. validate     -> python scripts/validate_dataset.py --ground-truth <file>")
        print("  3. score        -> python scripts/evaluate.py --ground-truth <file> --strip-tashkeel-view")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

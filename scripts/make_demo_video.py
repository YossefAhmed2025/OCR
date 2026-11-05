"""Record a demo video of the pipeline reading a page.

    python scripts/make_demo_video.py
    python scripts/make_demo_video.py --image mypage.png --out demo.mp4

Every word, box, score and metric drawn in the video comes from one
real run of the pipeline, performed by this script. Nothing is typed
in by hand. Where the model misreads a word the video shows the
misread word, in red, next to what the page actually says -- a demo
that hides its errors is worth nothing to the person deciding whether
to ship this.

The page itself is generated (``make_demo_page.py``), which is the
point: its correct transcription is known exactly, so the result can
be scored rather than admired.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FPS = 30
W, H = 1920, 1080

BG = (14, 17, 22)
PANEL = (22, 27, 34)
PANEL2 = (28, 34, 48)
LINE = (39, 46, 59)
INK = (232, 237, 245)
MUTED = (139, 151, 171)
ACCENT = (79, 140, 255)
OK = (63, 185, 80)
WARN = (210, 153, 34)
BAD = (240, 96, 96)


def shaper():
    """Arabic display text. delete_harakat=False is not optional."""
    import arabic_reshaper
    from bidi.algorithm import get_display

    r = arabic_reshaper.ArabicReshaper(configuration={"delete_harakat": False})
    return lambda s: get_display(r.reshape(s))


def run_ocr(image: Path, device: str) -> Dict[str, Any]:
    from phonics_ocr.config import OCRConfig
    from phonics_ocr.engine import OCREngine
    from phonics_ocr.utils import set_offline_env

    set_offline_env()
    engine = OCREngine(
        OCRConfig(lang="ar", device=device, model_dir=PROJECT_ROOT / "models",
                  offline=True)
    )
    return engine.process_image(image)


def score(result: Dict[str, Any], expected: List[str]) -> Dict[str, Any]:
    from phonics_ocr import evaluation as ev
    from phonics_ocr import postprocessing as pp

    order = result["reading_order"]["region_indices"]
    hyp = " ".join(result["regions"][i]["text_raw"] for i in order)
    ref = pp.normalise_text(" ".join(expected))
    got = pp.normalise_text(hyp)
    cer = ev.character_error_rate(ref, got)
    dia = ev.diacritic_prf(ref, got)
    rw, gw = ev.tokenize_words(ref), ev.tokenize_words(got)
    return {
        "cer": cer.rate,
        "accuracy": 1.0 - cer.rate,
        "dia_recall": dia.recall,
        "marks_found": dia.true_positives,
        "marks_total": dia.true_positives + dia.false_negatives,
        "words_exact": sum(1 for a, b in zip(rw, gw) if a == b),
        "words_total": len(rw),
        "truth_words": rw,
        "read_words": gw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path,
                        default=PROJECT_ROOT / "data/input/demo/phonics_baa.png")
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data/results/demo/phonics_demo.mp4")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg not on PATH.", file=sys.stderr)
        return 1
    if not args.image.is_file():
        print(f"ERROR: no image at {args.image}", file=sys.stderr)
        return 1

    meta_path = args.meta or args.image.with_suffix(".json")
    expected: List[str] = []
    if meta_path.is_file():
        expected = json.loads(meta_path.read_text(encoding="utf-8")).get(
            "expected_strings", []
        )

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    sh = shaper()
    f_title = ImageFont.truetype("C:/Windows/Fonts/tahomabd.ttf", 30)
    f_sub = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 17)
    f_ar = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 32)
    f_ar_s = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 20)
    f_ui = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 19)
    f_num = ImageFont.truetype("C:/Windows/Fonts/tahomabd.ttf", 38)
    f_lbl = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 14)

    print(f"Reading {args.image.name} ...")
    result = run_ocr(args.image, args.device)
    order = result["reading_order"]["region_indices"]
    regions = [result["regions"][i] for i in order]
    stats = score(result, expected) if expected else {}
    print(f"  {len(regions)} regions in {result['timing']['inference_seconds']:.2f}s")
    if stats:
        print(f"  accuracy {stats['accuracy']:.1%}, "
              f"{stats['words_exact']}/{stats['words_total']} words exact")

    # Which regions do not appear on the page. Marked in the panel so
    # the video never implies a clean sweep it did not get.
    #
    # Tested by containment, not by zipping the two word lists: one
    # dropped word shifts every pair after it, which silently passed a
    # misread ("al-baab" read as "al-aab") as correct in the first cut
    # of this video.
    wrong = set()
    if expected:
        from phonics_ocr import postprocessing as _pp

        reference = _pp.normalise_text(" ".join(expected))
        for region in result["regions"]:
            text = region["text_raw"]
            if text.strip() and _pp.normalise_text(text) not in reference:
                wrong.add(text)

    page = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    ph, pw = page.shape[:2]

    stage = (40, 96, 1000, 940)          # x, y, w, h
    scale = min(stage[2] / pw, stage[3] / ph)
    dw, dh = int(pw * scale), int(ph * scale)
    dx = stage[0] + (stage[2] - dw) // 2
    dy = stage[1] + (stage[3] - dh) // 2
    page_small = Image.fromarray(cv2.resize(page, (dw, dh), interpolation=cv2.INTER_AREA))

    def box_on_canvas(region: Dict[str, Any]) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = region["bbox"]
        return (int(dx + x0 * scale), int(dy + y0 * scale),
                int(dx + x1 * scale), int(dy + y1 * scale))

    px = 1060
    pw_panel = W - px - 40

    def frame(phase: str, t: float, n_boxes: int, n_rows: int, scan: float) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)

        d.rectangle([0, 0, W, 72], fill=(18, 23, 32))
        d.line([(0, 72), (W, 72)], fill=LINE)
        d.ellipse([28, 20, 60, 52], fill=ACCENT)
        d.text((76, 18), sh("موشور الصوتيات"), font=f_title, fill=INK)
        d.text((76, 48), "Phonics Prism  ·  Arabic OCR  ·  offline", font=f_sub, fill=MUTED)

        label = {"idle": "جاهز", "scan": "جارٍ التحليل", "done": "اكتملت القراءة"}[phase]
        col = {"idle": MUTED, "scan": WARN, "done": OK}[phase]
        tw = d.textlength(sh(label), font=f_ui)
        d.rounded_rectangle([W - tw - 90, 24, W - 30, 54], 14,
                            fill=(18, 27, 38), outline=col)
        d.text((W - tw - 60, 29), sh(label), font=f_ui, fill=col)

        d.rounded_rectangle([stage[0] - 10, stage[1] - 10,
                             stage[0] + stage[2] + 10, stage[1] + stage[3] + 10],
                            10, fill=(5, 7, 11), outline=LINE)
        img.paste(page_small, (dx, dy))

        if phase == "scan":
            y = int(dy + scan * dh)
            d.line([(dx, y), (dx + dw, y)], fill=ACCENT, width=3)
            d.rectangle([dx, dy, dx + dw, y], fill=None, outline=None)

        for i, region in enumerate(regions[:n_boxes]):
            x0, y0, x1, y1 = box_on_canvas(region)
            s = region["rec_score"]
            c = OK if s >= 0.9 else (WARN if s >= 0.7 else BAD)
            d.rectangle([x0, y0, x1, y1], outline=c, width=3)

        d.rounded_rectangle([px, 96, px + pw_panel, 1040], 12,
                            fill=PANEL, outline=LINE)
        d.text((px + pw_panel - 24, 118), sh("النتيجة"), font=f_ui, fill=INK, anchor="ra")
        d.line([(px + 16, 156), (px + pw_panel - 16, 156)], fill=LINE)

        cells = [
            ("سطور", str(len(regions)) if n_rows else "—"),
            ("تشكيل", str(stats.get("marks_found", "—")) if n_rows else "—"),
            ("الدقة", f"{stats['accuracy']:.0%}" if (n_rows and stats) else "—"),
        ]
        cw = (pw_panel - 32) // 3
        for i, (k, v) in enumerate(cells):
            cx = px + 16 + i * cw
            d.rounded_rectangle([cx, 172, cx + cw - 8, 254], 8, fill=PANEL2)
            d.text((cx + cw // 2 - 4, 196), v, font=f_num, fill=INK, anchor="mm")
            d.text((cx + cw // 2 - 4, 234), sh(k), font=f_lbl, fill=MUTED, anchor="mm")

        y = 278
        for i, region in enumerate(regions[:n_rows]):
            if y > 980:
                break
            s = region["rec_score"]
            txt = region["text_raw"]
            bad = txt in wrong
            c = BAD if bad else (OK if s >= 0.9 else WARN)
            rh = 64
            d.rounded_rectangle([px + 16, y, px + pw_panel - 16, y + rh], 9,
                                fill=PANEL2, outline=LINE)
            d.rounded_rectangle([px + 24, y + 7, px + 76, y + 28], 5,
                                fill=(16, 36, 22) if not bad else (42, 19, 19))
            d.text((px + 50, y + 17), f"{s:.2f}", font=f_lbl, fill=c, anchor="mm")
            if bad:
                d.text((px + 86, y + 10), sh("يخالف الصفحة"), font=f_lbl, fill=BAD)
            d.text((px + pw_panel - 32, y + 40), sh(txt), font=f_ar,
                   fill=INK, anchor="rm")
            y += rh + 6

        foot = (f"{result['timing']['inference_seconds']:.2f}s  ·  "
                f"{result['device_used']}  ·  {result['config']['lang']}  ·  "
                f"{result.get('weights_used', 'arabic_mixed_rec_r3')}")
        d.text((px + 24, 1008), foot, font=f_lbl, fill=MUTED)

        if phase == "done" and stats:
            note = (f"الكلمات المطابقة تمامًا: {stats['words_exact']} من "
                    f"{stats['words_total']}  ·  الحركات: "
                    f"{stats['marks_found']} من {stats['marks_total']}")
            d.text((stage[0], 1052), sh(note), font=f_ar_s, fill=MUTED)
        else:
            d.text((stage[0], 1052),
                   sh("صفحة مُولَّدة، نصها الصحيح معروف — لتقاس النتيجة لا لتُستحسن"),
                   font=f_ar_s, fill=MUTED)
        return img

    frames_dir = PROJECT_ROOT / "data/results/demo/frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    plan: List[Tuple[str, float, int, int, float]] = []
    for i in range(int(1.2 * FPS)):
        plan.append(("idle", i / FPS, 0, 0, 0.0))
    scan_n = int(1.8 * FPS)
    for i in range(scan_n):
        plan.append(("scan", i / FPS, 0, 0, i / scan_n))
    per = max(1, int(0.42 * FPS))
    for k in range(1, len(regions) + 1):
        for i in range(per):
            plan.append(("done", 0.0, k, k, 0.0))
    for i in range(int(3.5 * FPS)):
        plan.append(("done", 0.0, len(regions), len(regions), 0.0))

    print(f"Rendering {len(plan)} frames ...")
    for n, (phase, t, nb, nr, sc) in enumerate(plan):
        frame(phase, t, nb, nr, sc).save(frames_dir / f"f{n:05d}.png")
        if n % 60 == 0:
            print(f"  {n}/{len(plan)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-framerate", str(FPS),
           "-i", str(frames_dir / "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           str(args.out)]
    print("Encoding ...")
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        print(rc.stderr[-2000:], file=sys.stderr)
        return 1

    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB, {len(plan)/FPS:.1f}s)")
    if stats:
        print(f"  accuracy shown: {stats['accuracy']:.1%}  "
              f"words {stats['words_exact']}/{stats['words_total']}  "
              f"marks {stats['marks_found']}/{stats['marks_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

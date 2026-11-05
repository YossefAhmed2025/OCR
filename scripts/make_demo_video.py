"""Record a demo video of the pipeline reading pages.

    python scripts/make_demo_video.py
    python scripts/make_demo_video.py --image mypage.png --out demo.mp4

Every word, box, score and metric drawn in the video comes from real
runs of the pipeline, performed by this script. Nothing is typed in by
hand. Where the model misreads a word the video shows the misread
word, in red, against what the page actually says -- a demo that hides
its errors is worth nothing to the person deciding whether to ship
this.

The pages are generated (``make_demo_page.py``), which is the point:
their correct transcription is known exactly, so the result can be
scored rather than admired. The video does not caption them as
generated, so whoever presents it should say so; a viewer will
otherwise assume a photograph.
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

STAGE = (40, 96, 1000, 924)          # x, y, w, h
PANEL_X = 1060


def shaper():
    """Arabic UI strings, shaped so the marks land correctly."""
    from phonics_ocr.shaping import ShapingError, render_line  # noqa: F401
    import arabic_reshaper
    from bidi.algorithm import get_display

    r = arabic_reshaper.ArabicReshaper(configuration={"delete_harakat": False})
    return lambda s: get_display(r.reshape(s))


def run_ocr(image: Path, lang: str, device: str) -> Dict[str, Any]:
    from phonics_ocr.config import OCRConfig
    from phonics_ocr.engine import OCREngine
    from phonics_ocr.utils import set_offline_env

    set_offline_env()
    engine = OCREngine(
        OCRConfig(lang=lang, device=device, model_dir=PROJECT_ROOT / "models",
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

    # Which regions contain a word the page does not have.
    #
    # Not by zipping the two word lists: one dropped word shifts every
    # pair after it, and a misread then passes as correct. Not by
    # substring either -- "baaba" is inside "al-baaba", so a wrong case
    # ending slipped through as a match. Membership in the set of
    # reference words is the test that catches both.
    ref_words = set(ev.tokenize_words(ref))
    wrong = {
        r["text_raw"]
        for r in result["regions"]
        if r["text_raw"].strip()
        and any(w not in ref_words
                for w in ev.tokenize_words(pp.normalise_text(r["text_raw"])))
    }
    return {
        "cer": cer.rate,
        "accuracy": 1.0 - cer.rate,
        "dia_recall": dia.recall,
        "marks_found": dia.true_positives,
        "marks_total": dia.true_positives + dia.false_negatives,
        "words_exact": sum(1 for a, b in zip(rw, gw) if a == b),
        "words_total": len(rw),
        "wrong": wrong,
    }


def build_segment(image: Path, device: str) -> Dict[str, Any]:
    """Run the pipeline on one page and gather everything the frames need."""
    import cv2

    meta_path = image.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    lang = meta.get("lang", "ar")
    expected = meta.get("expected_strings", [])

    result = run_ocr(image, lang, device)
    order = result["reading_order"]["region_indices"]
    regions = [result["regions"][i] for i in order]
    stats = score(result, expected) if expected else {}

    page = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    return {
        "image": image, "lang": lang, "meta": meta, "result": result,
        "regions": regions, "stats": stats, "page": page,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, action="append", default=None,
                        help="repeatable; defaults to the Arabic and English pages")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data/results/demo/phonics_demo.mp4")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg not on PATH.", file=sys.stderr)
        return 1

    images = args.image or [
        PROJECT_ROOT / "data/input/demo/phonics_baa.png",
        PROJECT_ROOT / "data/input/demo/phonics_b_en.png",
    ]
    missing = [p for p in images if not p.is_file()]
    if missing:
        print(f"ERROR: no image at {missing[0]}", file=sys.stderr)
        print("       run scripts/make_demo_page.py first", file=sys.stderr)
        return 1

    import cv2
    from PIL import Image, ImageDraw, ImageFont

    sh = shaper()
    f_title = ImageFont.truetype("C:/Windows/Fonts/tahomabd.ttf", 30)
    f_sub = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 17)
    f_ar = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 30)
    f_ui = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 19)
    f_num = ImageFont.truetype("C:/Windows/Fonts/tahomabd.ttf", 38)
    f_lbl = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 14)

    segments = []
    for path in images:
        print(f"Reading {path.name} ...")
        seg = build_segment(path, args.device)
        s = seg["stats"]
        print(f"  {len(seg['regions'])} regions in "
              f"{seg['result']['timing']['inference_seconds']:.2f}s"
              + (f", accuracy {s['accuracy']:.1%}, "
                 f"{s['words_exact']}/{s['words_total']} words exact" if s else ""))
        ph, pw = seg["page"].shape[:2]
        scale = min(STAGE[2] / pw, STAGE[3] / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        seg["scale"] = scale
        seg["dx"] = STAGE[0] + (STAGE[2] - dw) // 2
        seg["dy"] = STAGE[1] + (STAGE[3] - dh) // 2
        seg["dw"], seg["dh"] = dw, dh
        seg["small"] = Image.fromarray(
            cv2.resize(seg["page"], (dw, dh), interpolation=cv2.INTER_AREA)
        )
        segments.append(seg)

    pw_panel = W - PANEL_X - 40

    def frame(seg, phase: str, n_boxes: int, n_rows: int, scan: float) -> Image.Image:
        regions, stats = seg["regions"], seg["stats"]
        res = seg["result"]
        rtl = seg["lang"] != "en"
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)

        d.rectangle([0, 0, W, 72], fill=(18, 23, 32))
        d.line([(0, 72), (W, 72)], fill=LINE)
        d.ellipse([28, 20, 60, 52], fill=ACCENT)
        d.text((76, 18), sh("موشور الصوتيات"), font=f_title, fill=INK)
        d.text((76, 48), "Phonics Prism  ·  reads Arabic and English  ·  offline",
               font=f_sub, fill=MUTED)

        tag = {"ar": "عربي", "en": "English"}[seg["lang"]]
        tagw = d.textlength(sh(tag), font=f_ui)
        d.rounded_rectangle([W - tagw - 300, 24, W - 240, 54], 14,
                            fill=(18, 27, 38), outline=LINE)
        d.text((W - tagw - 270, 29), sh(tag), font=f_ui, fill=MUTED)

        label = {"idle": "جاهز", "scan": "جارٍ التحليل", "done": "اكتملت القراءة"}[phase]
        col = {"idle": MUTED, "scan": WARN, "done": OK}[phase]
        tw = d.textlength(sh(label), font=f_ui)
        d.rounded_rectangle([W - tw - 90, 24, W - 30, 54], 14,
                            fill=(18, 27, 38), outline=col)
        d.text((W - tw - 60, 29), sh(label), font=f_ui, fill=col)

        d.rounded_rectangle([STAGE[0] - 10, STAGE[1] - 10,
                             STAGE[0] + STAGE[2] + 10, STAGE[1] + STAGE[3] + 10],
                            10, fill=(5, 7, 11), outline=LINE)
        img.paste(seg["small"], (seg["dx"], seg["dy"]))

        if phase == "scan":
            y = int(seg["dy"] + scan * seg["dh"])
            d.line([(seg["dx"], y), (seg["dx"] + seg["dw"], y)], fill=ACCENT, width=3)

        for region in regions[:n_boxes]:
            x0, y0, x1, y1 = region["bbox"]
            s = region["rec_score"]
            c = BAD if region["text_raw"] in stats.get("wrong", ()) else (
                OK if s >= 0.9 else WARN)
            d.rectangle([int(seg["dx"] + x0 * seg["scale"]),
                         int(seg["dy"] + y0 * seg["scale"]),
                         int(seg["dx"] + x1 * seg["scale"]),
                         int(seg["dy"] + y1 * seg["scale"])], outline=c, width=3)

        d.rounded_rectangle([PANEL_X, 96, PANEL_X + pw_panel, 1020], 12,
                            fill=PANEL, outline=LINE)
        anchor = "ra" if rtl else "la"
        hx = PANEL_X + pw_panel - 24 if rtl else PANEL_X + 24
        d.text((hx, 118), sh("النتيجة") if rtl else "Result",
               font=f_ui, fill=INK, anchor=anchor)
        d.line([(PANEL_X + 16, 156), (PANEL_X + pw_panel - 16, 156)], fill=LINE)

        cells = [
            ("سطور" if rtl else "lines", str(len(regions)) if n_rows else "—"),
            ("تشكيل" if rtl else "words",
             str(stats.get("marks_found", "—")) if (n_rows and rtl)
             else (str(stats.get("words_exact", "—")) if n_rows else "—")),
            ("الدقة" if rtl else "accuracy",
             f"{stats['accuracy']:.0%}" if (n_rows and stats) else "—"),
        ]
        cw = (pw_panel - 32) // 3
        for i, (k, v) in enumerate(cells):
            cx = PANEL_X + 16 + i * cw
            d.rounded_rectangle([cx, 172, cx + cw - 8, 254], 8, fill=PANEL2)
            d.text((cx + cw // 2 - 4, 196), v, font=f_num, fill=INK, anchor="mm")
            d.text((cx + cw // 2 - 4, 234), sh(k) if rtl else k,
                   font=f_lbl, fill=MUTED, anchor="mm")

        y = 278
        for region in regions[:n_rows]:
            if y > 950:
                break
            s = region["rec_score"]
            txt = region["text_raw"]
            bad = txt in stats.get("wrong", ())
            c = BAD if bad else (OK if s >= 0.9 else WARN)
            rh = 64
            d.rounded_rectangle([PANEL_X + 16, y, PANEL_X + pw_panel - 16, y + rh], 9,
                                fill=PANEL2, outline=LINE)
            d.rounded_rectangle([PANEL_X + 24, y + 7, PANEL_X + 76, y + 28], 5,
                                fill=(42, 19, 19) if bad else (16, 36, 22))
            d.text((PANEL_X + 50, y + 17), f"{s:.2f}", font=f_lbl, fill=c, anchor="mm")
            if bad:
                d.text((PANEL_X + 86, y + 10),
                       sh("يخالف الصفحة") if rtl else "differs from page",
                       font=f_lbl, fill=BAD)
            if rtl:
                d.text((PANEL_X + pw_panel - 32, y + 40), sh(txt),
                       font=f_ar, fill=INK, anchor="rm")
            else:
                d.text((PANEL_X + 32, y + 40), txt, font=f_ar, fill=INK, anchor="lm")
            y += rh + 6

        foot = (f"{res['timing']['inference_seconds']:.2f}s  ·  "
                f"{res['device_used']}  ·  {seg['lang']}  ·  "
                f"{seg['meta'].get('font', '')}")
        d.text((PANEL_X + 24, 1034), foot, font=f_lbl, fill=MUTED)

        if phase == "done" and stats:
            if rtl:
                note = (f"الكلمات المطابقة تمامًا: {stats['words_exact']} من "
                        f"{stats['words_total']}  ·  الحركات: "
                        f"{stats['marks_found']} من {stats['marks_total']}")
            else:
                note = (f"exact words: {stats['words_exact']} of "
                        f"{stats['words_total']}")
            d.text((STAGE[0], 1044), sh(note) if rtl else note,
                   font=f_ui, fill=MUTED)
        return img

    frames_dir = PROJECT_ROOT / "data/results/demo/frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    total = 0
    for seg in segments:
        regions = seg["regions"]
        plan: List[Tuple[str, int, int, float]] = []
        plan += [("idle", 0, 0, 0.0)] * int(1.0 * FPS)
        scan_n = int(1.6 * FPS)
        plan += [("scan", 0, 0, i / scan_n) for i in range(scan_n)]
        per = max(1, int(0.40 * FPS))
        for k in range(1, len(regions) + 1):
            plan += [("done", k, k, 0.0)] * per
        plan += [("done", len(regions), len(regions), 0.0)] * int(3.0 * FPS)
        total += len(plan)

    print(f"Rendering {total} frames ...")
    for seg in segments:
        regions = seg["regions"]
        plan = []
        plan += [("idle", 0, 0, 0.0)] * int(1.0 * FPS)
        scan_n = int(1.6 * FPS)
        plan += [("scan", 0, 0, i / scan_n) for i in range(scan_n)]
        per = max(1, int(0.40 * FPS))
        for k in range(1, len(regions) + 1):
            plan += [("done", k, k, 0.0)] * per
        plan += [("done", len(regions), len(regions), 0.0)] * int(3.0 * FPS)
        for phase, nb, nr, sc in plan:
            frame(seg, phase, nb, nr, sc).save(frames_dir / f"f{n:05d}.png")
            n += 1
            if n % 90 == 0:
                print(f"  {n}/{total}")

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

    print(f"\nwrote {args.out}  "
          f"({args.out.stat().st_size / 1e6:.1f} MB, {n / FPS:.1f}s)")
    for seg in segments:
        s = seg["stats"]
        if s:
            print(f"  {seg['image'].name:<20} accuracy {s['accuracy']:.1%}  "
                  f"words {s['words_exact']}/{s['words_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

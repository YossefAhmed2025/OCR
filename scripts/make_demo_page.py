"""Render a printed Arabic phonics page to read on camera.

This produces a page, not a photograph of one. It exists so a demo
runs on material whose correct transcription is known exactly, rather
than on a book page nobody has transcribed -- every word here has a
ground-truth string, so the result can be scored instead of admired.

    python scripts/make_demo_page.py
    python scripts/make_demo_page.py --font trado.ttf --out mypage.png

Typography follows what the measurements asked for: line height near
5% of frame height, which is where detection was found to work, and a
real Arabic typeface rather than a screen face.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

#: A phonics page for the letter baa: the three short vowels on the
#: letter, then the same letter inside words and sentences.
#:
#: Isolated single glyphs are deliberately absent. A bare "ba" is
#: below the detector's floor -- it was measured at 0.000 confidence
#: twice, at 104 px and at 150 px -- so the letter appears inside a
#: line the detector can find, which is also how a reader meets it.
WORDS = ["بَابٌ", "بِنْتٌ", "بُرْتُقَالٌ"]
SENTENCES = [
    "فَتَحَتِ الْبِنْتُ الْبَابَ",
    "أَكَلَ الْوَلَدُ الْبُرْتُقَالَ",
]
TITLE = "حَرْفُ الْبَاءِ"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--font", default="tahoma.ttf")
    parser.add_argument("--font-dir", default=r"C:\Windows\Fonts")
    parser.add_argument("--width", type=int, default=1240)
    parser.add_argument("--height", type=int, default=1754)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data/input/demo/phonics_baa.png")
    args = parser.parse_args()

    import arabic_reshaper
    from bidi.algorithm import get_display
    from PIL import Image, ImageDraw, ImageFont

    # delete_harakat=False is not optional. The default strips every
    # mark, which once produced a published conclusion that the model
    # ignored tashkeel -- see docs/baseline_findings.md section 2.
    reshaper = arabic_reshaper.ArabicReshaper(configuration={"delete_harakat": False})
    shape = lambda s: get_display(reshaper.reshape(s))

    font_path = Path(args.font_dir) / args.font
    if not font_path.is_file():
        print(f"ERROR: no font at {font_path}", file=sys.stderr)
        return 1

    W, H = args.width, args.height
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    f_title = ImageFont.truetype(str(font_path), 96)
    f_word = ImageFont.truetype(str(font_path), 104)
    f_sent = ImageFont.truetype(str(font_path), 78)

    margin = 90
    d.text((W // 2, 140), shape(TITLE), font=f_title, fill=(10, 10, 10), anchor="mm")
    d.line([(margin, 225), (W - margin, 225)], fill=(60, 60, 60), width=3)

    truth = [TITLE]
    right = W - margin

    y = 400
    for word in WORDS:
        d.text((right, y), shape(word), font=f_word, fill=(10, 60, 130), anchor="rm")
        truth.append(word)
        y += 190

    y += 60
    d.line([(margin, y), (W - margin, y)], fill=(190, 190, 190), width=2)
    y += 130

    for line in SENTENCES:
        d.text((right, y), shape(line), font=f_sent, fill=(10, 10, 10), anchor="rm")
        truth.append(line)
        y += 200

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)

    meta = args.out.with_suffix(".json")
    meta.write_text(json.dumps({
        "image": args.out.name,
        "font": args.font,
        "title": TITLE,
        "words": WORDS,
        "sentences": SENTENCES,
        "expected_strings": truth,
        "note": "Rendered page, not a photograph. Ground truth is exact.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {args.out}  ({W}x{H})")
    print(f"wrote {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

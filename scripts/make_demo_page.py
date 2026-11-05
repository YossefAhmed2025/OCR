"""Render printed phonics pages to read on camera.

    python scripts/make_demo_page.py                 # both pages
    python scripts/make_demo_page.py --lang ar
    python scripts/make_demo_page.py --font trado.ttf

These are pages, not photographs of pages. They exist so a demo runs
on material whose correct transcription is known exactly -- every line
has a ground-truth string, so the result can be scored instead of
admired.

Arabic is drawn through ``phonics_ocr.shaping``, which shapes with
HarfBuzz and rasterises with FreeType. The older path -- reshaper plus
Pillow -- could not place a harakah: only Tahoma and Arial survived
it, and every Naskh face printed books use drew the marks detached and
misplaced. See the module docstring for why.

Typography follows the measurements: line height near 5% of frame
height, where detection was found to work, and no isolated single
glyphs, which sit below the detector's floor (a bare vowelled "ba"
scored 0.000 confidence at both 104 px and 150 px).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ARABIC: Dict[str, Any] = {
    "lang": "ar",
    "rtl": True,
    "title": "حَرْفُ الْبَاءِ",
    "words": ["بَابٌ", "بِنْتٌ", "بُرْتُقَالٌ"],
    "lines": [
        "فَتَحَتِ الْبِنْتُ الْبَابَ",
        "أَكَلَ الْوَلَدُ الْبُرْتُقَالَ",
    ],
    # Sakkal Majalla: a Naskh-derived face that ships with Windows and
    # is widely set in Arabic documents. Measured best of eight here
    # at 93.0% character accuracy -- Traditional Arabic reached 85.0%
    # and Amiri 63.0%, which is the Naskh weakness this model still has.
    "font": "majalla.ttf",
    "font_dir": r"C:\Windows\Fonts",
    "stem": "phonics_baa",
}

ENGLISH: Dict[str, Any] = {
    "lang": "en",
    "rtl": False,
    "title": "The Letter B",
    "words": ["ball", "book", "butterfly"],
    "lines": [
        "The boy kicked the ball",
        "Bella reads a big blue book",
    ],
    "font": "georgia.ttf",
    "font_dir": r"C:\Windows\Fonts",
    "stem": "phonics_b_en",
}

WIDTH, HEIGHT = 1240, 1754


def render_page(spec: Dict[str, Any], font_name: str, out_dir: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    from phonics_ocr.shaping import render_line

    font_path = Path(spec["font_dir"]) / font_name
    if not font_path.is_file():
        raise SystemExit(f"ERROR: no font at {font_path}")

    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    margin = 90
    right = WIDTH - margin

    def place(text: str, size: int, y: int, colour) -> None:
        """Draw one line, right-aligned for Arabic, left for Latin."""
        if spec["rtl"]:
            strip = render_line(text, font_path, size, padding=6)
            mask = strip.point(lambda v: 255 - v)      # ink -> alpha
            x = right - strip.width
            img.paste(colour, (x, y - strip.height // 2), mask)
        else:
            font = ImageFont.truetype(str(font_path), size)
            draw.text((margin, y), text, font=font, fill=colour, anchor="lm")

    title_x = WIDTH // 2
    if spec["rtl"]:
        strip = render_line(spec["title"], font_path, 92, padding=6)
        mask = strip.point(lambda v: 255 - v)
        img.paste((10, 10, 10), (title_x - strip.width // 2, 140 - strip.height // 2), mask)
    else:
        f = ImageFont.truetype(str(font_path), 92)
        draw.text((title_x, 140), spec["title"], font=f, fill=(10, 10, 10), anchor="mm")

    draw.line([(margin, 236), (WIDTH - margin, 236)], fill=(60, 60, 60), width=3)

    truth: List[str] = [spec["title"]]
    y = 410
    for word in spec["words"]:
        place(word, 100, y, (10, 60, 130))
        truth.append(word)
        y += 190

    y += 60
    draw.line([(margin, y), (WIDTH - margin, y)], fill=(190, 190, 190), width=2)
    y += 130

    for line in spec["lines"]:
        place(line, 74, y, (10, 10, 10))
        truth.append(line)
        y += 200

    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{spec['stem']}.png"
    img.save(image_path)

    (out_dir / f"{spec['stem']}.json").write_text(
        json.dumps(
            {
                "image": image_path.name,
                "lang": spec["lang"],
                "font": font_name,
                "title": spec["title"],
                "words": spec["words"],
                "sentences": spec["lines"],
                "expected_strings": truth,
                "renderer": "HarfBuzz + FreeType" if spec["rtl"] else "Pillow",
                "note": "Rendered page, not a photograph. Ground truth is exact.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return image_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", choices=["ar", "en", "both"], default="both")
    parser.add_argument("--font", default=None, help="override the page font")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "data/input/demo")
    args = parser.parse_args()

    specs = {"ar": [ARABIC], "en": [ENGLISH], "both": [ARABIC, ENGLISH]}[args.lang]
    for spec in specs:
        font = args.font or spec["font"]
        path = render_page(spec, font, args.out_dir)
        print(f"wrote {path}  ({spec['lang']}, {font})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

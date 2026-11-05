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

WINDOWS_FONTS = r"C:\Windows\Fonts"

#: Sakkal Majalla: a Naskh-derived face that ships with Windows and is
#: widely set in Arabic documents. Measured best of eight at 93.0%
#: character accuracy -- Traditional Arabic reached 85.0% and Amiri
#: 63.0% on identical content, which is the Naskh weakness this model
#: still has. Two pages below use the harder faces on purpose: a set
#: that only contains the easy case cannot show anyone where the
#: limit is.
DEFAULT_AR_FONT = "majalla.ttf"


def _ar(stem: str, title: str, words: List[str], lines: List[str],
        font: str = DEFAULT_AR_FONT) -> Dict[str, Any]:
    return {"lang": "ar", "rtl": True, "title": title, "words": words,
            "lines": lines, "font": font, "font_dir": WINDOWS_FONTS,
            "stem": stem}


ARABIC_PAGES: List[Dict[str, Any]] = [
    _ar(
        "ar_01_baa", "حَرْفُ الْبَاءِ",
        ["بَابٌ", "بِنْتٌ", "بُرْتُقَالٌ"],
        ["فَتَحَتِ الْبِنْتُ الْبَابَ",
         "أَكَلَ الْوَلَدُ الْبُرْتُقَالَ"],
    ),
    _ar(
        "ar_02_meem", "حَرْفُ الْمِيمِ",
        ["مَدْرَسَةٌ", "مِفْتَاحٌ", "مُعَلِّمٌ"],
        ["ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ",
         "الْمُعَلِّمُ يَكْتُبُ الدَّرْسَ"],
    ),
    _ar(
        "ar_03_seen", "حَرْفُ السِّينِ",
        ["سَمَكٌ", "سِتَارٌ", "سُلَّمٌ"],
        ["السَّمَكُ يَسْبَحُ فِي الْمَاءِ",
         "صَعِدَ الْقِطُّ عَلَى السُّلَّمِ"],
    ),
    _ar(
        "ar_04_raa", "حَرْفُ الرَّاءِ",
        ["رَجُلٌ", "رِيشَةٌ", "رُمَّانٌ"],
        ["الرَّجُلُ يَقْرَأُ الْكِتَابَ",
         "أَكَلَ الطِّفْلُ الرُّمَّانَ"],
    ),
    _ar(
        "ar_05_dal", "حَرْفُ الدَّالِ",
        ["دَفْتَرٌ", "دِيكٌ", "دُبٌّ"],
        ["كَتَبْتُ الدَّرْسَ فِي الدَّفْتَرِ",
         "صَاحَ الدِّيكُ فِي الصَّبَاحِ"],
    ),
    # A reading page rather than a letter drill: longer lines, which
    # is what the product will actually be pointed at.
    _ar(
        "ar_06_reading", "قِرَاءَةٌ",
        ["الشَّمْسُ", "الْقَمَرُ", "النُّجُومُ"],
        ["تُشْرِقُ الشَّمْسُ فِي الصَّبَاحِ",
         "الْقَمَرُ مُنِيرٌ فِي اللَّيْلِ"],
    ),
    # The same content in the two faces the model reads worst. These
    # exist to be failed, so the gap is visible instead of hidden.
    _ar(
        "ar_07_naskh_trad", "حَرْفُ الْبَاءِ",
        ["بَابٌ", "بِنْتٌ", "بُرْتُقَالٌ"],
        ["فَتَحَتِ الْبِنْتُ الْبَابَ",
         "أَكَلَ الْوَلَدُ الْبُرْتُقَالَ"],
        font="trado.ttf",
    ),
    _ar(
        "ar_08_naskh_amiri", "حَرْفُ الْبَاءِ",
        ["بَابٌ", "بِنْتٌ", "بُرْتُقَالٌ"],
        ["فَتَحَتِ الْبِنْتُ الْبَابَ",
         "أَكَلَ الْوَلَدُ الْبُرْتُقَالَ"],
        font="Amiri-Regular.ttf",
    ),
]

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
    "font_dir": WINDOWS_FONTS,
    "stem": "phonics_b_en",
}

#: Kept so the demo video's default paths still resolve.
ARABIC = ARABIC_PAGES[0]

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
    parser.add_argument("--only", default=None,
                        help="render one page by stem, e.g. ar_02_meem")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "data/input/demo")
    args = parser.parse_args()

    specs = {
        "ar": list(ARABIC_PAGES),
        "en": [ENGLISH],
        "both": list(ARABIC_PAGES) + [ENGLISH],
    }[args.lang]
    if args.only:
        specs = [s for s in specs if s["stem"] == args.only]
        if not specs:
            print(f"ERROR: no page named {args.only!r}", file=sys.stderr)
            print("       have: " + ", ".join(s["stem"] for s in ARABIC_PAGES),
                  file=sys.stderr)
            return 1

    for spec in specs:
        font = args.font or spec["font"]
        # Amiri lives with the project, not with the system fonts.
        spec = dict(spec)
        if font.startswith("Amiri"):
            spec["font_dir"] = str(PROJECT_ROOT / "assets" / "fonts")
        path = render_page(spec, font, args.out_dir)
        print(f"wrote {path.name:<22} {spec['lang']}  {font}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

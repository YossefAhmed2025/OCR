"""Generate the synthetic smoke-test images.

    python scripts/make_synthetic_samples.py

These are **rendered font glyphs, not photographs**. They exist to
prove the pipeline runs end to end and to exercise the metrics. They
are clean, flat, evenly lit and over-sized compared with a real book
page, so **accuracy numbers from them do not generalise** and must not
be quoted as results for Phonics Prism. See
``docs/baseline_findings.md``.

Diacritics are the whole point of the Arabic samples, so this script
verifies that every tashkeel survives rendering and fails loudly if any
is lost. ``arabic-reshaper`` deletes harakat by default, and that
default already produced one wrong experimental conclusion in this
project before it was caught.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phonics_ocr import postprocessing as pp  # noqa: E402
from phonics_ocr.visualization import find_arabic_font, shape_for_display  # noqa: E402

# (text, is_rtl)
Line = Tuple[str, bool]

KATABA = "كَتَبَ الوَلَدُ"
QARAA = "قَرَأَ الطِفْلُ"
KATABA_PLAIN = "كتب الولد"
QARAA_PLAIN = "قرأ الطفل"
KITAB = "الكِتَابُ"

SAMPLES: dict[str, List[Line]] = {
    "test_arabic_tashkeel.png": [(KATABA, True), (QARAA, True)],
    "test_arabic_plain.png": [(KATABA_PLAIN, True), (QARAA_PLAIN, True)],
    "test_english.png": [("The boy wrote the lesson", False), ("Reading is fun", False)],
    # Scripts on SEPARATE lines -- the case mixed mode actually supports.
    "test_mixed.png": [(KITAB, True), ("The Book", False)],
}


def render(
    path: Path, lines: Sequence[Line], font_path: str, width: int, font_size: int
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, 110 + 95 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    y = 45
    for text, rtl in lines:
        display = shape_for_display(text) if rtl else text
        if rtl and pp.count_tashkeel(display) != pp.count_tashkeel(text):
            raise SystemExit(
                f"ABORT: shaping dropped diacritics for {path.name}.\n"
                f"  before: {pp.count_tashkeel(text)}  "
                f"after: {pp.count_tashkeel(display)}\n"
                "  arabic-reshaper must be configured with "
                "delete_harakat=False (see visualization.shape_for_display)."
            )
        box = draw.textbbox((0, 0), display, font=font)
        x = width - (box[2] - box[0]) - 55 if rtl else 55
        draw.text((x, y), display, font=font, fill="black")
        y += 95

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    marks = sum(pp.count_tashkeel(t) for t, _ in lines)
    print(f"  wrote {path}  {image.size}  ({marks} diacritics)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "input"
    )
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--font-size", type=int, default=52)
    parser.add_argument("--font", type=str, default=None, help="path to a .ttf")
    args = parser.parse_args()

    font_path = args.font or find_arabic_font()
    if not font_path:
        print("ERROR: no Arabic-capable font found. Pass one with --font.")
        return 1

    print("Generating SYNTHETIC smoke-test images (not photographs).")
    print(f"Font: {font_path}\n")
    for name, lines in SAMPLES.items():
        render(args.output_dir / name, lines, font_path, args.width, args.font_size)

    print(
        "\nThese prove the pipeline runs. They are NOT a benchmark -- see\n"
        "docs/baseline_findings.md before quoting any number from them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

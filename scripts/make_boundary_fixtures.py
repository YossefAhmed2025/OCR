"""Generate targeted word-boundary stress fixtures with exact ink boxes.

    python scripts/make_boundary_fixtures.py

Where `make_localization_fixtures.py` renders ordinary pages, this one
renders the cases that are *expected to be hard*, so that failures are
found deliberately rather than stumbled on:

* short vs long Arabic words
* tight inter-word gaps (the space the model must still detect)
* adjacent identical letters (CTC collapses repeats without a blank)
* punctuation attached to words
* diacritic-dense text
* several font sizes and line lengths
* mixed Arabic + English on one line

Boxes are exact by construction: each word is re-rendered alone at its
own pen position and its ink measured. Every record is tagged
`capture: synthetic_render`, so the dataset validator and the
evaluation scripts both refuse to treat results from them as
real-world evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from phonics_ocr import postprocessing as pp  # noqa: E402
from phonics_ocr.visualization import find_arabic_font, shape_for_display  # noqa: E402

KATABA = "كَتَبَ"
WALAD = "الوَلَدُ"
MUDARRIS = "مُدّرِسُ"   # shadda + kasra
KITABAN = "كِتَابًا"    # tanween
AB = "أب"                                            # very short
MAKTABA = "المَكتَبَةُ"  # long
KTB = "كتب"
WLD = "الولد"

#: name -> (words, rtl, gap_px_override, font_size, note)
CASES: List[Tuple[str, List[str], bool, int, int, str]] = [
    ("b_ar_short",      [AB, AB, AB],            True,  0, 54, "very short Arabic words"),
    ("b_ar_long",       [MAKTABA, MUDARRIS],     True,  0, 54, "long vowelled Arabic words"),
    ("b_ar_mixedlen",   [AB, MAKTABA, KTB],      True,  0, 54, "short + long + plain"),
    ("b_ar_diacritic",  [MUDARRIS, KITABAN],     True,  0, 54, "shadda and tanween"),
    ("b_ar_gap_tight",  [KTB, WLD],              True,  6, 54, "TIGHT 6px inter-word gap"),
    ("b_ar_gap_wide",   [KTB, WLD],              True, 60, 54, "wide 60px gap (control)"),
    ("b_en_repeat",     ["aa", "ll", "oo"],      False, 0, 54, "adjacent identical letters"),
    ("b_en_repeat_word",["committee", "letter"], False, 0, 54, "internal repeated letters"),
    ("b_en_punct",      ["Hello,", "world!", "(yes)"], False, 0, 54, "punctuation attached"),
    ("b_en_gap_tight",  ["one", "two"],          False, 6, 54, "TIGHT 6px gap, English"),
    ("b_size_small",    [KATABA, WALAD],         True,  0, 26, "small print"),
    ("b_size_large",    [KATABA, WALAD],         True,  0, 86, "large print"),
    ("b_long_line",     [KATABA, WALAD, KTB, WLD, AB], True, 0, 44, "5 words, long line"),
    ("b_mixed_line",    ["Book", KITABAN, "and", KTB], True, 0, 50, "mixed scripts, one line"),
]


def render_case(
    words: Sequence[str],
    rtl: bool,
    gap_override: int,
    font_size: int,
    font_path: str,
    margin: int = 50,
) -> Tuple[Any, List[Dict[str, Any]]]:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10), "white"))
    gap = gap_override if gap_override else probe.textlength(" ", font=font)

    shaped = [shape_for_display(w) if rtl else w for w in words]
    for original, display in zip(words, shaped):
        if pp.count_tashkeel(display) != pp.count_tashkeel(original):
            raise SystemExit(
                f"ABORT: shaping dropped diacritics for {original!r}."
            )
    widths = [probe.textlength(s, font=font) for s in shaped]
    total = sum(widths) + gap * (len(shaped) - 1)

    width = int(total + 2 * margin)
    height = int(font_size * 2.6)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = int(font_size * 0.5)
    x = float(margin)
    sequence = list(zip(words, shaped, widths))
    if rtl:
        sequence = list(reversed(sequence))  # draw left-to-right on screen
    placed: List[Tuple[str, str, float]] = []
    for original, display, w in sequence:
        draw.text((x, y), display, font=font, fill="black")
        placed.append((original, display, x))
        x += w + gap

    if rtl:
        placed = list(reversed(placed))  # back to logical order

    records: List[Dict[str, Any]] = []
    for original, display, pen_x in placed:
        solo = Image.new("RGB", (width, height), "white")
        ImageDraw.Draw(solo).text((pen_x, y), display, font=font, fill="black")
        ink = np.argwhere(np.asarray(solo.convert("L")) < 250)
        if ink.size == 0:
            continue
        y0, x0 = ink.min(axis=0)
        y1, x1 = ink.max(axis=0)
        records.append(
            {
                "text": original,
                "bbox": [float(x0), float(y0), float(x1 + 1), float(y1 + 1)],
                "line": 0,
                "n_tashkeel": pp.count_tashkeel(original),
            }
        )
    return image, records


def measure_gaps(records: Sequence[Dict[str, Any]]) -> List[float]:
    """Actual ink-to-ink horizontal gaps, in image order."""
    boxes = sorted((r["bbox"] for r in records), key=lambda b: b[0])
    return [round(boxes[i + 1][0] - boxes[i][2], 1) for i in range(len(boxes) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "localization_boundary"
    )
    parser.add_argument("--font", type=str, default=None)
    args = parser.parse_args()

    font_path = args.font or find_arabic_font()
    if not font_path:
        print("ERROR: no Arabic-capable font found. Pass --font.")
        return 1

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print("Generating SYNTHETIC boundary-stress fixtures (rendered, not photographed).")
    print(f"Font: {font_path}\n")
    print(f"{'case':<22}{'words':>6}{'size':>6}{'image':>14}  gaps(px)")
    print("-" * 72)

    annotations: List[Dict[str, Any]] = []
    for name, words, rtl, gap, size, note in CASES:
        image, records = render_case(words, rtl, gap, size, font_path)
        path = image_dir / f"{name}.png"
        image.save(path)
        gaps = measure_gaps(records)
        lang = "ar" if any(pp.is_arabic_char(c) for w in words for c in w) else "en"
        annotations.append(
            {
                "id": name,
                "image": f"images/{name}.png",
                "lang": lang,
                "group": "boundary_synthetic",
                "capture": "synthetic_render",
                "width": image.width,
                "height": image.height,
                "text": " ".join(words),
                "words": records,
                "box_source": "exact_render_metrics",
                "font_size": size,
                "measured_ink_gaps_px": gaps,
                "notes": f"SYNTHETIC boundary stress. {note}.",
            }
        )
        print(f"{name:<22}{len(records):>6}{size:>6}{str(image.size):>14}  {gaps}")

    out = args.output_dir / "boxes.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for record in annotations:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = sum(len(a["words"]) for a in annotations)
    print(f"\nWrote {out}")
    print(f"{len(annotations)} images, {total} reference word boxes.")
    print("\nRendered glyphs. Results are an UPPER BOUND; see "
          "docs/word_localization.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

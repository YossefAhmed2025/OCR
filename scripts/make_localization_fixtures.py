"""Generate synthetic pages with EXACT word boxes for localisation testing.

    python scripts/make_localization_fixtures.py

Why these boxes are trustworthy
-------------------------------
The word boxes here are not estimated, traced or eyeballed. Each line
is composed word by word: the renderer measures every word with the
font metrics, draws it at a known pen position, and records the box it
just drew into. The ground truth is therefore **exact by construction**
for the rendered image.

What that does and does not buy
--------------------------------
It gives a correct reference for asking "does CTC alignment put the box
in the right place?" on clean, flat, horizontal text. That is a real
and necessary first check.

It does **not** make these a camera benchmark. They are rendered
glyphs: no lens blur, no shadow, no perspective, no paper texture, no
print bleed. Localisation scores from them are an upper bound, and the
`--rotate` / `--perspective` variants only approximate geometric
distortion, not optical degradation. See `docs/camera_experiments.md`.

Output: images in `data/localization/images/` and annotations in
`data/localization/boxes.jsonl` using the v2 annotation schema.
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
QARAA = "قَرَأَ"
TIFL = "الطِفْلُ"
KITAB = "الكِتَابُ"
JADID = "الجَدِيدُ"
KTB_PLAIN = "كتب"
WLD_PLAIN = "الولد"

#: (name, [(words, rtl)], note)
PAGES: List[Tuple[str, List[Tuple[List[str], bool]], str]] = [
    ("loc_ar_tashkeel", [([KATABA, WALAD], True), ([QARAA, TIFL], True)],
     "Arabic, fully vowelled, 2 words per line"),
    ("loc_ar_plain", [([KTB_PLAIN, WLD_PLAIN], True)],
     "Arabic, no diacritics"),
    ("loc_ar_three", [([KITAB, JADID, KATABA], True)],
     "Arabic, 3 words, vowelled"),
    ("loc_en", [(["The", "boy", "wrote"], False), (["Reading", "is", "fun"], False)],
     "English, 3 words per line"),
    ("loc_en_punct", [(["Hello,", "world!", "Yes."], False)],
     "English with punctuation attached to words"),
]


def render_page(
    words_per_line: Sequence[Tuple[List[str], bool]],
    font_path: str,
    font_size: int,
    width: int,
    margin: int,
    line_height: int,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Draw each word individually and record the exact box it occupies."""
    from PIL import Image, ImageDraw, ImageFont

    height = 2 * margin + line_height * len(words_per_line)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    space_w = draw.textlength(" ", font=font)

    records: List[Dict[str, Any]] = []
    y = margin

    for line_index, (words, rtl) in enumerate(words_per_line):
        shaped = [shape_for_display(w) if rtl else w for w in words]
        for original, display in zip(words, shaped):
            if pp.count_tashkeel(display) != pp.count_tashkeel(original):
                raise SystemExit(
                    "ABORT: shaping dropped diacritics for "
                    f"{original!r}. arabic-reshaper must run with "
                    "delete_harakat=False."
                )
        widths = [draw.textlength(s, font=font) for s in shaped]
        total = sum(widths) + space_w * (len(shaped) - 1)

        # RTL lines are laid out right-aligned; the first logical word
        # sits at the RIGHT edge, so pen positions are assigned from
        # the right. Geometry stays left-to-right either way.
        x = width - margin - total if rtl else margin
        placed: List[Tuple[str, str, float]] = []
        sequence = list(zip(words, shaped, widths))
        if rtl:
            sequence = list(reversed(sequence))  # draw left-to-right on screen
        for original, display, w in sequence:
            draw.text((x, y), display, font=font, fill="black")
            placed.append((original, display, x))
            x += w + space_w

        # Reference boxes are the word's actual INK extent, measured by
        # re-rendering that word alone at the same pen position and
        # finding its non-white pixels. Using the nominal advance width
        # and the full line slot instead would pad every box with
        # whitespace -- which inflates the reference and makes a
        # correctly placed prediction score badly for the wrong reason.
        # Emit word records in LOGICAL reading order so that
        # " ".join(words) reproduces the transcription. `placed` is in
        # draw order, which for RTL runs right-to-left through the
        # logical sequence, so it is reversed back here. The boxes
        # themselves are untouched -- only the record order changes.
        if rtl:
            placed = list(reversed(placed))

        for original, display, pen_x in placed:
            probe = Image.new("RGB", (width, height), "white")
            ImageDraw.Draw(probe).text((pen_x, y), display, font=font, fill="black")
            ink = np.argwhere(np.asarray(probe.convert("L")) < 250)
            if ink.size == 0:
                continue  # whitespace-only token; nothing to localise
            y0, x0 = ink.min(axis=0)
            y1, x1 = ink.max(axis=0)
            records.append(
                {
                    "line": line_index,
                    "text": original,
                    "rtl": rtl,
                    "bbox": [float(x0), float(y0), float(x1 + 1), float(y1 + 1)],
                }
            )
        y += line_height

    return image, records


def apply_rotation(image, degrees: float):
    from PIL import Image

    return image.rotate(
        degrees, resample=Image.BICUBIC, expand=True, fillcolor="white"
    )


def rotate_boxes(
    records: List[Dict[str, Any]], degrees: float, old_size, new_size
) -> List[Dict[str, Any]]:
    """Rotate reference boxes to match PIL's expand=True rotation."""
    theta = np.deg2rad(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    ox, oy = old_size[0] / 2.0, old_size[1] / 2.0
    nx, ny = new_size[0] / 2.0, new_size[1] / 2.0
    out = []
    for record in records:
        x0, y0, x1, y1 = record["bbox"]
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        moved = []
        for px, py in corners:
            dx, dy = px - ox, py - oy
            # PIL rotates counter-clockwise for positive angles.
            rx = dx * cos + dy * sin
            ry = -dx * sin + dy * cos
            moved.append([round(rx + nx, 2), round(ry + ny, 2)])
        xs = [p[0] for p in moved]
        ys = [p[1] for p in moved]
        out.append(
            {
                **record,
                "polygon": moved,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "localization"
    )
    parser.add_argument("--font", type=str, default=None)
    parser.add_argument("--font-size", type=int, default=54)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="also emit a rotated copy of each page, in degrees",
    )
    args = parser.parse_args()

    font_path = args.font or find_arabic_font()
    if not font_path:
        print("ERROR: no Arabic-capable font found. Pass one with --font.")
        return 1

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotations: List[Dict[str, Any]] = []

    print("Generating SYNTHETIC localisation fixtures (rendered, not photographed).")
    print(f"Font: {font_path}\n")

    for name, lines, note in PAGES:
        image, records = render_page(
            lines, font_path, args.font_size, args.width, 45, 105
        )
        path = image_dir / f"{name}.png"
        image.save(path)

        words = [
            {
                "text": r["text"],
                "bbox": r["bbox"],
                "line": r["line"],
                "n_tashkeel": pp.count_tashkeel(r["text"]),
            }
            for r in records
        ]
        lang = "en" if name.startswith("loc_en") else "ar"
        text = " ".join(
            " ".join(ws) for ws, _ in lines
        )
        annotations.append(
            {
                "id": name,
                "image": f"images/{name}.png",
                "lang": lang,
                "group": "clean_synthetic",
                "capture": "synthetic_render",
                "width": image.width,
                "height": image.height,
                "text": text,
                "words": words,
                "box_source": "exact_render_metrics",
                "notes": f"SYNTHETIC. {note}. Boxes exact by construction.",
            }
        )
        print(f"  {path.name:24} {image.size}  {len(words)} words")

        if args.rotate:
            rotated = apply_rotation(image, args.rotate)
            rpath = image_dir / f"{name}_rot{args.rotate:g}.png"
            rotated.save(rpath)
            rrecords = rotate_boxes(
                records, args.rotate, image.size, rotated.size
            )
            annotations.append(
                {
                    "id": f"{name}_rot{args.rotate:g}",
                    "image": f"images/{rpath.name}",
                    "lang": lang,
                    "group": "rotated_synthetic",
                    "capture": "synthetic_render",
                    "width": rotated.width,
                    "height": rotated.height,
                    "text": text,
                    "words": [
                        {
                            "text": r["text"],
                            "bbox": r["bbox"],
                            "polygon": r["polygon"],
                            "line": r["line"],
                            "n_tashkeel": pp.count_tashkeel(r["text"]),
                        }
                        for r in rrecords
                    ],
                    "box_source": "exact_render_metrics_rotated",
                    "notes": (
                        f"SYNTHETIC, rotated {args.rotate} deg. Geometric "
                        "rotation only -- no optical degradation."
                    ),
                }
            )
            print(f"  {rpath.name:24} {rotated.size}  (rotated)")

    out = args.output_dir / "boxes.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for record in annotations:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_words = sum(len(a["words"]) for a in annotations)
    print(f"\nWrote {out}")
    print(f"{len(annotations)} images, {total_words} reference word boxes.")
    print(
        "\nThese are rendered glyphs. Localisation scores from them are an\n"
        "UPPER BOUND and do not predict camera performance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

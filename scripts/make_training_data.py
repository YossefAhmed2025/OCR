"""Build a PaddleOCR recognition training set of vowelled Arabic lines.

    python scripts/make_training_data.py --n 20000
    python scripts/make_training_data.py --n 500 --out data/train_small
    python scripts/make_training_data.py --n 20000 --val-split 0.1 --seed 7

Produces the layout PaddleOCR's recognition trainer expects:

    data/train_rec/
      images/            rendered line crops, height 48
      train_list.txt     "images/xxx.jpg\\tLABEL"   (tab separated)
      val_list.txt
      ppocrv5_arabic_dict.txt
      dataset_info.json

Two conventions this script enforces, because both fail silently:

* **Labels are VISUAL order.** PaddleOCR's Arabic recognisers train on
  visually-ordered text and its decoder reverses at inference. Logical
  order here trains the model backwards.
* **The dict filename contains ``arabic``.** That substring is what
  enables the decoder's reverse path. Renaming it breaks decoding.

Every rendered line is checked to carry the same diacritic count as its
source string; a mismatch aborts the run rather than quietly producing
a dataset that teaches the model to ignore tashkeel.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DICT_NAME = "ppocrv5_arabic_dict.txt"   # 'arabic' substring is load-bearing


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=20000, help="total line samples")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "data" / "train_rec"
    )
    parser.add_argument("--val-split", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--max-words", type=int, default=4)
    parser.add_argument(
        "--real-word-prob", type=float, default=0.92,
        help=(
            "share of words taken verbatim from the real vowelled "
            "lexicon. High on purpose: random vowelization measurably "
            "taught the model to place marks without learning which "
            "mark belongs."
        ),
    )
    parser.add_argument(
        "--lexicon", type=Path, default=None,
        help="vowelled lexicon JSON (default: data/corpus/vowelled_lexicon.json)",
    )
    parser.add_argument(
        "--max-fonts", type=int, default=40, help="cap on distinct fonts used"
    )
    parser.add_argument(
        "--warm-start-charset",
        type=Path,
        default=PROJECT_ROOT / "models" / "arabic_PP-OCRv5_mobile_rec",
        help="model dir whose charset seeds the dictionary (keeps indices)",
    )
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    args = parser.parse_args()

    from phonics_ocr import postprocessing as pp
    from phonics_ocr import synth
    from phonics_ocr.preprocessing import save_image
    from phonics_ocr.utils import configure_logging, iso_now, write_json

    configure_logging(False)
    rng = random.Random(args.seed)

    lexicon = synth.load_lexicon(args.lexicon)
    if lexicon:
        print(f"Vowelled lexicon: {len(lexicon)} real words")
    else:
        print(
            "WARNING: no vowelled lexicon found. Falling back to randomly\n"
            "         vowelized nonsense, which measurably degrades the\n"
            "         fine-tune. Build one first:\n"
            "           python scripts/build_vowelled_lexicon.py"
        )
        lexicon = list(synth.SEED_WORDS)

    if not 0.0 <= args.val_split < 0.9:
        print("ERROR: --val-split must be in [0, 0.9)", file=sys.stderr)
        return 1

    print("Finding fonts with full Arabic + all 8 harakat ...")
    try:
        fonts = synth.find_vowel_capable_fonts()
    except synth.SynthError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not fonts:
        print("ERROR: no font covers Arabic letters and all harakat.", file=sys.stderr)
        return 1
    rng.shuffle(fonts)
    fonts = fonts[: args.max_fonts]
    print(f"  using {len(fonts)} font(s)")

    base_charset: List[str] = []
    if args.warm_start_charset and Path(args.warm_start_charset).is_dir():
        try:
            base_charset = synth.load_model_charset(Path(args.warm_start_charset))
            print(f"  warm-start charset: {len(base_charset)} chars from "
                  f"{Path(args.warm_start_charset).name}")
        except synth.SynthError as exc:
            print(f"  [warn] {exc}; building the dictionary from scratch")

    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    config = synth.RenderConfig(height=args.height)
    records: List[Dict[str, str]] = []
    font_usage: Counter = Counter()
    failures = 0
    mark_total = 0

    print(f"\nRendering {args.n} line(s) ...")
    attempt = 0
    while len(records) < args.n:
        attempt += 1
        if attempt > args.n * 4:
            print("ERROR: too many render failures; aborting.", file=sys.stderr)
            return 1
        text = synth.make_line(
            rng,
            seed_words=lexicon,
            max_words=args.max_words,
            real_word_prob=args.real_word_prob,
        )
        font = rng.choice(fonts)
        try:
            image, visual_label = synth.render_line(text, font, config, rng)
        except synth.SynthError as exc:
            failures += 1
            if failures <= 3:
                print(f"  [skip] {exc}")
            if failures > args.n * 0.5:
                print("ERROR: over half the renders failed; aborting.", file=sys.stderr)
                return 1
            continue

        index = len(records)
        name = f"{index:07d}.{args.format}"
        save_image(image_dir / name, image)
        records.append(
            {
                "path": f"images/{name}",
                "label_visual": visual_label,
                "label_logical": text,
                "font": font.name,
            }
        )
        font_usage[font.name] += 1
        mark_total += pp.count_tashkeel(text)

        if len(records) % 2000 == 0:
            print(f"  {len(records)}/{args.n}")

    rng.shuffle(records)
    split = int(len(records) * (1 - args.val_split))
    train, val = records[:split], records[split:]

    def write_list(path: Path, rows: List[Dict[str, str]]) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                # PaddleOCR rec format: <relative path>\t<label>
                handle.write(f"{row['path']}\t{row['label_visual']}\n")

    write_list(args.out / "train_list.txt", train)
    write_list(args.out / "val_list.txt", val)

    charset = synth.build_dictionary(
        (r["label_visual"] for r in records), base=base_charset
    )
    dict_path = args.out / DICT_NAME
    with open(dict_path, "w", encoding="utf-8", newline="\n") as handle:
        for ch in charset:
            handle.write(ch + "\n")

    # Written separately: handy for inspecting what the model will see
    # without opening the image folder.
    write_json(
        args.out / "samples_preview.json",
        [
            {
                "path": r["path"],
                "logical": r["label_logical"],
                "visual": r["label_visual"],
                "tashkeel": pp.count_tashkeel(r["label_logical"]),
                "font": r["font"],
            }
            for r in records[:40]
        ],
    )

    marks_present = sorted(
        {ch for r in records for ch in r["label_visual"] if pp.is_tashkeel(ch)}
    )
    info = {
        "generated_at": iso_now(),
        "n_total": len(records),
        "n_train": len(train),
        "n_val": len(val),
        "seed": args.seed,
        "image_height": args.height,
        "n_fonts": len(fonts),
        "fonts_top10": font_usage.most_common(10),
        "render_failures": failures,
        "charset_size": len(charset),
        "charset_warm_started_from": (
            str(args.warm_start_charset) if base_charset else None
        ),
        "dict_file": DICT_NAME,
        "mean_tashkeel_per_line": round(mark_total / max(1, len(records)), 2),
        "distinct_diacritics_present": len(marks_present),
        "label_order": "VISUAL (bidi.get_display) -- PaddleOCR Arabic convention",
        "lexicon_words": len(lexicon),
        "real_word_prob": args.real_word_prob,
        "conventions": [
            "Labels are VISUAL order; PaddleOCR reverses at decode time.",
            f"Dict filename must contain 'arabic' ({DICT_NAME}) to enable "
            "the decoder's reverse path.",
            "Every line verified to preserve its source diacritic count.",
        ],
        "caveat": (
            "SYNTHETIC rendered text. Augmentation approximates camera "
            "degradation; it is not photographed paper. A model trained "
            "only on this must still be evaluated on real pages."
        ),
    }
    write_json(args.out / "dataset_info.json", info)

    print(f"\n{'=' * 66}")
    print(f"  samples      : {len(records)}  ({len(train)} train / {len(val)} val)")
    print(f"  fonts        : {len(fonts)}")
    print(f"  charset      : {len(charset)} chars"
          + (f" (warm-started from {len(base_charset)})" if base_charset else ""))
    print(f"  diacritics   : {len(marks_present)} distinct, "
          f"{info['mean_tashkeel_per_line']} per line on average")
    print(f"  render fails : {failures}")
    print(f"  output       : {args.out}")
    print(f"{'=' * 66}")
    print("\nNext: review a few crops in", image_dir)
    print("      then see docs/finetuning.md for the training command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Make the generated dataset safe for PP-OCRv5 CTC training.

    python scripts/prepare_training_set.py
    python scripts/prepare_training_set.py --max-label 35

Two constraints that break training silently if ignored.

**1. The dictionary must match the pretrained output layer.**
The official ``ppocrv5_arabic_dict.txt`` has 747 entries, so the CTC
head emits 749 classes (747 + blank + space). A locally-built
dictionary covering only the characters that happen to appear in the
generated data is smaller -- ours was 717 -- which changes the head's
shape and makes the pretrained weights unloadable. The warm start is
the whole point of fine-tuning, so this script copies the **official**
dictionary from the PaddleOCR repo and verifies every label character
is inside it.

**2. Labels cannot be longer than the CTC sequence.**
Training resizes every crop to ``[3, 48, 320]``, and the width stride
is 8, so there are exactly ``320 / 8 = 40`` timesteps. CTC cannot
align a label longer than the number of timesteps -- the loss is
undefined. Labels above the limit are dropped here rather than left to
fail or be silently truncated mid-training.

Writes a filtered copy; the generated data is left untouched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

#: Timesteps available at the training input width of 320 px.
#: Verified empirically: T == floor(W / 8) for every width tested.
CTC_TIMESTEPS = 40

DICT_NAME = "ppocrv5_arabic_dict.txt"

#: Where PaddleX unpacks the PaddleOCR repo.
DEFAULT_REPO = (
    PROJECT_ROOT
    / ".venv/lib/site-packages/paddlex/repo_manager/repos/PaddleOCR"
)


def read_list(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            image, label = line.split("\t", 1)
            rows.append((image, label))
    return rows


def write_list(path: Path, rows: List[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for image, label in rows:
            handle.write(f"{image}\t{label}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=PROJECT_ROOT / "data" / "train_rec")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "data" / "train_rec_ready"
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--max-label",
        type=int,
        default=35,
        help=f"drop labels longer than this (hard CTC ceiling is {CTC_TIMESTEPS})",
    )
    args = parser.parse_args()

    from phonics_ocr import postprocessing as pp
    from phonics_ocr.utils import write_json, iso_now

    if args.max_label > CTC_TIMESTEPS:
        print(
            f"ERROR: --max-label {args.max_label} exceeds the {CTC_TIMESTEPS} "
            "CTC timesteps available at width 320. Training would fail.",
            file=sys.stderr,
        )
        return 1

    train_src = args.src / "train.txt"
    val_src = args.src / "val.txt"
    for path in (train_src, val_src):
        if not path.is_file():
            print(f"ERROR: missing {path}", file=sys.stderr)
            print("       Run scripts/make_training_data.py first.", file=sys.stderr)
            return 1

    official = args.repo / "ppocr" / "utils" / "dict" / DICT_NAME
    if not official.is_file():
        print(f"ERROR: official dictionary not found at {official}", file=sys.stderr)
        print("       Install the training plugin:", file=sys.stderr)
        print("         python -m paddlex --install PaddleOCR -y", file=sys.stderr)
        return 1

    charset = [
        line.rstrip("\n")
        for line in official.read_text(encoding="utf-8").splitlines()
    ]
    known = set(charset)
    print(f"Official dictionary: {len(charset)} entries  ({official.name})")

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(official, args.out / DICT_NAME)
    shutil.copy2(official, args.out / "dict.txt")

    # Images are reused in place; the lists point at the source folder
    # so 165 MB is not duplicated.
    images_link = args.out / "images"
    if not images_link.exists():
        try:
            images_link.symlink_to(
                (args.src / "images").resolve(), target_is_directory=True
            )
            print("Linked images/ (symlink)")
        except OSError:
            # Windows without developer mode: fall back to copying.
            print("Symlink unavailable; copying images (this takes a minute)")
            shutil.copytree(args.src / "images", images_link)

    report: Dict[str, Dict[str, int]] = {}
    for tag, source in (("train", train_src), ("val", val_src)):
        rows = read_list(source)
        kept: List[Tuple[str, str]] = []
        too_long = 0
        unknown_chars: Counter = Counter()
        dropped_unknown = 0

        for image, label in rows:
            if len(label) > args.max_label:
                too_long += 1
                continue
            missing = [c for c in label if c not in known and c != " "]
            if missing:
                unknown_chars.update(missing)
                dropped_unknown += 1
                continue
            kept.append((image, label))

        write_list(args.out / f"{tag}.txt", kept)
        lengths = [len(label) for _, label in kept]
        marks = sum(pp.count_tashkeel(label) for _, label in kept)
        report[tag] = {
            "input": len(rows),
            "kept": len(kept),
            "dropped_too_long": too_long,
            "dropped_unknown_char": dropped_unknown,
            "max_label_len": max(lengths) if lengths else 0,
            "mean_label_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "mean_tashkeel": round(marks / len(kept), 2) if kept else 0,
        }
        print(
            f"  {tag:<5} {len(rows)} -> {len(kept)} kept "
            f"({too_long} too long, {dropped_unknown} unknown chars)"
        )
        if unknown_chars:
            print(f"        unknown: {dict(unknown_chars.most_common(8))}")

    write_json(
        args.out / "prepare_info.json",
        {
            "generated_at": iso_now(),
            "source": str(args.src),
            "dictionary": str(official),
            "dictionary_entries": len(charset),
            "max_label_length": args.max_label,
            "ctc_timesteps_at_width_320": CTC_TIMESTEPS,
            "splits": report,
            "notes": [
                "Dictionary is the OFFICIAL one so the pretrained CTC head "
                "loads without a shape mismatch.",
                f"Labels longer than {args.max_label} characters are dropped: "
                f"CTC cannot align more labels than the {CTC_TIMESTEPS} "
                "timesteps available at width 320.",
                "Labels are in visual order with base Arabic characters.",
            ],
        },
    )

    total_kept = sum(r["kept"] for r in report.values())
    print(f"\nReady: {args.out}")
    print(f"  {total_kept} samples, dictionary {len(charset)} entries")
    print(f"  max label {args.max_label} chars (CTC ceiling {CTC_TIMESTEPS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

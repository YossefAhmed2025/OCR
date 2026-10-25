"""Launch the Arabic vowelled fine-tune against the PaddleOCR repo.

    python scripts/train_arabic.py
    python scripts/train_arabic.py --check      # verify setup, do not train
    python scripts/train_arabic.py --epochs 5   # short run

PaddleOCR's trainer lives in the repository, not the pip package.
PaddleX unpacks that repo when the training plugin is installed, so
this finds it, sets PYTHONPATH, checks the preconditions that fail
confusingly mid-run, and then hands off to ``tools/train.py``.

Preconditions checked before launching, because each one costs an
hour to discover otherwise:

* the PaddleOCR repo and its ``tools/train.py`` exist
* the pretrained warm-start weights are present
* the dictionary has 747 entries, matching the pretrained CTC head
* train/val lists exist and no label exceeds the CTC timestep budget
* the GPU is actually usable
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_REPO = (
    PROJECT_ROOT / ".venv/lib/site-packages/paddlex/repo_manager/repos/PaddleOCR"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "arabic_vowelled_ppocr.yml"

#: Timesteps at the training width of 320 (stride 8). A label longer
#: than this cannot be aligned by CTC and the loss blows up.
CTC_TIMESTEPS = 40
EXPECTED_DICT_ENTRIES = 747


def check(config_path: Path, repo: Path) -> int:
    """Validate everything that would otherwise fail mid-training."""
    import yaml

    problems = []
    notes = []

    train_py = repo / "tools" / "train.py"
    if not train_py.is_file():
        problems.append(
            f"PaddleOCR trainer not found at {train_py}\n"
            "    Install it with:  python -m paddlex --install PaddleOCR -y"
        )
    else:
        notes.append(f"trainer      : {train_py}")

    if not config_path.is_file():
        problems.append(f"config not found: {config_path}")
        for item in notes:
            print(f"  {item}")
        for item in problems:
            print(f"  FAIL {item}")
        return 1

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    glob = config["Global"]

    weights = PROJECT_ROOT / (glob["pretrained_model"] + ".pdparams")
    if not weights.is_file():
        problems.append(
            f"pretrained weights missing: {weights}\n"
            "    Download arabic_PP-OCRv5_mobile_rec_pretrained.pdparams "
            "into pretrained/"
        )
    else:
        notes.append(f"warm start   : {weights.name} "
                     f"({weights.stat().st_size / 1e6:.0f} MB)")

    dict_path = PROJECT_ROOT / glob["character_dict_path"]
    if not dict_path.is_file():
        problems.append(f"dictionary missing: {dict_path}")
    else:
        entries = len(dict_path.read_text(encoding="utf-8").splitlines())
        if entries != EXPECTED_DICT_ENTRIES:
            problems.append(
                f"dictionary has {entries} entries, expected "
                f"{EXPECTED_DICT_ENTRIES}. A different size changes the CTC "
                "head width and the pretrained weights will not load."
            )
        else:
            notes.append(f"dictionary   : {entries} entries, {dict_path.name}")
        if "arabic" not in dict_path.name:
            problems.append(
                f"dictionary filename {dict_path.name!r} does not contain "
                "'arabic'; that substring enables the RTL decode path."
            )

    max_len = int(glob["max_text_length"])
    if max_len > CTC_TIMESTEPS:
        problems.append(
            f"max_text_length {max_len} exceeds the {CTC_TIMESTEPS} CTC "
            "timesteps available at width 320."
        )

    for key, section in (("train", "Train"), ("val", "Eval")):
        listed = config[section]["dataset"]["label_file_list"]
        for entry in listed:
            path = PROJECT_ROOT / entry
            if not path.is_file():
                problems.append(
                    f"{key} list missing: {path}\n"
                    "    Run:  python scripts/prepare_training_set.py"
                )
                continue
            rows = [
                line.split("\t", 1)
                for line in path.read_text(encoding="utf-8").splitlines()
                if "\t" in line
            ]
            longest = max((len(label) for _, label in rows), default=0)
            if longest > max_len:
                problems.append(
                    f"{key} contains a {longest}-character label but "
                    f"max_text_length is {max_len}."
                )
            notes.append(f"{key:<13}: {len(rows)} samples, longest label {longest}")

    try:
        from phonics_ocr.engine import detect_gpu

        gpu = detect_gpu()
        if gpu["usable"]:
            notes.append(
                f"gpu          : {gpu.get('device_name', 'ok')}, "
                f"{gpu.get('total_memory_mb', '?')} MB"
            )
        elif glob.get("use_gpu"):
            problems.append(f"use_gpu is true but the GPU is unusable: {gpu['detail']}")
    except Exception as exc:  # pragma: no cover - environment dependent
        notes.append(f"gpu          : could not probe ({exc})")

    print(f"\n{'=' * 66}")
    print("TRAINING PRECONDITIONS")
    print(f"{'=' * 66}")
    for item in notes:
        print(f"  OK   {item}")
    for item in problems:
        print(f"  FAIL {item}")
    print(f"{'=' * 66}")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--check", action="store_true", help="validate and exit")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch_num")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    status = check(args.config, args.repo)
    if args.check or status != 0:
        if status != 0:
            print("\nFix the failures above before training.")
        return status

    overrides = []
    if args.epochs is not None:
        overrides.append(f"Global.epoch_num={args.epochs}")
    if args.batch_size is not None:
        overrides += [
            f"Train.sampler.first_bs={args.batch_size}",
            f"Train.loader.batch_size_per_card={args.batch_size}",
        ]

    command = [
        sys.executable,
        str(args.repo / "tools" / "train.py"),
        "-c",
        str(args.config),
    ]
    if overrides:
        command += ["-o"] + overrides

    env = dict(os.environ)
    # The trainer imports ppocr.* from the repo root.
    env["PYTHONPATH"] = str(args.repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    # Paddle's default 8-worker loader deadlocks on Windows spawn.
    env.setdefault("FLAGS_call_stack_level", "2")

    print("\nLaunching training. Working directory:", PROJECT_ROOT)
    print("  " + " ".join(command) + "\n")
    return subprocess.call(command, cwd=str(PROJECT_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())

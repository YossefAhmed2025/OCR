"""Export the fine-tuned checkpoint to an inference model.

    python scripts/export_finetuned.py
    python scripts/export_finetuned.py --checkpoint output/arabic_vowelled/iter_epoch_35

Training writes ``.pdparams`` checkpoints, which the OCR pipeline
cannot load: inference needs an exported ``inference.json`` /
``inference.pdiparams`` pair plus the character dictionary. This runs
PaddleOCR's exporter and then copies the dictionary in beside it, so
the result drops straight into ``models/`` and works with the existing
``--model-dir`` plumbing.

The dictionary copy matters. An exported model carries its charset
inline, and if that charset does not match the one training used, the
model decodes to the wrong characters while looking perfectly healthy.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_REPO = (
    PROJECT_ROOT / ".venv/lib/site-packages/paddlex/repo_manager/repos/PaddleOCR"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "arabic_vowelled_ppocr.yml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "output" / "arabic_vowelled" / "best_accuracy",
        help="checkpoint prefix, without the .pdparams suffix",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "models" / "arabic_vowelled_rec",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()

    import yaml

    weights = Path(str(args.checkpoint) + ".pdparams")
    if not weights.is_file():
        print(f"ERROR: checkpoint not found: {weights}", file=sys.stderr)
        print("       Train first:  python scripts/train_arabic.py", file=sys.stderr)
        return 1

    export_py = args.repo / "tools" / "export_model.py"
    if not export_py.is_file():
        print(f"ERROR: exporter not found: {export_py}", file=sys.stderr)
        return 1

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dict_path = PROJECT_ROOT / config["Global"]["character_dict_path"]
    if not dict_path.is_file():
        print(f"ERROR: dictionary not found: {dict_path}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(export_py),
        "-c",
        str(args.config),
        "-o",
        f"Global.pretrained_model={args.checkpoint}",
        f"Global.save_inference_dir={args.out}",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"Exporting {weights.name} -> {args.out}")
    status = subprocess.call(command, cwd=str(PROJECT_ROOT), env=env)
    if status != 0:
        print("ERROR: export failed.", file=sys.stderr)
        return status

    # The pipeline looks for the dictionary beside the model.
    shutil.copy2(dict_path, args.out / dict_path.name)

    produced = sorted(p.name for p in args.out.iterdir())
    print(f"\nExported to {args.out}")
    for name in produced:
        print(f"  {name}")

    expected = ("inference.json", "inference.pdiparams")
    missing = [name for name in expected if not (args.out / name).is_file()]
    if missing:
        print(f"\nWARNING: expected files missing: {missing}")
        print("The exporter may use different names in this PaddleOCR version.")
        return 1

    print("\nCompare it against the baseline before trusting it:")
    print("  python scripts/compare_models.py \\")
    print("      --ground-truth data/ground_truth/book_page.jsonl \\")
    print(f"      --candidate {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

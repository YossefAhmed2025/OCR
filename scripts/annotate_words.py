"""Draw word-level boxes on real images and save them as v2 annotations.

    python scripts/annotate_words.py --image data/input/page.jpg --lang ar
    python scripts/annotate_words.py --input-dir data/input --lang ar
    python scripts/annotate_words.py --image page.jpg --review   # view existing

Controls
--------
  left-drag        draw a box around ONE word
  type + Enter     enter that word's text (typed in the terminal)
  u                undo the last box
  n / Enter        next image (saves)
  p                previous image
  s                save now
  r                redraw / clear all boxes on this image
  + / -            zoom in / out
  arrow keys       pan when zoomed
  q or Esc         quit (saves)

Annotation rules enforced here
------------------------------
* Boxes are entered in **logical reading order** -- for Arabic that
  means the right-most word on a line first. The tool shows the running
  transcription so the order is visible as you go.
* Draw around the **ink, including diacritics** above and below. A box
  that clips tashkeel is wrong for this project.
* Text is typed in the terminal, not captured from keystrokes in the
  window, so Arabic input uses your normal system keyboard and IME.

Output is schema v2 JSONL, which `validate_dataset.py` and
`evaluate_localization.py` both read directly. `capture` defaults to
`photograph`; pass `--capture` if that is wrong, because only
`photograph` counts as real-world evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HELP_LINES = [
    "drag=box  u=undo  r=reset  n=next  p=prev  s=save  +/-=zoom  q=quit",
]


class Annotator:
    """Minimal OpenCV box annotator with zoom and pan."""

    def __init__(self, image, title: str) -> None:
        import cv2

        self.cv2 = cv2
        self.image = image
        self.title = title
        self.boxes: List[Dict[str, Any]] = []
        self.scale = 1.0
        self.offset = [0, 0]
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_now: Optional[Tuple[int, int]] = None
        self._pending: Optional[Tuple[float, float, float, float]] = None

        screen_w, screen_h = 1500, 820
        h, w = image.shape[:2]
        self.scale = min(1.0, screen_w / w, screen_h / h)

    # -- coordinate helpers -------------------------------------

    def to_image(self, x: int, y: int) -> Tuple[float, float]:
        return (x / self.scale + self.offset[0], y / self.scale + self.offset[1])

    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int((x - self.offset[0]) * self.scale),
            int((y - self.offset[1]) * self.scale),
        )

    # -- mouse ---------------------------------------------------

    def on_mouse(self, event, x, y, flags, param) -> None:
        cv2 = self.cv2
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start:
            self._drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start:
            x0, y0 = self.to_image(*self._drag_start)
            x1, y1 = self.to_image(x, y)
            self._drag_start = None
            self._drag_now = None
            bx0, bx1 = sorted((x0, x1))
            by0, by1 = sorted((y0, y1))
            if bx1 - bx0 < 3 or by1 - by0 < 3:
                print("  (box too small, ignored)")
                return
            h, w = self.image.shape[:2]
            self._pending = (
                max(0.0, bx0), max(0.0, by0), min(float(w), bx1), min(float(h), by1),
            )

    def take_pending(self) -> Optional[Tuple[float, float, float, float]]:
        pending, self._pending = self._pending, None
        return pending

    # -- rendering -----------------------------------------------

    def render(self):
        import cv2
        import numpy as np

        h, w = self.image.shape[:2]
        view_w = int(w * self.scale)
        view_h = int(h * self.scale)
        canvas = cv2.resize(self.image, (max(1, view_w), max(1, view_h)))

        for i, box in enumerate(self.boxes):
            x0, y0 = self.to_screen(box["bbox"][0], box["bbox"][1])
            x1, y1 = self.to_screen(box["bbox"][2], box["bbox"][3])
            colour = [(0, 165, 255), (0, 200, 0), (220, 0, 180), (0, 0, 230)][i % 4]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
            cv2.putText(
                canvas, str(i + 1), (x0 + 3, max(14, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )

        if self._drag_start and self._drag_now:
            cv2.rectangle(canvas, self._drag_start, self._drag_now, (255, 0, 0), 1)

        banner = np.full((28, canvas.shape[1], 3), 32, dtype=np.uint8)
        cv2.putText(
            banner, f"{self.title}  |  {len(self.boxes)} boxes  |  {HELP_LINES[0]}",
            (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
        )
        return np.vstack([banner, canvas])


def annotate_image(path: Path, existing: Optional[Dict[str, Any]], review: bool):
    """Returns the word list for one image, or None if the user quit."""
    import cv2

    from phonics_ocr import postprocessing as pp
    from phonics_ocr.preprocessing import load_image

    image = load_image(path)
    annotator = Annotator(image, path.name)
    if existing:
        annotator.boxes = [dict(w) for w in existing.get("words", [])]

    window = "annotate_words"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, annotator.on_mouse)

    print(f"\n=== {path.name} ===")
    if review:
        print("  REVIEW mode: existing boxes shown. Edit or press n.")
    print("  Drag a box around one word, then type its text in THIS terminal.")
    print("  Arabic: enter words in logical order (right-most word first).")

    action = "next"
    while True:
        cv2.imshow(window, annotator.render())
        key = cv2.waitKey(20) & 0xFF

        pending = annotator.take_pending()
        if pending is not None:
            cv2.imshow(window, annotator.render())
            cv2.waitKey(1)
            try:
                text = input(f"  word #{len(annotator.boxes) + 1} text (blank=cancel): ")
            except EOFError:
                text = ""
            if text.strip():
                annotator.boxes.append(
                    {"text": text.strip(), "bbox": [round(v, 1) for v in pending],
                     "line": 0}
                )
                joined = " ".join(b["text"] for b in annotator.boxes)
                marks = pp.count_tashkeel(joined)
                print(f"    added. {len(annotator.boxes)} boxes, {marks} tashkeel")
                print(f"    running text: {joined}")
            else:
                print("    cancelled")

        if key in (ord("q"), 27):
            action = "quit"
            break
        if key in (ord("n"), 13, 10):
            action = "next"
            break
        if key == ord("p"):
            action = "prev"
            break
        if key == ord("s"):
            action = "save"
            break
        if key == ord("u") and annotator.boxes:
            removed = annotator.boxes.pop()
            print(f"    undid {removed['text']!r}")
        if key == ord("r"):
            annotator.boxes.clear()
            print("    cleared all boxes")
        if key in (ord("+"), ord("=")):
            annotator.scale = min(4.0, annotator.scale * 1.25)
        if key in (ord("-"), ord("_")):
            annotator.scale = max(0.1, annotator.scale / 1.25)
        if key == 81:
            annotator.offset[0] = max(0, annotator.offset[0] - 60)
        if key == 83:
            annotator.offset[0] += 60
        if key == 82:
            annotator.offset[1] = max(0, annotator.offset[1] - 60)
        if key == 84:
            annotator.offset[1] += 60

    cv2.destroyWindow(window)
    return annotator.boxes, action


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--lang", choices=["ar", "en", "mixed"], required=True)
    parser.add_argument(
        "--capture",
        choices=["photograph", "flatbed_scan", "screenshot", "synthetic_render"],
        default="photograph",
        help="how the image was produced; only 'photograph' counts as real-world",
    )
    parser.add_argument("--group", default="")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "ground_truth" / "manual_words.jsonl",
    )
    parser.add_argument("--review", action="store_true", help="load existing boxes")
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("ERROR: OpenCV is required. pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    from phonics_ocr.preprocessing import load_image
    from phonics_ocr.utils import find_images

    try:
        images = find_images(args.image if args.image else args.input_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not images:
        print("ERROR: no images found.", file=sys.stderr)
        return 1

    existing: Dict[str, Dict[str, Any]] = {}
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["id"]] = record
        print(f"Loaded {len(existing)} existing record(s) from {args.output}")

    results: Dict[str, Dict[str, Any]] = dict(existing)
    index = 0
    while 0 <= index < len(images):
        path = images[index]
        prior = results.get(path.stem) if args.review or path.stem in results else None
        boxes, action = annotate_image(path, prior, args.review)

        if boxes:
            image = load_image(path)
            height, width = image.shape[:2]
            results[path.stem] = {
                "id": path.stem,
                "image": str(
                    path.resolve().relative_to(args.output.parent.resolve().parent.parent)
                    if args.output.parent.resolve().parent.parent in path.resolve().parents
                    else path.resolve()
                ).replace("\\", "/"),
                "lang": args.lang,
                "group": args.group or f"{args.lang}_manual",
                "capture": args.capture,
                "width": width,
                "height": height,
                "text": " ".join(b["text"] for b in boxes),
                "words": boxes,
                "box_source": "manual_annotation",
                "notes": "Hand-annotated word boxes.",
            }

        # Save after every image so a crash never loses work.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            for record in results.values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if action == "quit":
            break
        if action == "prev":
            index = max(0, index - 1)
        elif action == "next":
            index += 1

    total_boxes = sum(len(r.get("words", [])) for r in results.values())
    print(f"\nSaved {len(results)} image(s), {total_boxes} word box(es)")
    print(f"  -> {args.output}")
    print("\nNext:")
    print(f"  python scripts/validate_dataset.py --ground-truth {args.output}")
    print(f"  python scripts/evaluate_localization.py --ground-truth {args.output}")
    if total_boxes < 100:
        print(f"\nNOTE: {total_boxes} boxes is a small sample. See")
        print("      docs/evaluation_protocol.md section 6 on sample size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

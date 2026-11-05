"""Launch the local web UI for camera OCR.

    python scripts/webui.py
    python scripts/webui.py --device cpu --port 8080
    python scripts/webui.py --no-save --words

Opens http://127.0.0.1:8000 in the browser. The camera is supplied by
the browser, the OCR runs here, and nothing leaves the machine: the
server binds to loopback, serves its own HTML, and the page loads no
external scripts or fonts.

Browsers only grant camera access on a secure origin, and
``http://127.0.0.1`` counts as one, so no certificate is needed. A
LAN address would not -- which is the other reason this stays on
loopback.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument(
        "--lang", choices=["ar", "en", "mixed"], default="ar",
        help="engine to preload; the page can switch languages anyway",
    )
    parser.add_argument("--no-save", action="store_true", help="write nothing to disk")
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument(
        "--words", action="store_true",
        help="also derive EXPERIMENTAL word spans from CTC alignment",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from phonics_ocr.config import OCRConfig
    from phonics_ocr.utils import configure_logging, run_id, set_offline_env
    from phonics_ocr.webapp import OCRService, serve

    configure_logging(args.verbose)
    offline = not args.no_offline
    if offline:
        set_offline_env()

    output_dir = args.output_dir or (
        PROJECT_ROOT / "data" / "results" / run_id("webui")
    )

    service = OCRService(
        device=args.device,
        offline=offline,
        output_dir=None if args.no_save else output_dir,
        save=not args.no_save,
        words=args.words,
    )

    print("=" * 62)
    print("  Phonics Prism - local OCR web UI")
    print("=" * 62)
    for row in service.info()["models"]:
        tag = "fine-tuned" if row["finetuned"] else "stock"
        print(f"  {row['lang']:<6} {row['weights']:<34} {tag}")
    print()

    print(f"Preloading the '{args.lang}' engine ...")
    try:
        engine = service.engine(args.lang)
    except Exception as exc:
        print(f"\nERROR: could not load the engine -- {exc}", file=sys.stderr)
        return 1
    print(f"  ready on {engine.device}")

    if args.no_save:
        print("  saving DISABLED (--no-save)")
    else:
        print(f"  saving to {output_dir}")
    if args.words:
        print("  word spans ON (experimental, derived from CTC timesteps)")

    try:
        server = serve(service, args.host, args.port)
    except OSError as exc:
        print(f"\nERROR: could not bind {args.host}:{args.port} -- {exc}", file=sys.stderr)
        print("       Try a different --port.", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"\n  open {url}")
    print("  Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

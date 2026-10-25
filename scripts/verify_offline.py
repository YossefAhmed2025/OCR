"""Verify that OCR really runs with no network access.

    python scripts/verify_offline.py
    python scripts/verify_offline.py --image data/input/my_page.jpg
    python scripts/verify_offline.py --no-block   # diagnostics only

What this does, in order:

1. Confirms every required model file is present in ``models/``.
2. Sets the offline environment switches.
3. **Installs a socket-level block** that makes any outbound TCP
   connection raise, then imports paddle/paddleocr and runs real
   inference through it.
4. Reports any connection that was attempted and blocked.

Step 3 is the point. "The code contains no HTTP calls" is not evidence
of offline operation: model hubs, telemetry and lazy downloads all live
inside dependencies. This blocks the socket layer underneath them, so a
pass means inference completed while outbound connections were
impossible.

Caveat, stated honestly: the block is installed in *this Python
process* and covers ``socket.socket.connect``/``connect_ex`` and
``socket.create_connection``. A dependency that opened a connection
from a C extension with its own socket stack, or in a subprocess, would
not be caught. For a stronger guarantee, disconnect the machine from
the network and run this script again -- it works either way, and the
report says which mode it ran in.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BLOCKED_ATTEMPTS: List[str] = []


class NetworkBlockedError(OSError):
    """Raised when blocked code attempts an outbound connection."""


def install_network_block() -> None:
    """Make every outbound socket connection in this process fail.

    Loopback is left open: it is not an external dependency, and some
    libraries use it for internal IPC.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def _is_loopback(address: Any) -> bool:
        try:
            host = address[0] if isinstance(address, (tuple, list)) else address
            return str(host) in ("127.0.0.1", "::1", "localhost")
        except (IndexError, TypeError):
            return False

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        if _is_loopback(address):
            return real_connect(self, address)
        BLOCKED_ATTEMPTS.append(f"socket.connect -> {address}")
        raise NetworkBlockedError(
            f"Outbound network blocked by verify_offline.py: {address}"
        )

    def guarded_connect_ex(self, address):  # type: ignore[no-untyped-def]
        if _is_loopback(address):
            return real_connect_ex(self, address)
        BLOCKED_ATTEMPTS.append(f"socket.connect_ex -> {address}")
        return 1

    def guarded_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        if _is_loopback(address):
            return real_create_connection(address, *args, **kwargs)
        BLOCKED_ATTEMPTS.append(f"socket.create_connection -> {address}")
        raise NetworkBlockedError(
            f"Outbound network blocked by verify_offline.py: {address}"
        )

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]


def self_test_block() -> bool:
    """Confirm the block actually blocks.

    The probe deliberately attempts a connection, so its own entry is
    removed from ``BLOCKED_ATTEMPTS`` afterwards. Leaving it in would
    make every report claim that something tried to reach the network,
    when the only thing that did was this check.
    """
    try:
        socket.create_connection(("example.com", 80), timeout=3)
        blocked = False
    except NetworkBlockedError:
        blocked = True
    except OSError:
        # Already offline at the OS level; also fine.
        blocked = True
    finally:
        BLOCKED_ATTEMPTS[:] = [
            a for a in BLOCKED_ATTEMPTS if "example.com" not in a
        ]
    return blocked


def make_test_image(path: Path) -> Path:
    """Render a small bilingual test page with PIL.

    Synthetic, so it needs no network and no bundled asset. It is a
    *smoke test* for the offline path, not an accuracy benchmark: the
    text is clean rendered glyphs, not a photographed book page.
    """
    from PIL import Image, ImageDraw, ImageFont

    from phonics_ocr.visualization import find_arabic_font, shape_for_display

    image = Image.new("RGB", (900, 320), "white")
    draw = ImageDraw.Draw(image)
    font_path = find_arabic_font()
    try:
        font = ImageFont.truetype(font_path, 48) if font_path else ImageFont.load_default()
        small = ImageFont.truetype(font_path, 34) if font_path else ImageFont.load_default()
    except Exception:
        font = small = ImageFont.load_default()

    draw.text((40, 40), "Reading Test Page", font=font, fill="black")
    draw.text((40, 120), shape_for_display("القراءة"), font=font, fill="black")
    draw.text(
        (40, 200),
        shape_for_display("كَتَبَ الوَلَدُ"),
        font=small,
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, help="image to run (default: synthetic)")
    parser.add_argument("--lang", choices=["ar", "en", "mixed"], default="ar")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="skip the socket block (use when already physically disconnected)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("OFFLINE VERIFICATION")
    print("=" * 70)

    # Step 1: models present. Done before any heavy import.
    from phonics_ocr.config import DEFAULT_MODEL_DIR, required_models

    print("\n[1/4] Checking local models ...")
    missing = []
    for name in required_models(args.lang):
        directory = DEFAULT_MODEL_DIR / name
        if directory.is_dir() and any(directory.iterdir()):
            size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
            print(f"      OK   {name}  ({size / 1e6:.1f} MB)")
        else:
            print(f"      MISS {name}")
            missing.append(name)
    if missing:
        print(f"\nFAILED: {len(missing)} model(s) missing from {DEFAULT_MODEL_DIR}.")
        print("Run (with network): python scripts/download_models.py")
        return 1

    # Step 2: offline env, before paddle is imported.
    from phonics_ocr.utils import set_offline_env, write_json, iso_now

    print("\n[2/4] Setting offline environment switches ...")
    for key, value in set_offline_env().items():
        print(f"      {key}={value}")

    # Step 3: block the network.
    blocking = not args.no_block
    if blocking:
        print("\n[3/4] Installing process-level network block ...")
        install_network_block()
        if self_test_block():
            print("      OK   self-test: outbound connection was refused")
        else:
            print("      FAIL self-test: an outbound connection SUCCEEDED.")
            print("      Offline cannot be verified in this process.")
            return 1
    else:
        print("\n[3/4] Network block SKIPPED (--no-block).")
        print("      This run only proves offline operation if the machine")
        print("      is genuinely disconnected right now.")

    # Step 4: real inference under the block.
    print("\n[4/4] Running inference with the network blocked ...")
    image_path = args.image
    if image_path is None:
        image_path = PROJECT_ROOT / "data" / "samples" / "offline_selftest.png"
        try:
            make_test_image(image_path)
            print(f"      generated test image: {image_path}")
        except Exception as exc:
            print(f"      Could not generate a test image: {exc}")
            print("      Pass one with --image instead.")
            return 1
    elif not image_path.is_file():
        print(f"      ERROR: image not found: {image_path}")
        return 1

    from phonics_ocr.config import OCRConfig
    from phonics_ocr.engine import EngineError, OCREngine

    try:
        config = OCRConfig(lang=args.lang, device=args.device, offline=True)
        engine = OCREngine(config)
        print(f"      engine loaded on '{engine.device}' with no network")
        result = engine.process_image(image_path)
    except (EngineError, NetworkBlockedError) as exc:
        print(f"\nFAILED: {exc}")
        if BLOCKED_ATTEMPTS:
            print("\nBlocked connection attempts:")
            for attempt in BLOCKED_ATTEMPTS:
                print(f"  - {attempt}")
            print("\nSomething still wants the network. Offline is NOT verified.")
        return 1
    except Exception as exc:
        print(f"\nFAILED (unexpected): {type(exc).__name__}: {exc}")
        return 1

    print(f"      inference OK: {result['n_regions']} region(s) in "
          f"{result['timing']['inference_seconds']:.3f}s")
    if result["regions"]:
        print("\n      Recognised text:")
        for region in result["regions"]:
            print(f"        [{region['rec_score']:.3f}] {region['text_raw']}")

    print("\n" + "=" * 70)
    if BLOCKED_ATTEMPTS:
        print("RESULT: inference SUCCEEDED, but connections were attempted:")
        for attempt in sorted(set(BLOCKED_ATTEMPTS)):
            print(f"  - {attempt}")
        print("They were blocked and inference still completed, so the code")
        print("has a network-dependent path that degrades gracefully.")
    else:
        print("RESULT: PASS -- inference completed with ZERO network attempts.")
    print("=" * 70)

    report: Dict[str, Any] = {
        "generated_at": iso_now(),
        "network_block_installed": blocking,
        "block_self_test_passed": blocking,
        "image": str(image_path),
        "lang": args.lang,
        "device_used": engine.device,
        "n_regions": result["n_regions"],
        "inference_seconds": result["timing"]["inference_seconds"],
        "blocked_connection_attempts": sorted(set(BLOCKED_ATTEMPTS)),
        "verdict": "PASS" if not BLOCKED_ATTEMPTS else "PASS_WITH_BLOCKED_ATTEMPTS",
        "scope_note": (
            "The block covers socket.connect, socket.connect_ex and "
            "socket.create_connection in this process only. Connections made "
            "from a C extension with its own socket stack, or from a "
            "subprocess, would not be observed. Disconnect the machine and "
            "re-run for a stronger guarantee."
        ),
    }
    out = PROJECT_ROOT / "data" / "results" / "offline_verification.json"
    write_json(out, report)
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Report the environment this project will actually run in.

Run this first, and run it again whenever something stops working:

    python scripts/check_environment.py

It prints findings and exits non-zero only if something that would stop
OCR from running at all is wrong. A missing GPU is reported, not
treated as a failure -- CPU inference is a supported configuration.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"
INFO = "[INFO]"

MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 13)

#: Minimum Windows NVIDIA driver for each PaddlePaddle CUDA build, from
#: https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html
DRIVER_REQUIREMENTS = {"cu118": 452.39, "cu126": 550.54, "cu129": 550.54}


def _section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def check_python(findings: List[str]) -> Dict[str, Any]:
    _section("Python interpreter")
    bits = struct.calcsize("P") * 8
    version = sys.version_info[:3]
    info = {
        "executable": sys.executable,
        "version": ".".join(map(str, version)),
        "bits": bits,
        "implementation": platform.python_implementation(),
        "in_virtualenv": sys.prefix != sys.base_prefix,
    }
    print(f"{INFO} Executable : {sys.executable}")
    print(f"{INFO} Version    : {info['version']} ({platform.python_implementation()})")

    if bits != 64:
        print(f"{FAIL} Architecture: {bits}-bit. PaddlePaddle requires 64-bit Python.")
        findings.append("Python is not 64-bit")
    else:
        print(f"{OK}   Architecture: 64-bit")

    if not (MIN_PYTHON <= version[:2] <= MAX_PYTHON):
        print(
            f"{FAIL} PaddlePaddle supports Python "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{MAX_PYTHON[0]}.{MAX_PYTHON[1]}."
        )
        findings.append(f"Unsupported Python version {info['version']}")
    else:
        print(f"{OK}   Version supported by PaddlePaddle")

    if info["in_virtualenv"]:
        print(f"{OK}   Running inside a virtual environment")
    else:
        print(f"{WARN} Not in a virtual environment. Activate .venv first.")
    return info


def check_os() -> Dict[str, Any]:
    _section("Operating system")
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }
    print(f"{INFO} {info['system']} {info['release']} (build {info['version']})")
    print(f"{INFO} Machine: {info['machine']}")
    return info


def check_disk() -> Dict[str, Any]:
    _section("Disk space")
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024**3)
    print(f"{INFO} Project drive free space: {free_gb:.1f} GB")
    if free_gb < 3:
        print(f"{WARN} Under 3 GB free. Model downloads and results may fail.")
    else:
        print(f"{OK}   Sufficient for models and results")
    return {"free_gb": round(free_gb, 2), "total_gb": round(usage.total / (1024**3), 2)}


def _parse_driver_version(raw: str) -> float:
    try:
        parts = raw.strip().split(".")
        return float(f"{parts[0]}.{parts[1]}") if len(parts) >= 2 else float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def check_nvidia(findings: List[str]) -> Dict[str, Any]:
    _section("NVIDIA GPU and driver")
    info: Dict[str, Any] = {"nvidia_smi_available": False}

    if shutil.which("nvidia-smi") is None:
        print(f"{INFO} nvidia-smi not found. No NVIDIA GPU usable; CPU mode only.")
        return info

    info["nvidia_smi_available"] = True
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"{WARN} nvidia-smi failed: {exc}")
        return info

    if query.returncode != 0 or not query.stdout.strip():
        print(f"{WARN} nvidia-smi returned no GPU data: {query.stderr.strip()}")
        return info

    gpus = []
    for line in query.stdout.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        gpu = {
            "name": fields[0] if len(fields) > 0 else "unknown",
            "driver_version": fields[1] if len(fields) > 1 else "unknown",
            "memory_total": fields[2] if len(fields) > 2 else "unknown",
            "compute_capability": fields[3] if len(fields) > 3 else "unknown",
        }
        gpus.append(gpu)
        print(f"{INFO} GPU    : {gpu['name']}")
        print(f"{INFO} Driver : {gpu['driver_version']}")
        print(f"{INFO} Memory : {gpu['memory_total']}")
        print(f"{INFO} Compute capability: {gpu['compute_capability']}")
    info["gpus"] = gpus

    driver = _parse_driver_version(gpus[0]["driver_version"])
    info["driver_numeric"] = driver
    compatible = [
        build for build, minimum in DRIVER_REQUIREMENTS.items() if driver >= minimum
    ]
    info["compatible_cuda_builds"] = compatible
    print(f"\n{INFO} PaddlePaddle CUDA builds this driver supports:")
    for build, minimum in DRIVER_REQUIREMENTS.items():
        mark = OK if driver >= minimum else FAIL
        print(f"  {mark} {build}: needs Windows driver >= {minimum}")
    if not compatible:
        print(f"{WARN} No CUDA build is supported by this driver. Use the CPU build.")
        findings.append("NVIDIA driver too old for any PaddlePaddle CUDA build")

    try:
        cap = float(gpus[0]["compute_capability"])
        if cap < 7.5:
            print(
                f"{WARN} Compute capability {cap} is below the 7.5 that "
                "PaddlePaddle's Windows GPU builds require."
            )
    except (ValueError, TypeError):
        pass
    return info


def check_paddle(findings: List[str]) -> Dict[str, Any]:
    _section("PaddlePaddle")
    info: Dict[str, Any] = {"installed": False}
    try:
        import paddle
    except ImportError:
        print(f"{FAIL} paddlepaddle is not installed.")
        print("       Install with scripts/setup_windows.bat, or see README.md")
        findings.append("paddlepaddle not installed")
        return info

    info["installed"] = True
    info["version"] = getattr(paddle, "__version__", "unknown")
    print(f"{OK}   paddle {info['version']}")

    try:
        info["compiled_with_cuda"] = bool(paddle.is_compiled_with_cuda())
    except Exception as exc:
        print(f"{WARN} Could not query CUDA support: {exc}")
        return info

    if not info["compiled_with_cuda"]:
        print(f"{INFO} CPU-only build. GPU inference is unavailable.")
        return info

    print(f"{OK}   Built with CUDA support")
    from phonics_ocr.engine import detect_gpu

    report = detect_gpu()
    info["gpu_probe"] = report
    if report["usable"]:
        print(f"{OK}   GPU allocation probe PASSED -- GPU inference is available")
        if "device_name" in report:
            print(f"{INFO} Paddle sees: {report['device_name']}")
    else:
        print(f"{WARN} GPU allocation probe FAILED: {report['detail']}")
        print("       CPU inference will still work (--device cpu).")
    return info


def check_packages(findings: List[str]) -> Dict[str, Any]:
    _section("Python packages")
    required = {
        "paddleocr": "PaddleOCR pipeline",
        "cv2": "OpenCV (opencv-python) image I/O",
        "numpy": "array support",
        "PIL": "Pillow, for visualisation labels",
    }
    optional = {
        "arabic_reshaper": "Arabic glyph shaping for visualisation labels",
        "bidi": "python-bidi, display ordering for visualisation labels",
        "pytest": "test runner",
    }
    info: Dict[str, Any] = {"required": {}, "optional": {}}

    for module, purpose in required.items():
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "unknown")
            print(f"{OK}   {module:<18} {version:<12} ({purpose})")
            info["required"][module] = version
        except ImportError:
            print(f"{FAIL} {module:<18} MISSING      ({purpose})")
            info["required"][module] = None
            findings.append(f"required package missing: {module}")

    for module, purpose in optional.items():
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "unknown")
            print(f"{OK}   {module:<18} {version:<12} ({purpose})")
            info["optional"][module] = version
        except ImportError:
            print(f"{WARN} {module:<18} missing      ({purpose})")
            info["optional"][module] = None
    return info


def check_models() -> Dict[str, Any]:
    _section("Local models")
    try:
        from phonics_ocr.config import DEFAULT_MODEL_DIR, all_required_models
    except ImportError as exc:
        print(f"{FAIL} Could not import phonics_ocr: {exc}")
        return {"error": str(exc)}

    info: Dict[str, Any] = {"model_dir": str(DEFAULT_MODEL_DIR), "models": {}}
    print(f"{INFO} Model directory: {DEFAULT_MODEL_DIR}")
    for name in all_required_models():
        directory = DEFAULT_MODEL_DIR / name
        present = directory.is_dir() and any(directory.iterdir())
        info["models"][name] = present
        if present:
            size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
            print(f"{OK}   {name}  ({size / 1e6:.1f} MB)")
        else:
            print(f"{WARN} {name}  not downloaded")
    if not all(info["models"].values()):
        print(f"{INFO} Run: python scripts/download_models.py")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write the report as JSON here")
    args = parser.parse_args()

    print("Phonics Prism OCR -- environment check")
    findings: List[str] = []
    report = {
        "python": check_python(findings),
        "os": check_os(),
        "disk": check_disk(),
        "nvidia": check_nvidia(findings),
        "paddle": check_paddle(findings),
        "packages": check_packages(findings),
        "models": check_models(),
    }

    _section("Summary")
    if findings:
        print(f"{FAIL} {len(findings)} blocking problem(s):")
        for item in findings:
            print(f"       - {item}")
    else:
        print(f"{OK}   No blocking problems found.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{INFO} JSON report written to {args.json}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

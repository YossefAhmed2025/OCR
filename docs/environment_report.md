# Environment report

Generated during the initial build of this project.
Regenerate the live view at any time with:

```bat
python scripts\check_environment.py --json data\results\env.json
```

---

## Host

| Item | Value |
|---|---|
| OS | Windows 11 Home, 10.0.26200 (build 26200) |
| Architecture | x64 |
| Project path | `C:\Users\Owner\Desktop\Phonics_Prism_Manufacturing_Design_Spec_Arabic` |
| Free disk at start | ~11 GB on C: (98% full) |

**Disk space was the binding constraint on this build.** The GPU
install came within ~840 MB of filling the drive during extraction, and
needed a `pip cache purge` (recovering 3.3 GB) to complete. Subsequent
installs used `--no-cache-dir`. Anyone reproducing this on a near-full
drive should plan for ~5 GB free, or use the CPU build (~2 GB).

## Python interpreters found

| Version | Path | Chosen |
|---|---|---|
| 3.13 / 3.13t | `...\Python313\python.exe` | No |
| 3.11 | `...\Python311\python.exe` | No |
| **3.10.11** | `...\Python310\python.exe` | **Yes** |

**Selected: Python 3.10.11, 64-bit (MSC v.1929 AMD64).**

Reason: all of 3.10–3.13 are supported by PaddlePaddle, and 3.10 has
the longest track record with the PaddleOCR 3.x / PaddleX dependency
stack. The project virtual environment is `.venv/` at the project root.

## GPU

| Item | Value |
|---|---|
| GPU | **NVIDIA GeForce RTX 3050 Ti Laptop GPU** |
| Driver | **546.30** |
| Driver's max CUDA | 12.3 (as reported by `nvidia-smi`) |
| VRAM | 4095 MB |
| Compute capability | **8.6** (Ampere) |
| SM count | 20 |
| CUDA Toolkit (`nvcc`) | **Not installed** — not required |

### Choosing the PaddlePaddle build

PaddlePaddle publishes Windows GPU wheels for cu118, cu126 and cu129,
each with a minimum NVIDIA driver:

| Build | Min Windows driver | Driver 546.30 satisfies? |
|---|---|---|
| **cu118** | 452.39 | **Yes** |
| cu126 | 550.54.14 | No |
| cu129 | 550.54.14 | No |

Sources:
<https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html>,
<https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip_en.html>

**Decision: `paddlepaddle-gpu==3.3.1`, cu118 build.**

The newer cu126/cu129 builds were rejected on a concrete requirement,
not a guess: the installed driver (546.30) is below their stated 550.54
floor. Compute capability 8.6 clears PaddlePaddle's ≥7.5 requirement.

No separate CUDA Toolkit installation was needed — the wheel pulls the
CUDA 11.8 runtime and cuDNN in as pip packages
(`nvidia-cudnn-cu11==8.9.4.19`, `nvidia-cublas-cu11==11.11.3.6`,
`nvidia-cufft-cu11`, `nvidia-cusparse-cu11`, `nvidia-cusolver-cu11`,
`nvidia-curand-cu11`, `nvidia-cuda-runtime-cu11`,
`nvidia-cuda-nvrtc-cu11`). The absence of `nvcc` on this machine is
therefore not a problem.

Install command used:

```bat
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### GPU verification — PASSED

Verified by executing a real kernel, not just by querying flags. A
build can report CUDA support and see a device while still failing on
the first kernel launch, so the check allocates and computes:

```
paddle 3.3.1
cuda compiled: True
device count: 1
GPU matmul sum: 33554432.0        <- 256x256 matmul of ones x twos; 512 * 65536, correct
device: name='NVIDIA GeForce RTX 3050 Ti Laptop GPU', major=8, minor=6,
        total_memory=4095MB, multi_processor_count=20
```

The same allocate-and-compute probe is what `engine.detect_gpu()` runs,
so `--device gpu` reflects actual capability rather than build flags.

### VRAM note

4 GB is modest. `--lang mixed` holds two recognition models
simultaneously and may exhaust it on large pages. Mitigations in
`docs/troubleshooting.md`.

## Software versions

| Component | Version | Source |
|---|---|---|
| Python | 3.10.11 (64-bit) | python.org |
| pip | upgraded to 26.2.2 | — |
| paddlepaddle-gpu | **3.3.1** (cu118) | PaddlePaddle package index |
| paddleocr | **3.7.0** | PyPI |
| OCR pipeline | PP-OCRv5 | — |

## Decisions and rejected alternatives

| Decision | Reason |
|---|---|
| Python 3.10 over 3.11/3.13 | All supported; 3.10 is the best-tested with the PaddleOCR 3.x stack. |
| cu118 over cu126/cu129 | Driver 546.30 < the 550.54 those builds require. |
| GPU over CPU build | Driver and compute capability both clear cu118's requirements, and a real kernel ran. |
| No CUDA Toolkit install | The wheel bundles the runtime and cuDNN as pip packages. |
| `PP-OCRv5` pinned explicitly via `ocr_version` | Newer PaddleOCR releases may default to PP-OCRv6; pinning keeps results comparable and model names valid. |
| No `.env.example` | No environment variables are genuinely needed. The offline switches (`PADDLE_PDX_OFFLINE_MODE` etc.) are set programmatically by `utils.set_offline_env()` before paddle is imported, which is more reliable than asking the user to export them. |

## Reproducing

```bat
scripts\setup_windows.bat
python scripts\check_environment.py
python scripts\download_models.py
python scripts\verify_offline.py
```

`setup_windows.bat` reads the driver version at run time and selects a
compatible build, so it does not hardcode this machine's answer.

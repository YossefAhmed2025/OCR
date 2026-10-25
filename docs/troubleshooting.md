# Troubleshooting

Start with:

```bat
python scripts\check_environment.py
```

It prints the Python build, driver version, which PaddlePaddle CUDA
builds that driver supports, whether the GPU probe passes, which
packages are installed and which models are present.

---

## Installation

### `ERROR: Could not find a version that satisfies the requirement paddlepaddle-gpu==3.3.1`

The `-i` index flag is missing. PaddlePaddle GPU wheels are **not on
PyPI** — the PyPI copy is stale.

```bat
pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### `No matching distribution found` on a supported-looking Python

Check the interpreter is 64-bit and in range (3.9–3.13):

```bat
python -c "import sys,struct;print(sys.version, struct.calcsize('P')*8)"
```

32-bit Python has no PaddlePaddle wheels at all.

### Disk fills up during install

The GPU build needs roughly 4 GB installed, plus pip caches every wheel
it downloads — about another 2.5 GB.

```bat
pip cache purge
```

To avoid the second copy in the first place:

```bat
pip install --no-cache-dir -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

If space is genuinely short, use the CPU build (~1 GB):

```bat
scripts\setup_windows.bat cpu
```

### Both `paddlepaddle` and `paddlepaddle-gpu` are installed

They both provide the `paddle` module and shadow each other
unpredictably. Remove both and reinstall one:

```bat
pip uninstall -y paddlepaddle paddlepaddle-gpu
pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

---

## GPU

### `GPU was requested with --device gpu but is not usable`

This is intentional — the engine refuses to silently fall back to CPU,
because a benchmark that ran on the wrong device is worse than no
benchmark. The message names the reason. Common ones:

**"This paddlepaddle build is CPU-only"** — the CPU wheel is installed.
Install the GPU one.

**"Paddle has CUDA support but reports 0 visible GPU devices"** — the
driver is not responding. Check `nvidia-smi` runs. On a laptop with
switchable graphics, confirm the NVIDIA GPU is not disabled for this
application in the NVIDIA Control Panel.

**"GPU allocation probe failed: ..."** — a driver/runtime mismatch. The
build compiled against CUDA 11.8 needs Windows driver ≥ 452.39. Check:

```bat
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

To work while sorting it out:

```bat
python scripts\run_ocr.py --image data\input\page.jpg --lang ar --device cpu
```

### Driver too old for cu126/cu129

Minimum Windows drivers, from the
[official install guide](https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html):

| Build | Minimum Windows driver |
|---|---|
| cu118 | 452.39 |
| cu126 | 550.54.14 |
| cu129 | 550.54.14 |

This project uses **cu118** precisely because it has the lowest driver
floor. Do not switch to cu126 without checking the driver first —
`check_environment.py` prints which builds the installed driver
supports.

### Out of memory on a 4 GB GPU

`--lang mixed` loads two recognition models at once. Options:

- Use `--lang ar` or `--lang en` instead.
- Lower `--det-limit-side-len` (e.g. `640`) to shrink the detection
  input.
- Avoid `--upscale` with large pages.
- Fall back to `--device cpu`.

---

## Models

### `Offline mode is on but these models are not present`

Run once with network access:

```bat
python scripts\download_models.py
```

### `these models were not found in the PaddleOCR cache`

The installed `paddleocr` release resolved different model names than
`config.py` expects. Do **not** substitute whatever is in the cache —
check the
[PP-OCRv5 multilingual model list](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
and update `RECOGNITION_MODELS` / `DETECTION_MODEL` in
`src/phonics_ocr/config.py` to the documented names for your version.

Cache locations searched:

- `%USERPROFILE%\.paddlex\official_models\`
- `%USERPROFILE%\.paddleocr\official_models\`

### Download is slow or times out

The model host can be slow from some regions. The download is resumable
by re-running the script — already-copied models are skipped. To start
clean:

```bat
python scripts\download_models.py --force
```

---

## Running OCR

### `Could not decode <path>. Is it a valid image file?`

Supported: png, jpg, jpeg, bmp, tif, tiff, webp. A renamed file keeps
its original format — check the real type. Note that loading uses
`np.fromfile` + `cv2.imdecode` specifically so that **non-ASCII paths
work**; `cv2.imread` returns `None` for those on Windows.

### No regions detected

In order of likelihood:

1. **Resolution too low.** `run_ocr.py` warns below 600px on the short
   side. Re-shoot larger, or try `--upscale 2.0`.
2. **Detection threshold too high.** Try `--det-box-thresh 0.3`.
3. **Page too large.** Detection downscales to `--det-limit-side-len`
   (default 960). For a dense page try `--det-limit-side-len 1600`.
4. **Low contrast.** Try `--clahe`.
5. **Rotated page.** Try `--deskew`, or `--textline-orientation` for
   upside-down lines.

### Text is detected but recognised as nonsense

Usually the wrong language model. Arabic pages need `--lang ar`; the
English model will return plausible-looking Latin garbage from Arabic
script.

If the language is right, the region is probably too small or too
blurred for the recognizer. Check `laplacian_variance` in the result
JSON's `image.inspection`.

### Diacritics are missing from the output

Work through this in order:

1. Check they are actually printed on the page — zoom into the image.
2. Check `hypothesis_tashkeel_count` in the JSON. If it is 0 across
   every sample, the model may not emit tashkeel at all. That is a
   model-capability question, not a tuning one — see
   `docs/arabic_diacritics.md` §8.
3. **Turn off `--denoise`.** It is the most likely cause of
   disappearing marks.
4. Try `--upscale 2.0`. More pixels per mark is the one reliable lever.
5. Re-shoot at higher resolution.

Do not "fix" this by stripping tashkeel from the reference. That hides
the problem and inflates the score.

### Arabic looks backwards in the console or a text editor

Almost always a *display* issue, not a data issue. Text is stored in
logical order, which is correct. Check the JSON in VS Code or a browser
— both apply the bidirectional algorithm properly.

If Arabic is genuinely reversed in the data, that is a bug: nothing in
this project reverses strings, and `visualization.shape_for_display()`
is the only function that produces display-ordered text, used solely
for drawing on PNGs.

### Arabic labels on the visualisation PNG are disconnected or reversed

Install the shaping packages:

```bat
pip install arabic-reshaper python-bidi
```

Without them the overlay draws raw logical-order glyphs. **The JSON is
unaffected** — this is cosmetic.

### Console shows `?????` instead of Arabic

A Windows console code page issue:

```bat
chcp 65001
```

Or just read the JSON/txt output files, which are always UTF-8.

---

## Evaluation

### `no .jsonl files in <dir>`

Create a labelled file — see `data/ground_truth/README.md`.
`example.jsonl` contains **template rows only**; its images do not
exist and it is not usable as real ground truth.

### `record N: image not found`

Paths in the JSONL resolve relative to the JSONL's own directory, then
the project root. Use a path relative to the JSONL file, or an absolute
one.

### CER is much worse than expected on Arabic

Expected, if the reference is vowelled and the model is not producing
diacritics. Every missing mark is a deletion. Confirm with:

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\my.jsonl --strip-tashkeel-view
```

A large gap between the primary and consonants-only figures localises
the problem to diacritics rather than letter recognition.

### A rate shows as `n/a`

The denominator was zero — an empty reference, or no diacritics on
either side. The report prints `n/a` rather than `0.0` so that "nothing
to measure" is never mistaken for "perfect".

---

## Offline verification

### `self-test: an outbound connection SUCCEEDED`

The socket block did not take effect — usually because something
imported and cached a socket reference before the patch. Run
`verify_offline.py` directly rather than importing it from another
script.

### `inference SUCCEEDED, but connections were attempted`

A dependency tried to reach the network, was blocked, and inference
completed anyway. Not a failure, but worth knowing. The report lists
the addresses attempted. Confirm `PADDLE_PDX_OFFLINE_MODE=1` is set
before paddle is imported.

### Verification passes but you want a stronger guarantee

The in-process block cannot see connections made from a C extension
with its own socket stack, or from a subprocess. Disconnect the machine
from the network and re-run:

```bat
python scripts\verify_offline.py --no-block
```

The report records which mode was used.

---

## Tests

### `ModuleNotFoundError: No module named 'phonics_ocr'`

Run pytest from the project root. `tests/conftest.py` puts `src/` on the
path.

```bat
cd <project root>
python -m pytest tests\ -v
```

---

## Known upstream issues (found during this build)

### CPU inference fails with `ConvertPirAttribute2RuntimeAttribute not support`

Full error:

```
NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
not support [pir::ArrayAttribute<pir::DoubleAttribute>]
(at ..\paddle\fluid\framework\new_executor\instruction\onednn\onednn_instruction.cc:118)
```

paddlepaddle-gpu 3.3.1's oneDNN (MKL-DNN) path fails on these PP-OCRv5
models, breaking **every** CPU run.

**Already worked around:** `enable_mkldnn` defaults to `False` in this
project, so CPU inference works out of the box. The cost is slower CPU
inference. If your build is unaffected, re-enable it:

```bat
python scripts\run_ocr.py --image page.jpg --lang ar --device cpu --enable-mkldnn
```

### Arabic diacritics vanish from rendered images

`arabic-reshaper` defaults to `delete_harakat=True` and strips every
diacritic while shaping text for display.

This is the most dangerous bug in the project's problem domain, because
it fails *silently* and makes accuracy numbers look **better**. It
already caused one wrong conclusion here — see
`docs/baseline_findings.md` §2.

**Already worked around:** `visualization.shape_for_display()`
configures `delete_harakat=False`, with a regression test in
`tests/test_visualization.py`.

If you write your own rendering or preprocessing code, check the
diacritic count on both sides:

```python
from phonics_ocr import postprocessing as pp
assert pp.count_tashkeel(after) == pp.count_tashkeel(before)
```

### Warning: "`lang` and `ocr_version` will be ignored"

Expected and harmless if you see it in older logs. PaddleOCR ignores
both when explicit model names are supplied. This project no longer
passes them — the explicit model names pin the exact PP-OCRv5 weights,
which matters because newer PaddleOCR releases default to PP-OCRv6.

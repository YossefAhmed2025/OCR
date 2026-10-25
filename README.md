# Phonics Prism — OCR Baseline

An offline OCR research baseline for reading printed **Arabic** (with
and without tashkeel) and **English**, built on PaddleOCR PP-OCRv5.

مشروع تجريبي للتعرّف الضوئي على النصوص العربية والإنجليزية المطبوعة،
يعمل محليًا بدون إنترنت، مع قياس دقّة قراءة التشكيل بشكل منفصل.

---

## What this is

Phonics Prism will eventually photograph a book page, recognise the
text, locate each word, and use a projector to highlight reading units
for children with dyslexia.

**This repository is the OCR proof of concept only.** It answers: how
well can a locally-run OCR stack read Arabic and English print, where
are the text regions, and — critically — does it read the **diacritics**
that determine pronunciation?

Not implemented here: projector calibration, camera capture, phonics
instruction, audio, word-level highlighting, hardware integration.

### The central concern: diacritics

Arabic `كتب` reads as *kataba* (he wrote), *kutiba* (it was written), or
*kutub* (books) depending on vowel marks that general OCR benchmarks
usually discard. A system that teaches a child to pronounce words
cannot discard them.

So: diacritics are never stripped by default, raw model output is never
modified, scores are always reported with tashkeel preserved, and a
model that outputs no diacritics scores **recall 0** rather than
looking good by omission. See [`docs/arabic_diacritics.md`](docs/arabic_diacritics.md).

---

## Status

| Feature | Status |
|---|---|
| Arabic OCR pipeline (`--lang ar`) | Implemented, runs |
| English OCR pipeline (`--lang en`) | Implemented, runs |
| Mixed-language (`--lang mixed`) | Implemented, **experimental** — see limits below |
| GPU acceleration | Implemented, **verified** on RTX 3050 Ti |
| CPU fallback | Implemented |
| Text-region localisation | Implemented |
| **Word-level** localisation | **Implemented and audited.** Given a correct line, 2-D IoU R@0.5 = 1.000 (synthetic). But only **38.9% of highlights** meet projector coverage - see below |
| Offline inference | Implemented, verified with a socket-level block |
| CER / WER / diacritic metrics | Implemented, unit-tested, **run end to end** |
| Diacritic recognition | **Possible** — 38 tashkeel in the model charset; partial recall measured |
| **Accuracy on real pages** | **Not measured** — no photographed dataset exists yet |
| Detection/localisation metrics | **Implemented** (IoU recall/precision, horizontal + 2-D) |
| Dataset schema v2 + validator | **Implemented** — provenance-aware |
| Preprocessing A/B harness | **Implemented**, first results in `docs/camera_experiments.md` |
| Resource measurement | **Implemented** — VRAM, RAM, latency, storage |
| Camera robustness study | **Plan only** — needs photographs |
| Webcam capture workflow | **Implemented**, tested via replay; live camera path **not run** |
| CTC alignment audit | **Done** — stride verified, mapping bug fixed |
| Manual word annotation tool | **Implemented** (`annotate_words.py`) |
| Projector coverage simulation | **Implemented** — error budget measured |
| Alternative localisation benchmark | **Done** — CTC beats projection profiling 5x |

**No accuracy claim is made for real pages.** The pipeline has been run
end to end and the metrics verified on **four synthetic images** (`n=1`
per group) — enough to prove the system works, nowhere near enough to
benchmark it. The 95% character-accuracy target is **not evaluated**.

On that synthetic smoke test the key result was:

| View | CER | "Accuracy" |
|---|---|---|
| Consonants only (tashkeel stripped) | 0.0435 | **95.7%** |
| **With tashkeel preserved** | 0.1028 | **89.7%** |

Same model, same run, same images: strip the diacritics and the target
is met; keep them and it is not. Full results, caveats and a correction
of an early wrong conclusion are in
[`docs/baseline_findings.md`](docs/baseline_findings.md).

---

## Requirements

- Windows 10/11, 64-bit
- Python 3.9–3.13, 64-bit (3.10 used here)
- ~5 GB disk for the GPU install, ~2 GB for CPU
- Optional: NVIDIA GPU, compute capability ≥ 7.5, driver ≥ 452.39

---

## Setup

```bat
cd <project root>
scripts\setup_windows.bat
```

The script finds a supported Python, creates `.venv`, reads your NVIDIA
driver version, and installs a PaddlePaddle build that driver actually
supports. Force a build with `scripts\setup_windows.bat cpu` or `gpu`.

### Manual setup

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

REM GPU (CUDA 11.8) -- the -i flag is REQUIRED
pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

REM ...or CPU
pip install -r requirements-cpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

pip install -r requirements.txt
```

> PaddlePaddle wheels are **not on PyPI** — the PyPI copy is stale. The
> `-i` index URL is mandatory.
> Source: [PaddlePaddle Windows install guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip_en.html)

### Why CUDA 11.8 and not 12.6

Minimum Windows drivers per build
([source](https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html)):

| Build | Min Windows driver |
|---|---|
| cu118 | **452.39** |
| cu126 | 550.54.14 |
| cu129 | 550.54.14 |

This machine runs driver **546.30** — above cu118's floor, below
cu126's. `check_environment.py` prints your driver and which builds it
supports, so verify before changing this.

No separate CUDA Toolkit install is needed: the wheel pulls the CUDA
runtime and cuDNN in as pip packages.

### Verify

```bat
python scripts\check_environment.py
```

---

## Download models (once, needs internet)

```bat
python scripts\download_models.py
```

| Role | Model |
|---|---|
| Detection | `PP-OCRv5_mobile_det` |
| Arabic recognition | `arabic_PP-OCRv5_mobile_rec` |
| English recognition | `en_PP-OCRv5_mobile_rec` |

Files land in `models/<name>/`, with `models/manifest.json` recording
every file, size and SHA-256.

```bat
python scripts\download_models.py --verify    REM check without downloading
python scripts\download_models.py --lang ar   REM Arabic only
python scripts\download_models.py --force     REM re-download
```

---

## Camera testing (real pages)

```bat
python scripts\camera_test.py --list          REM which cameras work
python scripts\camera_test.py --lang ar       REM live preview
python scripts\camera_test.py --lang ar --words --no-save
python scripts\camera_test.py --lang ar --replay data\input   REM no camera
```

`SPACE` captures and runs OCR, `s` toggles saving, `n` switches
camera, `q` stops. Each capture writes the frame, JSON, transcription
and an overlay to `data/results/camera_*/`; `--no-save` writes nothing
at all.

Solid boxes on the overlay are detector text lines. **Dashed boxes are
experimental word spans derived from CTC timesteps — not
detector-produced word boxes**, and not validated on photographs.

There is **no page detection and no perspective correction**: hold the
camera roughly parallel to the page. See
[`docs/camera_testing.md`](docs/camera_testing.md) for permissions and
troubleshooting, including the parts that have not been run on real
hardware.

## Running OCR

No sample page images are bundled (printed pages are usually
copyrighted). To generate synthetic ones for a quick smoke test:

```bat
python scripts\make_synthetic_samples.py
```

These are rendered font glyphs, **not photographs** — fine for checking
the pipeline runs, useless as a benchmark.

Put your own images in `data/input/`, then:

```bat
python scripts\run_ocr.py --image data\input\test_arabic.jpg --lang ar
python scripts\run_ocr.py --image data\input\test_english.jpg --lang en
python scripts\run_ocr.py --image data\input\test_mixed.jpg --lang mixed
python scripts\run_ocr.py --input-dir data\input --lang ar
```

### Device

```bat
python scripts\run_ocr.py --image page.jpg --lang ar --device gpu
python scripts\run_ocr.py --image page.jpg --lang ar --device cpu
```

`--device gpu` **fails loudly** if the GPU is unusable. It never
silently falls back — a benchmark that ran on the wrong device is worse
than no benchmark. `--device auto` (default) picks GPU if available and
reports which it used.

### Preprocessing — all OFF by default

| Flag | Effect | Risk to tashkeel |
|---|---|---|
| `--grayscale` | Luminance only | Low |
| `--clahe` | Adaptive contrast | Low–medium |
| `--denoise` | Non-local means | **High — erases diacritics** |
| `--deskew` | Rotation correction (≤15°) | Low |
| `--upscale N` | Resize before detection | **Measured harmful on clean images** (−0.20 diacritic recall at 2x); untested on degraded ones |

Nothing is applied automatically. Arabic diacritics are a few pixels
wide, and denoising routinely removes them while making consonants look
cleaner. Each run records which operations ran, so runs are comparable.
Original images are never modified.

### Tuning

```bat
--det-limit-side-len 1600   REM larger detection input for dense pages
--det-box-thresh 0.3        REM lower = more regions detected
--det-unclip-ratio 1.5      REM box padding
--rec-score-thresh 0.5      REM drop low-confidence regions
--textline-orientation      REM handle 180-degree-rotated lines
```

---

## Mixed-language: what it does and does not do

**Experimental.** PaddleOCR has no single model that reads Arabic and
English equally well in the same line.

`--lang mixed` runs both pipelines over the image, matches regions by
IoU, and picks the pass whose output script matches its own model.

> **Limit:** if one *text line* contains both scripts, only one model
> recognises that line, and the other script in it will be garbled.
> Detection returns one polygon per line; there is no per-word script
> routing.

Pages where Arabic and English sit on **separate lines or blocks** work
well. Mixing within a line does not. Mixed mode also costs ~2× the time
and holds two models in memory — watch this on a 4 GB GPU. Details in
[`docs/architecture.md`](docs/architecture.md).

---

## Results

Each run writes to `data/results/ocr_<lang>_<timestamp>/`:

| File | Contents |
|---|---|
| `<image>.json` | Full structured result |
| `<image>.txt` | Plain-text transcription in reading order |
| `<image>_viz.png` | Overlay of regions and labels |
| `run_metadata.json` | Config, device, timings, failures |

Timestamped directories mean no run overwrites another.

### Geometry — read this before using the coordinates

Per region the JSON stores `text_raw`, `text_normalised`, `rec_score`,
`polygon`, `bbox`, `source_pass`, `recognition_model`,
`dominant_script`, `has_tashkeel`, `tashkeel_count`.

- `text_raw` is the model's output, **unmodified**. `text_normalised`
  is a separate conservative view (NFC, tatweel removed, whitespace
  collapsed, **tashkeel preserved**).
- Coordinates are top-left origin, pixels, in the **original** image's
  space — `--upscale` is divided back out automatically.
- **`geometry_granularity` is `"text_region"`, not word-level.**
  PP-OCRv5 detection returns line/region polygons. Word-level
  localisation is **not implemented**, and these are not labelled as
  word boxes.
- These are **image** coordinates, not projector coordinates.
  Camera-to-page and page-to-projector calibration is a separate,
  unstarted stage.
- `rec_score` is PaddleOCR's raw score. It is **not** a calibrated
  probability of correctness and should not be compared across models.

The visualisation is a debugging aid; its labels are truncated and
display-reshaped. **The JSON is authoritative.**

---

## Word-level localisation (experimental)

```bat
python scripts
un_ocr.py --image page.jpg --lang ar --words
```

Adds a `words` array to the JSON, with text, polygon, bbox, confidence
and `flags`. These are **derived** geometry labelled
`word_ctc_aligned`, never presented as detection output. Not supported
with `--lang mixed` (the CLI rejects it rather than guessing).

Measure it against annotated boxes:

```bat
python scripts\make_localization_fixtures.py --rotate 4
python scripts\make_boundary_fixtures.py
python scripts\evaluate_localization.py --device gpu
python scripts\evaluate_localization.py --device gpu --oracle-lines
```

`--oracle-lines` bypasses detection and takes the line extent from the
reference boxes. Run both: the difference separates **detection**
failures from **alignment** failures, which are otherwise impossible to
tell apart. On the boundary set that difference is 0.054 vs 0.946
recall — the method looked broken when the input was.

Annotate your own images:

```bat
python scripts\annotate_words.py --image data\input\page.jpg --lang ar
```

Check what box error means for the projector:

```bat
python scripts\simulate_projector.py
```

Details, audit and limitations:
[`docs/word_localization.md`](docs/word_localization.md),
[`docs/projector_calibration.md`](docs/projector_calibration.md).

## Measuring preprocessing and resources

```bat
python scripts\compare_preprocessing.py --ground-truth data\ground_truth\my.jsonl --variants original clahe upscale2 denoise
python scripts\measure_resources.py --runs 5
```

The preprocessing harness prints CER **and** diacritic recall side by
side, because a variant that improves CER while lowering diacritic
recall is deleting tashkeel rather than reading better.

## Evaluation

### 1. Label your images

JSONL, one object per line — see
[`data/ground_truth/README.md`](data/ground_truth/README.md).

```json
{"id": "p1", "image": "../input/p1.jpg", "lang": "ar", "group": "arabic_tashkeel", "text": "كَتَبَ الْوَلَدُ"}
```

`data/ground_truth/example.jsonl` holds **template rows only** — the
images do not exist. It shows the schema; it is not data.

### 2. Run

Validate the labels first — this catches boxes outside the image,
word lists in the wrong order, and missing provenance:

```bat
python scriptsalidate_dataset.py --ground-truth data\ground_truth\my_labels.jsonl
```

Then evaluate:

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\my_labels.jsonl
python scripts\evaluate.py --ground-truth data\ground_truth\my_labels.jsonl --strip-tashkeel-view
```

### 3. What you get

CER, WER, exact-match rate, diacritic precision/recall/F1, per-class
confusable counts and mean inference time — **broken out by group**
(`arabic_tashkeel`, `arabic_plain`, `english`, `mixed`, `clean`,
`degraded`), never pooled into one headline number.

Every rate is reported with its denominator. A zero denominator prints
`n/a`, not `0.0`, so "nothing to measure" is never mistaken for
"perfect". Averaging is **micro** — distances and lengths summed before
dividing.

`--strip-tashkeel-view` adds a consonants-only score as an explicitly
labelled **secondary view**. It never replaces the vowelled figure.

Full protocol: [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

---

## Offline operation

Three phases, deliberately distinct:

| Phase | Network |
|---|---|
| Install packages | Required |
| Download models | Required, once |
| **Inference and evaluation** | **None** |

```bat
python scripts\verify_offline.py
```

This does not infer offline-ness from reading the code. It patches
`socket.connect`, `connect_ex` and `create_connection` to raise on any
non-loopback address, self-tests that the block works, then runs **real
inference** through it and reports any connection attempted.

Its limit is stated in the report: the block covers this Python process
only. For a stronger guarantee, disconnect the machine and run
`python scripts\verify_offline.py --no-block`.

No cloud OCR, no telemetry, no image upload — anywhere in this project.

---

## Tests

```bat
python -m pytest tests\ -v
```

Covers geometry conversion and bounds, coordinate round-trips, CER/WER
counting, diacritic P/R/F1, **diacritic preservation**, Unicode
normalisation behaviour, config validation, JSON/JSONL serialisation
and reading order.

These validate **code**, not OCR accuracy. All fixtures are
hand-written; none is model output.

---

## Known limitations

1. **No accuracy measurement on real pages.** No ground-truth dataset
   exists here. The 95% target is unevaluated.
2. **Word boxes are not yet good enough for a projector.** Audited in
   detail. Given a correct line, 2-D IoU recall is 1.000 @0.5 on
   rendered text — but IoU is the wrong bar. Converted into projector
   coverage, only **38.9%** of highlights cover >=90% of their word
   with <=5% spill, *before any calibration error*. A 1.25x highlight
   margin raises that to 77.8%. See `docs/projector_calibration.md`.
3. **Detection, not alignment, is the current bottleneck.** On the
   boundary stress set, detection fragmented short lines (it detected
   hamza marks as separate text regions) and word recall collapsed to
   0.054; with oracle lines the same alignment scored 0.946.
4. **Mixed script within one line**: measurements now suggest
   `--lang ar` handles mixed pages better than `--lang mixed`, because
   the Arabic charset covers Latin and detection already splits lines
   by script. Evidence is small (n=2-3, synthetic).
   See `docs/mixed_language.md`.
5. **Diacritic recognition is unverified.** Whether
   `arabic_PP-OCRv5_mobile_rec` emits tashkeel at all is the first
   question to answer with real data — `docs/arabic_diacritics.md` §8.
6. **Diacritic scoring is position-based**, so one missing consonant
   shifts indices and penalises later marks. Reported alongside CER,
   never alone.
7. **No localisation metrics on photographs** — the annotated sets
   are all synthetic renders.
8. **No perspective correction.** `four_point_transform()` exists as a
   hook but automatic corner detection is not implemented.
9. **4 GB VRAM is tight** for mixed mode.
10. **Confidence scores are not calibrated** and must not be read as
   probabilities of correctness.

---

## Next steps

1. Photograph and label 50–100 real pages per condition.
2. Answer the charset question: run one vowelled page, check
   `hypothesis_tashkeel_count`.
3. Measure the `arabic_tashkeel` vs `arabic_plain` gap on identical
   sentences.
4. Sweep `--upscale` against diacritic recall to set a camera
   resolution floor.
5. Decide on word-level localisation (options in
   `docs/architecture.md`).
6. Only then judge the 95% target.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/environment_report.md`](docs/environment_report.md) | The verified environment, versions, GPU findings |
| [`docs/architecture.md`](docs/architecture.md) | Pipeline, models, mixed-language design, geometry conventions |
| [`docs/arabic_diacritics.md`](docs/arabic_diacritics.md) | Why diacritics are the hard part, and how they are handled |
| [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) | Metric definitions, normalisation, reporting rules |
| [`docs/camera_testing.md`](docs/camera_testing.md) | **Webcam workflow, permissions, troubleshooting** |
| [`docs/dataset_preparation.md`](docs/dataset_preparation.md) | **Start here to capture and label real pages** |
| [`docs/word_localization.md`](docs/word_localization.md) | Word boxes from CTC alignment: method, scores, limits |
| [`docs/camera_experiments.md`](docs/camera_experiments.md) | Robustness experiment plan; preprocessing results |
| [`docs/mixed_language.md`](docs/mixed_language.md) | Arabic + English on one page |
| [`docs/projector_calibration.md`](docs/projector_calibration.md) | Camera→page→projector gap and the error budget |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Common errors and fixes |

## Official references

- PaddleOCR — <https://www.paddleocr.ai/>
- PaddleOCR GitHub — <https://github.com/PaddlePaddle/PaddleOCR>
- PaddlePaddle install — <https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip_en.html>
- PP-OCRv5 multilingual models — <https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html>

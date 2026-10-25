# Architecture

> **Written for:** engineers working on or integrating with this
> repository.

## Scope

This is an **OCR proof of concept**. It answers one question: how well
can an off-the-shelf, locally-run OCR stack read printed Arabic (with
and without tashkeel) and English, and where are the text regions on
the page?

Not in scope, and not implemented: projector calibration, camera
capture, phonics instruction, audio, word-level segmentation for
highlighting, any hardware integration.

---

## Pipeline

```
image file
    |
    v
[ load_image ]            OpenCV via np.fromfile/imdecode
    |                     (imread fails on non-ASCII Windows paths)
    v
[ inspect_image ]         dimensions, blur proxy, advisory warnings
    |                     -> recorded in JSON, acts on nothing
    v
[ apply_preprocessing ]   ALL OFF by default; original never mutated
    |                     -> reports which ops ran + scale factor
    v
[ PaddleOCR.predict ]     PP-OCRv5 detection + per-language recognition
    |                     one pipeline per language pass
    v
[ _merge_passes ]         mixed mode only: IoU-match regions, pick a
    |                     winner per region
    v
[ geometry ]              scale coordinates back to ORIGINAL image
    |                     space, clip to image bounds
    v
[ postprocessing ]        script detection, page reading order,
    |                     conservative normalised text view
    v
JSON + visualisation PNG + plain-text transcription
```

Stages live in `src/phonics_ocr/`; `scripts/` holds thin CLI wrappers
that parse arguments and print. Nothing in `scripts/` makes decisions
about OCR behaviour.

---

## Models

| Role | Model | Notes |
|---|---|---|
| Detection | `PP-OCRv5_mobile_det` | Script-agnostic. Shared by all languages. |
| Arabic recognition | `arabic_PP-OCRv5_mobile_rec` | Arabic **script**, not Arabic language only. |
| English recognition | `en_PP-OCRv5_mobile_rec` | English-tuned. |

Chosen explicitly in `config.py`, passed explicitly to PaddleOCR via
`text_detection_model_name` / `text_recognition_model_name`, and
recorded in every result file. There is no default-model path: if a
requested model is missing in offline mode, the engine raises.

The Arabic model's charset also covers Persian, Urdu, Uyghur, Pashto,
Kurdish, Sindhi and Balochi. A language-specific model is not the same
thing as a language-exclusive one, and this matters for mixed pages —
the Arabic recognizer will happily emit Latin characters.

### Why a separate English model

The Arabic model's charset includes Latin digits and some Latin
letters, so it *will* return something for English text. That is not a
reason to use it for English: it was trained predominantly on
Arabic-script line images, and its character-level priors are wrong for
English words. Measuring that difference is one of the things this
baseline is for.

---

## Device selection

`resolve_device()` in `engine.py`:

| Request | Behaviour |
|---|---|
| `cpu` | CPU. Always available. |
| `gpu` | GPU, or **raise**. Never silently falls back. |
| `auto` | GPU if the probe passes, else CPU. Reports which. |

The GPU probe does more than `paddle.is_compiled_with_cuda()`. A build
can be CUDA-compiled and see a device and still fail on the first
kernel launch, because of a driver/runtime mismatch. So `detect_gpu()`
actually allocates a tensor and runs an op on it. Only that counts as
"usable".

`--device gpu` raising rather than falling back is deliberate: a
benchmark that silently ran on the wrong device is worse than no
benchmark.

---

## Mixed-language handling (experimental)

**Status: implemented, experimental, with a known structural limit.**

PaddleOCR recognition models are per-script. There is no single model
in PP-OCRv5 that reads an Arabic word and an English word in the same
text line equally well.

### What `--lang mixed` does

1. Runs the full Arabic pipeline over the image.
2. Runs the full English pipeline over the same image.
3. Matches regions between the two runs by bounding-box IoU (≥ 0.5).
   Both runs share the same detection model and parameters, so their
   polygons are near-identical.
4. For each matched region, picks a winner with
   `choose_mixed_candidate()`.
5. Keeps regions that only one pass found.

### How the winner is chosen

Prefer the pass whose **output script matches the script its model is
built for** — Arabic output from the Arabic model, Latin output from
the English model. Fall back to raw confidence when neither matches.

This is a documented heuristic, not a probability judgement. Confidence
scores from two different recognition models are **not calibrated
against each other**: 0.95 from the English model and 0.95 from the
Arabic model do not mean the same thing. Comparing them directly would
be meaningless, which is why script agreement is the primary signal.

### The limit that does not go away

> If a **single text line** contains both Arabic and English, only one
> model recognises that line, and whichever model loses will have
> mangled part of it.

Detection returns one polygon for the whole line. There is no
per-word script routing. A line reading `الكتاب book` is recognised
either by the Arabic model (which may garble `book`) or the English
model (which may garble `الكتاب`) — the merge picks one, it does not
combine them.

Pages where Arabic and English occupy **separate lines or blocks** work
well, because each line goes to the right model. Pages that mix scripts
*within* a line do not.

Fixing it properly needs per-word script identification before
recognition: detect, split each line region into word-level crops,
classify the script of each crop, route each to the right recognizer.
That depends on reliable word segmentation, which this project does not
have (see below), so it has not been attempted.

### Cost

Two full passes: roughly 2× the inference time of a single-language
run, and both models resident in memory. On a 4 GB GPU this is worth
watching.

---

## Geometry and the word-level question

### What is returned

PP-OCRv5 detection produces **text-line / text-region** polygons. Every
record is labelled `"geometry_granularity": "text_region"`, and the
result JSON carries an explicit note saying these are not word boxes.

This matters because the eventual projector needs to highlight
**individual words**. Line boxes cannot do that.

**Word-level localisation is not implemented and not verified.** It is
named as a gap rather than approximated, because a plausible-looking
approximation is worse than an honest absence: downstream code would
treat guessed boxes as real and the error would surface as a projector
highlighting the wrong word at a child.

### Approaches available later, none implemented

| Approach | Note |
|---|---|
| Lower `--det-unclip-ratio` | Tightens boxes; may split lines into fragments, but the fragments are not reliably words. |
| Whitespace-gap analysis on the line crop | Plausible for English. Arabic's cursive joining means intra-word letter gaps and inter-word gaps are not cleanly separable by projection profile alone. |
| A word-level detection model | The real fix. Needs a model trained for it. |
| Recognition-time character alignment | Some CTC decoders expose per-character column alignment, which can be aggregated into word spans. Most promising, needs access to the raw decoder output rather than the pipeline API. |

Whichever is chosen, it needs its own annotated bounding-box dataset
and its own IoU-based evaluation before anything depends on it.

### Coordinate conventions

- Origin top-left, x rightward, y downward, pixels.
- **Always in the original input image's coordinate space.** If
  `--upscale` was used, coordinates are divided back down before being
  written, so consumers never need to know what preprocessing ran.
- Polygons keep the detector's own vertex order. They are not
  re-sorted.
- `bbox` is `[x_min, y_min, x_max, y_max]`, clipped to image bounds.
- Both the polygon and the derived AABB are stored; the polygon is the
  original, the AABB is derived.

These are **image coordinates**. They are not projector coordinates.
The JSON says so in a `coordinate_space.note` field. Camera-to-page and
page-to-projector calibration is a separate, unstarted stage, and
anything that treats these numbers as projector-ready will be wrong.

---

## Reading order

`sort_regions_for_reading()` buckets regions into rows by y (20px
tolerance by default), then orders within a row — right-to-left when
the page is predominantly Arabic, left-to-right otherwise.

This orders **regions on the page**. It never touches the characters
inside a string.

Arabic strings are stored in **logical order**, exactly as the model
emitted them. They are not reversed. Reversal is a common and
destructive bug: Arabic in a JSON file should be in logical order, and
the terminal, browser or renderer applies the Unicode bidirectional
algorithm at display time. A string that *looks* right in a non-bidi
viewer has usually been corrupted.

The one exception is `visualization.shape_for_display()`, which
produces a display-ordered, presentation-form string **purely to draw
pixels** on the overlay PNG. Its output is never stored or evaluated,
and the docstring says so.

---

## Preprocessing philosophy

Everything is off by default.

The reason is specific to this project: Arabic diacritics are a few
pixels. Denoising, binarisation and aggressive contrast all tend to
remove them while making consonants look cleaner — producing an image
that scores *better* on unvowelled text and *worse* on the thing we
care about.

| Operation | Flag | Risk to tashkeel |
|---|---|---|
| Grayscale | `--grayscale` | Low |
| CLAHE | `--clahe` | Low–medium (high clip limit amplifies grain into mark-sized specks) |
| Denoise | `--denoise` | **High — the most likely to erase marks** |
| Deskew | `--deskew` | Low; refuses estimates above 15°, which are almost always wrong |
| Upscale | `--upscale N` | **Measured harmful on clean images**: −0.20 diacritic recall at 2x. The "more pixels per mark" argument is untested on genuinely low-resolution input. See `camera_experiments.md` |

Each run records exactly which operations ran, so two runs can be
compared. `four_point_transform()` exists as a perspective-correction
hook but is deliberately **not** wired into the automatic path:
automatic page-corner detection is not implemented, and warping by
wrongly-guessed corners is far more destructive than leaving the page
alone.

---

## Offline operation

Three distinct phases, which the README keeps separate:

| Phase | Network |
|---|---|
| Install packages | Required |
| Download models | Required, once |
| Inference / evaluation | **None** |

`verify_offline.py` does not assert offline-ness from code inspection.
It monkey-patches `socket.socket.connect`, `connect_ex` and
`socket.create_connection` to raise on any non-loopback address,
self-tests that the block works, then runs real inference through it.

Its limits are stated in the script and in its report: the block covers
this Python process only, so a C extension with its own socket stack or
a subprocess would not be caught. Physically disconnecting and
re-running is the stronger check, and the script works either way.

---

## Result schema

`schema_version: "1.0"`. Top-level keys:

| Key | Contents |
|---|---|
| `image` | Path, dimensions, inspection report |
| `coordinate_space` | Origin, axes, units, and the projector caveat |
| `config` | Full configuration, including every model name |
| `device_used` | The resolved device, e.g. `gpu:0` |
| `preprocessing` | Which operations ran, scale factor, deskew angle |
| `timing` | Inference seconds, passes run |
| `geometry_granularity` | `text_region`, plus an explanatory note |
| `reading_order` | RTL flag and region indices in reading order |
| `regions[]` | Per region: `text_raw`, `text_normalised`, `rec_score`, `polygon`, `bbox`, `source_pass`, `recognition_model`, `dominant_script`, `has_tashkeel`, `tashkeel_count`, and `mixed_candidates` in mixed mode |
| `transcription_raw` | Regions joined in reading order |

`text_raw` is the model's output, unmodified. `text_normalised` is a
separate, conservatively-normalised view. Never overwrite the former
with the latter.

### Confidence scores

`rec_score` is reported exactly as PaddleOCR returns it. It is **not** a
calibrated probability of correctness, and this project does not treat
it as one. Use it for ranking within a single model's output; do not
compare it across models, and do not convert it into an accuracy claim.

---

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Explicit model/device/preprocessing configuration + validation |
| `engine.py` | PaddleOCR wrapper, device resolution, pass merging, record building |
| `preprocessing.py` | Image loading and optional, individually-switchable transforms |
| `postprocessing.py` | Arabic normalisation, script detection, reading order, candidate selection |
| `geometry.py` | Polygon/bbox conversion, clipping, IoU, scaling |
| `evaluation.py` | CER, WER, diacritic P/R/F1, confusable counts, aggregation |
| `visualization.py` | Overlay rendering (display only) |
| `utils.py` | Logging, JSON/JSONL I/O, run IDs, checksums, offline env |

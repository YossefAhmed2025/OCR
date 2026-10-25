# Fine-tuning the Arabic recogniser on vowelled text

> **Status:** training data generated and validated. **Training has
> not been run.** Nothing here reports a trained-model result.

---

## 1. Why fine-tune

Measured on this project's own tests, `arabic_PP-OCRv5_mobile_rec`
emits tashkeel but places them wrongly:

| Input | Diacritic recall |
|---|---|
| Clean synthetic render | 0.58 |
| Photo of a phone screen | 0.22 |
| **Photo of printed paper** | **0.00** |

Even with hand-cropped perfect line images from a printed page, the
character accuracy was **53%** against a vowelled reference. The spec
target (p.17) is **≥95% characters on vowelled text at 14–18pt**.

The model's charset already contains all 38 diacritics, so this is a
weights problem, not a capability problem — which is what makes
fine-tuning the right lever rather than swapping models.

**What fine-tuning will not fix:** detection. On a photographed page
with large print, detection returned garbage regardless of settings.
That is a separate problem — see §7.

---

## 2. Generate the data

```bat
python scripts\make_training_data.py --n 25000
```

Produces in `data/train_rec/`:

| File | Contents |
|---|---|
| `images/` | 25,000 line crops, height 48, JPEG |
| `train_list.txt` | 23,000 rows: `images/x.jpg<TAB>LABEL` |
| `val_list.txt` | 2,000 rows |
| `ppocrv5_arabic_dict.txt` | 717 characters |
| `dataset_info.json` | Provenance, fonts, counts |
| `samples_preview.json` | First 40 lines, logical + visual |

Current generated set: **26 fonts**, **9.57 diacritics per line**, all
8 classical harakat present, **0 render failures**.

---

## 3. Three conventions that fail silently

All three were hit during development. Each now has a test.

### 3.1 Labels are VISUAL order

The raw CTC sequence runs along the image's x-axis, left to right. For
Arabic that is the reverse of logical order. Verified directly against
the pretrained model:

| Source | Raw CTC output | `TextRecognition` output |
|---|---|---|
| `زَهْرَة` | `ةَرْهَز` | `زَهْرَة` |
| `قَلَم` | `ملق` | `قلم` |

The raw sequence is visual; PaddleOCR's decoder reverses it at
inference. **Training targets must match the raw sequence**, so labels
are stored visually. `synth.visual_label()` does this with
`bidi.get_display`.

### 3.2 Labels use BASE characters, not presentation forms

This one cost a whole dataset. Rendering needs `arabic_reshaper`,
which rewrites letters into presentation forms (U+FE70–U+FEFF). Those
are *display glyphs*. The recogniser's charset is base Arabic.

Labelling with reshaped text added **81 phantom classes** to the
dictionary (747 → 828) — classes the model would be trained to predict
and never scored on.

| | String |
|---|---|
| Render with (reshaped) | `ﺓَﺮْﻫَﺯ` |
| **Label with (base, reversed)** | **`ةَرْهَز`** |

After the fix the dictionary is **717 characters, a strict subset of
the pretrained 747** — zero new classes, which is what a warm start
should look like.

### 3.3 The dict filename must contain `arabic`

`ppocrv5_arabic_dict.txt`. That substring is what switches PaddleOCR's
label decoder into its reverse path. Rename the file and decoding
silently comes out backwards.

### Also: shaping must not delete harakat

`arabic-reshaper` strips every diacritic by default
(`delete_harakat=True`). It already produced one wrong conclusion in
this project. `visualization.shape_for_display` sets it `False`, and
every rendered line is checked to preserve its source mark count —
generation aborts on mismatch.

---

## 4. Font validation — base coverage is not enough

The first generated set contained rows of **tofu boxes** (□□□□) paired
with correct labels. The font check had tested base code points, but
fonts can declare the Arabic block while covering only part of the
presentation-form range that actually gets drawn.

`synth.font_covers_shaped()` now shapes six probe strings (covering
shadda, tanween, taa marbuta, hamza-on-ya, sukun) and requires every
resulting code point. That cut the usable fonts from **42 to 26** —
the 16 removed would have poisoned the data.

---

## 5. Training

PaddleOCR's training entry point lives in the PaddleOCR **repository**,
not the pip package. Clone it separately:

```bat
git clone https://github.com/PaddlePaddle/PaddleOCR
cd PaddleOCR
pip install -r requirements.txt
```

Start from a config for the PP-OCRv5 mobile recogniser and point it at
this dataset. The fields that matter:

```yaml
Global:
  character_dict_path: <ABS>/OCR/data/train_rec/ppocrv5_arabic_dict.txt
  max_text_length: 40
  use_space_char: true
  pretrained_model: <path to arabic_PP-OCRv5_mobile_rec weights>
  save_model_dir: ./output/ar_vowelled
  eval_batch_step: [0, 1000]

Train:
  dataset:
    name: SimpleDataSet
    data_dir: <ABS>/OCR/data/train_rec/
    label_file_list: [<ABS>/OCR/data/train_rec/train_list.txt]
  loader:
    batch_size_per_card: 64     # fits 4 GB VRAM at height 48

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: <ABS>/OCR/data/train_rec/
    label_file_list: [<ABS>/OCR/data/train_rec/val_list.txt]
```

```bat
python tools/train.py -c <your_config>.yml
```

Export when done:

```bat
python tools/export_model.py -c <your_config>.yml ^
  -o Global.pretrained_model=./output/ar_vowelled/best_accuracy ^
     Global.save_inference_dir=<ABS>/OCR/models/arabic_vowelled_rec
```

Then point this project at it — `config.py` already takes an explicit
recognition model directory, so no code change is needed beyond the
model name.

### Expectations, stated honestly

A community report fine-tuning Arabic PP-OCRv3 with 50k synthetic
samples plateaued around **81%** after 300 epochs. 25k samples will
not beat that by itself. Treat the first run as a baseline to measure,
not a solution, and expect to iterate on data volume and realism
before accuracy moves much.

---

## 6. Evaluate — do not trust the training metric

PaddleOCR's own accuracy figure is exact-string-match on the
validation split, which is synthetic and from the same generator. It
tells you the model fitted the data, not that it reads pages.

Score against real labelled photographs instead:

```bat
python scripts\validate_dataset.py --ground-truth data\ground_truth\camera.jsonl
python scripts\evaluate.py --ground-truth data\ground_truth\camera.jsonl --strip-tashkeel-view
```

Watch **diacritic recall**, not CER alone. A model that improves CER
while losing diacritics has learned to skip them — the exact failure
this project exists to avoid.

---

## 7. What this does not address

1. **Detection.** On a photographed page with large print, detection
   produced garbage under every setting tried. Recognition on
   hand-cropped lines still worked. Fine-tuning the recogniser cannot
   fix the detector.
2. **The real product geometry.** Both failing tests were handheld —
   a phone screen and a page held in the air. The spec puts the page
   under a **fixed 165×230 mm window** at a known distance, so text
   size in frame is controlled. That likely removes much of the
   detection failure, and it has not been tested.
3. **Camera.** The spec requires **8–12 MP autofocus**; the webcam
   used is 0.3 MP fixed-focus, and the phone image received was 1.4 MP
   downscaled with sharpness 65. No model compensates for that.
4. **Synthetic-to-real gap.** Augmentation approximates camera
   degradation; it is not photographed paper. The generated data is a
   starting point, and real labelled lines should be mixed in as soon
   as any exist.

---

## 8. If fine-tuning plateaus

| Option | Size | Reported tashkeel accuracy | Fits 4 GB SoM |
|---|---|---|---|
| Fine-tuned PP-OCRv5 | ~8 MB | unknown until measured | yes |
| Falcon-OCR-Arabic | 270 M | 81.9% | needs checking |
| QARI-OCR v0.2 | 2 B | CER 0.061 (≈93.9%) | **no** |

Note that **even QARI, the strongest published result, sits below the
spec's 95% target** — and it is a 2B-parameter model that cannot run
on a 4 GB SoM within a 500 ms latency budget. That gap is worth
raising with whoever owns the specification before more engineering
effort is spent chasing the number.

QARI remains useful off-device as a **teacher**: run it over real page
photographs to produce labels, then train the small model on those.

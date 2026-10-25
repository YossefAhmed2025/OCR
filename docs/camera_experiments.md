# Camera robustness experiment plan

> **Written for:** whoever runs the robustness study.
> **Status:** plan + harness implemented; **the study itself has not
> been run**, because it needs photographs that do not exist yet.

---

## 1. The question

Phonics Prism will point a camera at a book in a classroom or a home.
The baseline so far has only seen rendered glyphs. This study asks:
**which real-world conditions break it, and by how much?**

Not "does it work" — it demonstrably works on clean input. The useful
output is a table of what degrades accuracy and at what threshold, so
the hardware spec (camera, lighting, mount) can be driven by evidence.

---

## 2. First result, and a correction

The preprocessing harness has been run on the **clean synthetic** set.
Every variant made things worse:

| variant | CER | ΔCER | diacritic recall | Δ dia-R |
|---|---|---|---|---|
| **original** | **0.1028** | — | **0.5333** | — |
| clahe | 0.1121 | +0.0093 | 0.4000 | −0.1333 |
| upscale2 | 0.1121 | +0.0093 | 0.3333 | −0.2000 |
| upscale3 | 0.1495 | +0.0467 | 0.4667 | −0.0667 |
| denoise | 0.1215 | +0.0187 | 0.4000 | −0.1333 |

n=4 synthetic images, 107 reference characters. Reproduce with:

```bat
python scripts\compare_preprocessing.py --ground-truth data\ground_truth\synthetic_smoke.jsonl --device gpu --variants original clahe upscale2 upscale3 denoise
```

**This corrects an earlier claim.** `README.md` and `architecture.md`
previously described `--upscale` as the one preprocessing step that
"reliably helps" diacritics. Measured, it did the opposite here: it
*reduced* diacritic recall by 0.20 at 2× and still hurt at 3×. That
claim was a plausible hypothesis presented with more confidence than
the evidence supported, and it has been corrected in both documents.

The honest interpretation is narrower than either claim: **on clean,
high-resolution input there is nothing for preprocessing to fix, so
every transformation only adds resampling artefacts.** Whether it helps
on genuinely degraded camera images is exactly what this study is for,
and it remains open.

This is the whole reason the harness exists. "Enhancement improves
accuracy" is an assumption; the table above is a measurement.

---

## 3. Experimental design

### Capture protocol

Photograph **the same physical pages** under each condition, so that
differences are attributable to the condition rather than the content.
Use a tripod where possible, change one variable at a time, and record
the condition in the annotation's `conditions` field.

### Factors

| Factor | Levels | Why |
|---|---|---|
| **Lighting** | even / dim / harsh direct / shadowed / glare | Classroom and home lighting vary enormously; glare on glossy paper erases whole regions |
| **Blur** | none / slight / moderate / severe | A child moves the book; autofocus hunts |
| **Perspective** | flat / ~10° / ~25° / ~40° | A camera on a stand is rarely parallel to the page |
| **Curvature** | flat / slight / moderate (near spine) | Open books bow; `rectify_line` assumes a planar quad |
| **Print size** | large (children's book) / medium / small | Diacritic size scales with print size |
| **Resolution** | native / ½ / ¼ / ⅛ | Finds the pixel floor for diacritics |
| **Script** | Arabic + tashkeel / Arabic plain / English / mixed | The tashkeel gap is the headline metric |

### Minimum viable study

A full factorial is impractical. Run a **one-factor-at-a-time** study
from a clean baseline:

- 10 pages × clean baseline (reference condition)
- 10 pages × each of: dim, harsh, shadowed, blur-moderate,
  perspective-25°, curvature-moderate, print-small
- Resolution handled by downsampling the clean captures, so it needs
  no extra photography

That is ~80 photographs, each labelled once (the transcription does
not change between conditions of the same page — only the image does).

### Resolution sweep

The most actionable single number this study can produce is **the
pixel height at which a fatha stops being recognised**. It sets a hard
requirement on camera and mounting distance. Downsample the clean
captures by 1/2, 1/4, 1/8 and plot diacritic recall against measured
mark height.

---

## 4. Running it

Validate first:

```bat
python scripts\validate_dataset.py --ground-truth data\ground_truth\camera_study.jsonl
```

Baseline accuracy per condition group:

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\camera_study.jsonl --strip-tashkeel-view
```

Preprocessing comparison per condition — run it **separately for each
condition group**, because a variant that rescues a dim image can ruin
a clean one:

```bat
python scripts\compare_preprocessing.py --ground-truth data\ground_truth\dim_only.jsonl --variants original clahe clahe_strong grayscale denoise upscale2
```

Word localisation under distortion:

```bat
python scripts\evaluate_localization.py --ground-truth data\ground_truth\camera_study.jsonl
```

Resource cost:

```bat
python scripts\measure_resources.py --image data\input\a_real_photo.jpg --runs 10
```

---

## 5. Reading the results

**Always read CER and diacritic recall together.** A variant that
lowers CER while lowering diacritic recall is removing tashkeel, not
reading better — it looks like progress and is regression for this
project. `compare_preprocessing.py` prints both columns side by side
for exactly this reason.

Other traps:

- **Per-condition denominators.** 10 images is enough to spot a cliff,
  not to rank two similar variants.
- **Do not pool conditions.** The average over "clean" and "severe
  blur" describes no real situation.
- **Deltas do not transfer.** A preprocessing gain measured on dim
  images says nothing about blurred ones.
- **Detection failures cascade.** If a condition breaks detection,
  recognition and localisation scores for it are measuring detection,
  not themselves. Check `n_regions` before interpreting CER.

---

## 6. Expected outcomes, stated as hypotheses

These are predictions to test, **not findings**:

1. Diacritic recall will fall faster than character accuracy as
   resolution drops, because marks are small before letters are.
2. There is a resolution floor below which tashkeel are unrecoverable
   regardless of preprocessing.
3. CLAHE will help shadowed and dim images and hurt clean ones.
4. Denoising will hurt diacritics in every condition.
5. Perspective beyond ~25° will degrade word localisation more than
   recognition, because `rectify_line` assumes a planar quad.
6. Curvature near the spine will break word localisation first.

Hypothesis 3 is directly contradicted for *clean* images by §2, which
is consistent with it — the claim was about shadowed and dim images,
and those have not been tested.

---

## 7. What this study would unblock

- The camera and mounting specification (resolution floor, working
  distance, lighting requirement).
- Whether preprocessing should be on by default for any condition, and
  which.
- Whether the 95% character-accuracy target is reachable with the
  current model, or whether fine-tuning is required.
- Whether word localisation is good enough for a projector, under the
  distortions a real rig produces.

Until it runs, every accuracy figure in this repository is an upper
bound from rendered text.

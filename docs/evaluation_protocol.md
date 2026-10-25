# Evaluation protocol

> **Written for:** anyone who will quote a number from this project.

A score is only meaningful alongside its denominator, its
normalisation settings and its sample. This document defines all three
so that figures from different runs can be compared, and so that a
figure can never be read as meaning more than it does.

---

## 1. What gets measured

| Metric | Unit | Denominator |
|---|---|---|
| **CER** | Unicode code points | Reference character count |
| **WER** | Whitespace-separated tokens | Reference token count |
| **Exact match rate** | Whole transcriptions | Number of samples |
| **Diacritic precision / recall / F1** | `(base index, mark)` pairs | Predicted / reference marks |
| **Confusable count mismatch** | Characters per class | — (absolute counts) |
| **Mean inference time** | Seconds | Images successfully processed |

### CER

Levenshtein distance over code points, divided by reference length.

```
CER = (substitutions + deletions + insertions) / reference_characters
```

Each combining mark counts as **one unit**. A dropped fatha is one
deletion. This is deliberate and is the main reason CER here will look
worse than published Arabic OCR numbers computed on unvowelled text.

CER can exceed 1.0 when the hypothesis is much longer than the
reference. That is correct behaviour, not a bug.

### WER

The same edit distance over whitespace tokens. Tokenisation is
`str.split()` — naive, documented, and applied identically to both
sides. It does not separate punctuation and does not segment Arabic
clitics (`و`, `ال`, …), so WER here is coarser than a
linguistically-aware tokeniser would give. It is consistent, which is
what makes it usable for comparison between runs.

### Diacritic P / R / F1

See `arabic_diacritics.md` §6 for the full definition and its
limitations. The short version: a mark is a true positive only if the
same mark sits on the same base character position, and **a model that
outputs no diacritics scores recall 0**.

---

## 2. Micro-averaging, and why

Corpus figures are computed by **summing edit distances and reference
lengths, then dividing once**:

```
corpus_CER = sum(errors over all samples) / sum(reference_length over all samples)
```

Not by averaging per-sample CERs. A macro average would weight a
three-character caption the same as a three-hundred-character page, so
a few short, easy samples could dominate the headline number.

Every aggregate in the report includes `reference_length`, so the
denominator is always visible.

---

## 3. Normalisation applied before scoring

Both reference and hypothesis go through **the same** normalisation.
The settings are recorded in every report under `normalisation`.

### Primary scoring (always reported)

| Step | Setting | Why |
|---|---|---|
| Unicode form | **NFC** | Two transcriptions that look identical compare equal. Does not remove diacritics. |
| Tatweel (U+0640) | **removed** | Purely typographic elongation; carries no phonetic value. |
| Whitespace | **collapsed** | Line-break placement is a layout artefact, not a reading error. |
| Tashkeel | **PRESERVED** | A missing diacritic is an error. |
| Alif / taa marbuta / alif maqsura folding | **not applied** | These are real orthographic distinctions. |
| Arabic-Indic digits | **not mapped** | `١` and `1` are different printed characters. |

### Secondary view (`--strip-tashkeel-view`, optional)

Removes tashkeel from **both** sides and re-scores. Reported in a
separate block labelled `SECONDARY VIEW`.

It answers "did the model get the consonantal skeleton right?" — useful
for separating "misread the letters" from "read the letters, missed the
vowels". It is **not** an accuracy figure for this project and must
never be quoted on its own. The primary vowelled figure is always
reported too.

---

## 4. Evaluation groups

Results are broken out by group, never pooled into a single headline
number across conditions.

| Group | Contents | What it tells you |
|---|---|---|
| `arabic_tashkeel` | Arabic with diacritics | The core requirement |
| `arabic_plain` | Arabic without diacritics | Baseline for comparison |
| `english` | English only | A different model, different failure profile |
| `mixed` | Both scripts on one page | Experimental; see `architecture.md` |
| `clean` | Flat, even, high-res | Upper bound on achievable accuracy |
| `degraded` | Blur, perspective, shadow | What a camera rig actually produces |

The most informative comparison is **`arabic_tashkeel` vs
`arabic_plain` on the same sentences**. If CER is much worse on the
vowelled set, the gap is diacritic recognition specifically rather than
general Arabic difficulty. Labelling the same text both ways is worth
the extra effort.

---

## 5. Procedure

### 1. Collect images

Real photographs, not synthetic renders. See `data/samples/README.md`.
Cover each condition you intend to report on.

### 2. Label

One JSONL line per image. Schema and rules in
`data/ground_truth/README.md`. Transcribe what is printed, including
every diacritic that is printed and none that is not.

### 3. Hold out

Keep a held-out set that is **not** looked at while tuning thresholds,
preprocessing flags or anything else. Numbers from images you tuned on
are not a measure of performance — they are a measure of how well you
tuned.

### 4. Run

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\my_labels.jsonl
```

With the secondary consonants-only view:

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\my_labels.jsonl --strip-tashkeel-view
```

Pin the device when comparing runs:

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\my_labels.jsonl --device gpu
```

### 5. Read the report

`data/results/eval_<timestamp>.json` contains per-sample records and
group aggregates, plus the full configuration that produced them. The
console prints the summary table.

Check these before believing anything:

- `n_inference_failures` — failed images are **excluded** from every
  figure. A high CER with 5 silent failures is not a high CER.
- `reference_length` — a CER over 40 characters is noise.
- `diacritic.true_positives` — if 0 across the whole corpus, the model
  is producing no diacritics at all and the CER on vowelled text is
  measuring one single systematic failure, not a distribution of
  errors.
- `hypothesis_tashkeel_count` per sample — the fastest check of whether
  tashkeel are in the model's charset at all.

---

## 6. Sample size

Rough guidance, not statistics:

| Samples per group | Good for |
|---|---|
| 1–5 | Smoke-testing the pipeline. **Not** for quoting. |
| 10–30 | Spotting gross failures and direction of effects. |
| 50–100 | A number worth putting in a document, with the caveat stated. |
| 200+ | Comparing configurations whose difference is small. |

CER is more stable than WER at small n, because it has more units per
sample. Exact-match rate is the least stable: it is binary per sample.

---

## 7. Detection / localisation metrics

**Not implemented.** Scoring box localisation needs manually annotated
reference boxes, and there is no annotated box dataset in this project.

When one exists, the pieces are already here: `geometry.bbox_iou()`
computes IoU, and the standard protocol would be to match predicted to
reference boxes at IoU ≥ 0.5 and report precision/recall.

Note that reference boxes would have to be **line-level** to match what
the detector produces. Word-level localisation is not implemented (see
`architecture.md`), so word-level box metrics are not available.

---

## 8. Reporting rules

**Always state:**

- Number of samples and the total reference length.
- Which group(s) the figure covers.
- The normalisation settings (the report records them).
- Whether diacritics were included.
- Device and model versions.
- How many images failed inference.

**Never:**

- Quote a CER without its denominator.
- Quote the consonants-only figure as *the* accuracy.
- Pool groups into one number and call it overall accuracy.
- Quote numbers from the synthetic `offline_selftest.png`.
- Report figures from images used for tuning as held-out performance.

### On the 95% character-accuracy target

The project target is ~95% character accuracy (CER ≤ 0.05).

**That target has not been evaluated.** No ground-truth dataset exists
in this repository, so no CER has been computed on real data. The
evaluation framework is implemented and tested on synthetic fixtures;
it has not been run against labelled page images.

The claim may only be made when: a held-out set of real photographed
pages exists, with at least ~50 samples in the relevant group, scored
with tashkeel preserved, reported per group. Until then the honest
statement is "not yet measured" — not an estimate, and not a figure
borrowed from PaddleOCR's own published benchmarks, which were computed
on different data under different conditions.

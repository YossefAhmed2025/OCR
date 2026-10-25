# Word-level localisation

> **Written for:** engineers working on the projector path.
> **Status:** implemented, measured on synthetic pages, **not validated
> on photographs**.

The projector must highlight one word at a time. PP-OCRv5 detection
returns **line** polygons, so something has to turn a line into words.
This document records what was investigated, what was built, what it
scores, and where it breaks.

---

## 1. Options considered

| Approach | Verdict |
|---|---|
| Split the line box by character count | **Rejected.** Arabic is cursive with wildly varying glyph widths; diacritics add characters but no width. Boxes would look plausible and land on the wrong word. |
| Whitespace gap analysis (vertical projection profile) | **Rejected as primary.** In Arabic, intra-word letter gaps and inter-word gaps are not cleanly separable by a projection profile, because letters within a word join but some letters (`ا د ذ ر ز و`) do not join to the left, creating word-sized gaps inside words. |
| Lower `--det-unclip-ratio` to tighten detection | **Rejected.** Produces fragments, and fragments are not reliably words. |
| A dedicated word-level detection model | **Viable, not pursued.** No PP-OCRv5 word-level detector ships for Arabic; training one needs an annotated dataset that does not exist yet. |
| **CTC timestep alignment** | **Chosen.** See below. |

### Why CTC alignment

The recognition model is a CTC model. Its raw output is a matrix of
shape `(timesteps, classes)`, and each timestep corresponds to a
horizontal strip of the input image. Decoding while *keeping the
timestep* gives a model-derived horizontal position for every
character — measured by the network that actually read the glyph,
rather than inferred from character counts.

PaddleOCR's public `TextRecognition` API returns only `rec_text` and
`rec_score`, so this was initially assumed unavailable. It is reachable
by loading the **same** `inference.json` / `inference.pdiparams` through
`paddle.inference` directly. No different model, no retraining.

Verified on this model:

```
output shape (1, 40, 749)   # 40 timesteps, 747 charset + blank + space
```

---

## 2. How it works

```
line polygon (from detection)
    |
    v
rectify_line()        perspective-warp the quad to a horizontal strip,
    |                 keeping the inverse matrix
    v
CTCAligner.align()    resize to height 48, DYNAMIC width, no padding;
    |                 greedy CTC decode keeping each char's timestep;
    |                 timestep t -> columns [t, t+1) * (crop_w / T)
    v
segment_words()       split at the space class; diacritics never split
    |                 a word; reverse to logical order for RTL
    v
ink_rows()            measure the vertical ink extent inside each
    |                 word's own columns
    v
crop_rect_to_image_polygon()   map back through the inverse matrix
    |
    v
word polygon in ORIGINAL image coordinates
```

### Padding breaks the mapping

Three preprocessing variants were compared on the same crop:

| Variant | Timesteps | Decoded | Alignment usable? |
|---|---|---|---|
| Pad to fixed 320px | 40 | correct text | **No** — characters fire at timesteps 29–39, inside the zero-padded region, so a timestep no longer maps to a known column |
| **Dynamic width, no padding** | 13 | correct text | **Yes** — linear, ~8.3 px per timestep |
| Stretch to 320px | 40 | *more diacritics recovered* | Usable, distorts aspect ratio |

Dynamic width is used. The stretch result is a separate and
interesting lead — see §6.

### Vertical extent

CTC gives **no** vertical information: a timestep is a full-height
column. Left alone, every word box spans the whole line height.

`ink_rows()` therefore measures where dark pixels actually sit inside
each word's own column range and tightens the box to that. This uses
the image rather than an assumption, and it correctly includes
diacritics above and below the body — a box that clipped the tashkeel
would be wrong for this project specifically.

It assumes dark text on a light page. For an inverted or very
low-contrast page it finds no ink and returns `None`, and the caller
keeps the full line height and flags `no_ink_in_span` rather than
inventing a tighter box.

---

## 3. Audit of the alignment (2026-10-24)

The method was audited rather than trusted. Everything here is
measured.

### 3.1 Tensor semantics - VERIFIED

| Property | Finding |
|---|---|
| Head | `CTCLabelDecode` (from the model's own `inference.yml`) |
| Output | `(batch, T, 749)` = 747 charset + blank + space |
| Blank index | 0 |
| Space index | 748 (last) |
| Input | `[3, 48, W]`, height fixed at 48 |
| **Width stride** | **exactly 8**: `T == floor(W / 8)` for every width from 16 to 800 |

### 3.2 Linearity - VERIFIED, and a bug fixed

A single glyph was swept across x and the firing timestep recorded.
For confidently-recognised glyphs:

```
ink_centre = 7.98 .. 8.01 * timestep + constant     residual sd ~2px
```

Per-glyph error is a **constant offset independent of position**,
which is what linearity means. The mapping is sound.

**Bug found and fixed.** The implementation used `crop_width / T` as
the per-timestep width. That is correct only when the resized width is
an exact multiple of 8; the model never sees the final `W mod 8`
columns. Measured on non-multiple-of-8 widths with a glyph near the
right edge:

| Mapping | Mean error | sd | max |
|---|---|---|---|
| **stride 8 (now used)** | **+0.50px** | **0.00** | **0.50** |
| `crop_width / T` (was used) | +3.25px | 0.29 | 3.62 |

After the fix, horizontal recall at IoU 0.7 rose **0.750 -> 0.806**.

### 3.3 Decoding rules - VERIFIED

* **Repeated characters** decode correctly (`hello`, `aaa`,
  `committee`): separated glyphs put a blank between repeats, so CTC
  does not collapse them.
* **Diacritics consume their own timestep**, adjacent to but distinct
  from their base letter. A fatha therefore gets a *different column*
  from the letter it visually sits above. Within a word this largely
  cancels, but it widens the span slightly.
* **Diacritic recognition is lossy.** Vowelled `kataba` (3 marks)
  decoded with 2; `mudarris` lost its shadda. A recognition limit, not
  an alignment one.
* **Blank density falls with diacritics**: plain `ktb` gave 6/9 blank
  timesteps, the vowelled form only 2/9.

### 3.4 Conditions that invalidate the mapping

| Condition | Effect |
|---|---|
| Right-padding to a fixed width | **Breaks it.** Characters fire inside the padding. |
| Stretching to a fixed width | Still valid (uniform scale); aspect distorted. Recovers *more* diacritics - unexplored lead. |
| Low-confidence characters | **Position unreliable.** Isolated misread glyphs fired at wildly wrong timesteps (one at 247px error). |
| `W mod 8 != 0` | Final 0-7 resized columns are invisible to the model. |

---

## 4. Measured results

Two annotated sets, both **synthetic renders**, boxes exact by
construction. 95% Wilson intervals shown, because 36 boxes cannot
support a tight claim.

### 4.1 The decisive decomposition: detection vs alignment

Same images with detected lines, then with **oracle lines** (line
extent from the union of reference boxes, detection bypassed):

| Set | Lines | Horiz R@0.5 | 2-D R@0.5 |
|---|---|---|---|
| Main (36 boxes) | detected | 0.889 | 0.889 |
| Main | **oracle** | **1.000** [0.904, 1.000] | **1.000** |
| Boundary (37 boxes) | detected | **0.054** | **0.054** |
| Boundary | **oracle** | **0.946** [0.823, 0.985] | **0.919** |

**CTC alignment is not the weak link; detection is.** On the boundary
set, PP-OCRv5 detection fragmented short sparse images - for a line of
three two-letter Arabic words it detected the three **hamza marks** as
separate text regions (2px-wide boxes). Word localisation then
faithfully localised the fragments.

A headline recall of 0.054 looks like a broken method and is actually
a broken input.

### 4.2 Alignment quality given a correct line (oracle, main set)

| IoU | Horizontal recall | 95% CI | 2-D recall | 95% CI |
|---|---|---|---|---|
| 0.30 | 1.000 | [0.904, 1.000] | 1.000 | [0.904, 1.000] |
| 0.50 | 1.000 | [0.904, 1.000] | 1.000 | [0.904, 1.000] |
| 0.70 | 0.917 | [0.782, 0.971] | 0.806 | [0.650, 0.902] |

Mean IoU on matched pairs: 0.86 horizontal, 0.82 2-D.

**Text agreement on matched boxes: 24/36 (66.7%)** on the main set,
16/36 (44.4%) on the boundary set. Geometry is substantially better
than recognition: a correctly placed box often carries wrong text.

### 4.3 Vertical tightening - audited

| Check | Result |
|---|---|
| Marks **above** (fatha) | Captured, 0px clipping |
| Marks **below** (kasra) | Captured, 0px clipping |
| Both at once (shadda + kasra) | Captured, box correctly taller |
| English descenders / ascenders | Captured, 0px clipping |
| Punctuation below baseline | Captured |
| Neighbouring word in other columns | **Not** included |
| Light print (grey >= 200, contrast <= 55) | No ink -> falls back to line height, flagged |
| **Inverted print (light on dark)** | **DEFECT, now fixed** |
| Two lines inside one crop | One tall box spanning both (documented limit) |

**Defect found and fixed.** White-on-black returned the *full crop
height* with no warning, because the dark background fell below the
threshold everywhere - a silently wrong box, which is worse than no
box. `ink_rows` now returns `(extent, status)` and refuses with
`suspect_polarity` when more than 60% of the window reads as ink.

Both boxes are preserved separately: `bbox` (tightened) and
`bbox_untightened` (raw CTC column range at line height), plus
`vertical_source` recording which path ran.

Tightening remains a large win, A/B measured:

| 2-D IoU | with tightening | without |
|---|---|---|
| R@0.5 | 0.889 | 0.722 |
| R@0.7 | **0.667** | **0.028** |

### 4.4 Alternatives benchmarked

Projection-profile segmentation (no model, pure CV) implemented as
`segment_by_projection`, run on the same oracle lines:

| Set | CTC 2-D R@0.5 | Projection 2-D R@0.5 |
|---|---|---|
| Main | **1.000** | 0.194 |
| Boundary | **0.919** | 0.351 |

The theoretical objection is confirmed by measurement: Arabic's
non-joining letters create word-sized gaps *inside* words, which
projection profiling cannot distinguish from real boundaries.
**CTC alignment stays the baseline.** No new model was added.

### 4.5 The projector-relevant result

IoU is not what the projector needs.
`scripts/simulate_projector.py` converts boxes into coverage and
spill. With **real predicted boxes and zero calibration error**, at
coverage >= 0.90 and spill <= 0.05:

| Highlight scale | Success | Coverage | Spill |
|---|---|---|---|
| 1.00 | **0.389** | 0.834 | 0.000 |
| 1.10 | 0.667 | 0.907 | 0.000 |
| **1.25** | **0.778** | 0.960 | 0.005 |
| 1.50 | 0.778 | 0.990 | 0.028 |

**Only 38.9% of highlights are acceptable at 1:1 scale, before any
calibration error exists.** A deliberate 1.25x margin raises that to
77.8%. See `projector_calibration.md` for the full error budget.

### What these numbers are not

Every annotated image is `capture: synthetic_render` - rendered Arial
glyphs, flat, evenly lit, no lens blur, no paper texture. Both the
evaluation script and the dataset validator say so. **These are an
upper bound and do not predict camera performance.** 36 and 37 boxes
are small samples; the Wilson intervals above are the honest width.

---

## 5. Known failure modes

| Mode | Flag emitted | Notes |
|---|---|---|
| Model never emits a space, so two words merge | `single_span_line` | The span is still returned, flagged, not hidden |
| Low character confidence | `low_confidence` | Threshold 0.5 on the weakest character in the word |
| A lone diacritic became its own "word" | `orphan_diacritic` | Usually a decode artefact |
| No ink found in the word's columns | `vertical_no_ink` | Falls back to full line height |
| Inverted / dark-background print | `vertical_suspect_polarity` | Refuses to tighten rather than returning a full-height box |
| Degenerate line polygon from detection | logged, line skipped | Seen on fixtures where detection returned a 2×1 polygon |

Detection quality cascades. On one fixture the detector split a line
into spurious fragments (`M`, `−`), and word localisation faithfully
localised the garbage. Word boxes can never be better than the lines
they come from.

---

## 6. Honest limitations

1. **CTC peaks approximate position.** The network emits a character at
   whichever timestep maximises likelihood, which need not be the
   column where the ink sits. Peaks typically lead or lag slightly.
   This is why horizontal IoU plateaus around 0.85 rather than
   approaching 1.0.
2. **Vertical extent is measured, not predicted.** It comes from pixel
   thresholding, so it inherits that method's assumptions.
3. **Not validated on photographs.** Perspective, curvature and blur
   are untested for this path beyond a ±4° synthetic rotation.
4. **Line curvature is not handled.** `rectify_line` applies a single
   perspective transform, which straightens a *planar* quad. A curved
   line on a bowed page is not unwarped, and words near the curve
   extremes will drift.
5. **`--lang mixed` is unsupported** for word localisation: each line
   would need a script decision before alignment. The CLI rejects the
   combination rather than guessing.

---

## 7. Next steps

1. **Annotate word boxes on real photographs** and re-run
   `evaluate_localization.py`. Everything above is synthetic.
2. **Test the stretch-to-fixed-width variant.** It recovered more
   diacritics in the preprocessing comparison. If that holds, it may
   be worth a second recognition pass purely for diacritic recall,
   with alignment taken from the dynamic-width pass.
3. **Handle curvature** — piecewise rectification along the line, or a
   thin-plate spline, if photographed pages show the drift.
4. **Consider a word-level detector** once an annotated set exists.
   CTC alignment is a good interim answer; a model trained for the job
   would be better.
5. **Decide the accuracy bar the projector needs.** Highlighting the
   wrong word is a worse failure for a dyslexic reader than
   highlighting nothing, so the product requirement should be set
   before this is shipped — and it is a product decision, not a
   technical one.

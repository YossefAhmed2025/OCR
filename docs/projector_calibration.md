# Camera → page → projector calibration

> **Written for:** whoever builds the projector rig.
> **Status: NOTHING in this document is implemented.** This is a
> statement of what is still missing between a word box and a lit word.

---

## 1. The gap

Every coordinate this project produces is in **camera image pixels**:
origin top-left of the photograph, x right, y down. The result JSON
says so in `coordinate_space.note`.

A projector needs **projector framebuffer pixels**. These are not the
same space, not related by a constant, and not relatable by any amount
of OCR improvement. Three transforms sit between them, none built:

```
word box in CAMERA pixels
      |  H_cam->page    camera pose, lens distortion, page flatness
      v
position on the PHYSICAL PAGE (mm)
      |  H_page->proj   projector pose, throw, keystone, lens
      v
PROJECTOR framebuffer pixels
      |  warp/blend     focus, brightness, colour
      v
light on the page
```

Treating camera pixels as projector pixels would light an arbitrary
region. This is not a tuning problem.

---

## 2. What each stage needs

### Camera → page

A homography maps a plane in the camera image to the physical page,
**if** the page is flat and the lens is rectified.

Needs:
- **Intrinsic calibration** of the camera (focal length, principal
  point, radial/tangential distortion). Standard chequerboard
  procedure, done once per camera model.
- **Page corner detection** at run time, or a fixed mechanical
  registration so the page always sits in a known place.
  `preprocessing.four_point_transform()` exists as the hook and is
  deliberately **not** wired in, because automatic corner detection is
  not implemented.
- **A flatness assumption.** A homography cannot model a bowed page
  near the spine. Measured curvature beyond a few millimetres needs
  either a mechanical flattener (glass platen) or a non-planar model.

### Page → projector

Also a homography for a flat page, but it must be calibrated
**in situ**, because it depends on where the projector physically sits.

Needs:
- **A calibration pattern projected and observed.** The standard
  approach: project a known grid, photograph it with the same camera,
  solve for the homography. This closes the loop and absorbs projector
  keystone and camera pose in one step.
- **Re-calibration whenever anything moves.** Bumping the rig
  invalidates it. The system should detect this, not fail silently.
- **Projector lens distortion**, which is usually smaller than the
  camera's but not zero for short-throw units.

### Closing the loop

The robust design calibrates **camera ↔ projector directly** and never
materialises page millimetres at all:

1. Projector displays a structured-light or chequerboard pattern.
2. Camera photographs it.
3. Solve H_proj→cam from the correspondences.
4. Invert to map an OCR box into projector pixels.

This is one homography instead of two and removes page-corner
detection from the critical path, as long as the page stays in the
plane used for calibration.

---

## 3. The error budget

`scripts/simulate_projector.py` measures how much total error a
highlight survives. **Measured on synthetic pages with exact boxes**,
with a highlight counted acceptable at coverage ≥ 0.90 and spill ≤ 0.05:

| Perturbation | Success rate |
|---|---|
| none | 1.000 |
| dx = 0.05 × word height (2.4px) | 1.000 |
| dx = 0.10 × word height (4.8px) | 0.944 |
| dx = 0.20 × word height (9.6px) | 0.389 |
| **dy = 0.10 × word height** | **0.556** |
| dy = 0.20 × word height | 0.000 |

Two things follow.

**The budget is small.** Roughly **0.05–0.10 × word height** of total
residual error, for everything combined: OCR box error, camera
calibration, projector calibration, and mechanical drift. On a page
where a word is 48px tall that is about 2–5px.

**Vertical precision matters more than horizontal.** At the same
offset, dy costs far more than dx (0.556 vs 0.944), because words are
wider than they are tall, so a vertical shift loses a larger fraction
of the box.

### Scale error is asymmetric — exploit it

| Highlight scale | Success | Coverage | Spill |
|---|---|---|---|
| 0.90 | 0.000 | 0.688 | 0.000 |
| 1.00 | 0.389 | 0.834 | 0.000 |
| 1.10 | 0.667 | 0.907 | 0.000 |
| **1.25** | **0.778** | **0.960** | **0.005** |
| 1.50 | 0.778 | 0.990 | 0.028 |

(These rows use the **real CTC-predicted boxes**, not perfect ones.)

Under-scaling is fatal; over-scaling is nearly free until about 1.5×,
where spill starts. **Design the highlight with a deliberate ~1.25×
margin.** A slightly generous highlight is a much better failure mode
than a clipped one.

---

## 4. The result that matters

With **real predicted boxes and zero calibration error**, only
**38.9%** of highlights met the coverage/spill criteria. With a 1.25×
margin that rises to **77.8%**.

So the current word boxes are not yet good enough for a projector,
*before any calibration error is added at all*. The headline IoU
figures (recall 1.000 at IoU 0.5 with oracle lines) do not contradict
this — IoU 0.5 simply is not a demanding enough bar for a highlight
that must cover ≥90% of a word.

**This is the single most important finding of the validation.** It
should be read as "the metric was wrong", not "the method failed":
coverage, not IoU, is the product-relevant quantity, and it is now
measured.

---

## 5. Caveats on these numbers

- **Synthetic pages.** Rendered glyphs, flat, evenly lit. Real camera
  input will be worse.
- **36 reference boxes.** Wilson intervals are reported, and they are
  wide. These are directional, not specifications.
- **Rigid perturbations only.** Translation and uniform scale.
  Rotation, keystone and lens distortion are not modelled.
- **Axis-aligned boxes.** A highlight on a rotated line needs the
  polygon, not the bbox. The polygon is already in the JSON.
- **Filled-rectangle highlight assumed.** An underline or outline has
  different failure modes and probably a different budget.

---

## 6. Recommended order of work

1. **Do not start calibration yet.** Fix word-box coverage first —
   §4 shows boxes are the current limit, and calibration error only
   subtracts from an already-insufficient budget.
2. Get real photographed pages with manual word boxes
   (`scripts/annotate_words.py`), and re-run the coverage simulation
   on those.
3. Decide the product thresholds. Is 0.90 coverage right? Is any spill
   acceptable? These are pedagogical questions about a dyslexic reader,
   not engineering ones, and the whole budget depends on them.
4. Then calibrate camera ↔ projector directly (§2), and measure the
   residual against the budget in §3.

### A note on failure modes

For a child learning to read, **highlighting the wrong word is worse
than highlighting nothing**. A system that is confident and wrong
teaches the wrong word. Whatever design emerges should be able to say
"I am not sure" and decline — the `flags` and `confidence` fields on
every word span exist for that, and nothing downstream uses them yet.

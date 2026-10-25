# Camera testing

> **Written for:** whoever points a camera at a real book page.
> **Status:** workflow implemented and tested through a replay source.
> **The live-camera path has not been run** — no camera was reachable
> on the build machine (see §7).

This is the step that turns the project from a synthetic benchmark
into evidence. Everything runs locally against the models already in
`models/`.

---

## 1. Launch

```bat
.venv\Scripts\activate

REM which cameras actually work?
python scripts\camera_test.py --list

REM Arabic, default camera
python scripts\camera_test.py --lang ar

REM English, second camera, nothing written to disk
python scripts\camera_test.py --lang en --camera 1 --no-save

REM with experimental word spans
python scripts\camera_test.py --lang ar --words

REM no camera available? drive the same workflow from image files
python scripts\camera_test.py --lang ar --replay data\input
```

### Controls

| Key | Action |
|---|---|
| `SPACE` or `c` | **Capture & run OCR** on the current frame |
| `n` | next camera (when more than one was found) |
| `s` | toggle saving on/off mid-session |
| `w` | toggle experimental word localisation |
| `q` or `ESC` | **Stop camera** and exit |

The preview banner shows live resolution, a sharpness number and
advisory hints ("too dark", "soft focus", "low resolution"). Fixing
those *before* capturing is far cheaper than discovering them
afterwards.

---

## 2. What you get per capture

Terminal: frame quality, region count, inference time, **mean and
lowest confidence**, and the recognised text per region with its
script and tashkeel count. A warning prints when any region is below
0.50.

On disk, under `data/results/camera_<lang>_<timestamp>/`:

| File | Contents |
|---|---|
| `capture_NNN.png` | the frame exactly as captured |
| `capture_NNN.json` | full OCR result + capture metadata |
| `capture_NNN.txt` | plain transcription, UTF-8 |
| `capture_NNN_overlay.png` | regions and word spans drawn |
| `session.json` | all captures, confidences, timings |

`--no-save` writes **nothing at all** — no image, no JSON, not even a
temporary file left behind. There is a test asserting that.

### Reading the overlay

* **Solid boxes** — the detector's own text-line regions. Real model
  output.
* **Dashed boxes** — **experimental word spans derived from CTC
  recognition timesteps.** These are *not* detector-produced word
  boxes and are not validated on photographed pages. Blue = ok,
  amber = carries a flag, red = confidence below 0.50.

The distinction is deliberate and is repeated in the JSON
(`granularity: "word_ctc_aligned"`, plus `word_geometry_note`).

---

## 3. Camera permissions on Windows

If `--list` finds nothing, work through these in order.

**1. Windows camera privacy settings.**
*Settings → Privacy & security → Camera*. Turn on:
- "Camera access"
- "Let apps access your camera"
- **"Let desktop apps access your camera"** — this last one is the
  usual culprit, and Python counts as a desktop app.

**2. Another application is holding it.** Teams, Zoom, Chrome and the
Windows Camera app all claim the device exclusively. Close them and
retry.

**3. The device is disabled.** *Device Manager → Cameras*. If the
entry has a down-arrow, right-click → Enable. If its status is
"Unknown", the camera is not currently attached or is disabled in
firmware.

**4. A privacy shutter or function key.** Many laptops have an
`Fn+F8`-style camera kill switch, or a physical slider.

**5. Laptop firmware.** Some BIOS/UEFI menus can disable the
integrated camera entirely.

---

## 4. Troubleshooting

### "No working camera found"

The probe only reports indices that actually **return a frame**. A
device that opens but never reads is excluded on purpose — it would
hang the preview. Work through §3.

### Preview takes several seconds to appear

Windows Media Foundation is slow to fail. The code tries DirectShow
first for this reason. To force a backend:

```bat
python scripts\camera_test.py --lang ar --backend CAP_DSHOW
```

### "opened but returned no frame"

Usually a permissions problem (§3) or another app holding the device.
It can also mean the requested resolution is unsupported — try:

```bat
python scripts\camera_test.py --lang ar --width 1280 --height 720
```

### "cannot open a preview window"

No display is available (remote session, service account). Capture
one frame without a window:

```bat
python scripts\camera_test.py --lang ar --no-preview
```

### Camera stops mid-session

"stopped returning frames" means it was unplugged or claimed by
another application. Restart the script.

### No text detected

In order of likelihood: too far away (fill the frame with the page),
out of focus, too dark, or the page is at too steep an angle. The live
banner flags the first three before you capture.

### Arabic shows as `?????` in the terminal

A console code-page issue, not a data issue:

```bat
chcp 65001
```

The `.txt` and `.json` files are UTF-8 regardless.

### Out of memory on the 4 GB GPU

`--lang mixed` holds two recognition models. Use `--lang ar` or
`--lang en`, or add `--device cpu` (about 20x slower).

---

## 5. What this workflow does NOT do

**No page detection. No perspective correction.**
`preprocessing.four_point_transform()` exists, but nothing finds page
corners automatically, so the frame goes to OCR as captured. This is
stated here because it would be easy to assume otherwise.

What does exist:
- `--deskew` — corrects small **in-plane rotation** only (≤15°).
- `--clahe` — adaptive contrast.

Both are off by default. On the synthetic set every preprocessing
variant made accuracy *worse* (`docs/camera_experiments.md` §2), so
turn them on only when you can measure the effect.

Practical consequence: **hold the camera roughly parallel to the
page.** Perspective is not corrected and will degrade both recognition
and word localisation.

Also not done here, deliberately: projector calibration, and model
fine-tuning.

---

## 6. From captures to a real measurement

Capturing is not evaluating. To get an actual accuracy number:

1. **Capture** 40+ pages across conditions — see
   `docs/dataset_preparation.md` §2 for the minimum viable set.
2. **Label** them. Transcribe what is printed, keeping every printed
   diacritic and adding none. Same document, §4.
3. **Validate**:
   ```bat
   python scripts\validate_dataset.py --ground-truth data\ground_truth\camera.jsonl
   ```
   Set `"capture": "photograph"` — only that counts as real-world
   evidence, and the validator says so plainly.
4. **Score**:
   ```bat
   python scripts\evaluate.py --ground-truth data\ground_truth\camera.jsonl --strip-tashkeel-view
   ```

Until step 4 runs on photographs, every accuracy figure in this
repository is still an upper bound from rendered text.

---

## 7. Verification status — read this

| Part | Status |
|---|---|
| Frame sources, assessment, capture loop | **Tested** (46 automated tests) |
| Saving, `--no-save` guarantee | **Tested** |
| Arabic diacritic + RTL preservation | **Tested** |
| Word-span attachment and warnings | **Tested** |
| Full OCR path, GPU, offline | **Run** via `--replay` |
| Overlay rendering | **Run**, inspected visually |
| **Opening a real camera** | **NOT RUN** |
| **Live preview window** | **NOT RUN** |
| **Camera switching (`n`)** | **NOT RUN** |

The build machine reported an "Integrated Camera" with status
`Unknown`, and OpenCV could not open index 0–3 on **any** backend
(DSHOW, MSMF, ANY, VFW). So the device-handling code is written
carefully and unit-tested through an injected frame source, but the
live path has never executed.

Expect to hit something on first run. §3 and §4 cover the likely
causes; `--replay` lets you confirm the rest of the pipeline is
healthy while you sort the camera out.

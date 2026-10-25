# Baseline findings

> **Written for:** the Phonics Prism team.
> **Date:** 2026-10-24
> **Status of every number below: SMOKE TEST, not a benchmark.**

## Read this before quoting anything here

Every measurement on this page comes from **four synthetic images** of
text rendered from Arial by PIL — one image per group, `n=1`.

They are clean, flat, perfectly lit, perfectly focused, at a font size
far larger than a real book page. They are **not photographs**, and
real camera input will be harder in every respect.

These numbers demonstrate that the pipeline runs end to end and that
the metrics behave correctly. **They are not an accuracy result for
Phonics Prism**, and the 95% target is not evaluated by them. See
`evaluation_protocol.md` §6 for the sample sizes a real figure needs.

---

## 1. The Arabic model's charset contains tashkeel

Checked directly against the shipped character dictionary in
`models/arabic_PP-OCRv5_mobile_rec/inference.yml`:

- **747 characters total**
- **38 of them are Arabic diacritics**, including all eight classical
  harakat: fathatan, dammatan, kasratan, fatha, damma, kasra, shadda,
  sukun — plus maddah, hamza above/below, dagger alif, and a set of
  Quranic annotation marks.

This answers the gating question in `arabic_diacritics.md` §8:
**diacritic recognition is structurally possible with this model.** It
is not excluded by the charset, so it is worth measuring rather than
abandoning.

---

## 2. A correction: an early conclusion here was wrong

The first run of this smoke test reported that the model recognised the
consonants perfectly and returned **zero** diacritics, which looked
like a clean "the model ignores tashkeel" result.

**That conclusion was wrong, and the cause was a bug in our own test
rig, not in the model.**

`arabic-reshaper` — used to shape Arabic glyphs for rendering —
defaults to `delete_harakat=True`. It had stripped every diacritic
while generating the "vowelled" test image. The model was scored on an
image that carried no tashkeel at all, so a recall of 0 was the only
possible outcome.

The shaper is now configured with `delete_harakat=False`
(`visualization.shape_for_display`), there is a regression test for it
(`tests/test_visualization.py::TestDisplayShaping`), and all numbers
below come from re-running against images whose diacritics are
visibly present.

The general lesson is the one in `arabic_diacritics.md` §2: Arabic
tooling removes diacritics by default, quietly, at every layer — and a
pipeline that loses them produces a *better-looking* number, not an
error. Verify the input before trusting the output.

---

## 3. Measured results (synthetic, n=1 per group)

Command:

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\synthetic_smoke.jsonl --device gpu --strip-tashkeel-view
```

Device `gpu:0` (RTX 3050 Ti). Tashkeel preserved; NFC, tatweel removed,
whitespace collapsed.

| Group | n | CER | WER | exact | diacritic recall | diacritic F1 |
|---|---|---|---|---|---|---|
| `arabic_plain` | 1 | **0.0000** | 0.0000 | 1.00 | n/a | n/a |
| `arabic_tashkeel` | 1 | **0.2581** | 0.7500 | 0.00 | **0.5833** | 0.6667 |
| `english` | 1 | **0.0000** | 0.0000 | 1.00 | n/a | n/a |
| `mixed` | 1 | 0.1667 | 0.3333 | 0.00 | 0.3333 | 0.4000 |
| **overall** | 4 | 0.1028 | 0.2105 | 0.50 | 0.5333 | 0.6154 |

Denominators: **107 reference characters, 19 reference words**.
Diacritics: **TP=8, FP=3, FN=7**.

### The finding that matters most

| View | CER | "Accuracy" |
|---|---|---|
| Consonants only (tashkeel stripped from both sides) | 0.0435 | **95.7%** |
| **With tashkeel preserved** | 0.1028 | **89.7%** |

On the same images, the same run, the same model:

> Strip the diacritics and the pipeline **meets** the 95%
> character-accuracy target. Keep them and it **does not**.

This is precisely the trap `arabic_diacritics.md` §2 describes, now
demonstrated with real output rather than argued in the abstract. Any
report that quotes the first number without the second is describing a
system that cannot tell a child how to pronounce a word.

### Secondary observations

1. **Unvowelled Arabic and English are read perfectly** (CER 0.0000)
   on clean synthetic text. The baseline works.
2. **Vowelled Arabic costs consonant accuracy too.** `arabic_plain`
   scored CER 0.0000; the same sentences with tashkeel scored
   consonant errors as well as diacritic errors. The marks do not just
   go unrecognised — they interfere with letter recognition.
   Example: reference `قَرَأَ الطِفْلُ` was read as `فَرَا الطِلُ` —
   `ق`→`ف`, and `ف` dropped entirely.
3. **Confidence drops on vowelled text**: ~0.92 unvowelled vs ~0.63–0.75
   vowelled for the same words. The model is measurably less certain,
   which is at least an honest signal.
4. **Diacritic placement is unstable.** `الوَلَدُ` came back as
   `َالولدُ` — marks present but attached to the wrong letters. Position
   matters for pronunciation, and the position-based metric catches
   this where a bag-of-marks count would not.
5. **Mixed mode routed correctly** on separate lines: `الكِتَابُ` to the
   Arabic model, `The Book` to the English model.
6. **Upscaling did not recover diacritics** when they were absent from
   the image (`--upscale 3.0` on the pre-fix image changed nothing) —
   consistent with the marks genuinely not being there, and an
   indication that `--upscale` is not a substitute for input quality.

---

## 4. Verified operational facts

| Fact | Evidence |
|---|---|
| GPU inference works | Real kernel executed; 256×256 matmul numerically correct. RTX 3050 Ti, compute 8.6, CUDA runtime 11.8 on driver API 12.3. |
| CPU fallback works | Verified after disabling oneDNN (below). |
| GPU ≈ 5× faster than CPU | 0.686 s vs 3.449 s on the same image. |
| Offline inference works | `verify_offline.py` ran inference with `socket.connect`/`create_connection` blocked. Verdict **PASS, zero connection attempts**. |
| Models load from local disk | Log shows explicit local paths for both det and rec models. |
| Models are small | 20.2 MB total for all three. |
| Word-level boxes | **Not available.** Detection returns line/region polygons, labelled `text_region`. |

### Two upstream issues found and worked around

**oneDNN crashes CPU inference.** paddlepaddle-gpu 3.3.1 on CPU fails
with `ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]`. Every CPU run errored
out. `enable_mkldnn` now defaults to `False`, which fixes it at the
cost of slower CPU inference. `--enable-mkldnn` re-enables it for
builds that are unaffected.

**`lang` and `ocr_version` are ignored when explicit model names are
passed.** PaddleOCR warns about this. Both arguments were removed; the
explicit model names are the stronger guarantee anyway, since newer
PaddleOCR releases default to PP-OCRv6 and would otherwise silently
change which weights are used.

---

## 5. What to do next

In priority order:

1. **Replace these synthetic images with photographed pages.** Nothing
   here generalises to camera input. This is the single highest-value
   next step.
2. **Measure diacritic recall properly** — 50+ vowelled pages, same
   sentences labelled with and without tashkeel so the gap is
   attributable.
3. **Investigate the interference effect** (observation 2). If vowelled
   text systematically degrades consonant recognition, that shapes
   whether fine-tuning is needed.
4. **Find the resolution floor** at which a fatha stops being
   resolvable. This sets a hard requirement on the camera rig.
5. **Decide on word-level localisation** — options in
   `architecture.md`. The projector depends on it and nothing in this
   baseline provides it.
6. **Re-assess the 95% target** against real data, reported with
   tashkeel preserved.

---

## 6. Reproducing

```bat
python scripts\check_environment.py
python scripts\download_models.py
python scripts\verify_offline.py
python -m pytest tests\ -q
python scripts\evaluate.py --ground-truth data\ground_truth\synthetic_smoke.jsonl --device gpu --strip-tashkeel-view
```

Regenerate the synthetic images with:

```bat
python scripts\make_synthetic_samples.py
```

That script aborts if shaping loses a diacritic, so the bug in §2
cannot silently recur. `data/ground_truth/synthetic_smoke.jsonl` labels
the images and marks every row `SYNTHETIC`.

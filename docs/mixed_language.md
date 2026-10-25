# Mixed Arabic–English pages

> **Status:** investigated with measurements; the recommendation below
> **changes** the earlier design. Evidence is synthetic and small
> (n=2–3 images), so it is a lead, not a settled result.

---

## 1. The earlier design, and why it is now in doubt

`--lang mixed` runs the Arabic pipeline and the English pipeline over
the same page, matches regions by IoU, and picks a winner per region
with `choose_mixed_candidate()`.

That arbitration has a problem documented in
`postprocessing.choose_mixed_candidate`: when both models return text
in their own script, nothing in the output distinguishes "this region
really is Arabic" from "the Arabic model hallucinated Arabic for a
Latin region". Confidence does not separate them either, because
scores from two different models are not calibrated against each
other.

So the mode costs ~1.6× the latency, holds two models in memory, and
resolves the genuinely ambiguous case by heuristic.

---

## 2. What the measurements show

### The Arabic model reads English too

Its charset includes Latin letters, digits and punctuation. Tested on
rendered English pages:

| Image | Pipeline | CER |
|---|---|---|
| `loc_en.png` | `en` | **0.0000** |
| `loc_en.png` | `ar` | **0.0000** |
| `loc_en_punct.png` | `en` | 0.1111 |
| `loc_en_punct.png` | `ar` | 0.1667 |

On clean English the Arabic pipeline matched the English pipeline
exactly. With punctuation it was slightly worse. (Both models inserted
a spurious digit on the punctuation page — a shared failure, not an
Arabic-model one.)

### Detection already separates scripts

A single line reading `The Book الكِتابُ` was split by the **detector**
into three regions — `The`, `Book`, `الكِتابُ` — because the scripts are
separated by normal word spacing. Recognition then ran per region:

| Pipeline | `The` | `Book` | `الكِتابُ` |
|---|---|---|---|
| `ar` | 1.000 ✓ | 0.999 ✓ | 0.789 ✓ (`الكَيتابُ`, minor error) |
| `en` | 1.000 ✓ | 1.000 ✓ | **0.388 ✗** (`U`) |

The Arabic pipeline alone read the entire mixed line. The English
pipeline failed the Arabic region — and **flagged it with low
confidence (0.388)**, which is a usable signal.

The earlier architecture note said a single line containing both
scripts would be recognised by one model and the other script garbled.
That is true in the sense that one *model* handles the line — but the
pessimism was misplaced, because detection splits the line into
per-script regions first, and the Arabic model covers both charsets.

---

## 3. Recommendation

**For mixed pages, prefer `--lang ar` over `--lang mixed`.**

| | `--lang ar` | `--lang mixed` |
|---|---|---|
| Latency | 0.306 s | 0.498 s |
| Peak VRAM | 676 MiB | 862 MiB |
| Arbitration heuristic | none needed | required, and unreliable |
| English accuracy | 0.0000 CER (clean) | same or worse |
| Word localisation | supported | **rejected by the CLI** |

`--lang mixed` is kept because it is implemented, tested and may still
win on English-heavy pages where the English model's punctuation
handling matters. It is no longer the default advice.

### What would change this

- Real photographs, where the Arabic model's Latin handling may
  degrade differently from the English model's.
- English-dominant pages, which the n=2 test above does not cover.
- A line where the two scripts are **not** space-separated (e.g.
  `كتاب(book)` with no space). Detection would not split that, and it
  is untested.

---

## 4. Honest limitations

1. **n=2 rendered English images and one mixed line.** This is a
   direction, not a conclusion. `evaluate.py` on real labelled pages
   is what would settle it.
2. **The arbitration weakness in `mixed` mode is unfixed.** It is
   documented in code, has an asserting test for the failure case, and
   both candidates are preserved in the JSON under `mixed_candidates`
   so no reading is lost.
3. **Low confidence is a signal, not a guarantee.** The 0.388 on the
   Arabic region was informative, but confidence is not calibrated and
   must not be converted into a probability of correctness.
4. **No script-level routing within a region.** If detection ever
   returns one region containing both scripts, whichever model runs
   will mangle part of it.
5. **RTL/LTR ordering on a genuinely bidirectional line** is untested.
   Region reading order uses the page's dominant script, which is a
   simplification for a line that mixes directions.

---

## 5. Reproducing

```bat
python scripts\run_ocr.py --image data\input\test_mixed_sameline.png --lang ar --device gpu
python scripts\run_ocr.py --image data\input\test_mixed_sameline.png --lang en --device gpu
python scripts\run_ocr.py --image data\input\test_mixed.png --lang mixed --device gpu
python scripts\measure_resources.py --configs ar:gpu en:gpu mixed:gpu
```

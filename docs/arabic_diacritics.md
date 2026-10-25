# Arabic diacritics and OCR

> **Written for:** the Phonics Prism team — engineers and the Arabic
> literacy specialists who will read the evaluation reports.

This is the hardest part of the project and the easiest part to get
quietly wrong. The failure mode is not a crash. It is a pipeline that
reports 97% accuracy while being unable to tell a child how to
pronounce a single word.

---

## 1. Two different recognition problems

Arabic text has two layers:

**The consonantal skeleton (rasm).** The letters: `ك ت ب`.

**The vocalisation (tashkeel).** Small marks above and below the
letters that carry the short vowels: `كَتَبَ`, `كُتِبَ`, `كُتُب`.

Standard OCR benchmarks measure the first. Phonics Prism needs the
second.

`كتب` is three characters. Depending on the vowelling it reads as:

| Written | Pronounced | Meaning |
|---|---|---|
| `كَتَبَ` | kataba | he wrote |
| `كُتِبَ` | kutiba | it was written |
| `كُتُب` | kutub | books |

All three share an identical consonantal skeleton. A model that outputs
`كتب` has produced a **correct unvowelled transcription** and has told
us **nothing about pronunciation**.

For a general-purpose OCR system that is fine — a human reader supplies
the vowels from context. For a system that tells a dyslexic child how
to sound out a word, it is the whole problem. We cannot supply the
vowels from context, because supplying the vowels is the job.

### The consequence for how we report accuracy

> A correct unvowelled word does **not** mean the pronunciation is
> correct. It means the pronunciation is **unknown**.

Any CER figure computed after stripping diacritics measures the first
layer only. This project therefore reports the vowelled score as
primary, and the consonants-only score as an explicitly-labelled
secondary view (`--strip-tashkeel-view`), never on its own.

---

## 2. How normalisation silently destroys tashkeel

Arabic text pipelines routinely include a "normalisation" step. Typical
published Arabic NLP preprocessing does some or all of:

```python
text = re.sub(r'[ً-ْ]', '', text)   # remove tashkeel
text = re.sub('[إأآا]', 'ا', text)             # fold alif variants
text = text.replace('ة', 'ه')                  # fold taa marbuta
```

Every one of those lines is standard practice for Arabic search and
classification, where vowels are noise. Every one of them is
catastrophic here. The first deletes the signal we are measuring. The
second erases the hamza, which is a real consonant (a glottal stop).
The third merges two letters a child must learn to distinguish.

The danger is that applying them *improves* the headline number. Strip
tashkeel from both reference and hypothesis and CER drops, because the
hardest characters have been removed from the denominator. The system
looks better and does less.

### What this project does instead

Three rules, enforced in code:

1. **Raw text is never modified.** Every result record stores
   `text_raw`, exactly the bytes the model returned. Normalised views
   live in separate keys (`text_normalised`).
   — `src/phonics_ocr/engine.py`

2. **Every lossy fold is opt-in and named.** `normalise_text()` defaults
   to NFC + tatweel removal + whitespace collapsing. `remove_tashkeel`,
   `fold_alif_variants`, `fold_taa`, `fold_maqsura` all default to
   `False` and must be requested explicitly.
   — `src/phonics_ocr/postprocessing.py`

3. **The evaluation report states which folds were applied**, next to
   the score they produced. A number without that context is not a
   result.
   — `scripts/evaluate.py`

There is a regression test for rule 1 specifically
(`tests/test_postprocessing.py::TestDiacriticPreservation`), because
this is the kind of bug that gets reintroduced by a well-meaning
refactor.

---

## 3. Why tashkeel are combining characters

Tashkeel are **combining marks**: separate Unicode code points that
render attached to the preceding base letter.

`كَتَبَ` is six code points, not three:

| # | Code point | Name | Renders as |
|---|---|---|---|
| 1 | U+0643 | ARABIC LETTER KAF | ك |
| 2 | U+064E | ARABIC FATHA | ◌َ |
| 3 | U+062A | ARABIC LETTER TEH | ت |
| 4 | U+064E | ARABIC FATHA | ◌َ |
| 5 | U+0628 | ARABIC LETTER BEH | ب |
| 6 | U+064E | ARABIC FATHA | ◌َ |

Practical consequences:

- **`len()` is not what you see.** `len("كَتَبَ") == 6`. Any code that
  assumes one code point per visible glyph will mis-slice Arabic.
- **CER denominators change.** Scoring over code points,
  `كَتَبَ` → `كتب` is 3 deletions out of 6 — a CER of 0.5, not 0.0.
  That is the correct and intended behaviour here.
- **Marks are invisible when missing.** A dropped fatha produces a
  string that still looks like valid Arabic. Only a comparison against
  a vowelled reference reveals it.

The main diacritic ranges:

| Range | Contents |
|---|---|
| U+064B–U+0652 | Tanween, the short vowels, shadda, sukun |
| U+0653–U+0655 | Maddah, hamza above, hamza below |
| U+0670 | Superscript (dagger) alif |
| U+06D6–U+06ED | Quranic annotation marks |

`src/phonics_ocr/postprocessing.py::TASHKEEL` enumerates these
explicitly rather than using a range regex, so that adding or excluding
a mark is a visible, reviewable change.

---

## 4. Choosing a Unicode normalisation form

Four normalisation forms exist. The choice matters.

| Form | Effect on Arabic | Verdict |
|---|---|---|
| **NFC** | Composes `alif + combining hamza above` (U+0627 U+0654) into U+0623. Leaves fatha, damma, kasra, shadda untouched — they have no precomposed form. | **Used as the project default.** |
| NFD | Decomposes U+0623 back into two code points. Harmless but inflates code-point counts. | Not used. |
| NFKC | *Compatibility* composition. Rewrites Arabic presentation forms (U+FE70–U+FEFF, U+FB50–U+FDFF) into base letters and expands ligatures such as lam-alif. | **Rejected as a default.** |
| NFKD | As NFKC, decomposed. | Not used. |

NFC is chosen because two transcriptions that look identical should
compare equal. Without it, a reference typed as U+0623 and a hypothesis
emitted as U+0627 U+0654 render identically and score as an error.

NFKC is rejected because it silently changes which characters are being
measured. If a model emits presentation-form glyphs, that is a fact
about the model worth knowing, not something to normalise away before
anyone sees it.

**NFC does not remove diacritics.** This is load-bearing and has its own
test (`test_nfc_does_not_strip_combining_vowel_marks`). If it ever
stopped holding, every diacritic metric in the project would silently
read zero.

### Mark ordering

Unicode does not reorder combining marks within the same combining
class during NFC. `shadda + fatha` and `fatha + shadda` stay distinct
sequences even though they render the same. Label consistently — see
`data/ground_truth/README.md`.

---

## 5. Visually similar characters that need their own tests

Beyond the vowel marks, a set of Arabic characters differ by a dot or a
hamza. These are exactly the distinctions a dyslexic reader struggles
with, and exactly the ones a recognition model is most likely to drop.

| Class | Characters | What is at stake |
|---|---|---|
| **Hamza** | ء أ إ ؤ ئ آ | A consonant (glottal stop). `أكل` (he ate) vs `اكل`. Losing it changes the word. |
| **Alif variants** | ا أ إ آ ٱ | Differ only by hamza/maddah placement. Carry vowel and glottal information. |
| **Taa marbuta / haa** | ة vs ه | Identical but for two dots. Usually a feminine ending; affects pronunciation and grammar. |
| **Alif maqsura / yaa** | ى vs ي | Identical but for two dots. `على` (on) vs `علي` (Ali). |
| **Shadda** | ◌ّ | Gemination — the consonant is doubled and held. `درس` vs `درّس` are different verbs. |
| **Tanween** | ◌ً ◌ٌ ◌ٍ | Case endings, pronounced `-an`, `-un`, `-in`. |

`evaluation.confusable_error_counts()` reports a per-class count
mismatch for each of these.

**What that metric is and is not.** It compares *how many* characters of
each class appear in the reference versus the hypothesis. It does not
align them positionally. So it answers "did the model produce the right
number of hamzas?" and not "did it put them in the right places". A
reading where one hamza is dropped and a spurious one is added
elsewhere scores zero mismatch. It is a diagnostic pointer toward which
classes to look at, not a standalone accuracy figure, and the report
labels it as such.

---

## 6. How diacritics are scored

`evaluation.diacritic_prf()` treats each diacritic as a pair:

```
(index of its base character in the tashkeel-stripped string, the mark)
```

`كَتَبَ` becomes `[(0, fatha), (1, fatha), (2, fatha)]`.

A hypothesis mark is a true positive only if the same mark sits at the
same base index. From that: precision, recall, F1.

### What the numbers mean in the cases that matter

| Situation | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Perfect vowelling | 3 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| **Model outputs no tashkeel at all** | 0 | 0 | 3 | **undefined** | **0.0** | **undefined** |
| Model invents tashkeel | 0 | 3 | 0 | 0.0 | undefined | undefined |
| Right mark, wrong letter | 0 | 1 | 1 | 0.0 | 0.0 | undefined |
| Wrong mark, right letter | 0 | 1 | 1 | 0.0 | 0.0 | undefined |
| 2 of 3 marks correct | 2 | 1 | 0 | 1.0 | 0.67 | 0.80 |

The second row is the important one. **A model that never outputs a
diacritic scores recall 0.** It does not score well by omission, and it
does not score `None` on recall — only precision is undefined, because
it genuinely made no predictions. This is the single most likely
outcome for an off-the-shelf Arabic OCR model, and the metric is built
so that outcome is impossible to mistake for success.

### The honest limitation

Position-based matching is brittle to consonant errors. If the
hypothesis drops or adds a **letter**, every later base index shifts and
all subsequent diacritics are scored wrong — even ones the model read
correctly.

This is a real weakness, and it is why the diacritic metric is always
reported **alongside** the full vowelled CER rather than instead of it.
When the two disagree sharply — low CER but near-zero diacritic F1 — the
cause is usually an index shift from one missing consonant, and the
per-sample records in the report let you check.

---

## 7. What this project will not do

**No automatic diacritisation.** There are good tools (Mishkal,
Farasa, neural diacritisers) that add tashkeel to unvowelled Arabic
using a language model. None of them is in this pipeline, and none of
their output may be reported as OCR output.

The reason: such a tool predicts the most probable vowelling given
context. A child learning to read needs the vowelling that is
**printed on the page in front of them**. A statistical guess that
disagrees with the page teaches the wrong pronunciation, and it would
arrive with the same confidence as a correct reading.

If diacritisation is ever added, it belongs in a clearly separate
field, produced by a clearly separate stage, with its own accuracy
figures, never merged into `text_raw`.

**No scoring that discards diacritics by default.** The
consonants-only view exists, is useful for diagnosis, and is labelled
`SECONDARY VIEW` in every report that contains it.

---

## 8. Open questions this baseline is meant to answer

None of these is settled, and the code is written so that the answers
come from measurement rather than assumption:

1. ~~**Does `arabic_PP-OCRv5_mobile_rec` have tashkeel in its character
   set at all?**~~ **ANSWERED: yes.** The shipped charset holds 747
   characters, **38 of them diacritics**, including all eight classical
   harakat. Diacritic recognition is not structurally excluded. See
   [`baseline_findings.md`](baseline_findings.md) §1.
2. **What recall do they get** on clean print, and how does it degrade
   with resolution? *Partially answered:* **0.58 recall on a synthetic
   smoke test (n=1, TP=8 FP=3 FN=7)** — real, but far too small a
   sample to generalise. Resolution sensitivity is untested.
3. **At what pixel height does a fatha stop being resolvable?** This
   sets a hard minimum for the camera rig.
4. **Does `--upscale` improve diacritic recall**, and past what factor
   does it stop helping?
5. **Which confusable classes fail most?** That determines whether
   fine-tuning is worth it and on what data.

Question 1 gated the rest, and it came back **yes** — so the remaining
questions are worth real effort rather than being moot.

A sixth question was added by the smoke test: **does vowelling degrade
consonant recognition?** On the synthetic sample, the same sentences
scored CER 0.0000 unvowelled and had consonant errors once tashkeel
were present (`قَرَأَ` → `فَرَا`). If that holds on real pages, the marks
are not merely unread — they actively interfere — and fine-tuning
becomes more attractive than post-processing.

### A practical warning for anyone extending this

The single most dangerous bug class in this area is **tooling that
deletes diacritics by default**. `arabic-reshaper` does
(`delete_harakat=True`), and it silently produced a wrong experimental
result in this very project before it was caught — see
[`baseline_findings.md`](baseline_findings.md) §2. The failure is
quiet, and it makes the numbers look *better*. Check the data at each
stage, not just the output.

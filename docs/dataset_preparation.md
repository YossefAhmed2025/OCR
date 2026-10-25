# Dataset preparation guide

> **Written for:** whoever captures and labels the Phonics Prism test
> pages — an engineer, a teacher, or an Arabic literacy specialist.
> No command-line experience is assumed beyond copy-paste.

This is the bottleneck. Everything else in the repository is built and
tested; what it lacks is **real photographed pages with correct
labels**. Until those exist, no accuracy figure here describes the real
product.

---

## 1. What you are making

A folder of page images, plus one text file listing what each image
says. That text file is the "ground truth".

```
data/
  input/                  <- your images go here
    ar_photo_001.jpg
    ar_photo_002.jpg
  ground_truth/
    my_labels.jsonl       <- one line per image
```

Start from the template:

```bat
copy data\ground_truth\template_v2.jsonl data\ground_truth\my_labels.jsonl
```

Then replace every `TEMPLATE ROW`. The template rows point at images
that do not exist; they are there to show the shape, not to be scored.

---

## 2. Capturing the photographs

**Photograph real printed pages with a real camera.** Not screenshots,
not PDF exports, not rendered text. The system will face a camera
pointed at a book, so the test set must be camera images of books.

### Minimum viable set

If you can only do a little, do this:

| Pages | Condition |
|---|---|
| 10 | Arabic **with** tashkeel, clean and flat |
| 10 | The *same sentences* without tashkeel |
| 10 | English, clean and flat |
| 10 | Arabic with tashkeel, realistic conditions (held book, room light, slight angle) |

40 pages is enough to find gross problems and to measure the
tashkeel gap, which is the single most important number for this
project. It is **not** enough to certify a 95% target — see
`evaluation_protocol.md` §6.

The "same sentences with and without tashkeel" pairing is worth the
extra effort: it is the only way to attribute an error to diacritics
rather than to general difficulty.

### Capture settings

- **Highest resolution the camera offers.** Diacritics need pixels. If
  a fatha is under about 4 px tall it is unlikely to survive.
- **Fill the frame with the page.** Do not photograph a page from
  across the room and crop later.
- **Camera roughly parallel to the page.** `--deskew` fixes small
  in-plane rotation; it does not fix perspective.
- **No flash on glossy paper.** The specular highlight erases a whole
  region.
- **Do not edit the photos.** No sharpening, no auto-enhance, no
  filters. Feed the original; the pipeline has its own preprocessing
  flags so that any enhancement is recorded and measurable.

### For the robustness experiments

Shoot the *same page* under several conditions so results are
comparable. See `camera_experiments.md` for the full matrix.

---

## 3. Writing the labels

One JSON object per line. Minimum:

```json
{"id": "ar_photo_001", "image": "../input/ar_photo_001.jpg", "lang": "ar", "group": "arabic_tashkeel", "capture": "photograph", "text": "كَتَبَ الوَلَدُ"}
```

### Fields

| Field | Required | Meaning |
|---|---|---|
| `image` | **yes** | Path to the image, relative to the label file |
| `text` | **yes** | Exact transcription of what is printed |
| `lang` | **yes** | `ar`, `en`, or `mixed` |
| `capture` | **strongly** | `photograph`, `flatbed_scan`, `screenshot`, `synthetic_render` |
| `id` | no | Defaults to the filename |
| `group` | no | Evaluation group (below) |
| `width`, `height` | no | Image pixel size; validated against the file |
| `words` | no | Word boxes — see §5 |
| `lines` | no | Line polygons |
| `conditions` | no | Lighting, blur, perspective, curvature, print size |
| `notes` | no | Free text, carried into reports |

### `capture` matters more than it looks

It decides whether a score may be called real-world performance.
Only `photograph` counts. Leave it out and it defaults to `unknown`,
which is also **not** counted — the default is deliberately
conservative so that an unlabelled set cannot claim real-world
evidence by omission.

A flatbed scan is clean, flat and evenly lit. It is useful as an upper
bound, and it is not what a camera produces.

### Groups

`arabic_tashkeel`, `arabic_plain`, `english`, `mixed`, `clean`,
`degraded`. Results are reported per group and never pooled into one
headline number.

---

## 4. Transcribing: the rules that matter

**Transcribe what is printed. Not what it should say.**

1. **Keep every diacritic that is printed.** `كَتَبَ` and `كتب` are
   different labels. Do not normalise one into the other.
2. **Do not add diacritics that are not printed.** If the page is
   partially vowelled, label exactly that.
3. **Logical order, not visual order.** Type the Arabic normally in a
   normal editor. Do not copy text out of a PDF viewer that has
   already applied the bidirectional algorithm — that produces
   reversed text that *looks* right and is wrong.
4. **Be consistent about mark order.** `shadda + fatha` and
   `fatha + shadda` are different code-point sequences that render
   identically. NFC does not reorder them. Pick one and stick to it.
5. **Join lines with a single space**, or label line by line — but be
   consistent, because whitespace affects WER.
6. **Save as UTF-8.** On Windows, Notepad's "ANSI" default corrupts
   Arabic. Use VS Code, or *Save As → Encoding: UTF-8*.

### Why this is strict

The whole point of the project is reading diacritics correctly so a
child is taught the right pronunciation. If the labels lose them, the
evaluation silently measures the wrong thing and the score goes *up*.
`arabic_diacritics.md` §2 has the full argument, including a case
where this exact failure produced a wrong conclusion in this
repository.

---

## 5. Word boxes (optional, for the projector work)

Only needed to evaluate **word-level localisation**. Transcription-only
labels remain perfectly valid without them.

```json
"words": [
  {"text": "كَتَبَ",      "bbox": [2100, 880, 2460, 1010], "line": 0},
  {"text": "الوَلَدُ", "bbox": [1680, 880, 2060, 1010], "line": 0}
]
```

- `bbox` is `[x_min, y_min, x_max, y_max]` in **image pixels**, origin
  top-left.
- List words in **logical reading order**, so that joining their
  `text` fields reproduces the `text` field of the record. The
  validator enforces this.
- For Arabic that means the right-most word on a line comes **first**.
  Its box still has the larger x — record order and geometry are
  independent.
- Draw the box around the **ink**, including diacritics above and
  below. A box that clips the tashkeel is wrong for this project.
- Add `"box_source": "manual_annotation"` so a reader knows the boxes
  were drawn by hand rather than generated.

Any tool that exports rectangles works (LabelImg, CVAT, Label Studio);
convert its output to the fields above.

---

## 6. Validate before evaluating

```bat
python scripts\validate_dataset.py --ground-truth data\ground_truth\my_labels.jsonl
```

This checks the schema, opens every image, verifies declared
dimensions, confirms word boxes fall inside the image, and confirms the
word list matches the transcription. It then prints a provenance
summary telling you plainly whether this set can support a real-world
claim.

Fix every error before running an evaluation. A box one pixel outside
the image, or a word list in the wrong order, will quietly corrupt
every IoU score computed from it.

---

## 7. Run the evaluation

Transcription accuracy (CER, WER, diacritics):

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\my_labels.jsonl --strip-tashkeel-view
```

Word localisation (needs `words`):

```bat
python scripts\evaluate_localization.py --ground-truth data\ground_truth\my_labels.jsonl
```

Read the results with `evaluation_protocol.md` open. In particular,
check the sample counts and denominators before quoting anything.

---

## 8. Common mistakes

| Mistake | Consequence |
|---|---|
| Labelling from memory rather than the page | Measures the labeller, not the OCR |
| Dropping tashkeel "to keep it simple" | Deletes the signal; score rises; project goal unmeasured |
| Copying Arabic out of a PDF viewer | Reversed text; everything scores as wrong |
| Saving as ANSI | Mojibake |
| `capture` left blank on real photos | Counted as `unknown`; real-world claim not credited |
| Word list in visual order | Validator rejects it (correctly) |
| Tuning thresholds on the same images you report | Numbers are not held-out performance |
| Reporting one pooled number | Hides that Arabic-with-tashkeel is the hard case |

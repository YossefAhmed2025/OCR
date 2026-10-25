# Ground truth

Labels live here as JSONL — one JSON object per line, UTF-8, no BOM.

**`example.jsonl` contains template rows, not real data.** Every row is
marked `TEMPLATE ROW` and points at an image that does not exist. It
shows the schema. Delete it or replace it before evaluating; it exists
so you can see the format, not so you can score against it.

No accuracy figure in this project comes from invented labels.

## Schema

| Field | Required | Meaning |
|---|---|---|
| `id` | no | Unique label for the sample. Defaults to the image filename stem. |
| `image` | **yes** | Path to the image. Relative paths resolve against this directory, then the project root. |
| `text` | **yes** | The correct transcription, exactly as printed. |
| `lang` | **yes** | `ar`, `en` or `mixed`. Selects the pipeline used for this sample. |
| `group` | no | Evaluation group (below). Defaults to `<lang>_ungrouped`. |
| `notes` | no | Free text; carried into the report. |

## Evaluation groups

Report rows are broken out by `group`, so results for vowelled Arabic
never get averaged into results for clean English:

| Group | Contents |
|---|---|
| `arabic_tashkeel` | Arabic **with** diacritics |
| `arabic_plain` | Arabic **without** diacritics |
| `english` | English only |
| `mixed` | Both scripts on one page |
| `clean` | High-quality, flat, evenly lit scans |
| `degraded` | Blur, perspective, shadow, low contrast |

A sample belongs to exactly one group. To measure both "is it Arabic
with tashkeel?" and "is it degraded?", make two passes over the same
images in separate JSONL files, or pick the axis that matters for the
question being asked.

## How to label your own images

1. Put the page image in `data/input/`.
2. Transcribe what is **printed on the page**, not what you think it
   should say. Keep every diacritic that is actually printed; do not add
   any that is not.
3. Transcribe in **logical order** — the order the letters are typed and
   read, not the order they appear left-to-right on screen. Any normal
   Arabic text editor produces logical order; this is only a problem if
   you copy text out of a PDF viewer that has already applied the
   bidirectional algorithm.
4. Join the lines of a multi-line page with a single space, or keep one
   line per JSONL record if you are labelling line by line. Be
   consistent: whitespace affects WER.
5. Add one line to your `.jsonl`.

### Diacritics: the thing to get right

The point of this dataset is to measure whether the model reads
tashkeel. That only works if the labels carry them.

- `كَتَبَ` (with fatha on each consonant) and `كتب` (bare) are **different
  labels**. Do not normalise one into the other.
- If a page prints partial vowelling, label exactly the partial
  vowelling. Do not complete it.
- Writing `shadda + fatha` in that order vs `fatha + shadda` produces
  different code-point sequences. Unicode NFC does not reorder them.
  Pick one order and stay consistent; the evaluator reports diacritic
  position matches, so an inconsistent order shows up as errors.

### Encoding

Save as UTF-8. On Windows, Notepad's "ANSI" default will corrupt Arabic —
use VS Code, or Notepad with *Save As → Encoding: UTF-8*.

The evaluator applies NFC normalisation to both sides before scoring, so
small composition differences (bare alif + combining hamza vs
precomposed alif-hamza) will not be counted as errors. Diacritics are
**not** stripped.

## Running an evaluation

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\my_labels.jsonl
```

Add `--strip-tashkeel-view` to also get a secondary consonants-only
score alongside the primary vowelled one:

```bat
python scripts\evaluate.py --ground-truth data\ground_truth\my_labels.jsonl --strip-tashkeel-view
```

See `docs/evaluation_protocol.md` for what the numbers mean and how
many samples you need before a figure is worth quoting.

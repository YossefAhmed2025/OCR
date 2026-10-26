"""Extract a clean vowelled-word lexicon from the Tashkeela corpus.

    python scripts/build_vowelled_lexicon.py
    python scripts/build_vowelled_lexicon.py --max-words 150000

Why this replaces random vowelization
--------------------------------------
The first fine-tune doubled diacritic recall (0.188 -> 0.344) but made
CER worse, and the cause was in the data: ``synth.vowelize`` assigns
harakat at random, so the same word appeared as كِتِبٌ, كَتَبُ and
كِتُبُ across samples. Only about 41% of generated words carried
linguistically correct vowelling, so the model learned *where* marks
go without learning *which* mark belongs.

This pulls real, fully vowelled words out of a real corpus instead.

Corpus and licence
------------------
Tashkeela (Zerrouki & Balla, Data in Brief 11, 2017), ~75M vocalized
words from 97 books, distributed under **GPL-2.0**.

GPL-2 is a copyleft licence. For a research baseline that is fine. For
a shipped commercial product -- which Phonics Prism is, with a patent
behind it -- the obligations need checking with whoever owns the IP
before any model trained on it goes into a device. The lexicon written
here records its source and licence so that question cannot be lost.

Quality filtering
-----------------
Published work on Tashkeela reports missing diacritics, more than two
marks on one character, inconsistent Unicode and stray foreign
characters. Every word kept here must therefore:

* be Arabic letters and harakat only,
* carry a diacritic on a reasonable share of its letters,
* have no letter bearing more than two marks,
* be short enough to fit the CTC timestep budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CORPUS_LICENCE = "GPL-2.0"
CORPUS_CITATION = (
    "Zerrouki, T. & Balla, A. (2017). Tashkeela: Novel corpus of Arabic "
    "vocalized texts. Data in Brief 11. doi:10.1016/j.dib.2017.01.011"
)


def iter_corpus_text(path: Path, max_books: int) -> Iterator[str]:
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    seen = 0
    for batch in handle.iter_batches(batch_size=1, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if not text:
                continue
            yield text
            seen += 1
            if seen >= max_books:
                return


def clean_words(
    text: str,
    *,
    min_letters: int,
    max_letters: int,
    min_mark_ratio: float,
) -> Iterator[str]:
    """Yield fully-vowelled words that pass the quality filters."""
    from phonics_ocr import postprocessing as pp

    for raw in text.split():
        word = raw.strip("،؛؟.,;:!?()[]{}\"'«»-—/\\|*")
        if not word:
            continue

        letters = 0
        marks = 0
        marks_on_current = 0
        bad = False
        for ch in word:
            if pp.is_tashkeel(ch):
                marks += 1
                marks_on_current += 1
                # Shadda plus a vowel is legal; three marks is corruption.
                if marks_on_current > 2:
                    bad = True
                    break
            elif pp.is_arabic_char(ch):
                letters += 1
                marks_on_current = 0
            else:
                bad = True  # latin, digits, punctuation inside the word
                break
        if bad or not (min_letters <= letters <= max_letters):
            continue
        if marks == 0 or marks / letters < min_mark_ratio:
            continue
        yield word


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "tashkeela_part0.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "vowelled_lexicon.json",
    )
    parser.add_argument("--max-words", type=int, default=120000)
    parser.add_argument("--max-books", type=int, default=33)
    parser.add_argument("--min-letters", type=int, default=2)
    parser.add_argument(
        "--max-letters",
        type=int,
        default=9,
        help="long words crowd out variety and strain the CTC budget",
    )
    parser.add_argument(
        "--min-mark-ratio",
        type=float,
        default=0.5,
        help="minimum diacritics per letter; rejects partly-vowelled words",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="a word must appear at least this often (drops typos)",
    )
    args = parser.parse_args()

    from phonics_ocr import postprocessing as pp
    from phonics_ocr.utils import iso_now, write_json

    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        print(
            "       Download a Tashkeela shard, e.g.\n"
            "       https://huggingface.co/datasets/community-datasets/tashkeela",
            file=sys.stderr,
        )
        return 1

    print(f"Reading {args.corpus.name} ...")
    counts: Counter = Counter()
    books = 0
    for text in iter_corpus_text(args.corpus, args.max_books):
        books += 1
        counts.update(
            clean_words(
                text,
                min_letters=args.min_letters,
                max_letters=args.max_letters,
                min_mark_ratio=args.min_mark_ratio,
            )
        )
        print(f"  book {books}: {len(counts)} distinct words so far")
        if len(counts) >= args.max_words * 4:
            print("  enough material; stopping early")
            break

    frequent = [(w, c) for w, c in counts.items() if c >= args.min_count]
    frequent.sort(key=lambda item: -item[1])
    kept = frequent[: args.max_words]

    if not kept:
        print("ERROR: no words survived filtering.", file=sys.stderr)
        return 1

    words = [w for w, _ in kept]
    marks_total = sum(pp.count_tashkeel(w) for w in words)
    letters_total = sum(len(pp.strip_tashkeel(w)) for w in words)
    mark_kinds = Counter(c for w in words for c in w if pp.is_tashkeel(c))

    write_json(
        args.out,
        {
            "generated_at": iso_now(),
            "source": "Tashkeela",
            "source_file": str(args.corpus),
            "licence": CORPUS_LICENCE,
            "citation": CORPUS_CITATION,
            "licence_note": (
                "GPL-2.0 is copyleft. Fine for a research baseline; the "
                "obligations must be reviewed with the IP owner before a "
                "model trained on this ships in a product."
            ),
            "books_read": books,
            "distinct_words_seen": len(counts),
            "words_kept": len(words),
            "min_count": args.min_count,
            "mean_letters": round(letters_total / len(words), 2),
            "mean_marks": round(marks_total / len(words), 2),
            "marks_per_letter": round(marks_total / letters_total, 3),
            "distinct_mark_types": len(mark_kinds),
            "mark_histogram": {k: v for k, v in mark_kinds.most_common()},
            "words": words,
        },
    )

    print(f"\n{'=' * 62}")
    print(f"  books read      : {books}")
    print(f"  distinct words  : {len(counts)}")
    print(f"  kept (freq >={args.min_count}) : {len(words)}")
    print(f"  mean letters    : {letters_total / len(words):.2f}")
    print(f"  mean marks      : {marks_total / len(words):.2f}")
    print(f"  marks per letter: {marks_total / letters_total:.3f}")
    print(f"  distinct marks  : {len(mark_kinds)}")
    print(f"{'=' * 62}")
    print("\n  sample:")
    for word, count in kept[:10]:
        print(f"    {word}   (x{count})")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

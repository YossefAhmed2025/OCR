"""Text post-processing: Arabic-aware normalisation and region merging.

Two rules govern this module.

1. **Raw text is never modified.** Everything here produces *additional*
   representations stored under separate keys. The string the model
   returned is preserved byte-for-byte so that diacritic evaluation has
   something truthful to score against.
2. **Tashkeel are never invented.** No function here adds a diacritic
   that the model did not output. Stripping tashkeel is available, but
   only as an explicitly named, opt-in comparison view.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Sequence, Set, Tuple

# ---------------------------------------------------------------------
# Arabic Unicode inventory
# ---------------------------------------------------------------------

#: Arabic combining marks (tashkeel / harakat) in the Arabic block.
#: U+064B..U+0652 are the classical harakat and tanween; U+0653..U+0655
#: are maddah/hamza-above/hamza-below; U+0670 is the dagger alif;
#: U+06D6..U+06ED are Quranic annotation marks.
TASHKEEL: frozenset = frozenset(
    "ً"  # FATHATAN
    "ٌ"  # DAMMATAN
    "ٍ"  # KASRATAN
    "َ"  # FATHA
    "ُ"  # DAMMA
    "ِ"  # KASRA
    "ّ"  # SHADDA
    "ْ"  # SUKUN
    "ٓ"  # MADDAH ABOVE
    "ٔ"  # HAMZA ABOVE
    "ٕ"  # HAMZA BELOW
    "ٖ"  # SUBSCRIPT ALEF
    "ٗ"  # INVERTED DAMMA
    "٘"  # MARK NOON GHUNNA
    "ٙ"  # ZWARAKAY
    "ٚ"  # VOWEL SIGN SMALL V ABOVE
    "ٛ"  # VOWEL SIGN INVERTED SMALL V ABOVE
    "ٜ"  # VOWEL SIGN DOT BELOW
    "ٝ"  # REVERSED DAMMA
    "ٞ"  # FATHA WITH TWO DOTS
    "ٟ"  # WAVY HAMZA BELOW
    "ٰ"  # SUPERSCRIPT ALEF (dagger alif)
)

#: Quranic annotation signs, kept separate because they are editorial
#: marks rather than vowel marks.
QURANIC_MARKS: frozenset = frozenset(chr(c) for c in range(0x06D6, 0x06ED + 1))

#: Tatweel / kashida -- a purely typographic elongation carrying no
#: phonetic value. Removing it is safe for phonics; it is still only
#: done on request.
TATWEEL: str = "ـ"

#: Arabic-Indic and extended Arabic-Indic digits, mapped to ASCII.
_ARABIC_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}
_EXT_ARABIC_DIGITS = {chr(0x06F0 + i): str(i) for i in range(10)}

#: Alif variants. Folding these destroys a real orthographic (and for
#: hamza, phonemic) distinction, so it is opt-in and reported separately.
ALIF_VARIANTS: frozenset = frozenset("آأإٱ")  # آ أ إ ٱ
ALIF_PLAIN: str = "ا"  # ا

HAMZA_FORMS: frozenset = frozenset(
    "ءآأؤإئٕٔ"  # ء آ أ ؤ إ ئ + combining
)
TAA_MARBUTA: str = "ة"  # ة
HAA: str = "ه"  # ه
ALIF_MAQSURA: str = "ى"  # ى
YAA: str = "ي"  # ي
SHADDA: str = "ّ"
TANWEEN: frozenset = frozenset("ًٌٍ")

#: Character classes tracked individually by the diacritic/confusable
#: error report in :mod:`phonics_ocr.evaluation`.
CONFUSABLE_CLASSES: Dict[str, frozenset] = {
    "hamza": HAMZA_FORMS,
    "alif_variants": ALIF_VARIANTS | {ALIF_PLAIN},
    "taa_marbuta_haa": frozenset({TAA_MARBUTA, HAA}),
    "alif_maqsura_yaa": frozenset({ALIF_MAQSURA, YAA}),
    "shadda": frozenset({SHADDA}),
    "tanween": TANWEEN,
}

_ARABIC_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def is_arabic_char(ch: str) -> bool:
    """Whether ``ch`` belongs to an Arabic Unicode block."""
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ARABIC_RANGES)


def is_latin_char(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def is_tashkeel(ch: str) -> bool:
    return ch in TASHKEEL or ch in QURANIC_MARKS


def count_tashkeel(text: str) -> int:
    return sum(1 for ch in text if is_tashkeel(ch))


def has_tashkeel(text: str) -> bool:
    return any(is_tashkeel(ch) for ch in text)


def strip_tashkeel(text: str) -> str:
    """Remove every diacritic. Opt-in only; never applied to raw text.

    Used to answer "did the model get the consonantal skeleton right?",
    which is a *different* question from "did it get the vowelling
    right?". Both are reported separately.
    """
    return "".join(ch for ch in text if not is_tashkeel(ch))


def strip_tatweel(text: str) -> str:
    return text.replace(TATWEEL, "")


def extract_tashkeel_positions(text: str) -> List[Tuple[int, str]]:
    """Diacritics as ``(base_index, mark)`` pairs.

    ``base_index`` is the index of the preceding base character in the
    *tashkeel-stripped* string, which lets two transcriptions be compared
    position-by-position even when their consonant counts differ.
    A mark with no preceding base character gets index ``-1``.
    """
    pairs: List[Tuple[int, str]] = []
    base_index = -1
    for ch in text:
        if is_tashkeel(ch):
            pairs.append((base_index, ch))
        else:
            base_index += 1
    return pairs


def fold_alif(text: str) -> str:
    """Map every alif variant to bare alif. Destructive; opt-in."""
    return "".join(ALIF_PLAIN if ch in ALIF_VARIANTS else ch for ch in text)


def fold_taa_marbuta(text: str) -> str:
    return text.replace(TAA_MARBUTA, HAA)


def fold_alif_maqsura(text: str) -> str:
    return text.replace(ALIF_MAQSURA, YAA)


def map_arabic_digits(text: str) -> str:
    """Map Arabic-Indic digits to ASCII digits."""
    return "".join(_ARABIC_DIGITS.get(ch, _EXT_ARABIC_DIGITS.get(ch, ch)) for ch in text)


def normalise_unicode(text: str, form: str = "NFC") -> str:
    """Apply a Unicode normalisation form.

    NFC is the project default. It composes sequences such as
    ``alif + combining hamza above`` into the single precomposed
    character ``U+0623``, so that two transcriptions that *look*
    identical compare equal. It does **not** remove diacritics: combining
    marks with no precomposed form (fatha, damma, kasra, shadda, ...)
    survive NFC unchanged.

    NFKC is deliberately *not* the default: it rewrites Arabic
    presentation forms and ligatures, which would silently alter the
    character inventory being measured.
    """
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"Unknown normalisation form {form!r}")
    return unicodedata.normalize(form, text)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends."""
    return " ".join(text.split())


def normalise_text(
    text: str,
    *,
    unicode_form: str = "NFC",
    collapse_whitespace: bool = True,
    remove_tatweel: bool = True,
    remove_tashkeel: bool = False,
    fold_alif_variants: bool = False,
    fold_taa: bool = False,
    fold_maqsura: bool = False,
    arabic_digits_to_ascii: bool = False,
) -> str:
    """Build a normalised *view* of ``text``.

    Defaults are deliberately conservative: NFC, whitespace collapsing
    and tatweel removal only. Every lossy fold -- including tashkeel
    removal -- must be switched on by name, and each one is recorded in
    the evaluation report so a score can never be read without knowing
    which distinctions were discarded to obtain it.
    """
    out = normalise_unicode(text, unicode_form)
    if remove_tatweel:
        out = strip_tatweel(out)
    if remove_tashkeel:
        out = strip_tashkeel(out)
    if fold_alif_variants:
        out = fold_alif(out)
    if fold_taa:
        out = fold_taa_marbuta(out)
    if fold_maqsura:
        out = fold_alif_maqsura(out)
    if arabic_digits_to_ascii:
        out = map_arabic_digits(out)
    if collapse_whitespace:
        out = normalise_whitespace(out)
    return out


# ---------------------------------------------------------------------
# Script detection and mixed-language merging
# ---------------------------------------------------------------------


def script_profile(text: str) -> Dict[str, int]:
    """Count Arabic, Latin, digit and other characters in ``text``."""
    counts = {"arabic": 0, "latin": 0, "digit": 0, "other": 0}
    for ch in text:
        if ch.isspace():
            continue
        if is_arabic_char(ch):
            counts["arabic"] += 1
        elif is_latin_char(ch):
            counts["latin"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        else:
            counts["other"] += 1
    return counts


def dominant_script(text: str) -> str:
    """Return ``"arabic"``, ``"latin"`` or ``"unknown"`` for ``text``.

    Digits and punctuation do not vote: both scripts share them.
    """
    counts = script_profile(text)
    if counts["arabic"] == 0 and counts["latin"] == 0:
        return "unknown"
    return "arabic" if counts["arabic"] >= counts["latin"] else "latin"


def choose_mixed_candidate(
    candidates: Sequence[Tuple[str, str, float]],
) -> Tuple[str, str, float]:
    """Pick between per-language recognitions of the *same* image region.

    ``candidates`` is a sequence of ``(pass_lang, text, score)``.

    The rule rests on an **asymmetry between the two charsets**, not on
    comparing their confidence scores:

    * The English model's charset contains no Arabic characters, so it
      *cannot* emit Arabic script. Arabic-script output from the Arabic
      pass is therefore strong evidence that the region really is
      Arabic.
    * The Arabic model's charset does include Latin letters and digits,
      so Latin output from the English pass is only *weak* evidence --
      the Arabic model could have produced Latin too.

    So: if the Arabic pass read Arabic script, take it. Otherwise the
    region is presumed Latin and the English pass wins. Confidence is
    consulted only as a last resort, because scores from two different
    recognition models are not calibrated against each other and 0.95
    from one does not mean what 0.95 from the other means.

    Candidates that recognised nothing lose to any that read something.

    **Known failure mode.** The Arabic model can also *hallucinate*
    Arabic glyphs for a Latin region, and this rule will believe it. So
    an English word that the Arabic pass misreads as Arabic script will
    be taken over the correct English reading. There is no signal in
    the recognition output alone that separates that case from a
    genuinely Arabic region, and raw confidence does not separate it
    either, because the two models' scores are not comparable.

    Both candidates are therefore kept in the result record under
    ``mixed_candidates`` so the alternative reading is never lost. See
    ``docs/architecture.md``.
    """
    if not candidates:
        raise ValueError("choose_mixed_candidate requires at least one candidate")

    def tier(candidate: Tuple[str, str, float]) -> int:
        pass_lang, text, _ = candidate
        script = dominant_script(text)
        if not text.strip():
            return 0  # read nothing at all
        if pass_lang == "ar" and script == "arabic":
            return 3  # only the Arabic model can produce this
        if pass_lang == "en" and script == "latin":
            return 2  # plausible, but the Arabic model could also do it
        return 1  # a model emitting the other script: likely a misread

    return max(candidates, key=lambda c: (tier(c), c[2], len(c[1].strip())))


def group_regions_into_lines(
    boxes: Sequence[Sequence[float]], line_tolerance: float = 20.0
) -> List[List[int]]:
    """Cluster region indices into reading lines by vertical centre.

    Boxes are sorted by vertical centre and grown greedily: a box joins
    the current line when its centre is within ``line_tolerance`` pixels
    of that line's running mean centre, otherwise it starts a new line.

    Greedy clustering rather than bucketing ``y`` on a fixed grid,
    because a grid splits a line whenever its boxes straddle a bucket
    boundary -- two boxes 4px apart can land in different buckets while
    two boxes 19px apart share one. Real detections on a single text
    line jitter by several pixels, so that happens constantly.
    """
    if line_tolerance <= 0:
        raise ValueError("line_tolerance must be > 0")

    def centre_y(index: int) -> float:
        return (boxes[index][1] + boxes[index][3]) / 2.0

    lines: List[List[int]] = []
    centres: List[float] = []
    for index in sorted(range(len(boxes)), key=centre_y):
        cy = centre_y(index)
        # Sorted input means only the most recent line can still match.
        if lines and abs(cy - centres[-1]) <= line_tolerance:
            lines[-1].append(index)
            centres[-1] = sum(centre_y(i) for i in lines[-1]) / len(lines[-1])
        else:
            lines.append([index])
            centres.append(cy)
    return lines


def sort_regions_for_reading(
    boxes: Sequence[Sequence[float]], *, rtl: bool, line_tolerance: float = 20.0
) -> List[int]:
    """Indices of ``boxes`` in page reading order.

    Lines run top to bottom; within a line, regions run right-to-left
    when ``rtl`` is set, otherwise left-to-right.

    This orders *regions on the page*. It never reverses the characters
    inside a string: Arabic text is stored in logical order exactly as
    the model emitted it, and the terminal or renderer applies the
    bidirectional algorithm for display.
    """
    order: List[int] = []
    for line in group_regions_into_lines(boxes, line_tolerance):
        order.extend(
            sorted(line, key=lambda i: -boxes[i][2] if rtl else boxes[i][0])
        )
    return order


def diacritic_char_set(texts: Sequence[str]) -> Set[str]:
    """Every distinct diacritic appearing across ``texts``."""
    found: Set[str] = set()
    for text in texts:
        found.update(ch for ch in text if is_tashkeel(ch))
    return found

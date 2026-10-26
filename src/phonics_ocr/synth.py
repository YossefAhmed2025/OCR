"""Synthetic vowelled-Arabic line generator for recognition fine-tuning.

Why this exists
---------------
Measured on real pages, ``arabic_PP-OCRv5_mobile_rec`` emits tashkeel
but places them wrongly: diacritic recall ran 0.58 on clean synthetic
input, 0.22 on a phone screen and 0.00 on a photographed page. The
charset contains all 38 marks, so the capability is there and the
weights are not. That is a fine-tuning problem, and fine-tuning needs
line images paired with exact vowelled labels.

Two conventions that are easy to get wrong, both load-bearing
----------------------------------------------------------
1. **Labels are written in VISUAL order**, produced by
   ``bidi.get_display``. PaddleOCR's Arabic recognisers are trained
   that way, and its CTC decoder reverses predictions back to logical
   order at inference. Writing logical-order labels trains the model
   backwards.
2. **The character dictionary filename must contain ``arabic``.** That
   substring is what switches PaddleOCR's decoder into its
   ``pred_reverse`` path. Rename the file and decoding silently comes
   out reversed.

Both are asserted by the tests rather than left as comments.

Third rule, specific to this project: **shaping must not delete
harakat.** ``arabic-reshaper`` strips every diacritic by default, which
once produced a wrong experimental result here. Every rendered line is
verified to carry the same mark count as its source string, and
generation aborts if it does not.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import postprocessing as pp
from .utils import get_logger

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------
# Character inventory
# ---------------------------------------------------------------------

#: Arabic consonants a child-facing reader will meet.
CONSONANTS: Tuple[str, ...] = tuple(
    "\u0628\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631\u0632"
    "\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a\u0641\u0642"
    "\u0643\u0644\u0645\u0646\u0647\u0648\u064a"
)

#: Letters that do NOT join to the following letter. They create
#: word-internal gaps, which is why whitespace-based word segmentation
#: fails on Arabic -- worth having well represented in training data.
NON_JOINING: Tuple[str, ...] = tuple("\u0627\u062f\u0630\u0631\u0632\u0648")

#: Hamza and alif family, plus taa marbuta and alif maqsura: the
#: confusable classes the evaluation tracks separately.
SPECIAL_FORMS: Tuple[str, ...] = tuple(
    "\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0629\u0649"
)

#: The eight classical harakat. Tanween only ever ends a word, and
#: sukun never starts one, so placement is constrained below.
SHORT_VOWELS: Tuple[str, ...] = ("\u064e", "\u064f", "\u0650")   # fatha damma kasra
TANWEEN: Tuple[str, ...] = ("\u064b", "\u064c", "\u064d")
SUKUN: str = "\u0652"
SHADDA: str = "\u0651"

#: Curated vowelled words from early-reader vocabulary. Real words
#: matter because letter-pair frequencies in genuine Arabic differ
#: sharply from uniform random, and a model trained only on random
#: strings learns the wrong priors.
SEED_WORDS: Tuple[str, ...] = (
    "\u0623\u064e\u0631\u0652\u0646\u064e\u0628\u064c",          # arnabun
    "\u0625\u0650\u0628\u0652\u0631\u0650\u064a\u0642\u064c",    # ibreequn
    "\u0623\u064f\u0630\u064f\u0646\u064c",                      # udhunun
    "\u0643\u064e\u062a\u064e\u0628\u064e",                      # kataba
    "\u0642\u064e\u0631\u064e\u0623\u064e",                      # qara'a
    "\u0627\u0644\u0652\u0648\u064e\u0644\u064e\u062f\u064f",    # al-waladu
    "\u0627\u0644\u0637\u0650\u0641\u0652\u0644\u064f",          # at-tiflu
    "\u0645\u064e\u062f\u0652\u0631\u064e\u0633\u064e\u0629\u064c",  # madrasatun
    "\u0643\u0650\u062a\u064e\u0627\u0628\u064c",                # kitabun
    "\u0642\u064e\u0644\u064e\u0645\u064c",                      # qalamun
    "\u0628\u064e\u064a\u0652\u062a\u064c",                      # baytun
    "\u0634\u064e\u0645\u0652\u0633\u064c",                      # shamsun
    "\u0642\u064e\u0645\u064e\u0631\u064c",                      # qamarun
    "\u0645\u064f\u0639\u064e\u0644\u0651\u0650\u0645\u064c",    # mu'allimun (shadda)
    "\u0633\u064e\u0645\u064e\u0643\u064c",                      # samakun
    "\u0632\u064e\u0647\u0652\u0631\u064e\u0629\u064c",          # zahratun
    "\u0639\u064e\u064a\u0652\u0646\u064c",                      # 'aynun
    "\u064a\u064e\u062f\u064c",                                  # yadun
    "\u0628\u064e\u0627\u0628\u064c",                            # babun
    "\u062c\u064e\u0645\u064e\u0644\u064c",                      # jamalun
    "\u0646\u064e\u062c\u0652\u0645\u064c",                      # najmun
    "\u0637\u064e\u0627\u0626\u0650\u0631\u064c",                # ta'irun (hamza on ya)
    "\u0645\u0650\u0641\u0652\u062a\u064e\u0627\u062d\u064c",    # miftahun
    "\u0633\u064e\u0628\u0651\u064f\u0648\u0631\u064e\u0629\u064c",  # sabburatun
    "\u062a\u0650\u0644\u0652\u0645\u0650\u064a\u0630\u064c",    # tilmeedhun
    "\u0623\u064f\u0633\u0652\u0631\u064e\u0629\u064c",          # usratun
    "\u0645\u064e\u0627\u0621\u064c",                            # ma'un
    "\u062e\u064f\u0628\u0652\u0632\u064c",                      # khubzun
    "\u0644\u064e\u0628\u064e\u0646\u064c",                      # labanun
    "\u0648\u064e\u0631\u064e\u062f\u064e\u0629\u064c",          # waradatun
)


class SynthError(RuntimeError):
    """Raised when a sample cannot be generated faithfully."""


@dataclasses.dataclass(frozen=True)
class RenderConfig:
    """Rendering and augmentation ranges for one dataset."""

    height: int = 48            # PP-OCRv5 recognition input height
    pad_x: int = 6
    pad_y: int = 4
    font_size_range: Tuple[int, int] = (28, 56)
    #: Rotation in degrees. Lines reaching the recogniser are already
    #: rectified by detection, so only small residual skew is realistic.
    rotation_range: Tuple[float, float] = (-2.0, 2.0)
    blur_prob: float = 0.45
    blur_range: Tuple[float, float] = (0.4, 1.3)
    noise_prob: float = 0.35
    noise_sigma_range: Tuple[float, float] = (2.0, 9.0)
    #: Uneven lighting across the crop, as a page under a lamp gives.
    gradient_prob: float = 0.3
    gradient_strength: Tuple[float, float] = (0.10, 0.35)
    contrast_range: Tuple[float, float] = (0.65, 1.0)
    jpeg_prob: float = 0.4
    jpeg_quality_range: Tuple[int, int] = (45, 92)
    invert_prob: float = 0.0    # the product only targets dark-on-light

    def __post_init__(self) -> None:
        if self.height < 16:
            raise SynthError("height must be >= 16")
        lo, hi = self.font_size_range
        if lo < 8 or hi < lo:
            raise SynthError(f"bad font_size_range {self.font_size_range}")


# ---------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------


def vowelize(word: str, rng: random.Random, *, full: bool = True) -> str:
    """Attach plausible harakat to an unvowelled word.

    Placement follows the rules that make Arabic vowelling legal rather
    than decorative: tanween only on the final letter, sukun never on
    the first, and never two marks on one letter except shadda plus a
    short vowel.
    """
    out: List[str] = []
    letters = [c for c in word if not pp.is_tashkeel(c)]
    for i, ch in enumerate(letters):
        out.append(ch)
        last = i == len(letters) - 1
        if ch in ("\u0627", "\u0649"):
            continue  # long alif carries no vowel of its own
        if not full and rng.random() < 0.45:
            continue
        if rng.random() < 0.12:
            out.append(SHADDA)
        if last:
            out.append(rng.choice(TANWEEN if rng.random() < 0.55 else SHORT_VOWELS))
        elif i > 0 and rng.random() < 0.25:
            out.append(SUKUN)
        else:
            out.append(rng.choice(SHORT_VOWELS))
    return "".join(out)


def random_word(rng: random.Random, length: Optional[int] = None) -> str:
    """A pronounceable-looking nonsense word, fully vowelled.

    Nonsense words are deliberate: they force coverage of
    letter-diacritic pairs that a finite real-word list would never
    contain, which is what a recogniser actually needs to generalise.
    """
    length = length or rng.randint(2, 6)
    letters: List[str] = []
    for i in range(length):
        pool = CONSONANTS
        if i == 0 and rng.random() < 0.18:
            pool = SPECIAL_FORMS
        elif rng.random() < 0.22:
            pool = NON_JOINING
        elif i == length - 1 and rng.random() < 0.12:
            pool = ("\u0629", "\u0649")  # taa marbuta / alif maqsura endings
        letters.append(rng.choice(pool))
    return vowelize("".join(letters), rng)


DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "data" / "corpus" / "vowelled_lexicon.json"


def load_lexicon(path: Optional[Path] = None) -> List[str]:
    """Load real vowelled words extracted from a corpus.

    Strongly preferred over :func:`vowelize`. The first fine-tune
    showed why: random vowelization taught the model *where* marks go
    (diacritic recall doubled) but not *which* mark belongs (false
    positives rose and CER got worse). Real words carry the
    morphological patterns that make a vowel predictable from context,
    which is the signal the recogniser needs.

    Returns an empty list when the lexicon has not been built, so the
    caller can fall back and say so rather than failing.
    """
    import json

    path = Path(path) if path else DEFAULT_LEXICON
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("words", []))


def make_line(
    rng: random.Random,
    *,
    seed_words: Sequence[str] = SEED_WORDS,
    max_words: int = 4,
    real_word_prob: float = 0.92,
) -> str:
    """One training line: between one and ``max_words`` vowelled words.

    ``real_word_prob`` defaults high because random vowelization is
    actively harmful: measured, it taught the model to place marks
    without learning which mark belongs. The small remainder of
    synthetic words exists only to cover letter-diacritic pairs a
    finite lexicon misses.
    """
    count = rng.randint(1, max_words)
    pool = list(seed_words)
    words: List[str] = []
    for _ in range(count):
        if pool and rng.random() < real_word_prob:
            # Real corpus words are used VERBATIM. Re-vowelizing them
            # would reintroduce exactly the problem this fixes.
            words.append(rng.choice(pool))
        else:
            words.append(random_word(rng))
    return " ".join(words)


# ---------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------

#: Code points a font must cover to be usable: a spread of letters plus
#: all eight classical harakat. A font missing a mark renders it as
#: .notdef and silently teaches the model a wrong glyph.
_REQUIRED_LETTERS = (0x0643, 0x062A, 0x0628, 0x0627, 0x0631, 0x0646, 0x0642, 0x0630)
_REQUIRED_MARKS = (0x064B, 0x064C, 0x064D, 0x064E, 0x064F, 0x0650, 0x0651, 0x0652)


#: Strings whose SHAPED forms a font must render completely.
#:
#: Checking base code points is not enough and produced a dataset of
#: tofu boxes paired with correct labels. ``arabic_reshaper`` rewrites
#: letters into presentation forms (U+FE70-U+FEFF), and many fonts
#: declare the base Arabic block while covering only part of that
#: range. The only safe test is to shape the text first and then
#: require every resulting code point.
_COVERAGE_PROBES: Tuple[str, ...] = (
    "كَتَبَ",                       # joined + fatha
    "إِبْرِيقٌ",     # tanween + sukun
    "مُعَلِّمٌ",     # shadda + kasra
    "أُذُنٌ",                       # hamza forms
    "مَدْرَسَةٌ",  # taa marbuta
    "طَائِرٌ",                 # hamza on ya
)


def font_covers_shaped(font_path: Path, probes: Sequence[str] = _COVERAGE_PROBES) -> bool:
    """Whether a font can render every glyph of the SHAPED probes.

    This is the check that matters. A font passing the base-code-point
    test can still render presentation forms as .notdef boxes, which
    yields training images of empty rectangles carrying correct labels
    -- the most damaging kind of silent dataset corruption.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise SynthError("fontTools is required. pip install fonttools") from exc

    from .visualization import shape_for_display

    try:
        font = TTFont(font_path, fontNumber=0, lazy=True)
        cmap: set = set()
        for table in font["cmap"].tables:
            cmap |= set(table.cmap.keys())
        font.close()
    except Exception:
        return False

    for probe in probes:
        shaped = shape_for_display(probe)
        for ch in shaped:
            if ch in (" ", "‏", "‎"):
                continue
            if ord(ch) not in cmap:
                return False
    return True


def find_vowel_capable_fonts(directories: Optional[Iterable[Path]] = None) -> List[Path]:
    """System fonts that fully render SHAPED vowelled Arabic."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise SynthError(
            "fontTools is required to validate font coverage. "
            "pip install fonttools"
        ) from exc

    if directories is None:
        import platform

        directories = (
            [Path(r"C:\Windows\Fonts")]
            if platform.system() == "Windows"
            else [Path("/usr/share/fonts"), Path.home() / ".fonts"]
        )

    found: List[Path] = []
    for directory in directories:
        if not Path(directory).is_dir():
            continue
        for path in sorted(Path(directory).rglob("*.ttf")):
            try:
                font = TTFont(path, fontNumber=0, lazy=True)
                cmap: set = set()
                for table in font["cmap"].tables:
                    cmap |= set(table.cmap.keys())
                font.close()
            except Exception:
                continue
            if not (
                all(c in cmap for c in _REQUIRED_LETTERS)
                and all(c in cmap for c in _REQUIRED_MARKS)
            ):
                continue
            # Base coverage is necessary but NOT sufficient -- the
            # shaped presentation forms are what actually get drawn.
            if font_covers_shaped(path):
                found.append(path)
    return found


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


def visual_label(text: str) -> str:
    """Logical text reordered to visual order, in BASE Arabic forms.

    This is the training label, and it is **not** the same string used
    for rendering. Rendering needs ``arabic_reshaper``, which rewrites
    letters into presentation forms (U+FE70-U+FEFF). Those are display
    glyphs, not the characters the recogniser predicts: the model's
    charset is base Arabic, and its raw CTC sequence comes out as base
    letters in visual order.

    Labelling with presentation forms silently adds ~80 unseen classes
    to the dictionary and trains the model to predict glyph codes it
    will never be scored on. Verified directly against the pretrained
    model: for ``زَهْرَة`` the raw CTC sequence is ``ةَرْهَز`` (base,
    reversed), while the reshaped render string is ``ﺓَﺮْﻫَﺯ``.

    So: bidi reordering only, no reshaping.
    """
    try:
        from bidi.algorithm import get_display
    except ImportError as exc:
        raise SynthError("python-bidi is required. pip install python-bidi") from exc

    ordered = get_display(text)
    if pp.count_tashkeel(ordered) != pp.count_tashkeel(text):
        raise SynthError(
            f"bidi reordering changed the diacritic count for {text!r} "
            f"({pp.count_tashkeel(text)} -> {pp.count_tashkeel(ordered)})"
        )
    return ordered


def render_line(
    text: str,
    font_path: Path,
    config: RenderConfig,
    rng: random.Random,
) -> Tuple[np.ndarray, str]:
    """Render one vowelled line and return ``(image_bgr, label)``.

    The label is in **visual order using base Arabic characters** --
    see :func:`visual_label`. The *rendering* uses reshaped
    presentation forms, which is a different string.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    from .visualization import shape_for_display

    display = shape_for_display(text)
    if pp.count_tashkeel(display) != pp.count_tashkeel(text):
        raise SynthError(
            f"shaping dropped diacritics for {text!r} "
            f"({pp.count_tashkeel(text)} -> {pp.count_tashkeel(display)}). "
            "arabic-reshaper must run with delete_harakat=False."
        )
    label = visual_label(text)

    size = rng.randint(*config.font_size_range)
    try:
        font = ImageFont.truetype(str(font_path), size)
    except Exception as exc:
        raise SynthError(f"could not load {font_path}: {exc}") from exc

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8), "white"))
    try:
        box = probe.textbbox((0, 0), display, font=font)
    except Exception as exc:
        raise SynthError(f"textbbox failed for {font_path}: {exc}") from exc
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width < 4 or height < 4:
        raise SynthError(f"degenerate render for {text!r} in {font_path.name}")

    paper = rng.randint(226, 255)
    ink = rng.randint(0, 70)
    canvas = Image.new(
        "RGB",
        (int(width + 2 * config.pad_x), int(height + 2 * config.pad_y)),
        (paper, paper, paper),
    )
    ImageDraw.Draw(canvas).text(
        (config.pad_x - box[0], config.pad_y - box[1]),
        display,
        font=font,
        fill=(ink, ink, ink),
    )

    image = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    image = augment(image, config, rng, paper)

    # Final resize to the recogniser's input height, aspect preserved.
    h, w = image.shape[:2]
    target_w = max(8, int(round(config.height * w / float(h))))
    image = cv2.resize(
        image, (target_w, config.height), interpolation=cv2.INTER_AREA
    )
    return image, label


def augment(
    image: np.ndarray, config: RenderConfig, rng: random.Random, paper: int
) -> np.ndarray:
    """Apply camera-like degradation.

    Ranges are deliberately mild. These crops stand in for *rectified
    detected lines*, not raw frames, so heavy perspective and rotation
    would not be realistic -- detection has already removed them.
    """
    import cv2

    out = image

    angle = rng.uniform(*config.rotation_range)
    if abs(angle) > 0.05:
        h, w = out.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(
            out, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(paper, paper, paper),
        )

    if rng.random() < config.gradient_prob:
        h, w = out.shape[:2]
        strength = rng.uniform(*config.gradient_strength)
        ramp = np.linspace(1.0, 1.0 - strength, w, dtype=np.float32)
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        out = np.clip(out.astype(np.float32) * ramp[None, :, None], 0, 255).astype(
            np.uint8
        )

    contrast = rng.uniform(*config.contrast_range)
    if contrast < 0.999:
        mean = float(out.mean())
        out = np.clip((out.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(
            np.uint8
        )

    if rng.random() < config.blur_prob:
        sigma = rng.uniform(*config.blur_range)
        out = cv2.GaussianBlur(out, (0, 0), sigma)

    if rng.random() < config.noise_prob:
        sigma = rng.uniform(*config.noise_sigma_range)
        noise = rng.gauss  # keep determinism tied to the passed rng
        grain = np.array(
            [noise(0, sigma) for _ in range(out.shape[0] * out.shape[1])],
            dtype=np.float32,
        ).reshape(out.shape[:2])
        out = np.clip(out.astype(np.float32) + grain[:, :, None], 0, 255).astype(
            np.uint8
        )

    if rng.random() < config.jpeg_prob:
        quality = rng.randint(*config.jpeg_quality_range)
        ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    return out


# ---------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------


def build_dictionary(labels: Iterable[str], base: Optional[Iterable[str]] = None) -> List[str]:
    """Character dictionary covering every symbol in ``labels``.

    ``base`` seeds it with the pretrained model's charset so that
    fine-tuning can warm-start without re-indexing every class. Order
    matters: existing entries keep their indices, new ones append.
    """
    seen: List[str] = []
    known: set = set()
    for ch in base or ():
        if ch not in known:
            known.add(ch)
            seen.append(ch)
    extra: List[str] = []
    for label in labels:
        for ch in label:
            if ch not in known:
                known.add(ch)
                extra.append(ch)
    return seen + sorted(extra)


def load_model_charset(model_dir: Path) -> List[str]:
    """Charset from a PaddleOCR inference.yml, for warm-starting."""
    import yaml

    path = Path(model_dir) / "inference.yml"
    if not path.is_file():
        raise SynthError(f"missing {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    def find(node: Any) -> Optional[List[str]]:
        if isinstance(node, dict):
            for key, value in node.items():
                if "character" in str(key).lower() and isinstance(value, list):
                    return value
                found = find(value)
                if found:
                    return found
        return None

    charset = find(config)
    if not charset:
        raise SynthError(f"no character dictionary in {path}")
    return list(charset)

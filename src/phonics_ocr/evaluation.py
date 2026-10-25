"""Error-rate metrics with explicit, reportable denominators.

Every metric returned here carries the counts it was computed from.
A bare rate with no denominator is not a result, and this module never
produces one: :class:`ErrorRate` always exposes ``reference_length``,
and an aggregate over a corpus is computed by summing edit distances
and reference lengths (a *micro* average), not by averaging per-item
rates, which would weight a three-character line as heavily as a
three-hundred-character page.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import postprocessing as pp


@dataclasses.dataclass(frozen=True)
class ErrorRate:
    """An error rate together with the counts that produced it."""

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> Optional[float]:
        """Errors divided by reference length.

        ``None`` when the reference is empty: a rate with a zero
        denominator is undefined, and reporting 0.0 there would silently
        turn "nothing to measure" into "perfect score".
        """
        if self.reference_length == 0:
            return None
        return self.errors / self.reference_length

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rate": self.rate,
            "errors": self.errors,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
        }

    def __add__(self, other: "ErrorRate") -> "ErrorRate":
        return ErrorRate(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )


def _levenshtein_counts(
    reference: Sequence[Any], hypothesis: Sequence[Any]
) -> Tuple[int, int, int]:
    """Edit operation counts turning ``reference`` into ``hypothesis``.

    Returns ``(substitutions, deletions, insertions)`` from one optimal
    alignment. Deletions are reference tokens absent from the
    hypothesis; insertions are hypothesis tokens absent from the
    reference. All three operations cost 1.

    When several optimal alignments exist the operation *mix* can differ
    between them, but the total is identical, so ``errors`` and the rate
    are well defined even though the individual counts are one valid
    decomposition rather than the only one.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return (0, 0, m)
    if m == 0:
        return (0, n, 0)

    # Each cell holds (cost, subs, dels, ins) for the best path to it.
    previous: List[Tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        current: List[Tuple[int, int, int, int]] = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost, s, d, ins = previous[j - 1]
                current.append((cost, s, d, ins))
                continue
            sub_cost, sub_s, sub_d, sub_i = previous[j - 1]
            del_cost, del_s, del_d, del_i = previous[j]
            ins_cost, ins_s, ins_d, ins_i = current[j - 1]
            best = min(
                (sub_cost + 1, sub_s + 1, sub_d, sub_i),
                (del_cost + 1, del_s, del_d + 1, del_i),
                (ins_cost + 1, ins_s, ins_d, ins_i + 1),
                key=lambda t: t[0],
            )
            current.append(best)
        previous = current

    _, subs, dels, ins = previous[m]
    return (subs, dels, ins)


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """CER over Unicode code points.

    The denominator is the number of code points in the reference, so a
    combining mark counts as its own unit. That is intentional: for this
    project a missed fatha *is* a character error.
    """
    subs, dels, ins = _levenshtein_counts(reference, hypothesis)
    return ErrorRate(subs, dels, ins, len(reference))


def tokenize_words(text: str) -> List[str]:
    """Whitespace tokenisation.

    Deliberately naive and documented as such: it does not split
    punctuation from words and does not segment Arabic clitics. Both
    reference and hypothesis go through the same function, so the
    comparison is fair even though the token definition is coarse.
    """
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """WER over whitespace-separated tokens."""
    ref_tokens = tokenize_words(reference)
    hyp_tokens = tokenize_words(hypothesis)
    subs, dels, ins = _levenshtein_counts(ref_tokens, hyp_tokens)
    return ErrorRate(subs, dels, ins, len(ref_tokens))


@dataclasses.dataclass(frozen=True)
class PRF:
    """Precision / recall / F1 with the counts behind them."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> Optional[float]:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else None

    @property
    def recall(self) -> Optional[float]:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }

    def __add__(self, other: "PRF") -> "PRF":
        return PRF(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )


def diacritic_prf(reference: str, hypothesis: str) -> PRF:
    """Precision/recall/F1 over Arabic diacritics.

    A diacritic is treated as the pair ``(index of its base character in
    the tashkeel-stripped string, the mark itself)``. A hypothesis mark
    counts as a true positive only when the same mark sits on the same
    base position.

    Consequences, stated plainly:

    * A model that outputs no tashkeel at all scores recall 0 and
      precision ``None`` (undefined -- it made no predictions). It does
      **not** score well by default.
    * If the hypothesis drops or adds a *consonant*, every subsequent
      base index shifts and the diacritics after that point are counted
      wrong. That is a real limitation of position-based matching, and
      it is why the diacritic metric is reported alongside, never
      instead of, CER on the fully vowelled text.
    """
    ref_marks = pp.extract_tashkeel_positions(reference)
    hyp_marks = pp.extract_tashkeel_positions(hypothesis)

    remaining = list(ref_marks)
    true_positives = 0
    for mark in hyp_marks:
        if mark in remaining:
            remaining.remove(mark)
            true_positives += 1

    return PRF(
        true_positives=true_positives,
        false_positives=len(hyp_marks) - true_positives,
        false_negatives=len(ref_marks) - true_positives,
    )


def confusable_error_counts(reference: str, hypothesis: str) -> Dict[str, int]:
    """Per-class counts of characters present in one string but not the other.

    For each tracked Arabic class (hamza, alif variants, taa marbuta vs
    haa, alif maqsura vs yaa, shadda, tanween) this reports the absolute
    difference in multiset occurrence between reference and hypothesis.

    This is a *count mismatch*, not an alignment, so it answers "did the
    model produce the right number of hamzas?" rather than "did it put
    them in the right places". It is a diagnostic pointer toward which
    character classes need attention, not a standalone accuracy figure.
    """
    counts: Dict[str, int] = {}
    for class_name, members in pp.CONFUSABLE_CLASSES.items():
        ref_n = sum(1 for ch in reference if ch in members)
        hyp_n = sum(1 for ch in hypothesis if ch in members)
        counts[class_name] = abs(ref_n - hyp_n)
    return counts


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference == hypothesis


@dataclasses.dataclass
class SampleScore:
    """All metrics for one image, on one text representation."""

    sample_id: str
    group: str
    cer: ErrorRate
    wer: ErrorRate
    exact: bool
    diacritic: Optional[PRF]
    confusables: Dict[str, int]
    reference: str
    hypothesis: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "group": self.group,
            "cer": self.cer.to_dict(),
            "wer": self.wer.to_dict(),
            "exact_match": self.exact,
            "diacritic": self.diacritic.to_dict() if self.diacritic else None,
            "confusable_count_mismatches": self.confusables,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
        }


def score_sample(
    sample_id: str,
    group: str,
    reference: str,
    hypothesis: str,
    *,
    score_diacritics: bool = True,
) -> SampleScore:
    """Score one reference/hypothesis pair.

    Both strings should already be in whatever normalised representation
    the caller intends to measure; this function applies no further
    normalisation of its own, so the denominator stays the one the
    caller can describe.
    """
    return SampleScore(
        sample_id=sample_id,
        group=group,
        cer=character_error_rate(reference, hypothesis),
        wer=word_error_rate(reference, hypothesis),
        exact=exact_match(reference, hypothesis),
        diacritic=diacritic_prf(reference, hypothesis) if score_diacritics else None,
        confusables=confusable_error_counts(reference, hypothesis),
        reference=reference,
        hypothesis=hypothesis,
    )


def aggregate(scores: Sequence[SampleScore]) -> Dict[str, Any]:
    """Micro-average a set of sample scores.

    Edit distances and reference lengths are summed before dividing, so
    long pages carry proportionally more weight than short ones. The
    sums are included in the output so the denominator is always visible.
    """
    if not scores:
        return {
            "n_samples": 0,
            "cer": None,
            "wer": None,
            "exact_match_rate": None,
            "diacritic": None,
            "confusable_count_mismatches": {},
        }

    cer_total = ErrorRate(0, 0, 0, 0)
    wer_total = ErrorRate(0, 0, 0, 0)
    for s in scores:
        cer_total = cer_total + s.cer
        wer_total = wer_total + s.wer

    dia_scores = [s.diacritic for s in scores if s.diacritic is not None]
    dia_total: Optional[PRF] = None
    if dia_scores:
        dia_total = PRF(0, 0, 0)
        for d in dia_scores:
            dia_total = dia_total + d

    confusables: Dict[str, int] = {}
    for s in scores:
        for key, value in s.confusables.items():
            confusables[key] = confusables.get(key, 0) + value

    return {
        "n_samples": len(scores),
        "cer": cer_total.to_dict(),
        "wer": wer_total.to_dict(),
        "exact_match_rate": sum(1 for s in scores if s.exact) / len(scores),
        "exact_matches": sum(1 for s in scores if s.exact),
        "diacritic": dia_total.to_dict() if dia_total else None,
        "confusable_count_mismatches": confusables,
    }


def aggregate_by_group(scores: Sequence[SampleScore]) -> Dict[str, Dict[str, Any]]:
    """Micro-averages per evaluation group, plus an ``overall`` entry."""
    groups: Dict[str, List[SampleScore]] = {}
    for s in scores:
        groups.setdefault(s.group, []).append(s)
    out = {name: aggregate(items) for name, items in sorted(groups.items())}
    out["overall"] = aggregate(scores)
    return out

def wilson_interval(
    successes: int, trials: int, z: float = 1.96
) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion.

    Used for recall and precision, which are proportions of a countable
    number of boxes. Wilson rather than the normal approximation
    because the normal interval is badly wrong at small n and at
    proportions near 0 or 1 -- exactly where small localisation
    evaluations sit. A recall of 36/36 has a normal interval of width
    zero, which is obviously false; Wilson gives a sensible lower
    bound.

    Returns ``(None, None)`` when there were no trials.

    The interval accounts for **sampling** error only. It says nothing
    about whether the sample is representative, so a tight interval on
    rendered images is still an interval about rendered images.
    """
    if trials <= 0:
        return (None, None)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    ) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))

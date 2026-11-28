"""Result distance d and the two coverage measures (paper §3).

`d` is paper §3's "Result distance": relative difference per label set, the maximum over
a common domain, and infinity when the label sets differ at all. §3's lemma "d is a
metric" is what the admissibility proposition of §3.1 needs, since that argument is a
triangle inequality; `tests/test_metrics.py` checks the lemma's case analysis.

`kappa` below is the *differential* coverage of §3's remark "Two coverages" -- the share
of queries whose translation was observed to agree with the source system -- so it takes
per-query outcomes and not translator verdicts. It is the stronger measure, it needs both
systems live, and it is not what the paper reports: §7's κ is §3's "Coverage", the share
the translator is willing to translate at all, computed in notebook 03 as translated over
total. The two coincide exactly when every emitted expression is correct.

Sections are cited by name here, not by number: the numbering shifts whenever a
definition is added, and a stale cross-reference in a docstring is invisible to the
build.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def reldiff(a: float, b: float) -> float:
    """|a-b| / max(|a|,|b|): symmetric and satisfies the triangle inequality, so d is a metric."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    den = max(abs(a), abs(b))
    return 0.0 if den == 0 else abs(a - b) / den


def d(r1: Mapping[tuple, float], r2: Mapping[tuple, float]) -> float:
    """Results are {label-set key: value}. ∞ if the label sets differ, else max reldiff.

    NaN if any label set is incomparable, which is outside §3's definition -- that one
    is over real-valued results -- but inside this module's, because `reldiff` returns
    NaN by design and the only caller of it that is not a test can be handed one: an
    expression that selects nothing at an instant reads back as NaN.

    Taken over the distances as a list rather than left to `max`, which is where this
    went wrong. `max` keeps whichever operand it saw first when the comparison involves
    a NaN, because every comparison with one is false -- so `d` returned 0.0 for one
    incomparable label set among identical ones, and NaN for the same two results in
    the other dictionary order. Reporting *the results agree* for a comparison one side
    of which produced no value is the failure §3.1 exists to detect.
    """
    if set(r1) != set(r2):
        return math.inf
    diffs = [reldiff(r1[k], r2[k]) for k in r1]
    if any(math.isnan(x) for x in diffs):
        return math.nan
    return max(diffs, default=0.0)


def kappa(outcomes: Iterable[str]) -> float:
    """Coverage over a corpus. outcomes: 'pass', 'fail', 'bottom', or 'untested'.
    Untested queries are excluded from the denominator and reported separately."""
    xs = [o for o in outcomes if o != "untested"]
    return sum(o == "pass" for o in xs) / len(xs) if xs else math.nan

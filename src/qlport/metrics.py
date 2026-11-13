"""Result distance d (proposal eq. 1, revised) and coverage κ (eq. 2)."""
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
    """Results are {label-set key: value}. ∞ if the label sets differ, else max reldiff."""
    if set(r1) != set(r2):
        return math.inf
    return max((reldiff(r1[k], r2[k]) for k in r1), default=0.0)


def kappa(outcomes: Iterable[str]) -> float:
    """Coverage over a corpus. outcomes: 'pass', 'fail', 'bottom', or 'untested'.
    Untested queries are excluded from the denominator and reported separately."""
    xs = [o for o in outcomes if o != "untested"]
    return sum(o == "pass" for o in xs) / len(xs) if xs else math.nan

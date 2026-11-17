"""The ranking workload, defined once, so both halves of the `top()` comparison share it.

Paper §7 settles whether Datadog's `top(q, n, 'BY', 'DIR')` is expressible in PromQL by
comparing two things: what a real Prometheus returns for the `@`-pinned candidate
(`scripts/verify_topk.py`), and what a real Datadog returns for `top()` itself
(`scripts/datadog_top.py`). The comparison is only valid if both systems were shown the
*same* series, so the series live here rather than in either script, and
`tests/test_harness.py` checks the cached runs against this definition.

Five series chosen so that no two rankings coincide by accident, and so that ranking over
a window disagrees with ranking at an instant -- which is the whole difference between
the two operators:

    a  flat and high      wins on mean, loses on max
    b  rises to the top   wins on last and max
    c  flat runner-up     wins at early instants, never over the window
    d  flat and low       never wins
    e  one early spike    wins on max, loses on mean

`e` is why the window must include the first sample: the spike at index 0 is the only
thing separating `max` from `mean`, and a range selector that drops it makes the two
rankings agree and the experiment prove nothing.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = ["N", "STEP", "SERIES", "PROMQL_OVER_TIME", "REDUCERS",
           "values", "score", "model_rank"]

N, STEP = 20, 60

SERIES: dict[str, Callable[[int], float]] = {
    "a": lambda i: 100.0,
    "b": lambda i: 10.0 + 9 * i,
    "c": lambda i: 95.0,
    "d": lambda i: 1.0,
    "e": lambda i: 200.0 if i == 0 else 1.0,
}

# Datadog's ranking function -> the PromQL range function that computes the same score.
# `min` is here because the candidate expression has to be checked for it; the corpus
# does not use it (paper §7 says which four it does use).
PROMQL_OVER_TIME = {"max": "max_over_time", "mean": "avg_over_time",
                    "min": "min_over_time", "sum": "sum_over_time",
                    "last": "last_over_time"}

REDUCERS: dict[str, Callable[[Sequence[float]], float]] = {
    "max": max, "min": min, "sum": sum,
    "mean": lambda v: sum(v) / len(v), "last": lambda v: v[-1],
}


def values(name: str, lo: int = 0, hi: int = N) -> list[float]:
    """Series `name` over indices [lo, hi)."""
    f = SERIES[name]
    return [f(i) for i in range(lo, hi)]


def score(by: str, name: str, lo: int = 0, hi: int = N) -> float:
    return REDUCERS[by](values(name, lo, hi))


def model_rank(by: str, n: int, direction: str = "desc",
               lo: int = 0, hi: int = N) -> list[str]:
    """Datadog's documented rule: aggregate each series over the queried range, take n.

    `direction` is 'desc' for the largest n and 'asc' for the smallest -- the corpus uses
    both. Returned sorted by name, because the question is which series were selected and
    not what order they came back in.

    Ties are not broken, they are refused: the workload is designed so that no two series
    score alike under any of the five functions, and a tie at the cut would make the
    prediction a coin flip dressed as a model.
    """
    if direction not in ("desc", "asc"):
        raise ValueError(f"direction must be 'desc' or 'asc', got {direction!r}")
    scored = {k: score(by, k, lo, hi) for k in SERIES}
    order = sorted(scored, key=lambda k: scored[k], reverse=direction == "desc")
    if n < len(order) and scored[order[n - 1]] == scored[order[n]]:
        raise ValueError(f"{by} ties at the n={n} cut over [{lo},{hi}): {scored}")
    return sorted(order[:n])

"""Reference evaluator for Datadog COUNT metrics.

The bucket convention here is measured, not assumed. `scripts/trial_checks.py e2`
submits one point at an exact multiple of the rollup interval and one strictly inside a
bucket, and reads back which bucket each is reported under; the recorded answer is in
`data/raw/datadog/e2.json` and `tests/test_harness_datadog.py` regression-tests it.

  A bucket is **left-closed and labelled by its start**: the bucket reported at
  timestamp `b` aggregates the points in `[b, b + interval)`.

An earlier version of this file assumed the opposite --- right-closed `(t - interval, t]`
--- which is the convention PromQL's own range selectors use and therefore the easy guess.
It mattered less than it looks: a model and a candidate both evaluated under the same
convention agree with each other, so every agreement figure the paper reports is
unchanged. What was wrong was the claim about Datadog, not the arithmetic.

It does matter for one thing, and sharply. A PromQL range selector covers `(u - w, u]`,
so no evaluation instant makes it cover a left-closed bucket exactly when a sample lands
on a boundary: Datadog counts a sample at `b`, PromQL at `u = b + interval` counts one at
`b + interval` instead. With an aligned scrape -- the common case, since the scrape
interval usually divides the rollup interval -- that is every bucket. The translation must
therefore be evaluated strictly *inside* the bucket, at `b + interval - eps` for any eps
smaller than the sample spacing; `eval_instant` below is that instant. Evaluating at the
bucket's right edge, which is the obvious correction once the closure is known, is wrong
on almost every bucket.
"""
from __future__ import annotations

import numpy as np

# Small enough to stay inside any realistic sample spacing, large enough to survive
# float64 arithmetic on epoch-scale timestamps.
EVAL_EPSILON = 1e-3


def eval_instant(bucket_start: float, interval: float, eps: float = EVAL_EPSILON) -> float:
    """The PromQL evaluation time whose window `(u - interval, u]` matches `[b, b+interval)`.

    Any instant at or after the bucket's last sample and strictly before its right edge
    works; this picks one just inside the edge so it does not depend on the sample grid.
    """
    return bucket_start + interval - eps


def rollup_sum(ts: np.ndarray, deltas: np.ndarray, b: float, interval: float) -> float:
    """Datadog's `.rollup(sum, interval)` for the bucket reported at `b`."""
    m = (ts >= b) & (ts < b + interval)
    return float(deltas[m].sum()) if m.any() else np.nan


def as_rate(ts, deltas, b, interval) -> float:
    return rollup_sum(ts, deltas, b, interval) / interval


def as_count(ts, deltas, b, interval) -> float:
    return rollup_sum(ts, deltas, b, interval)


def bucket_starts(t0: float, t1: float, interval: float) -> np.ndarray:
    """Epoch-aligned bucket start times in [t0, t1).

    Epoch alignment is documented; which edge the reported timestamp names is not, and
    is what E2 measured.
    """
    return np.arange(np.ceil(t0 / interval) * interval, t1, interval)

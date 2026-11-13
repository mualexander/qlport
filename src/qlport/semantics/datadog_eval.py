"""Reference evaluator for Datadog COUNT metrics.

ASSUMPTIONS to verify on the trial account (paper §7):
  * rollup buckets are epoch-aligned and right-closed: (t - interval, t]
  * as_rate() on a COUNT = sum of deltas in the bucket / interval (docs: "sets the time
    aggregator to SUM" and "divides the result post-aggregation by the sampling interval")
"""
from __future__ import annotations

import numpy as np


def rollup_sum(ts: np.ndarray, deltas: np.ndarray, t: float, interval: float) -> float:
    m = (ts > t - interval) & (ts <= t)
    return float(deltas[m].sum()) if m.any() else np.nan


def as_rate(ts, deltas, t, interval) -> float:
    return rollup_sum(ts, deltas, t, interval) / interval


def as_count(ts, deltas, t, interval) -> float:
    return rollup_sum(ts, deltas, t, interval)


def bucket_ends(t0: float, t1: float, interval: float) -> np.ndarray:
    """Epoch-aligned bucket end times in [t0, t1)."""
    return np.arange(np.ceil(t0 / interval) * interval, t1, interval)

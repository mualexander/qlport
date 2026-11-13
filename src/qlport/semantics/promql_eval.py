"""Reference evaluators for the PromQL constructs used in translations.

Window semantics follow Prometheus 3.x: range selectors cover (t - w, t];
instant selectors return the last sample in (t - lookback, t].
"""
from __future__ import annotations

import numpy as np

LOOKBACK = 300.0  # Prometheus default lookback delta, seconds


def instant(ts: np.ndarray, vs: np.ndarray, t: float, lookback: float = LOOKBACK) -> float:
    m = (ts > t - lookback) & (ts <= t)
    return float(vs[m][-1]) if m.any() else np.nan


def offset_diff_rate(ts, vs, t, interval, lookback=LOOKBACK) -> float:
    """(x - x offset interval) / interval   — not reset-aware."""
    return (instant(ts, vs, t, lookback) - instant(ts, vs, t - interval, lookback)) / interval


def rate(ts: np.ndarray, vs: np.ndarray, t: float, w: float) -> float:
    """Port of Prometheus extrapolatedRate (promql/functions.go), counter, isRate=True."""
    m = (ts > t - w) & (ts <= t)
    st, sv = ts[m], vs[m]
    if len(st) < 2:
        return np.nan
    result = sv[-1] - sv[0]
    prev = sv[0]
    for v in sv[1:]:
        if v < prev:          # counter reset: assume restart from zero
            result += prev
        prev = v
    to_start = st[0] - (t - w)
    to_end = t - st[-1]
    sampled = st[-1] - st[0]
    avg = sampled / (len(st) - 1)
    thr = avg * 1.1
    if to_start >= thr:
        to_start = avg / 2
    if result > 0 and sv[0] >= 0:   # "duration to zero": reads the absolute first value
        to_zero = sampled * (sv[0] / result)
        if to_zero < to_start:
            to_start = to_zero
    if to_end >= thr:
        to_end = avg / 2
    return float(result * (sampled + to_start + to_end) / sampled / w)


def increase_anchored(ts, vs, t, w, lookback=LOOKBACK) -> float:
    """MODEL of `increase(x[w] anchored)` (experimental, proposal 0052).

    Assumed semantics: left boundary pinned to the last real sample at or before t - w
    (within lookback), right boundary to the last sample in the window, reset-aware,
    no extrapolation. VERIFY against a real Prometheus with
    --enable-feature=promql-extended-range-selectors before relying on it.
    """
    left = (ts > t - w - lookback) & (ts <= t - w)
    inside = (ts > t - w) & (ts <= t)
    if not inside.any():
        return np.nan
    pts = np.concatenate([vs[left][-1:], vs[inside]])
    if len(pts) < 2:
        return np.nan
    total, prev = 0.0, pts[0]
    for v in pts[1:]:
        total += v - prev if v >= prev else v
        prev = v
    return float(total)

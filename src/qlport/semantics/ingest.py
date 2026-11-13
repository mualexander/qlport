"""Ingestion maps between what each system stores (paper §3).

phi:    cumulative counter (Prometheus)  ->  per-interval deltas (Datadog COUNT)
cumsum: per-interval deltas (DogStatsD)  ->  cumulative counter (statsd_exporter -> Prometheus)

Datadog documents that the first cumulative point is omitted. Its behaviour on a counter
reset is undocumented, so the reset policy is a parameter to be fixed empirically.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .model import Database, Series

ResetPolicy = Literal["zero", "drop"]


def phi_series(s: Series, reset: ResetPolicy = "zero") -> Series:
    if s.kind != "counter":
        return s  # gauges map by the identity
    ts, ds = [], []
    for i in range(1, len(s.ts)):
        d = s.vs[i] - s.vs[i - 1]
        if d < 0:
            if reset == "zero":
                d = s.vs[i]
            else:  # "drop"
                continue
        ts.append(s.ts[i])
        ds.append(d)
    return Series(dict(s.labels), np.array(ts), np.array(ds), kind="delta")


def phi(db: Database, reset: ResetPolicy = "zero") -> Database:
    return Database([phi_series(s, reset) for s in db.series])


def cumsum_series(s: Series, start: float = 0.0) -> Series:
    if s.kind != "delta":
        return s
    return Series(dict(s.labels), s.ts.copy(), start + np.cumsum(s.vs), kind="counter")

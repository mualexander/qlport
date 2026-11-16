"""Ingestion maps between what each system stores (paper §3).

phi:    cumulative counter (Prometheus)  ->  per-interval deltas (Datadog COUNT)
cumsum: per-interval deltas (DogStatsD)  ->  cumulative counter (statsd_exporter -> Prometheus)

Datadog documents that the first cumulative point is omitted. Its behaviour on a counter
reset is undocumented, but its agent is open source, so the policy is read off
`pkg/metrics/monotonic_count.go` rather than guessed (paper Remark 1):

  zero   a decrease contributes the new value, on the assumption that the counter
         restarted from 0 -- what the OpenMetrics check does from its second scrape
         onward, and what PromQL's own reset handling assumes;
  clamp  the decrease contributes nothing and the point is still emitted -- what the
         same code does with the flag clear, i.e. for non-OpenMetrics checks;
  drop   the interval is discarded. Not a reset policy in the agent at all: it is what
         happens to a series' first interval, and to the interval after a failed scrape
         once the aggregator's predecessor has expired. Kept here because the
         trial-account experiment predicts against it, so a surprising answer can be
         named rather than merely failing to match.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .model import Database, Series

ResetPolicy = Literal["zero", "clamp", "drop"]


def phi_series(s: Series, reset: ResetPolicy = "zero") -> Series:
    if s.kind != "counter":
        return s  # gauges map by the identity
    ts, ds = [], []
    for i in range(1, len(s.ts)):
        d = s.vs[i] - s.vs[i - 1]
        if d < 0:
            if reset == "zero":
                d = s.vs[i]
            elif reset == "clamp":
                d = 0.0
            elif reset == "drop":
                continue
            else:
                raise ValueError(f"unknown reset policy {reset!r}")
        ts.append(s.ts[i])
        ds.append(d)
    return Series(dict(s.labels), np.array(ts), np.array(ds), kind="delta")


def phi(db: Database, reset: ResetPolicy = "zero") -> Database:
    return Database([phi_series(s, reset) for s in db.series])


def cumsum_series(s: Series, start: float = 0.0) -> Series:
    if s.kind != "delta":
        return s
    return Series(dict(s.labels), s.ts.copy(), start + np.cumsum(s.vs), kind="counter")

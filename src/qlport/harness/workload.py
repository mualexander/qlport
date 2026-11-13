"""Seeded synthetic counter workloads. Same seed -> same series, so experiments are reproducible."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..semantics.model import Series

RateFn = Callable[[float, float, np.random.Generator], float]  # increment over (a, b]


def steady(r: float) -> RateFn:
    return lambda a, b, rng: r * (b - a)


def poisson(r: float) -> RateFn:
    return lambda a, b, rng: float(rng.poisson(r * (b - a)))


def bursty(r: float, burst_every: float, burst_len: float, factor: float) -> RateFn:
    """Poisson traffic with periodic spikes (deploys, incidents): rate × factor during bursts."""
    def f(a, b, rng):
        mid = (a + b) / 2
        mult = factor if (mid % burst_every) < burst_len else 1.0
        return float(rng.poisson(r * mult * (b - a)))
    return f


def scrape(rate_fn: RateFn, interval: float, duration: float, *, seed: int = 681,
           start_value: float = 0.0, jitter: float = 0.0, restarts: Sequence[float] = (),
           labels: dict | None = None) -> Series:
    """Cumulative counter scraped every `interval` s. A restart zeroes the counter, so the next
    scrape holds only the increments since the restart."""
    rng = np.random.default_rng(seed)
    n = int(duration // interval)
    ts = np.arange(1, n + 1) * interval + rng.uniform(-jitter, jitter, n)
    ts = np.sort(ts)
    vs, v, last = [], start_value, 0.0
    for t in ts:
        inc = rate_fn(last, t, rng)
        if any(last < r <= t for r in restarts):
            frac = (t - max(r for r in restarts if last < r <= t)) / (t - last)
            v = inc * frac
        else:
            v += inc
        vs.append(v)
        last = t
    return Series(labels or {}, ts, np.array(vs), kind="counter")

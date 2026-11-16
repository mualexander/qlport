"""Rig A: a counter whose exposed value follows a script, for settling $\\rho$ (E1).

$\\rho$ is the one assumption of paper §3.1 that is read rather than observed. The
agent's OpenMetrics source reports a counter decrease as the new value --- $\\rho =
\\mathsf{zero}$ (Remark `rem:rho`) --- but reading source establishes intent, not what the
backend stores. Settling it end to end needs a counter that actually resets, which is
precisely what the replay of §7 cannot supply: a real request counter does not restart
inside half an hour.

Two decisions here are forced rather than chosen.

The exposed value is a function of **wall-clock time**, not of how many times the
endpoint has been scraped. Two scrapers read the same endpoint --- Prometheus and the
agent --- and a value that advanced per request would hand each of them a different
series, so neither side's reading would be about the series the other saw.

The predictions are computed from the exposition **as actually served**, not from the
ideal schedule. The agent chooses its own scrape phase and its period drifts, so two
scrapes can land in one step of the grid (a zero increment) or skip one (an increment of
two). Scoring against the ideal schedule would read that drift as a disagreement about
$\\rho$. The server therefore logs what value it served at what time, and the policies are
scored against that log. `docs/trial_account_checks.md` records the ideal table the run
was committed to in advance, and the runner reports both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..semantics import datadog_eval as dd
from ..semantics.ingest import phi_series
from ..semantics.model import Series

__all__ = ["Schedule", "value_at", "POLICIES", "deltas_under", "bucket_table",
           "consistent_policies"]

# `raw` is not a reset policy. It is the null hypothesis that no reset handling happens
# at all and the backend stores the signed difference, which would make a restart look
# like a large negative count. It is here to be ruled out, not chosen.
POLICIES = ("zero", "clamp", "drop", "raw")


@dataclass(frozen=True)
class Schedule:
    """The exposed value as a step function of time.

    `step` is the grid the value advances on, chosen equal to the scrape interval so that
    consecutive scrapes see consecutive values and each increment is $1$. `reset_at` is
    the first instant at which the reset value is exposed; the step containing it is the
    one whose increment carries the reset.
    """

    t0: float
    step: float
    reset_at: float | None
    reset_to: int
    pre_reset: int = 1000

    @property
    def reset_index(self) -> int | None:
        if self.reset_at is None:
            return None
        return int((self.reset_at - self.t0) // self.step)

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive")
        k = self.reset_index
        if k is not None and k < 1:
            raise ValueError("the reset needs at least one step of warm-up before it")


def value_at(t: float, s: Schedule) -> int:
    """The counter value a scrape at `t` reads.

    Before the reset the value climbs by one per step and reaches `pre_reset` on the last
    step before it, so the reset's signed difference is a known, large negative number.
    After the reset it climbs by one per step again from `reset_to`.
    """
    k = int((t - s.t0) // s.step)
    if k < 0:
        raise ValueError(f"time {t} precedes the schedule's start {s.t0}")
    K = s.reset_index
    if K is None:
        return s.pre_reset + k
    if k < K:
        return s.pre_reset - (K - 1 - k)
    return s.reset_to + (k - K)


def deltas_under(series: Series, policy: str) -> Series:
    """The per-interval deltas `policy` would have the agent submit for `series`.

    `zero`, `clamp` and `drop` are the policies of `semantics/ingest.py`, so what this
    experiment scores Datadog against is the paper's own $\\phi$ and not a second
    implementation of it.
    """
    if policy == "raw":
        ts = np.asarray(series.ts, dtype=float)[1:]
        vs = np.diff(np.asarray(series.vs, dtype=float))
        return Series(dict(series.labels), ts, vs, kind="delta")
    if policy not in ("zero", "clamp", "drop"):
        raise ValueError(f"unknown policy {policy!r}")
    return phi_series(series, policy)


def bucket_table(ts, vs, interval: float, t0: float, t1: float) -> dict:
    """`{bucket_start: (sum, count)}` over the buckets of `[t0, t1)`.

    The buckets are the ones E2 measured: epoch-aligned, left-closed `[b, b+interval)`
    and named by their start. A bucket holding no delta is omitted rather than recorded
    as zero, because Datadog reports no point for it and a zero would compare equal to a
    real zero sum.
    """
    ts = np.asarray(ts, dtype=float)
    vs = np.asarray(vs, dtype=float)
    out = {}
    for b in dd.bucket_starts(t0, t1, interval):
        m = (ts >= b) & (ts < b + interval)
        if m.any():
            out[float(b)] = (float(vs[m].sum()), int(m.sum()))
    return out


def consistent_policies(exposed_ts, exposed_vs, observed: dict, interval: float,
                        *, labels: dict | None = None) -> dict:
    """Which policies the observed buckets are consistent with.

    `observed` is `{bucket_start: (sum, count)}` as read back from Datadog. Only buckets
    present in both the prediction and the observation are scored: the first bucket of a
    run is short by the sample the agent omits on startup, and the last may be clipped by
    when the run stopped, so neither is evidence about $\\rho$.
    """
    series = Series(dict(labels or {}), np.asarray(exposed_ts, dtype=float),
                    np.asarray(exposed_vs, dtype=float), kind="counter")
    t0, t1 = float(series.ts[0]), float(series.ts[-1]) + interval
    out = {}
    for policy in POLICIES:
        d = deltas_under(series, policy)
        predicted = bucket_table(d.ts, d.vs, interval, t0, t1)
        shared = sorted(set(predicted) & set(observed))
        disagree = [
            {"bucket": b, "predicted": predicted[b], "observed": observed[b]}
            for b in shared
            if not (abs(predicted[b][0] - observed[b][0]) < 1e-6
                    and predicted[b][1] == observed[b][1])
        ]
        out[policy] = {"buckets_scored": len(shared),
                       "disagreements": disagree[:5],
                       "consistent": bool(shared) and not disagree}
    return out

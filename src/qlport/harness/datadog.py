"""Talk to a real Datadog: submit COUNT points at chosen timestamps, read a query back.

This is the "Rig B" side of `docs/trial_account_checks.md`: direct submission through
`/api/v2/series`, which bypasses the agent entirely. That makes it useless for the reset
policy -- rho is a property of the agent's conversion, not of the intake -- and exactly
right for everything else, because it is the only way to control the *timestamp* of a
sample. E2 (are rollup buckets right- or left-closed?) is a question about which bucket a
point at a known instant lands in, and the agent cannot be asked that: it timestamps
samples itself.

Nothing here reaches the network except `submit` and `query`, and nothing here starts an
experiment; `scripts/trial_checks.py` does that and writes every response under
`data/raw/datadog/` before anything reads it, so the notebooks stay offline (README
rule 3).

Credentials come from `.env` (gitignored) and are never logged, never cached, and never
put in a URL: both keys travel as headers. `credentials()` reports *which* variable is
missing and nothing about the values.

Two unit asymmetries live in this file because they are where this experiment is easiest
to get silently wrong:

  * submission timestamps are whole **seconds**;
  * the `pointlist` that comes back is in **milliseconds**.

A factor of 1000 at a bucket edge looks exactly like a closure convention, which is the
thing E2 is trying to measure, so `pointlist` converts once, here.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .prometheus import read_raw, write_raw  # same cache convention, one implementation

__all__ = [
    "COUNT", "GAUGE", "RATE", "UNSPECIFIED",
    "Credentials", "credentials", "count_payload", "submit", "query",
    "pointlist", "reported_interval", "bucket_of", "interpret_closure",
    "consistent_conventions", "interpret_default_aggregator",
    "interpret_default_aggregator",
    "read_raw", "write_raw",
]

UA = "qlport-harness/0.1 (academic query-language study)"

# `/api/v2/series` metric types. Checked against the published client rather than
# recalled: datadog-api-client-python, v2/model/metric_intake_type.py. The ordering is a
# trap -- 3 is GAUGE, not COUNT -- and submitting a gauge would make E3 meaningless,
# since `.as_count()` and the enforced `avg` default are type-dependent.
UNSPECIFIED, COUNT, RATE, GAUGE = 0, 1, 2, 3


@dataclass(frozen=True)
class Credentials:
    """An API key, an application key and a site. The keys never reach `repr`."""

    api_key: str
    app_key: str
    site: str

    def __repr__(self) -> str:  # pragma: no cover - trivial, but load-bearing
        return f"Credentials(site={self.site!r}, api_key=<hidden>, app_key=<hidden>)"

    @property
    def base_url(self) -> str:
        """`https://api.<site>` -- the site matters: US1 and EU are different hosts."""
        return f"https://api.{self.site.strip().strip('/')}"

    def headers(self) -> dict[str, str]:
        return {
            "DD-API-KEY": self.api_key,
            "DD-APPLICATION-KEY": self.app_key,
            "Content-Type": "application/json",
            "User-Agent": UA,
        }


def credentials(env: dict[str, str] | None = None) -> Credentials:
    """Read `DD_API_KEY`, `DD_APP_KEY` and `DD_SITE`, loading `.env` if present.

    Raises on a missing variable rather than falling back to a default, because every
    plausible default is somebody's production org. The message names the variable and
    never the value.
    """
    if env is None:
        try:
            from dotenv import load_dotenv
        except ImportError:  # pragma: no cover - dotenv is a declared dependency
            pass
        else:
            load_dotenv(override=False)
        env = dict(os.environ)

    missing = [k for k in ("DD_API_KEY", "DD_APP_KEY", "DD_SITE") if not env.get(k)]
    if missing:
        raise RuntimeError(
            "missing from the environment or .env: " + ", ".join(missing)
            + " (DD_SITE is the bare site, e.g. datadoghq.com or datadoghq.eu)"
        )
    return Credentials(env["DD_API_KEY"], env["DD_APP_KEY"], env["DD_SITE"])


def count_payload(metric: str, points: list[tuple[float, float]], *,
                  tags: list[str] | None = None,
                  interval: int | None = None,
                  metric_type: int = COUNT) -> dict:
    """A `/api/v2/series` body submitting `points` as `(timestamp_seconds, value)`.

    Timestamps are coerced to whole seconds, which is the unit the intake accepts; a
    fractional timestamp is a silent rounding at exactly the bucket edge E2 probes, so
    it is refused here instead.

    `interval` is required by the intake for RATE and optional for COUNT, where it is
    only metadata -- it does not rescale the value. Keep `tags` to one or two series:
    custom metrics are the billed axis on a trial.
    """
    if not points:
        raise ValueError("no points to submit")
    series_points = []
    for t, v in points:
        if float(t) != int(t):
            raise ValueError(f"submission timestamps must be whole seconds, got {t!r}")
        series_points.append({"timestamp": int(t), "value": float(v)})

    series: dict = {"metric": metric, "type": metric_type, "points": series_points}
    if tags:
        series["tags"] = list(tags)
    if interval is not None:
        series["interval"] = int(interval)
    return {"series": [series]}


def _post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:  # body carries the reason; the key does not
        raise RuntimeError(f"{exc.code} {exc.reason}: {exc.read()[:400].decode(errors='replace')}") from None
    return json.loads(raw) if raw else {}


def submit(creds: Credentials, payload: dict, *, timeout: float = 30.0) -> dict:
    """`POST /api/v2/series`. Returns the decoded body ({} on an empty 202).

    The intake accepts a point only if its timestamp is within roughly an hour of now,
    so every experiment built on this is wall-clock-bound rather than a batch job.
    """
    return _post(f"{creds.base_url}/api/v2/series", json.dumps(payload).encode(),
                 creds.headers(), timeout)


def query(creds: Credentials, q: str, start: float, end: float,
          *, timeout: float = 60.0) -> dict:
    """One `GET /api/v1/query`: the raw decoded JSON, cached by the caller.

    `from`/`to` are whole seconds. Datadog picks the rollup interval itself unless the
    query says otherwise, and reports its choice in the response -- see
    `reported_interval`, which is what E4 reads.
    """
    params = urllib.parse.urlencode({"query": q, "from": int(start), "to": int(end)})
    req = urllib.request.Request(f"{creds.base_url}/api/v1/query?{params}",
                                 headers=creds.headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{exc.code} {exc.reason}: {exc.read()[:400].decode(errors='replace')}") from None
    if raw.get("status") == "error":
        raise ValueError(f"query failed: {raw.get('error')}")
    return raw


def _one_series(raw: dict) -> dict | None:
    series = raw.get("series") or []
    if not series:
        return None
    if len(series) > 1:
        raise ValueError(f"expected one series, got {len(series)}")
    return series[0]


def pointlist(raw: dict) -> dict[float, float | None]:
    """A query response as {timestamp_seconds: value}, keeping `None` for empty buckets.

    Nulls are kept rather than dropped because they are the measurement: a bucket that
    exists and holds nothing is what shows where the grid falls, which is half of E2's
    answer. Keyed by timestamp for the same reason as the Prometheus side -- comparing
    two responses by list position would line up different instants.
    """
    s = _one_series(raw)
    if s is None:
        return {}
    out: dict[float, float | None] = {}
    for t_ms, v in s.get("pointlist") or []:
        out[float(t_ms) / 1000.0] = None if v is None else float(v)
    return out


def reported_interval(raw: dict) -> int | None:
    """The rollup interval Datadog says it applied, in seconds.

    This is E4's instrument: queried with no `.rollup()`, the value should match the
    documented time-frame table, and the documentation publishes that table for widgets
    without promising the API agrees.
    """
    s = _one_series(raw)
    return None if s is None or s.get("interval") is None else int(s["interval"])


# --- E2: which bucket does an instant belong to, and which edge is it labelled by? ---
#
# Two conventions are unknown at once, and conflating them is the easy mistake: whether a
# bucket is right-closed (t-D, t] or left-closed [t, t+D), and whether the timestamp
# Datadog reports for a bucket is its start or its end. A single point on the grid cannot
# separate them -- under "right-closed, labelled by end" and "left-closed, labelled by
# start" a point at a multiple of D is reported at the same timestamp either way.
#
# A second point in mid-bucket does separate them, because its bucket is the same under
# both closures, so where it is reported depends only on the labelling. With the
# labelling known, the on-grid point then fixes the closure. Hence E2 submits two probes.


def bucket_of(t: float, delta: float, *, closure: str, label: str) -> float:
    """The timestamp a bucket containing instant `t` would be reported under.

    `closure` is "right" for (b, b+delta] or "left" for [b, b+delta); `label` is "start"
    or "end" for which edge names the bucket.
    """
    if closure not in ("right", "left"):
        raise ValueError(f"closure must be 'right' or 'left', got {closure!r}")
    if label not in ("start", "end"):
        raise ValueError(f"label must be 'start' or 'end', got {label!r}")
    if closure == "right":
        # t exactly on the grid belongs to the bucket *below* it.
        b = (math.ceil(t / delta) - 1) * delta
    else:
        b = math.floor(t / delta) * delta
    return float(b if label == "start" else b + delta)


def consistent_conventions(observations) -> list[dict]:
    """Which (closure, label) conventions explain every observation.

    Each observation is `(submitted_at, delta, reported_bucket)`: a point submitted at
    one instant, read back under a rollup of that width, and reported under that bucket
    timestamp. Returns every hypothesis consistent with all of them, which is the honest
    output -- one surviving hypothesis is an answer, several means the probes did not
    separate them, and none means the server does something this model does not cover.
    """
    return [
        {"closure": c, "label": l}
        for c in ("right", "left")
        for l in ("start", "end")
        if all(bucket_of(float(s), float(d), closure=c, label=l) == float(b)
               for s, d, b in observations)
    ]


def interpret_closure(t_star: float, delta: float, ongrid_ts: float, mid_ts: float,
                      *, mid_submitted_at: float | None = None) -> dict:
    """Read E2's two probes back as a (closure, label) pair.

    `t_star` is a multiple of `delta` and `ongrid_ts` the bucket reported for a point
    submitted at exactly `t_star`. `mid_ts` is the bucket reported for the second probe,
    which must sit strictly inside a bucket; it defaults to `t_star + delta/2` and
    `mid_submitted_at` states it when the probe was placed elsewhere.

    That argument exists because the mid-bucket probe is only informative while it lies
    inside the same bucket the model assumes. Reading a probe placed 1.5 buckets away as
    though it were half a bucket away yields a confident and wrong answer, which is a
    mistake made once already.
    """
    mid_at = t_star + delta / 2.0 if mid_submitted_at is None else float(mid_submitted_at)
    if not 0 < (mid_at - t_star) % delta < delta:
        raise ValueError(
            f"the second probe must fall strictly inside a bucket; {mid_at - t_star:+g}s "
            f"from t* is on a {delta:g}s boundary")
    observations = [(t_star, delta, ongrid_ts), (mid_at, delta, mid_ts)]
    consistent = consistent_conventions(observations)
    return {
        "t_star": float(t_star),
        "delta": float(delta),
        "ongrid_reported_at": float(ongrid_ts),
        "midbucket_submitted_at": mid_at,
        "midbucket_reported_at": float(mid_ts),
        "consistent_with": consistent,
        "conclusive": len(consistent) == 1,
    }


def interpret_default_aggregator(interval: int | None, points, values, defined: dict) -> dict:
    """Name the aggregator a bare query behaved as, or say why it cannot be named.

    E3. All four samples must sit strictly inside one bucket of the interval the
    server reports, or the comparison is between a partial bucket and a whole
    prediction -- which would name an aggregator confidently and wrongly.
    """
    if interval is None:
        return {"conclusive": False, "why": "the response reported no interval"}
    delta = float(interval)
    # All four samples must sit strictly inside one bucket, under either closure, or the
    # comparison is between a partial bucket and a full prediction.
    buckets = {bucket_of(t, delta, closure=c, label="start")
               for t, _ in points for c in ("right", "left")}
    on_edge = [t for t, _ in points if t % delta == 0]
    if len(buckets) != 1 or on_edge:
        return {"conclusive": False, "interval": interval,
                "why": f"the four samples do not share one {interval}s bucket "
                       f"(buckets={sorted(buckets)}, on_edge={on_edge})"}
    got = [v for v in defined.values() if v]
    if len(got) != 1:
        return {"conclusive": False, "interval": interval,
                "why": f"expected one non-zero bucket, got {len(got)}"}
    value, total = got[0], sum(values)
    predictions = {
        "avg": statistics.fmean(values),
        "sum (as if .as_count())": total,
        "sum/interval (as if .as_rate())": total / delta,
    }
    matches = [k for k, p in predictions.items() if math.isclose(value, p, rel_tol=1e-6)]
    return {"conclusive": len(matches) == 1, "interval": interval, "value": value,
            "predictions": predictions, "behaved_as": matches}

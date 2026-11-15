"""Talk to a real Prometheus: render a workload for backfill, and read a time grid back.

Nothing here starts a server or reaches for Docker; `scripts/verify_anchored.py` does
that and writes every response to `data/raw/` before anything reads it, so the
notebooks stay offline (README rule 3).

The point of going to a real server at all is that `semantics/promql_eval.py` is only a
*model* of PromQL. The `anchored` modifier in particular was written from the proposal
text, so until a real Prometheus answers the same questions on the same samples, any
claim about it is a claim about our own code.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from ..semantics.model import Series

UA = "qlport-harness/0.1 (academic query-language study)"


def quantise(s: Series) -> Series:
    """Round timestamps to whole milliseconds.

    Prometheus stores sample times as integer milliseconds. Rounding here, once, means
    the server and the reference evaluators are given the *same* series rather than two
    series that differ in the sixth decimal place -- otherwise a disagreement of
    1e-7 s at a window boundary looks like a semantic difference.
    """
    ts = np.round(np.asarray(s.ts, dtype=float) * 1000.0) / 1000.0
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):
        raise ValueError("rounding to milliseconds collided two samples")
    return Series(dict(s.labels), ts, np.asarray(s.vs, dtype=float), kind=s.kind)


def openmetrics(s: Series, metric: str, t0: float) -> str:
    """An OpenMetrics exposition of `s`, shifted to start at `t0`.

    `promtool tsdb create-blocks-from openmetrics` reads this. OpenMetrics names a
    counter's sample series `<metric>_total`, so the TYPE line carries the bare name
    and the samples carry the suffix.
    """
    if s.kind != "counter":
        raise ValueError(f"expected a counter series, got {s.kind!r}")
    labels = "".join(f'{k}="{v}",' for k, v in sorted(s.labels.items()))
    sel = f"{metric}_total{{{labels.rstrip(',')}}}" if labels else f"{metric}_total"
    # %.17g round-trips a float64 exactly. Writing fewer digits hands the server a
    # slightly different series from the one the reference evaluators see, which then
    # shows up as a ~1e-10 "disagreement" that is really our own rounding.
    lines = [f"# TYPE {metric} counter"]
    lines += [f"{sel} {v:.17g} {t0 + t:.3f}" for t, v in zip(s.ts, s.vs)]
    lines.append("# EOF")
    return "\n".join(lines) + "\n"


def query_range(base_url: str, expr: str, start: float, end: float, step: float,
                *, timeout: float = 120.0) -> dict:
    """One `/api/v1/query_range` call: the raw decoded JSON, cached by the caller.

    A range query evaluates at `start, start+step, ...`, so choosing `start` as a
    multiple of `step` puts the evaluation grid exactly on Datadog's epoch-aligned
    bucket ends -- which is the only grid on which the two systems are comparable at
    all (the ROLLUP_ALIGNMENT caveat of emit/promql.py).
    """
    body = urllib.parse.urlencode(
        {"query": expr, "start": f"{start:.3f}", "end": f"{end:.3f}", "step": f"{step:g}"}
    ).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/v1/query_range", data=body,
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def series_from_response(raw: dict) -> dict[float, float]:
    """A range response as {timestamp: value}.

    Keyed by timestamp on purpose: Prometheus simply omits the steps where an
    expression has no value (a rate over fewer than two samples, say), so aligning two
    candidates by list position would silently compare different instants.
    """
    if raw.get("status") != "success":
        raise ValueError(f"query failed: {raw.get('errorType')}: {raw.get('error')}")
    result = raw["data"]["result"]
    if not result:
        return {}
    if len(result) > 1:
        raise ValueError(f"expected one series, got {len(result)}")
    return {float(t): float(v) for t, v in result[0]["values"]}


def write_raw(path: Path | str, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


def read_raw(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())

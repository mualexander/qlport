"""E6: where `.as_rate()` sits relative to rollup and to space aggregation.

Three questions the corpus-scale differential test raised, none of them answerable from
the documentation, all of them translation-critical:

  (a) Order against an explicit rollup. `.as_rate().rollup(avg, R)` could divide each
      raw delta by its own submission interval and then average those rates, or average
      the deltas and divide once by `R`. The two agree whenever the submission grid
      equals `R`, which is why a harness that scrapes at the rollup width cannot see the
      difference; this one submits every 10 s and rolls up over 30 s so they separate.

  (b) What `.as_rate()` does to a GAUGE. Datadog documents the modifier for COUNT and
      RATE metrics and says nothing about gauges, so a translation that divides anyway
      is guessing.

  (c) Order against space aggregation. `sum:` commutes with dividing by a constant, so
      `sum(x)/R` and `sum(x/R)` agree and the question stays invisible. `count:` does
      not commute: `count(x/R) = count(x)`. Whichever side Datadog applies the modifier
      on is therefore observable through `count:` and only through `count:`.

Writes data/raw/datadog/as_rate_order.json; interpretation lives with the caller.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlport.harness.datadog import (                                   # noqa: E402
    COUNT, GAUGE, count_payload, credentials, query, submit,
)

OUT = Path(__file__).resolve().parents[1] / "data/raw/datadog/as_rate_order.json"

STEP = 10.0          # submission grid
ROLLUP = 30.0        # explicit rollup width: deliberately not equal to STEP
SPAN = 1800.0        # half an hour of points
DELTA = 10.0         # per-point value for the COUNT metrics
LEVEL = 50.0         # value for the GAUGE metric


def series_points(t0: float, value: float) -> list[tuple[float, float]]:
    n = int(SPAN // STEP)
    return [(t0 + k * STEP, value) for k in range(n)]


def multi(raw: dict) -> dict[str, dict[float, float | None]]:
    """Every series in a response, keyed by its scope string."""
    out = {}
    for s in raw.get("series") or []:
        pts = {float(t) / 1000.0: (None if v is None else float(v))
               for t, v in (s.get("pointlist") or [])}
        out[s.get("scope", "")] = pts
    return out


def interval_of(raw: dict) -> int | None:
    """The rollup interval Datadog reports, read off a possibly multi-series response.

    `harness.datadog.reported_interval` refuses more than one series on purpose --- E4
    measures a single probe and two series there would mean the query was wrong. Here a
    grouped query returns one series per group by design, and they share the interval.
    """
    for s in raw.get("series") or []:
        if s.get("interval") is not None:
            return int(s["interval"])
    return None


def defined(pts: dict[float, float | None]) -> list[float]:
    return [v for v in pts.values() if v is not None]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="re-query an earlier run's metrics instead of "
                                  "submitting new ones (the rNNNNNNNNNN tag)")
    args = ap.parse_args(argv)

    creds = credentials()
    if args.run:
        run = args.run
        t0 = float(run.lstrip("r"))
    else:
        t0 = float(int(time.time() - SPAN - 240.0))
        run = f"r{int(t0)}"
    cnt = f"qlport.ar.{run}.cnt"
    gau = f"qlport.ar.{run}.gau"
    grp = f"qlport.ar.{run}.grp"

    if args.run:
        print(f"re-querying {run} without resubmitting")
    else:
        print(f"submitting {run}: COUNT {DELTA}/{STEP:.0f}s, GAUGE {LEVEL}, "
              f"3 grouped COUNT series")
    # one COUNT series
    pts = series_points(t0, DELTA)
    if args.run:
        pts = []
    for i in range(0, len(pts), 200):
        submit(creds, count_payload(cnt, pts[i:i + 200], tags=["probe:a"],
                                    metric_type=COUNT))
    # one GAUGE series
    gpts = series_points(t0, LEVEL) if not args.run else []
    for i in range(0, len(gpts), 200):
        submit(creds, count_payload(gau, gpts[i:i + 200], tags=["probe:b"],
                                    metric_type=GAUGE))
    # three grouped COUNT series
    for k in (() if args.run else range(3)):
        for i in range(0, len(pts), 200):
            submit(creds, count_payload(grp, pts[i:i + 200], tags=[f"g:g{k}"],
                                        metric_type=COUNT))

    queries = {
        # (a) rollup order, COUNT
        "cnt.rollup_avg": f"sum:{cnt}{{*}}.rollup(avg, {int(ROLLUP)})",
        "cnt.rollup_sum": f"sum:{cnt}{{*}}.rollup(sum, {int(ROLLUP)})",
        "cnt.as_rate": f"sum:{cnt}{{*}}.as_rate()",
        "cnt.as_rate.rollup_avg": f"sum:{cnt}{{*}}.as_rate().rollup(avg, {int(ROLLUP)})",
        "cnt.as_rate.rollup_sum": f"sum:{cnt}{{*}}.as_rate().rollup(sum, {int(ROLLUP)})",
        "cnt.as_count.rollup_avg": f"sum:{cnt}{{*}}.as_count().rollup(avg, {int(ROLLUP)})",
        # (b) as_rate on a gauge
        "gau.plain": f"sum:{gau}{{*}}",
        "gau.rollup_avg": f"sum:{gau}{{*}}.rollup(avg, {int(ROLLUP)})",
        "gau.as_rate": f"sum:{gau}{{*}}.as_rate()",
        "gau.as_rate.rollup_avg": f"sum:{gau}{{*}}.as_rate().rollup(avg, {int(ROLLUP)})",
        # (b') is `.as_count()` ignored on a gauge too? If both modifiers are no-ops
        # there, neither is evidence of the metric's type --- only of what the
        # dashboard's author believed it to be.
        "gau.as_count": f"sum:{gau}{{*}}.as_count()",
        "gau.as_count.rollup_avg": f"sum:{gau}{{*}}.as_count().rollup(avg, {int(ROLLUP)})",
        "gau.as_rate.rollup_sum": f"sum:{gau}{{*}}.as_rate().rollup(sum, {int(ROLLUP)})",
        # (a') the same ordering question against a summing rollup: dividing each delta
        # by its own 10 s interval and summing gives 3, rolling up and dividing once by
        # 30 gives 1.
        "grp.sum.as_rate.rollup_sum":
            f"sum:{grp}{{*}} by {{g}}.as_rate().rollup(sum, {int(ROLLUP)})",
        "grp.sum.as_count": f"sum:{grp}{{*}} by {{g}}.as_count().rollup(avg, {int(ROLLUP)})",
        "grp.count.as_count":
            f"count:{grp}{{*}} by {{g}}.as_count().rollup(avg, {int(ROLLUP)})",
        # (c) as_rate against space aggregation
        "grp.count": f"count:{grp}{{*}} by {{g}}.rollup(avg, {int(ROLLUP)})",
        "grp.count.as_rate": f"count:{grp}{{*}} by {{g}}.as_rate().rollup(avg, {int(ROLLUP)})",
        "grp.sum": f"sum:{grp}{{*}} by {{g}}.rollup(avg, {int(ROLLUP)})",
        "grp.sum.as_rate": f"sum:{grp}{{*}} by {{g}}.as_rate().rollup(avg, {int(ROLLUP)})",
    }

    start, end = t0 - 60.0, t0 + SPAN + 60.0
    results: dict[str, dict] = {}
    for attempt in range(1, 9):
        missing = [k for k in queries if k not in results]
        if not missing:
            break
        print(f"  pass {attempt}: {len(missing)} outstanding")
        for k in missing:
            try:
                raw = query(creds, queries[k], start, end)
            except Exception as exc:                       # noqa: BLE001
                results[k] = {"query": queries[k], "error": str(exc)[:200]}
                continue
            got = multi(raw)
            vals = {scope: defined(p) for scope, p in got.items()}
            if not any(vals.values()):
                continue                                   # not indexed yet
            results[k] = {
                "query": queries[k],
                "reported_interval": interval_of(raw),
                "series": len(got),
                "values": {scope: sorted({round(v, 6) for v in vs})[:6]
                           for scope, vs in vals.items()},
                "n_points": {scope: len(vs) for scope, vs in vals.items()},
            }
        if any(k not in results for k in queries):
            time.sleep(20)

    for k in queries:
        results.setdefault(k, {"query": queries[k], "error": "no data after retries"})

    payload = {
        "run": run, "t0": t0, "step": STEP, "rollup": ROLLUP,
        "count_delta": DELTA, "gauge_level": LEVEL,
        "queries": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}\n")
    for k in queries:
        r = results[k]
        if "error" in r:
            print(f"{k:28s} ERROR {r['error']}")
        else:
            print(f"{k:28s} iv={r['reported_interval']} series={r['series']} "
                  f"{r['values']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

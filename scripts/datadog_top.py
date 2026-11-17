#!/usr/bin/env python3
"""Does Datadog's `top()` do what its documentation says? Asked of a real Datadog.

`scripts/verify_topk.py` settles the PromQL half of paper §7's `top()` argument: the
`@`-pinned expression

    q and on(svc) topk(n, <agg>_over_time(q[R] @ T))

reproduces, on a real Prometheus, the selection Datadog's *documented* rule predicts ---
aggregate each series over the whole queried range, take the top n, return those series'
whole trajectories. §7 says so and then says what is wrong with it: "the Prometheus side
is measured and the Datadog side is a model of the documentation, which is the weaker
half of the comparison and the half this paper has twice found wrong". The two prior
occasions were the aggregation order and the bucket closure, both read off the
documentation, both wrong.

So this is Rig B pointed at the other half. The same five series (`harness/ranking.py`,
shared with the Prometheus script so the two halves cannot drift apart) go to a real
Datadog as a GAUGE, and `top()` itself is asked for its selection. Three claims, each
separately falsifiable:

  1. **The ranking is over the whole queried range.** Every (function, n, direction) the
     model predicts is compared against what comes back. The workload is built so no two
     series score alike, so a disagreement names a different rule rather than a tie.
  2. **Whole trajectories come back, not winning instants.** Series `e` spikes at index 0
     and sits at 1 afterwards. Under `top(..., 'max', 'desc')` it is selected; a
     `topk`-shaped operator would return it only where it is winning. Counting its
     non-null points separates the two.
  3. **The selection depends on the queried range.** Over all 20 buckets the top two by
     `max` are `b` and `e`; over the last ten, `e`'s spike is outside the window and the
     answer is `a` and `b`. If Datadog returns the same pair for both windows, the
     ranking is not range-relative and §7's claim that an implementation needs to be told
     the dashboard's time range is overstated.

Claim 3 is the one that prices §8's argument rather than restating it: `R` and `T` are
preconditions only if the vendor's own answer moves when they move.

Costs five custom metric series for one run (`docs/trial_account_checks.md` asks for one
or two; five is what a ranking needs, and they are one metric name used once).

  python scripts/datadog_top.py [--dry-run]

Writes data/raw/datadog/top.json; tests/test_harness_datadog.py reads it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport.harness.datadog import (                                  # noqa: E402
    GAUGE, count_payload, credentials, query, submit,
)
from qlport.harness.ranking import (                                  # noqa: E402
    N, REDUCERS, SERIES, STEP, model_rank,
)

OUT = ROOT / "data" / "raw" / "datadog" / "top.json"
ROLLUP = STEP            # one submitted point per bucket, so the trajectory is the input
SETTLE = 180.0           # before the first readiness poll; the intake is not synchronous
READY_TIMEOUT = 420.0
DIRECTIONS = ("desc", "asc")
LIMITS = (2, 3)


def inner(metric: str) -> str:
    """The query `top()` ranks. One series per group, so the space aggregation is a no-op
    and what is ranked is exactly what was submitted."""
    return f"avg:{metric}{{*}} by {{svc}}.rollup(avg, {int(ROLLUP)})"


def top_query(metric: str, n: int, by: str, direction: str) -> str:
    return f"top({inner(metric)}, {n}, '{by}', '{direction}')"


def _svc(series: dict) -> str:
    """Which of the five series a response series is.

    Datadog reports a scope as a comma-separated tag list (the mistake E5 made was
    matching it whole), and `tag_set` carries the same tags as a list. Read either.
    """
    for token in list(series.get("tag_set") or []) + \
            str(series.get("scope") or "").split(","):
        token = token.strip()
        if token.startswith("svc:"):
            return token[4:]
    return "?"


def read_series(raw: dict) -> dict[str, dict[str, float | None]]:
    """A multi-series response as {svc: {timestamp: value}}, keeping nulls.

    `harness/datadog.pointlist` refuses more than one series by design -- for E2 a second
    series means the probe is wrong. Here n series is the answer, so this is the one place
    that reads them all.
    """
    out: dict[str, dict[str, float | None]] = {}
    for s in raw.get("series") or []:
        pts = {str(float(t_ms) / 1000.0): (None if v is None else float(v))
               for t_ms, v in (s.get("pointlist") or [])}
        out[_svc(s)] = pts
    return out


def defined(points: dict[str, float | None]) -> list[float]:
    return [v for v in points.values() if v is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the payload and queries; no keys, no network")
    args = ap.parse_args()

    # Whole window in the past and inside the intake's ~1 h backfill, with room for the
    # last point to be ingested before the first poll.
    t0 = math.floor((time.time() - N * STEP - 300.0) / STEP) * STEP
    end = t0 + N * STEP
    half = t0 + (N // 2) * STEP
    metric = f"qlport.top.r{int(t0)}.q"

    points = {name: [(t0 + i * STEP, f(i)) for i in range(N)]
              for name, f in SERIES.items()}
    payloads = [count_payload(metric, pts, tags=[f"svc:{name}", "rig:top"],
                              metric_type=GAUGE)
                for name, pts in points.items()]

    windows = {"full": (t0, end, 0, N), "second half": (half, end, N // 2, N)}
    plan: list[dict] = []
    for by in sorted(REDUCERS):
        for n in LIMITS:
            for direction in DIRECTIONS:
                plan.append({"window": "full", "by": by, "n": n,
                             "direction": direction})
    # Claim 3 needs the same question asked of a narrower window. `max` is the only
    # function whose selection the model says moves; `mean` is the control that says it
    # does not, so a run where *everything* moves is distinguishable from a real result.
    for by in ("max", "mean"):
        plan.append({"window": "second half", "by": by, "n": 2, "direction": "desc"})

    if args.dry_run:
        print(f"metric {metric}, {N} points per series at {STEP}s from {t0}")
        print(f"  windows: " + ", ".join(
            f"{k}=[{a}, {b}) indices [{lo},{hi})" for k, (a, b, lo, hi) in windows.items()))
        print(f"  control: {inner(metric)}")
        for case in plan:
            a, b, lo, hi = windows[case["window"]]
            print(f"  {case['window']:<11} {top_query(metric, case['n'], case['by'], case['direction']):<72}"
                  f" model={model_rank(case['by'], case['n'], case['direction'], lo, hi)}")
        print(f"  {len(payloads)} series, {sum(len(p) for p in points.values())} points, "
              f"{len(plan) + 1} queries")
        return 0

    creds = credentials()
    for payload in payloads:
        submit(creds, payload)
    print(f"{metric}: {len(payloads)} series x {N} points at {STEP}s from {t0}")
    print(f"  waiting for ingestion (settling {SETTLE:.0f}s)")
    time.sleep(SETTLE)

    control_raw: dict = {}
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        control_raw = query(creds, inner(metric), t0, end)
        got = read_series(control_raw)
        if len(got) == len(SERIES) and all(len(defined(p)) == N for p in got.values()):
            break
        if time.monotonic() > deadline:
            print(f"  giving up: {len(got)} series, "
                  f"{ {k: len(defined(v)) for k, v in got.items()} } points", file=sys.stderr)
            return 2
        time.sleep(15.0)
    control = read_series(control_raw)
    print(f"  all {len(control)} series present with {N} points each")

    # The values Datadog stored are what every ranking below is about; if they differ from
    # the submission, the model is predicting over numbers the server does not hold.
    stored_matches_input = {
        name: [v for _, v in sorted(control[name].items(), key=lambda kv: float(kv[0]))]
        == [v for _, v in points[name]]
        for name in SERIES}

    cases: list[dict] = []
    raw_responses: dict[str, dict] = {"control": control_raw}
    for case in plan:
        a, b, lo, hi = windows[case["window"]]
        q = top_query(metric, case["n"], case["by"], case["direction"])
        raw = query(creds, q, a, b)
        got = read_series(raw)
        want = model_rank(case["by"], case["n"], case["direction"], lo, hi)
        key = f"{case['window']}|{case['by']}|{case['n']}|{case['direction']}"
        raw_responses[key] = raw
        cases.append({
            **case, "expr": q, "from": a, "to": b, "indices": [lo, hi],
            "model": want, "selected": sorted(got),
            "matches_model": sorted(got) == want,
            "points_per_series": {k: len(defined(v)) for k, v in sorted(got.items())},
            # Claim 2: a selected series must come back over the whole window, not only
            # where it is winning. `hi - lo` buckets is the whole window.
            "whole_trajectories": bool(got) and all(
                len(defined(v)) == hi - lo for v in got.values()),
            "reported_interval": (raw.get("series") or [{}])[0].get("interval"),
        })
        mark = "ok " if cases[-1]["matches_model"] else "NO "
        print(f"  {mark} {case['window']:<11} by={case['by']:<5} n={case['n']} "
              f"{case['direction']}  model={want} got={sorted(got)}")

    full = [c for c in cases if c["window"] == "full"]
    narrow = {c["by"]: c for c in cases if c["window"] == "second half"}
    range_relative = {
        by: {"full": next(c["selected"] for c in full
                          if c["by"] == by and c["n"] == 2 and c["direction"] == "desc"),
             "second half": narrow[by]["selected"],
             "model_moves": model_rank(by, 2, "desc", 0, N)
                            != model_rank(by, 2, "desc", N // 2, N)}
        for by in narrow}
    for by, obs in range_relative.items():
        obs["moved"] = obs["full"] != obs["second half"]
        obs["as_model_predicts"] = obs["moved"] == obs["model_moves"]

    agree = sum(c["matches_model"] for c in cases)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "provenance": {
            "site": creds.site,
            "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "metric": metric, "t0": t0, "step": STEP, "n_points": N,
            "rollup_seconds": int(ROLLUP),
            "windows": {k: {"from": a, "to": b, "indices": [lo, hi]}
                        for k, (a, b, lo, hi) in windows.items()},
            "series": {k: [v for _, v in pts] for k, pts in points.items()},
            "submitted": payloads,
        },
        "control": {"expr": inner(metric), "series": control,
                    "stored_matches_input": stored_matches_input},
        "cases": cases,
        "range_relative": range_relative,
        "agreement": {"cases": len(cases), "matching_model": agree},
        "raw": raw_responses,
    }, indent=1, sort_keys=True) + "\n")

    print(f"\n{agree}/{len(cases)} selections match the documented rule")
    for by, obs in range_relative.items():
        print(f"  by={by}: full={obs['full']} second half={obs['second half']}  "
              f"moved={obs['moved']} (model says {obs['model_moves']})")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if agree == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

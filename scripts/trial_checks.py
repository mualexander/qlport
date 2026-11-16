#!/usr/bin/env python3
"""Settle Datadog's undocumented rollup behaviour against a real trial account.

The experiments are specified in docs/trial_account_checks.md; this runs them and caches
every raw response under data/raw/datadog/ so the notebooks stay offline (README rule 3).

  python scripts/trial_checks.py e2            # right- or left-closed buckets?
  python scripts/trial_checks.py e3            # default time aggregator for a COUNT
  python scripts/trial_checks.py e4            # the automatic rollup interval
  python scripts/trial_checks.py e5            # does a glob's `*` match empty?
  python scripts/trial_checks.py e2 --dry-run  # print what would be sent, touch nothing

Needs DD_API_KEY, DD_APP_KEY and DD_SITE in .env (gitignored). Keys are read at call
time, travel as headers and are never written to the cache.

Everything here is wall-clock-bound: the intake rejects points older than roughly an
hour, so these submit into the recent past and then wait for ingestion. E4 backfills a
full hour in one go, which is why it does not need the two hours of real time the plan
first assumed -- the interval Datadog reports is a function of the query range, not of
how long the data took to arrive.

Billing: three custom metrics, one series each, a few hundred points. Custom metrics are
the billed axis, so the tag set is deliberately `rig:b` and nothing else.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qlport.harness.datadog import (  # noqa: E402
    COUNT, Credentials, count_payload, credentials, interpret_closure,
    interpret_default_aggregator, pointlist, query, reported_interval, submit,
    write_raw,
)

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog"
TAGS = ["rig:b"]
DELTA = 60.0           # the rollup interval E2 probes
SETTLE = 300           # submit this far into the past, so the point is queryable
POLL_TIMEOUT = 600.0   # a brand-new custom metric can take minutes to appear
POLL_EVERY = 10.0


def _now() -> float:
    return time.time()


def _grid(t: float, delta: float = DELTA) -> float:
    """The largest multiple of `delta` at or below `t`."""
    return math.floor(t / delta) * delta


def _provenance(creds: Credentials, **extra) -> dict:
    """What the cache records about the run. Deliberately no key material."""
    return {
        "site": creds.site,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api": {"submit": "/api/v2/series", "query": "/api/v1/query"},
        "metric_type": COUNT,
        **extra,
    }


def _submit_all(creds: Credentials, payloads: list[dict], *, dry: bool) -> list[dict]:
    if dry:
        for p in payloads:
            print(json.dumps(p, indent=1))
        return []
    acks = []
    for p in payloads:
        acks.append(submit(creds, p))
    return acks


def _await_all(creds: Credentials, queries: dict[str, str], start: float,
               end: float) -> dict[str, dict]:
    """Poll every query until each has a non-null point, or the budget runs out.

    One pass per query per round, rather than one query polled to exhaustion and then
    the next: a newly created custom metric can take minutes to become queryable, and
    serial polling spends the whole budget on whichever probe happens to be first.
    """
    deadline = _now() + POLL_TIMEOUT
    out: dict[str, dict] = {}
    while True:
        for name, q in queries.items():
            if name in out:
                continue
            raw = query(creds, q, start, end)
            if any(v for v in pointlist(raw).values()):
                out[name] = raw
            else:
                out.setdefault("_last_" + name, raw)
        if len(set(queries) & set(out)) == len(queries) or _now() >= deadline:
            # Fall back to the last empty response so the cache records what was said.
            for name in queries:
                out.setdefault(name, out.get("_last_" + name, {}))
            return {k: v for k, v in out.items() if not k.startswith("_last_")}
        time.sleep(POLL_EVERY)


def _await_points(creds: Credentials, q: str, start: float, end: float, *,
                  want: int = 1) -> dict:
    """Query until at least `want` non-null points come back, or give up.

    Giving up returns the last response rather than raising, so the cache records what
    the server actually said; an experiment with too few points is reported inconclusive
    by the caller rather than silently interpreted.
    """
    deadline = _now() + POLL_TIMEOUT
    raw: dict = {}
    while True:
        raw = query(creds, q, start, end)
        if sum(1 for v in pointlist(raw).values() if v is not None) >= want:
            return raw
        if _now() >= deadline:
            return raw
        time.sleep(POLL_EVERY)


def _defined(raw: dict) -> dict[float, float]:
    return {t: v for t, v in pointlist(raw).items() if v is not None}


# ------------------------------------------------------------------ E2

def e2(creds: Credentials, *, dry: bool) -> dict:
    """Are rollup buckets right- or left-closed?

    Two probes, because one cannot answer it: a point exactly on the grid is reported at
    the same timestamp under "right-closed, labelled by end" as under "left-closed,
    labelled by start". A mid-bucket point sits in the same bucket under either closure,
    so it fixes the labelling, and the on-grid point then fixes the closure.
    """
    t_star = _grid(_now() - SETTLE)
    # Every run gets its own tag. Without one, a re-run submits to the same metric names
    # and the wide query window picks up the *previous* run's points -- which is exactly
    # what happened the first time this was repeated, and it produced two probes both
    # reporting a bucket from the earlier run. The run tag makes each run self-contained
    # at the cost of one extra series per run rather than one extra custom metric.
    run = f"r{int(t_star)}"
    tags = TAGS + [f"run:{run}"]
    probes = {
        "ongrid": ("qlport.e2.ongrid", t_star),
        "midbucket": ("qlport.e2.midbucket", t_star + DELTA / 2.0),
    }
    payloads = [count_payload(m, [(t, 1000.0)], tags=tags) for m, t in probes.values()]
    acks = _submit_all(creds, payloads, dry=dry)
    if dry:
        for name, (metric, t) in probes.items():
            print(f"# then query  sum:{metric}{{run:{run}}}.rollup(sum, 60)"
                  f"  from {t_star - 900:.0f} to {t_star + 900:.0f}  ({name} at {t:.0f})")
        return {}

    # A wide window on purpose: the bucket a probe lands in is the measurement, and a
    # narrow window risks clipping it if the convention is the opposite of the assumed
    # one. Both probes are polled together -- polling one to exhaustion before starting
    # the other starved the second and made the first run of this inconclusive.
    start, end = t_star - 900, t_star + 900
    queries = {name: f"sum:{metric}{{run:{run}}}.rollup(sum, {DELTA:.0f})"
               for name, (metric, _) in probes.items()}
    raws = _await_all(creds, queries, start, end)

    out: dict = {"provenance": _provenance(creds, experiment="e2", t_star=t_star, run=run,
                                           delta=DELTA, probes={k: v[1] for k, v in probes.items()}),
                 "submitted": payloads, "acks": acks, "responses": {}, "reported_at": {}}
    for name, raw in raws.items():
        out["responses"][name] = {"query": queries[name], "from": start, "to": end,
                                  "raw": raw}
        out["reported_at"][name] = [t for t, v in _defined(raw).items() if v != 0.0]

    ongrid, mid = out["reported_at"]["ongrid"], out["reported_at"]["midbucket"]
    if len(ongrid) == 1 and len(mid) == 1:
        out["interpretation"] = interpret_closure(
            t_star, DELTA, ongrid[0], mid[0],
            mid_submitted_at=probes["midbucket"][1])
    else:
        out["interpretation"] = {
            "conclusive": False,
            "why": "expected exactly one non-zero bucket per probe; "
                   f"got {len(ongrid)} on-grid and {len(mid)} mid-bucket",
        }
    return out


# ------------------------------------------------------------------ E3

def e3(creds: Credentials, *, dry: bool) -> dict:
    """What time aggregator does a bare API query apply to a COUNT?

    `avg` is the documented enforced default, but COUNT and RATE metrics are documented
    to get `.as_count()` appended *in the UI*, and whether a bare API query does the same
    is not stated. METRIC_TYPE_UNKNOWN depends on the answer.

    The four points are packed into four consecutive seconds well inside a bucket, so
    they share one bucket under every rollup interval Datadog is likely to pick; the
    prediction is then read against the interval it reports rather than against an
    assumed 60.
    """
    t_star = _grid(_now() - SETTLE)
    offsets, values = (31, 32, 33, 34), (1.0, 2.0, 3.0, 10.0)
    metric = "qlport.e3.deltas"
    points = [(t_star + o, v) for o, v in zip(offsets, values)]
    payloads = [count_payload(metric, points, tags=TAGS)]
    acks = _submit_all(creds, payloads, dry=dry)

    variants = {"bare": "", "as_count": ".as_count()", "as_rate": ".as_rate()"}
    # The automatic interval is a function of the query range (E4's subject), and a
    # narrow range buys a *finer* one: a 240 s window returned a 1 s rollup, which put
    # each of the four samples in its own bucket and made the comparison meaningless.
    # An hour is coarse enough to hold them together and is what a dashboard would use.
    start, end = t_star - 1800, t_star + 1800
    if dry:
        for name, suffix in variants.items():
            print(f"# then query  sum:{metric}{{*}}{suffix}"
                  f"  from {start:.0f} to {end:.0f}  ({name})")
        return {}

    out: dict = {"provenance": _provenance(creds, experiment="e3", t_star=t_star,
                                           points=points, sum_of_values=sum(values),
                                           mean_of_values=statistics.fmean(values)),
                 "submitted": payloads, "acks": acks, "responses": {}}
    for name, suffix in variants.items():
        q = f"sum:{metric}{{*}}{suffix}"
        raw = _await_points(creds, q, start, end)
        interval = reported_interval(raw)
        out["responses"][name] = {
            "query": q, "from": start, "to": end, "raw": raw,
            "interval": interval,
            "defined": {str(t): v for t, v in _defined(raw).items()},
        }

    interval = out["responses"]["bare"]["interval"]
    out["interpretation"] = interpret_default_aggregator(
        interval, points, values, out["responses"]["bare"]["defined"])
    return out


# ------------------------------------------------------------------ E4

def e4(creds: Credentials, *, dry: bool) -> dict:
    """Confirm the automatic rollup interval, and what a too-fine manual one does.

    The documented time-frame table is published for widgets, and its API column is not
    promised to match `/api/v1/query`. The interval Datadog reports is a function of the
    query range rather than of the data, so an hour of backfill is enough to ask about a
    one-day range too -- the far buckets simply come back null.
    """
    now = _grid(_now() - 60)
    metric = "qlport.e4.dense"
    spacing, span = 10, 3500           # an hour, inside the intake's backfill window
    points = [(now - span + i * spacing, 1.0)
              for i in range(span // spacing + 1)]
    payloads = [count_payload(metric, points[i:i + 100], tags=TAGS)
                for i in range(0, len(points), 100)]
    ranges = {"1m": 60, "5m": 300, "1h": 3600, "4h": 14400, "1d": 86400}
    acks = _submit_all(creds, payloads, dry=dry)
    if dry:
        print(f"# {len(points)} points every {spacing}s over {span}s, "
              f"in {len(payloads)} request(s)")
        for name, width in ranges.items():
            print(f"# then query  sum:{metric}{{*}}  over {name} ({width}s)")
        print(f"# and          sum:{metric}{{*}}.rollup(sum, 1)  over 1d")
        return {}

    # Wait once, on the range most likely to show the data, before timing the rest.
    _await_points(creds, f"sum:{metric}{{*}}", now - 3600, now, want=3)

    out: dict = {"provenance": _provenance(creds, experiment="e4", spacing=spacing,
                                           span=span, n_points=len(points), anchor=now),
                 "submitted_request_count": len(payloads), "acks": acks,
                 "responses": {}}
    for name, width in ranges.items():
        q = f"sum:{metric}{{*}}"
        raw = query(creds, q, now - width, now)
        ts = sorted(pointlist(raw))
        gaps = [round(b - a, 3) for a, b in zip(ts, ts[1:])]
        out["responses"][name] = {
            "query": q, "range_seconds": width, "raw": raw,
            "reported_interval": reported_interval(raw),
            "n_points": len(ts),
            "observed_spacing": statistics.mode(gaps) if gaps else None,
        }

    q_fine = f"sum:{metric}{{*}}.rollup(sum, 1)"
    try:
        raw = query(creds, q_fine, now - 86400, now)
        ts = sorted(pointlist(raw))
        gaps = [round(b - a, 3) for a, b in zip(ts, ts[1:])]
        out["too_fine_rollup"] = {
            "query": q_fine, "accepted": True, "raw": raw,
            "reported_interval": reported_interval(raw),
            "observed_spacing": statistics.mode(gaps) if gaps else None,
        }
    except (ValueError, RuntimeError) as exc:
        out["too_fine_rollup"] = {"query": q_fine, "accepted": False, "error": str(exc)}

    out["interpretation"] = {
        "reported_by_range": {k: v["reported_interval"] for k, v in out["responses"].items()},
        "observed_by_range": {k: v["observed_spacing"] for k, v in out["responses"].items()},
        "finer_than_enforced": ("silently widened"
                                if out["too_fine_rollup"].get("accepted")
                                else "rejected"),
    }
    return out


# ------------------------------------------------------------------ E5

def _series_by_scope(raw: dict) -> dict[str, float]:
    """Total per returned series, keyed by the scope Datadog names it with.

    `pointlist` deliberately refuses a multi-series response; this experiment wants one,
    because the question is *which* series a glob selects.
    """
    out: dict[str, float] = {}
    for s in raw.get("series") or []:
        pts = [v for _t, v in (s.get("pointlist") or []) if v is not None]
        out[str(s.get("scope", "?"))] = float(sum(pts))
    return out


def e5(creds: Credentials, *, dry: bool) -> dict:
    """Does a Datadog glob's `*` match the empty string?

    Paper §6 needs this and the documentation does not give it. "Prefix, suffix, and
    substring wildcard tag value filtering are supported", with `pod_name: web-*` as the
    prefix example, says nothing about whether `web-*` also matches `web-` itself.

    It decides which of the two most common PromQL regexes has no glob. A glob has one
    wildcard, which either requires a character or does not, so `.*` and `.+` cannot both
    be expressible: under one reading `.+` has none, under the other `.*` has none.

    Two series of one metric, distinguished by a `probe` tag: one valued `web-`, the
    other `web-1`. A control query with no glob establishes that both were stored --
    without it, a glob that selects nothing is indistinguishable from a point that never
    arrived, which is the failure mode that made E2's first run inconclusive.
    """
    t_star = _grid(_now() - SETTLE)
    run = f"r{int(t_star)}"
    metric = "qlport.e5.star"
    probes = {"empty-suffix": "web-", "one-char-suffix": "web-1"}
    payloads = [count_payload(metric, [(t_star, 1.0)],
                              tags=TAGS + [f"run:{run}", f"probe:{v}"])
                for v in probes.values()]
    acks = _submit_all(creds, payloads, dry=dry)

    control = f"sum:{metric}{{run:{run}}} by {{probe}}"
    globbed = f"sum:{metric}{{probe:web-*,run:{run}}} by {{probe}}"
    start, end = t_star - 900, t_star + 900
    if dry:
        print(f"# then query  {control}   from {start:.0f} to {end:.0f}  (control)")
        print(f"# then query  {globbed}   from {start:.0f} to {end:.0f}  (the question)")
        return {}

    # `pointlist` deliberately refuses a multi-series response and the control query is
    # `by {probe}`, so this experiment needs its own wait rather than `_await_all`.
    deadline = _now() + POLL_TIMEOUT
    raws = {"control": query(creds, control, start, end)}
    while len(_series_by_scope(raws["control"])) < len(probes) and _now() < deadline:
        time.sleep(POLL_EVERY)
        raws["control"] = query(creds, control, start, end)
    raws["globbed"] = query(creds, globbed, start, end)

    stored = _series_by_scope(raws["control"])
    selected = _series_by_scope(raws["globbed"])
    out: dict = {
        "provenance": _provenance(creds, experiment="e5", t_star=t_star, run=run,
                                  probes=probes),
        "submitted": payloads, "acks": acks,
        "responses": {k: {"query": control if k == "control" else globbed,
                          "from": start, "to": end, "raw": v}
                      for k, v in raws.items()},
        "stored": stored, "selected": selected,
    }
    out["interpretation"] = _interpret_star(probes, stored, selected)
    return out


def _probe_values(scopes) -> set[str]:
    """The `probe` tag values in a set of Datadog scope strings.

    A scope comes back as a comma-separated tag list -- `probe:web-,run:r123` -- and the
    run tag is there precisely because runs must not see each other, so matching a scope
    string whole finds nothing.
    """
    out = set()
    for scope in scopes:
        for term in str(scope).split(","):
            key, sep, value = term.strip().partition(":")
            # A filtered query echoes its own filter term back in the scope, so
            # `probe:web-*` appears beside the real tag values; a glob is not one.
            if sep and key == "probe" and "*" not in value:
                out.add(value)
    return out


def _interpret_star(probes: dict, stored: dict, selected: dict) -> dict:
    """Did `web-*` select the series whose tag value is exactly `web-`?"""
    stored_v, selected_v = _probe_values(stored), _probe_values(selected)
    missing = [v for v in probes.values() if v not in stored_v]
    if missing:
        return {"conclusive": False,
                "why": f"the control query did not return probe values {missing}; "
                       "nothing can be concluded from a glob that selects nothing"}
    empty_v, one_v = probes["empty-suffix"], probes["one-char-suffix"]
    if one_v not in selected_v:
        return {"conclusive": False,
                "why": f"`web-*` did not even select probe:{one_v}, so it is not "
                       "behaving as a prefix wildcard at all"}
    matches_empty = empty_v in selected_v
    return {"conclusive": True,
            "star_matches_empty": matches_empty,
            "selected": sorted(selected_v),
            "reading": ("`*` matches the empty string: `.*` has a glob and `.+` does not"
                        if matches_empty else
                        "`*` requires at least one character: `.+` has a glob and `.*` does not"),
            "regular_py_constant_should_be": matches_empty}


# ------------------------------------------------------------------ E1

def e1(creds: Credentials, *, dry: bool) -> dict:
    """The reset policy. Needs Rig A (an agent container scraping a scripted endpoint).

    Not runnable from here, and no longer the priority: the agent source settles rho at
    `zero` on the OpenMetrics path (docs/trial_account_checks.md, Step 0), so this is a
    confirmation rather than a four-way discrimination. Rig A needs
    `src/qlport/harness/expose.py` and `harness/dualwrite.py`, both still stubs.
    """
    raise SystemExit(
        "e1 needs Rig A (agent + scripted /metrics endpoint), which is not built.\n"
        "rho is settled from the agent source as `zero`; see docs/trial_account_checks.md."
    )


EXPERIMENTS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(EXPERIMENTS))
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be submitted and queried; no network, no keys")
    args = ap.parse_args(argv)

    if args.dry_run:
        creds = Credentials("<unused>", "<unused>", "datadoghq.com")
    else:
        creds = credentials()

    result = EXPERIMENTS[args.experiment](creds, dry=args.dry_run)
    if args.dry_run:
        return 0

    path = write_raw(OUT / f"{args.experiment}.json", result)
    interp = result.get("interpretation", {})
    print(f"wrote {path}")
    print(json.dumps(interp, indent=1, sort_keys=True))
    return 0 if interp.get("conclusive", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

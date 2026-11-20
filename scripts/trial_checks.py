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
    COUNT, RATE, Credentials, count_payload, credentials, interpret_closure,
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
    # Deltas chosen so that every candidate aggregator predicts a different reading. The
    # first version used 1, 2, 3, 10, whose mean is 4 and whose *count* is also 4, so the
    # run named `avg` conclusively on a reading `count` explained equally well. With
    # 1, 14, 2, 3: avg 5, sum 20, count 4, sum/20s 1.0, min 1, max 14, last 3, median 2.5.
    offsets, values = (31, 32, 33, 34), (1.0, 14.0, 2.0, 3.0)
    # And a per-run tag, for the reason E2 above gives: without one this experiment read
    # an earlier run's bucket back through its own 3600 s window and interpreted it.
    run = f"r{int(t_star)}"
    tags = TAGS + [f"run:{run}"]
    metric = "qlport.e3.deltas"
    points = [(t_star + o, v) for o, v in zip(offsets, values)]
    payloads = [count_payload(metric, points, tags=tags)]
    acks = _submit_all(creds, payloads, dry=dry)

    variants = {"bare": "", "as_count": ".as_count()", "as_rate": ".as_rate()"}
    scope = f"{{run:{run}}}"
    # The automatic interval is a function of the query range (E4's subject), and a
    # narrow range buys a *finer* one: a 240 s window returned a 1 s rollup, which put
    # each of the four samples in its own bucket and made the comparison meaningless.
    # An hour is coarse enough to hold them together and is what a dashboard would use.
    start, end = t_star - 1800, t_star + 1800
    if dry:
        for name, suffix in variants.items():
            print(f"# then query  sum:{metric}{scope}{suffix}"
                  f"  from {start:.0f} to {end:.0f}  ({name})")
        return {}

    out: dict = {"provenance": _provenance(creds, experiment="e3", t_star=t_star, run=run,
                                           points=points, sum_of_values=sum(values),
                                           mean_of_values=statistics.fmean(values)),
                 "submitted": payloads, "acks": acks, "responses": {}}
    for name, suffix in variants.items():
        q = f"sum:{metric}{scope}{suffix}"
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


def e12(creds: Credentials, *, dry: bool) -> dict:
    """Does a Datadog glob have anything other than `*`?

    Definition 14 of the paper says it does not --- "prefix, suffix and substring
    wildcards and no other construct" --- and that sentence is read from the
    documentation and has never been asked of the system. It is load-bearing in the
    reverse direction: Table 5 refuses 140 matcher occurrences because no glob denotes a
    single arbitrary character, and the paper's own method note says a rule taken from a
    manual and not put to the system is the error it keeps making.

    Three constructs, one submission. Three series distinguished by a `probe` tag ---
    `web-1`, `web-12`, `web-a` --- and a control with no glob to establish all three
    were stored, which is the discipline E5 added after E2's first run could not tell a
    glob that selects nothing from a point that never arrived.

      `probe:web-?`       a single-character wildcard selects web-1 and web-a and not
                          web-12; a literal `?` selects nothing
      `probe:web-[0-9]`   a character class selects web-1; a literal selects nothing
      `probe:web-1|web-a` an alternation selects two; a literal selects nothing

    A negative on all three confirms Definition 14 and the 140 refusals that rest on it.
    A positive on the first costs the paper that row.
    """
    t_star = _grid(_now() - SETTLE)
    run = f"r{int(t_star)}"
    metric = "qlport.e12.wild"
    probes = {"one-digit": "web-1", "two-digit": "web-12", "one-letter": "web-a"}
    payloads = [count_payload(metric, [(t_star, 1.0)],
                              tags=TAGS + [f"run:{run}", f"probe:{v}"])
                for v in probes.values()]
    acks = _submit_all(creds, payloads, dry=dry)

    control = f"sum:{metric}{{run:{run}}} by {{probe}}"
    asked = {
        "question-mark": f"sum:{metric}{{probe:web-?,run:{run}}} by {{probe}}",
        "character-class": f"sum:{metric}{{probe:web-[0-9],run:{run}}} by {{probe}}",
        "alternation": f"sum:{metric}{{probe:web-1|web-a,run:{run}}} by {{probe}}",
    }
    start, end = t_star - 900, t_star + 900
    if dry:
        print(f"# then query  {control}   from {start:.0f} to {end:.0f}  (control)")
        for name, q in asked.items():
            print(f"# then query  {q}   ({name})")
        return {}

    deadline = _now() + POLL_TIMEOUT
    raws = {"control": query(creds, control, start, end)}
    while len(_series_by_scope(raws["control"])) < len(probes) and _now() < deadline:
        time.sleep(POLL_EVERY)
        raws["control"] = query(creds, control, start, end)
    # A refusal is an answer here, and the most informative one: if the construct is
    # not in Datadog's query grammar at all, the request fails to parse rather than
    # selecting nothing. `query` raises on that, so it is caught and recorded.
    rejected: dict[str, str] = {}
    for name, q in asked.items():
        try:
            raws[name] = query(creds, q, start, end)
        except ValueError as exc:
            rejected[name] = str(exc)

    stored = _series_by_scope(raws["control"])
    selected = {name: _series_by_scope(raws[name])
                for name in asked if name in raws}
    out: dict = {
        "provenance": _provenance(creds, experiment="e12", t_star=t_star, run=run,
                                  probes=probes),
        "submitted": payloads, "acks": acks,
        "responses": {k: {"query": control if k == "control" else asked[k],
                          "from": start, "to": end, "raw": v}
                      for k, v in raws.items()},
        "stored": stored, "selected": selected,
    }
    out["rejected"] = rejected
    if len(stored) != len(probes):
        out["interpretation"] = {"all": "inconclusive: not every probe was stored"}
    else:
        out["interpretation"] = {
            **{name: "rejected by the query parser" for name in rejected},
            **{name: ("construct is live" if sel else "accepted, selects nothing")
               for name, sel in selected.items()},
        }
    return out


def e13(creds: Credentials, *, dry: bool) -> dict:
    """Does Datadog's `or` bind more loosely than its comma?

    Remark~8 asserts that it does, so `{a:x or a:y,*}` parses with the comma inside the
    right branch. The justification given is an analogy with the Lucene-style filter
    syntax Datadog's editor accepts, which is not evidence about this grammar. The
    claim decides how 160 SCOPE_BOOL occurrences are *translated* rather than whether
    they are, and no corpus query discriminates: the only shape mixing the two
    operators has match-all terms on the other side, so both readings agree on it.

    Three series over two tag keys:

        s1  aa:x bb:2      s2  aa:y bb:1      s3  aa:x bb:1

    and one question, `{aa:x or aa:y, bb:1}`. Under the looser-`or` reading this parses
    as `aa:x OR (aa:y AND bb:1)` and selects s1, because s1 satisfies `aa:x` whatever
    its `bb`. Under the tighter reading it is `(aa:x OR aa:y) AND bb:1` and s1 is
    excluded. So the presence of s1 in the answer is the whole experiment.
    """
    t_star = _grid(_now() - SETTLE)
    run = f"r{int(t_star)}"
    metric = "qlport.e13.prec"
    series = {"s1": ("x", "2"), "s2": ("y", "1"), "s3": ("x", "1")}
    payloads = [count_payload(metric, [(t_star, 1.0)],
                              tags=TAGS + [f"run:{run}", f"aa:{a}", f"bb:{b}"])
                for a, b in series.values()]
    acks = _submit_all(creds, payloads, dry=dry)

    control = f"sum:{metric}{{run:{run}}} by {{aa,bb}}"
    asked = f"sum:{metric}{{aa:x or aa:y,bb:1,run:{run}}} by {{aa,bb}}"
    start, end = t_star - 900, t_star + 900
    if dry:
        print(f"# then query  {control}  (control)")
        print(f"# then query  {asked}  (the question)")
        return {}

    deadline = _now() + POLL_TIMEOUT
    raws = {"control": query(creds, control, start, end)}
    while len(_series_by_scope(raws["control"])) < len(series) and _now() < deadline:
        time.sleep(POLL_EVERY)
        raws["control"] = query(creds, control, start, end)
    rejected = {}
    try:
        raws["asked"] = query(creds, asked, start, end)
    except ValueError as exc:
        rejected["asked"] = str(exc)

    stored = _series_by_scope(raws["control"])
    selected = _series_by_scope(raws["asked"]) if "asked" in raws else {}
    s1_present = any("aa:x" in s and "bb:2" in s for s in selected)
    if len(stored) != len(series):
        verdict = "inconclusive: not every probe was stored"
    elif rejected:
        verdict = "inconclusive: the query was rejected"
    else:
        verdict = ("or binds more loosely than the comma"
                   if s1_present else "the comma binds more loosely than or")
    return {
        "provenance": _provenance(creds, experiment="e13", t_star=t_star, run=run,
                                  series=series),
        "submitted": payloads, "acks": acks,
        "responses": {k: {"query": control if k == "control" else asked,
                          "from": start, "to": end, "raw": v}
                      for k, v in raws.items()},
        "stored": stored, "selected": selected, "rejected": rejected,
        "interpretation": {"verdict": verdict, "s1_selected": s1_present},
    }


def e14(creds: Credentials, *, dry: bool) -> dict:
    """What does Datadog do with `{*,!*}`?

    Nine corpus occurrences carry it and the paper puts them in the unimplemented
    column on the strength of a sentence: that PromQL can denote an empty selection, and
    that `{*,!*}` means empty. The second half was never asked. It matters beyond nine
    occurrences because §C.3 now calls the 99.1% ceiling tight, and tightness needs
    every unimplemented occurrence to be genuinely reachable. If Datadog *rejects* the
    scope rather than returning nothing, these belong in the fourth refusal kind that
    Remark~11 calls possible and empty here.
    """
    t_star = _grid(_now() - SETTLE)
    run = f"r{int(t_star)}"
    metric = "qlport.e14.negall"
    payloads = [count_payload(metric, [(t_star, 1.0)], tags=TAGS + [f"run:{run}"])]
    acks = _submit_all(creds, payloads, dry=dry)

    control = f"sum:{metric}{{run:{run}}}"
    asked = {
        "match-all-and-negated": f"sum:{metric}{{*,!*,run:{run}}}",
        "negated-alone": f"sum:{metric}{{!*,run:{run}}}",
    }
    start, end = t_star - 900, t_star + 900
    if dry:
        print(f"# then query  {control}  (control)")
        for name, q in asked.items():
            print(f"# then query  {q}  ({name})")
        return {}

    deadline = _now() + POLL_TIMEOUT
    raws = {"control": query(creds, control, start, end)}
    while not _defined(raws["control"]) and _now() < deadline:
        time.sleep(POLL_EVERY)
        raws["control"] = query(creds, control, start, end)
    rejected, points = {}, {}
    for name, q in asked.items():
        try:
            raws[name] = query(creds, q, start, end)
            points[name] = len(_defined(raws[name]))
        except ValueError as exc:
            rejected[name] = str(exc)

    stored = len(_defined(raws["control"]))
    return {
        "provenance": _provenance(creds, experiment="e14", t_star=t_star, run=run),
        "submitted": payloads, "acks": acks,
        "responses": {k: {"query": control if k == "control" else asked.get(k),
                          "from": start, "to": end, "raw": v}
                      for k, v in raws.items()},
        "stored_points": stored, "selected_points": points, "rejected": rejected,
        "interpretation": (
            {"all": "inconclusive: the control stored nothing"} if not stored else
            {**{n: "rejected by the query parser" for n in rejected},
             **{n: ("selects the series" if k else "accepted, selects nothing")
                for n, k in points.items()}}),
    }


def e15(creds: Credentials, *, dry: bool) -> dict:
    """Is a Datadog RATE metric mapped by the identity, as Definition 7 says?

    Definition 7 sends every non-counter series through phi unchanged, and Remark 27
    counts 134 corpus occurrences naming metrics the vendor declares as `rate`. Those
    occurrences are inside kappa and their exactness rests on the identity mapping,
    which is asserted rather than measured: a RATE is submitted as a per-second value
    over a declared interval, and whether what Datadog stores and reduces is that value
    or the count it implies is exactly the question phi exists to ask.

    One RATE series, interval 60, five points of a known constant, and two rollups: a
    `sum` and an `avg` over a window containing all five. If storage is the identity,
    `avg` returns the submitted value and `sum` returns five times it. If the intake
    rescales by the interval, both are off by a factor of 60 in a direction this tells
    us. A COUNT series carrying the same numbers is submitted alongside as the control,
    because the question is what RATE does *differently*.
    """
    t_star = _grid(_now() - SETTLE) - 300
    run = f"r{int(t_star)}"
    value, n = 7.0, 5
    points = [(t_star + i * DELTA, value) for i in range(n)]
    rate_m, count_m = "qlport.e15.rate", "qlport.e15.count"
    payloads = [
        count_payload(rate_m, points, tags=TAGS + [f"run:{run}"],
                      interval=int(DELTA), metric_type=RATE),
        count_payload(count_m, points, tags=TAGS + [f"run:{run}"],
                      interval=int(DELTA), metric_type=COUNT),
    ]
    acks = _submit_all(creds, payloads, dry=dry)

    span = int(n * DELTA)
    asked = {
        f"{kind}-{op}": f"{op}:{m}{{run:{run}}}.rollup({op}, {span})"
        for kind, m in (("rate", rate_m), ("count", count_m))
        for op in ("avg", "sum")
    }
    start, end = t_star - 900, t_star + span + 900
    if dry:
        for name, q in asked.items():
            print(f"# then query  {q}  ({name})")
        return {}

    deadline = _now() + POLL_TIMEOUT
    raws: dict = {}
    for name, q in asked.items():
        while True:
            raws[name] = query(creds, q, start, end)
            if _defined(raws[name]) or _now() > deadline:
                break
            time.sleep(POLL_EVERY)

    values = {name: sorted(_defined(raw).values()) for name, raw in raws.items()}
    return {
        "provenance": _provenance(creds, experiment="e15", t_star=t_star, run=run,
                                  submitted_value=value, points=n,
                                  interval=int(DELTA)),
        "submitted": payloads, "acks": acks,
        "responses": {k: {"query": asked[k], "from": start, "to": end, "raw": v}
                      for k, v in raws.items()},
        "values": values,
        "interpretation": {
            "note": ("compare rate-avg against the submitted value and rate-sum against "
                     f"{n} times it; the count series is the control"),
            "submitted_value": value, "points": n,
        },
    }


def e16(creds: Credentials, *, dry: bool) -> dict:
    """What does a keyless scope term like `*-location` glob over?

    §6 and §7.3 read it as a glob over the whole `key:value` string, which is why
    LABEL_NAME_GLOB is called an expressiveness gap: a pattern over names is something
    no PromQL matcher takes. The reading has never been measured, and it has an odd
    consequence --- `cluster-location:us-east1` does not end in `-location`, so under
    that reading the velero panels carrying `{*-location,*-name,*-uid,*}` select
    nothing, which is a strange thing for a shipped dashboard to do. The alternative
    reading, that the glob is over the key, makes those panels select what their
    `by {cluster-location,...}` clause groups by.

    Three series, one per tag key, so the two readings disagree on the answer:

        cluster-location:eu      site-location:us      region:ap

    A key glob selects the first two. A glob over `key:value` selects none, since no
    tag string ends in `-location`. A third possibility the control will expose is that
    the term is rejected outright.
    """
    t_star = _grid(_now() - SETTLE)
    run = f"r{int(t_star)}"
    metric = "qlport.e16.keyglob"
    series = {"cluster-location": "eu", "site-location": "us", "region": "ap"}
    payloads = [count_payload(metric, [(t_star, 1.0)],
                              tags=TAGS + [f"run:{run}", f"{k}:{v}"])
                for k, v in series.items()]
    acks = _submit_all(creds, payloads, dry=dry)

    control = f"sum:{metric}{{run:{run}}} by {{cluster-location,site-location,region}}"
    asked = {
        "keyless-glob": f"sum:{metric}{{*-location,run:{run}}}",
        "glob-with-value": f"sum:{metric}{{*-location:*,run:{run}}}",
    }
    start, end = t_star - 900, t_star + 900
    if dry:
        print(f"# then query  {control}  (control)")
        for name, q in asked.items():
            print(f"# then query  {q}  ({name})")
        return {}

    deadline = _now() + POLL_TIMEOUT
    raws = {"control": query(creds, control, start, end)}
    while len(_series_by_scope(raws["control"])) < len(series) and _now() < deadline:
        time.sleep(POLL_EVERY)
        raws["control"] = query(creds, control, start, end)
    rejected, picked = {}, {}
    for name, q in asked.items():
        try:
            raws[name] = query(creds, q, start, end)
            picked[name] = len(_defined(raws[name]))
        except ValueError as exc:
            rejected[name] = str(exc)

    stored = _series_by_scope(raws["control"])
    return {
        "provenance": _provenance(creds, experiment="e16", t_star=t_star, run=run,
                                  series=series),
        "submitted": payloads, "acks": acks,
        "responses": {k: {"query": control if k == "control" else asked.get(k),
                          "from": start, "to": end, "raw": v}
                      for k, v in raws.items()},
        "stored": stored, "selected_points": picked, "rejected": rejected,
        "interpretation": (
            {"all": "inconclusive: not every probe was stored"}
            if len(stored) < len(series) else
            {**{n: "rejected by the query parser" for n in rejected},
             **{n: ("selects something --- the glob is over the key"
                    if k else "selects nothing --- consistent with a glob over key:value")
                for n, k in picked.items()}}),
    }


# ------------------------------------------------------------------ E18


def e18(creds: Credentials, *, dry: bool) -> dict:
    r"""The automatic rollup interval for every time span the corpus actually uses.

    E4 asked this for five ranges and found the interval is a function of the query
    range: 1s at a minute, 2s at five, 20s at an hour, 60s at four hours, 300s at a day.
    \S\ref{sec:threats} then says the 60s default behind `ROLLUP_DEFAULT` is
    "systematic rather than arbitrary" --- but 60s is the four-hour row, and the Datadog
    dashboards in the corpus declare their own span: 460 `live_span` values across 197
    files, of which four hours is 28.3 per cent and one hour, the modal value, is 38.3.
    So the default is wrong wherever a widget says something else, and five of the ten
    spans the corpus uses were never measured at all.

    This measures all ten, which turns `ROLLUP_DEFAULT` from an assumption into a
    lookup. E4's five are re-asked rather than taken on trust: they are the control, and
    a run where they do not reproduce says the instrument moved and is reported
    inconclusive rather than merged with E4's numbers.

    Same shape and same billing as E4 --- one custom metric, one series, an hour of
    points at 10s --- because the interval depends on the range and not on the data, so
    an hour of backfill is enough to ask about a month. The far buckets come back null,
    and a null bucket still carries the interval Datadog chose.
    """
    now = _grid(_now() - 60)
    metric = "qlport.e18.spans"
    spacing, span = 10, 3500
    points = [(now - span + i * spacing, 1.0)
              for i in range(span // spacing + 1)]
    payloads = [count_payload(metric, points[i:i + 100], tags=TAGS)
                for i in range(0, len(points), 100)]
    # Every `live_span` the corpus's Datadog dashboards declare, in seconds. E4 measured
    # the five marked below; the rest are why this experiment exists.
    ranges = {"1m": 60, "5m": 300, "10m": 600, "15m": 900, "30m": 1800,
              "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1mo": 2592000}
    e4_measured = {"1m": 1, "5m": 2, "1h": 20, "4h": 60, "1d": 300}

    acks = _submit_all(creds, payloads, dry=dry)
    if dry:
        print(f"# {len(points)} points every {spacing}s over {span}s, "
              f"in {len(payloads)} request(s)")
        for name, width in ranges.items():
            seen = "E4" if name in e4_measured else "new"
            print(f"# then query  sum:{metric}{{*}}  over {name} ({width}s)  [{seen}]")
        return {}

    _await_points(creds, f"sum:{metric}{{*}}", now - 3600, now, want=3)

    out: dict = {"provenance": _provenance(creds, experiment="e18", spacing=spacing,
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

    reported = {k: v["reported_interval"] for k, v in out["responses"].items()}
    # The control: E4's five must come back as E4 found them, or the two runs cannot be
    # read as one table.
    replicated = {k: (reported.get(k), v) for k, v in e4_measured.items()}
    disagreed = {k: got for k, (got, want) in replicated.items() if got != want}
    unreadable = [k for k, v in reported.items() if v is None]
    out["interpretation"] = {
        "reported_by_range": reported,
        "observed_by_range": {k: v["observed_spacing"]
                              for k, v in out["responses"].items()},
        "e4_expected": e4_measured,
        "e4_disagreed": disagreed,
        "intervals_unreadable": unreadable,
        "spans_whose_interval_is_60s": [k for k, v in reported.items() if v == 60],
        "conclusive": not disagreed and not unreadable,
    }
    return out


EXPERIMENTS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e12": e12,
               "e13": e13, "e14": e14, "e15": e15, "e16": e16,
               "e18": e18}


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

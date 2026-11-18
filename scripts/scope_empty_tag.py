#!/usr/bin/env python3
"""Does Datadog's `!k:` mean what PromQL's `{k!=""}` means? It does not.

Paper §5's Lemma lem:scope claims the matchers the lowering builds select the same
series as the Datadog scope, for every scope lowering admits. The proof's first clause
reads "a literal tag is an equality matcher; a negated tag its negation", which is sound
for a non-empty value and unsound for the empty one -- and the eighth review pass found
five translated corpus occurrences of `!k:` emitting `{k!=""}`.

The two systems divide tag states differently. Datadog distinguishes three: the tag is
absent, present with an empty value, or present with a value. Prometheus has two, because
its data model *defines* a label with an empty value to be the same as an absent label.
So a Datadog scope that separates "absent" from "present and empty" asks for a
distinction the target does not represent, and no PromQL selector recovers it.

Three series of one gauge, distinguished by whether they carry `k` at all:

  has_k     k:v     the tag is present with a value
  no_k      --      the tag is absent
  empty_k   k:      the tag is present with an empty value

and five scopes read back against them. The PromQL column is not measured here: it is
Prometheus's documented data model, and `tests/test_lower.py` pins the emitted matchers.

  python scripts/scope_empty_tag.py [--dry-run]

Writes data/raw/datadog/scope_empty_tag.json.
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

OUT = ROOT / "data" / "raw" / "datadog" / "scope_empty_tag.json"
STEP, N = 60.0, 10
SETTLE = 180.0
READY_TIMEOUT = 420.0

# What each scope should select if Datadog separates the three tag states, and what the
# PromQL matcher the lowering emits selects under Prometheus's data model. Recorded as a
# prediction table so a disagreement names a different rule rather than a surprise.
PREDICTIONS = {
    "all":      (["empty_k", "has_k", "no_k"], None,         ["empty_k", "has_k", "no_k"]),
    "!k:":      (["has_k", "no_k"],            'k!=""',      ["has_k"]),
    "k:":       (["empty_k"],                  'k=""',       ["empty_k", "no_k"]),
    "k:*":      (["empty_k", "has_k"],         'k=~".+"',    ["has_k"]),
    "!k:*":     (["no_k"],                     'k!~".+"',    ["empty_k", "no_k"]),
}
SERIES = {"has_k": ["k:v"], "no_k": [], "empty_k": ["k:"]}


def scope_of(form: str, run: str) -> str:
    return f"run:{run}" if form == "all" else f"run:{run},{form}"


def selected(raw: dict) -> list[str]:
    out = []
    for s in raw.get("series") or []:
        tags = list(s.get("tag_set") or []) + str(s.get("scope") or "").split(",")
        for t in (x.strip() for x in tags):
            if t.startswith("probe:"):
                out.append(t[6:])
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = math.floor((time.time() - N * STEP - 300.0) / STEP) * STEP
    run = f"r{int(t0)}"
    metric = f"qlport.scope.{run}.g"
    points = [(t0 + i * STEP, 1.0) for i in range(N)]
    payloads = [count_payload(metric, points,
                              tags=[f"run:{run}", f"probe:{name}"] + extra,
                              metric_type=GAUGE)
                for name, extra in SERIES.items()]

    if args.dry_run:
        print(f"{metric}: {len(payloads)} series x {N} points at {STEP:g}s from {t0:.0f}")
        for name, extra in SERIES.items():
            print(f"  probe:{name:<8} tags {extra or '(no k tag)'}")
        for form, (dd, pq, pq_sel) in PREDICTIONS.items():
            print(f"  sum:{metric}{{{scope_of(form, run)}}} by {{probe}}")
            print(f"      Datadog should select {dd}; PromQL {pq or '--'} selects {pq_sel}")
        return 0

    creds = credentials()
    for payload in payloads:
        submit(creds, payload)
    print(f"{metric}: {len(payloads)} series -> {sorted(SERIES)}")
    print(f"  settling {SETTLE:.0f}s")
    time.sleep(SETTLE)

    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        raw = query(creds, f"sum:{metric}{{run:{run}}} by {{probe}}", t0 - 60, t0 + N * STEP)
        if len(selected(raw)) == len(SERIES):
            break
        if time.monotonic() > deadline:
            print(f"  giving up: only {selected(raw)} present", file=sys.stderr)
            return 2
        time.sleep(15.0)

    probes, agree = {}, True
    for form, (dd_pred, pq_matcher, pq_sel) in PREDICTIONS.items():
        q = f"sum:{metric}{{{scope_of(form, run)}}} by {{probe}}"
        raw = query(creds, q, t0 - 60, t0 + N * STEP)
        got = selected(raw)
        ok = got == sorted(dd_pred)
        agree &= ok
        probes[form] = {
            "query": q, "selected": got, "datadog_predicted": sorted(dd_pred),
            "matches_prediction": ok,
            "promql_matcher": pq_matcher, "promql_selects": sorted(pq_sel),
            "same_series_set": pq_matcher is None or got == sorted(pq_sel),
            "raw": raw,
        }
        mark = "ok " if ok else "NO "
        same = "" if pq_matcher is None else (
            "  == PromQL" if got == sorted(pq_sel) else "  != PromQL " + str(sorted(pq_sel)))
        print(f"  {mark} {form:<6} -> {got}{same}")

    diverging = [f for f, p in probes.items() if not p["same_series_set"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "provenance": {
            "site": creds.site, "metric": metric, "run": run, "t0": t0,
            "step": STEP, "n_points": N,
            "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "series": SERIES,
        },
        "probes": probes,
        "datadog_matches_prediction": bool(agree),
        "forms_whose_translation_selects_a_different_set": sorted(diverging),
        "submitted": payloads,
    }, indent=1, sort_keys=True) + "\n")
    print(f"\nDatadog matched the three-state prediction on all five scopes: {agree}")
    print(f"forms whose emitted matcher denotes a different series set: {sorted(diverging)}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
r"""The three differential runs §7.6 reports, recovered from the caches that held them.

`data/raw/shapes/differential.json` holds one run: the newest. §7.6 compares three, and
until this script there was no source in the repository for the two it no longer holds --
so the sentences about them were the only record of them, and one of them was wrong. It
claimed the second run was over 332 shapes and therefore had three uncompared, which is
run *one's* shape count divided into run *two's* compared count; and it explained the
discrepancy that arithmetic produced by asserting the second run's cache was gone. The
cache is `580f802:data/raw/shapes/differential.json`, which says four.

A figure nothing can recompute is a figure that drifts, and history is the worst case of
it, because a historical value cannot go stale: nothing will ever fail on it. So the
three runs are recovered from git here and written to a cache the tests read, with the
commit that held each one named. Nothing is re-measured -- the Datadog trial these ran
against has lapsed and could not be -- and nothing here needs to be: every number below
is already in a committed file.

The uncompared rows record only why they could not be compared, not what they were, so
naming them means re-deriving the shape list at each commit. `representatives()` is pure
-- corpus in, shapes out, no network -- so each commit is extracted and asked, in its own
process, because the three disagree about `corpus._TMPL` and the first import of
`qlport` wins. That is what lets the claims about *which* shapes those were be checked
rather than asserted, which is the half of the broken sentence that no arithmetic would
have caught.

    python scripts/differential_run_history.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "shapes" / "differential.json"
OUT = ROOT / "data" / "raw" / "shapes" / "run_history.json"
REGISTRY = ROOT / "paper" / "quoted_figures.json"

# The runs §7.6 counts, oldest first, each named by the commit whose tree held its cache.
# `git log --follow -- data/raw/shapes/differential.json` lists ten; the first seven are
# the rig being built and debugged, and Remark~\ref{rem:differential-fixes} accounts for
# them. These three are the ones reported as runs: the last before `corpus._TMPL` was
# fixed, and the two after it, each on its own set of synthetic series.
RUNS: tuple[tuple[str, str], ...] = (
    ("3f995e3", "first"),
    ("580f802", "second"),
    ("9089766", "third"),
)

# Run in a child process, once per commit: that commit's shape list, by index. Each
# commit lowers and normalises the corpus differently, so this cannot be done in one
# process -- `sys.modules` would hand the second commit the first one's `qlport`.
_DUMP = r"""
import json, sys, importlib.util
root = sys.argv[1]
sys.path.insert(0, root + "/src")
spec = importlib.util.spec_from_file_location("sd", root + "/scripts/shape_differential.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(json.dumps({r["index"]: r["datadog_original"] for r in m.representatives("rX")}))
"""


def _cache_at(sha: str) -> dict:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{sha}:data/raw/shapes/differential.json"],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _shapes_at(sha: str) -> dict[int, str]:
    """That commit's shape list: index to the corpus query standing for the shape."""
    with tempfile.TemporaryDirectory() as tmp:
        tar = subprocess.run(["git", "-C", str(ROOT), "archive", sha],
                             check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=tar, check=True)
        out = subprocess.run([sys.executable, "-c", _DUMP, tmp],
                             check=True, capture_output=True, text=True).stdout
    return {int(k): v for k, v in json.loads(out).items()}


def _summary(sha: str, name: str, cache: dict, shapes: dict[int, str]) -> dict:
    rows = cache["rows"]
    compared = [r for r in rows if r["status"] == "compared"]
    assert len(shapes) == len(rows), (sha, len(shapes), len(rows))
    return {
        "commit": sha,
        "ordinal": name,
        "run_tag": cache["provenance"]["run"],
        "collected_at": cache["provenance"]["collected_at"],
        "series": cache["provenance"]["series"],
        "shapes": len(rows),
        "compared": len(compared),
        "agreed": sum(1 for r in compared if r["exact"] == 1.0),
        "uncompared": [
            {"index": r["index"], "status": r["status"], "detail": r["detail"],
             "query": shapes[r["index"]]}
            for r in sorted(rows, key=lambda r: r["index"]) if r["status"] != "compared"
        ],
        "bucket_comparisons": sum(int(r["buckets"]) for r in compared),
        "occurrences_compared": cache["occurrences_compared"],
        "bit_identical": sum(1 for r in compared
                             if r.get("bit_identical") in (1.0, True)),
        "max_deviation": max(r["max_d"] for r in compared),
        # By the query the shape stands for, not by index: the indices shift when a
        # shape is added, which is why the no-data row is 299 in the first run and 300
        # in the other two and is the same envoy query in all three.
        "status_by_query": {shapes[r["index"]]: r["status"] for r in rows},
    }


def main() -> int:
    recovered = [_summary(sha, name, _cache_at(sha), _shapes_at(sha))
                 for sha, name in RUNS]

    # The newest run is the one the working tree holds. If it is not, this file is
    # describing a history that the live cache has already left.
    live = json.loads(CACHE.read_text())
    assert live["provenance"]["run"] == recovered[-1]["run_tag"], (
        f"{CACHE.name} holds {live['provenance']['run']}, not "
        f"{recovered[-1]['run_tag']}; add the newer run to RUNS")

    runs = []
    for i, r in enumerate(recovered):
        prev = recovered[i - 1]["status_by_query"] if i else {}
        now = r["status_by_query"]
        rec = {k: v for k, v in r.items() if k != "status_by_query"}
        rec["shapes_added_since_previous"] = sorted(set(now) - set(prev)) if i else []
        rec["shapes_removed_since_previous"] = sorted(set(prev) - set(now)) if i else []
        rec["status_changes_since_previous"] = [
            {"query": q, "from": prev[q], "to": now[q]}
            for q in sorted(set(prev) & set(now)) if prev[q] != now[q]
        ]
        runs.append(rec)

    first, second, third = runs

    fig = {
        # Run one's shape count. The prose quotes it in two places and until this key
        # existed the census licensed it from `rs.span_declared_wrong`, a rollup-span
        # figure of the same value and no relation -- so a wrong shape count here would
        # have passed.
        "dr.first_shapes": first["shapes"],
        # Compared, and agreed, in each of the first two runs. The same number twice is
        # not a coincidence: the second run reached every shape the first did and the
        # one shape it added came back `unmatched`, so the two compared the same set.
        "dr.first_compared": first["compared"],
        # And over the same buckets, for the same reason. Quoted beside the third run's
        # count to say the agreement did not move.
        "dr.shared_buckets": first["bucket_comparisons"],
    }
    assert second["compared"] == fig["dr.first_compared"]
    assert second["bucket_comparisons"] == fig["dr.shared_buckets"]

    OUT.write_text(json.dumps({"figures": fig, "runs": runs}, indent=1,
                              sort_keys=True) + "\n")

    reg = json.loads(REGISTRY.read_text())
    reg.update(fig)
    REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")

    for r in runs:
        print(f"{r['ordinal']:>6} {r['commit']} {r['run_tag']} "
              f"{r['shapes']:>4} shapes  {r['compared']} compared "
              f"({r['agreed']} agreed)  {len(r['uncompared'])} uncompared  "
              f"{r['bucket_comparisons']:>6,} buckets  max {r['max_deviation']:.3g}")
        for u in r["uncompared"]:
            print(f"         {u['status']:<18} {u['query'][:88]}")
        for c in r["status_changes_since_previous"]:
            print(f"         {c['from']} -> {c['to']}: {c['query'][:72]}")
        for q in r["shapes_added_since_previous"]:
            print(f"         added: {q[:88]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

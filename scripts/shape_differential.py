#!/usr/bin/env python3
"""The corpus-scale differential test: every translated shape, both systems, compared.

paper §7 reports that the corpus collapses to 331 distinct query shapes once metric
names and tag values are abstracted away, and that every shape class emits a single
PromQL form -- so one representative stands for its class. This runs them.

For each representative: synthesise the series its own filters select
(`harness/shapes.py`), write them to Prometheus as the cumulative counter and to Datadog
as the per-interval deltas that are their image under φ, then evaluate the Datadog query
against Datadog and its PromQL translation against Prometheus and compare bucket by
bucket.

Each shape gets its own metric namespace. Two shapes naming the same corpus metric would
otherwise share data, and if they disagreed about whether it is a counter the comparison
would measure that disagreement instead of the translation. Renaming is safe here in a
way it is not in general: §7's congruence check establishes that a shape determines the
emitted form, so a metric name is not load-bearing.

The query range is four hours. Datadog picks the rollup interval from the range and E4
measured four hours as 60 s, which is the interval the emitter assumes when a query
states none -- so the default path is exercised rather than bypassed.

Needs Docker and DD_API_KEY / DD_APP_KEY / DD_SITE in .env.

  python scripts/shape_differential.py              # every shape
  python scripts/shape_differential.py --limit 20   # the 20 most frequent
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport import ir                                            # noqa: E402
from qlport.corpus import read_queries, normalise                # noqa: E402
from qlport.diagnostics import Bottom                            # noqa: E402
from qlport.emit.promql import emit_detail                       # noqa: E402
from qlport.harness.datadog import (                             # noqa: E402
    COUNT, GAUGE, count_payload, credentials, query, submit,
)
from qlport.harness.shapes import instantiate                    # noqa: E402
from qlport.ir import shape                                      # noqa: E402
from qlport.lower.datadog import label_name, lower_all, metric_name          # noqa: E402
from qlport.metrics import reldiff                               # noqa: E402
from lark import Token                                          # noqa: E402
from qlport.parse import parse                                   # noqa: E402
from qlport.semantics import datadog_eval as dd                  # noqa: E402

IMAGE = "prom/prometheus:v3.15.0"
PORT = 19095
OUT = ROOT / "data" / "raw" / "shapes" / "differential.json"
PREFIX = "qlport.sd"
ROLLUP = 60.0
SCRAPE = 15.0
WINDOW = 1800.0
SETTLE = 180.0
RANGE = 4 * 3600.0          # four hours -> a 60 s automatic interval (E4)
INGEST_WAIT = 900.0
# Two different engines computing the same mean sum in different orders, so agreement
# is to float64 precision rather than bit-identical. The paper's own model-versus-server
# check uses the same bound.
TOLERANCE = 1e-12
# Subtracting two near-equal values loses most of the significand, so a shape whose
# query is a difference of two aggregates carries far more rounding than one that is
# not. 1e-9 still catches every real disagreement here by three orders of magnitude --
# the smallest genuine one found was 0.077 -- while not reporting cancellation as one.
CANCELLATION_TOLERANCE = 1e-9


def run(*cmd: str, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _ids() -> str:
    import os
    return f"{os.getuid()}:{os.getgid()}"


# ------------------------------------------------------------------ representatives

_DD_METRIC = re.compile(r"(?<![\w.])([a-z_]+:)?([A-Za-z][\w.]*)\s*\{")


def dd_spellings(text: str) -> dict[str, str]:
    """The query's own spelling of every identifier that lowers to a different name.

    Keyed by the name the IR and PromQL use, valued by what the query wrote. Datadog
    tag keys admit dots, hyphens and slashes; PromQL label names admit none of them, so
    the IR holds `cluster_location` where the query says `cluster-location` and
    `openai_organization_name` where it says `openai.organization.name`. Submitting the
    normalised spelling while asking Datadog to group by the original makes Datadog
    group by a tag that is not there: it silently returns one ungrouped series, and the
    comparison then reports groups that did not line up.

    Taken off the parse tree, because the grammar already knows which token is which.
    Only three kinds of identifier are ever mapped --- a METRIC, a grouping KEY, and the
    key half of a scope TAG --- and the tree labels all three. The regex this replaces
    matched identifier-shaped runs anywhere in the query text, which pulled in
    hyphenated tag *values* as well: 283 spurious entries over this corpus, harmless
    only because no value's normalised form collided with a real label name. That was a
    property of the corpus rather than of the code, and a query carrying the value
    `foo-bar` beside the key `foo_bar` would have collided.

    `*` is excluded explicitly: it is a TAG token with no key half, and `label_name`
    sends it to `_`.
    """
    out: dict[str, str] = {}

    def add(name: str) -> None:
        normalised = label_name(name)
        if normalised != name:
            out[normalised] = name

    for tok in parse("datadog", text).scan_values(lambda x: isinstance(x, Token)):
        if tok.type in ("METRIC", "KEY"):
            add(str(tok))
        elif tok.type == "TAG":
            key, _, has_value = str(tok).partition(":")
            if has_value or _:          # `k:v` or `k:`; a bare `*` has neither
                add(key)
    return out


def representatives(run_tag: str) -> list[dict]:
    """One instantiable representative per shape, renamed into a private namespace."""
    qs = [q for q in read_queries(ROOT / "data" / "corpus" / "queries.jsonl")
          if q.lang == "datadog"]
    occ = collections.Counter(normalise("datadog", q.text) for q in qs)

    best: dict = {}
    for text, n in sorted(occ.items()):
        try:
            tree = parse("datadog", text, "v2")
        except Exception:
            continue
        lowered = lower_all(tree)
        if len(lowered) != 1 or isinstance(lowered[0], Bottom):
            continue
        e = lowered[0]
        if isinstance(emit_detail(e, default_rollup=ROLLUP), Bottom):
            continue
        s = shape(e)
        rec = best.get(s)
        if rec is None:
            best[s] = {"text": text, "ir": e, "n": 0}
            rec = best[s]
        elif len(text) < len(rec["text"]):
            rec.update(text=text, ir=e)
        rec["n"] += n

    out = []
    for i, (s, rec) in enumerate(sorted(best.items(), key=lambda kv: -kv[1]["n"])):
        inst = instantiate(rec["ir"])
        if not inst.ok:
            continue
        # Rename every metric this shape touches into its own namespace.
        originals = sorted({sp.metric for sp in inst.series})
        # The run tag is in the *metric name*, not only in a tag. The query range is
        # four hours and a previous run's points sit inside it, so sharing a name would
        # let Datadog see two runs' data while Prometheus, rebuilt each time, sees one.
        # Filtering by a run tag instead would mean editing each scope, and a boolean
        # scope cannot take a comma-joined term.
        rename = {m: f"{PREFIX}.{run_tag}.s{i}_{j}" for j, m in enumerate(originals)}
        dd_text = _DD_METRIC.sub(
            lambda m: f"{m.group(1) or ''}{rename.get(metric_name(m.group(2)), m.group(2))}{{",
            rec["text"])
        renamed_ir = _rename_ir(rec["ir"], {metric_name(k): metric_name(v)
                                            for k, v in rename.items()})
        em = emit_detail(renamed_ir, default_rollup=ROLLUP)
        if isinstance(em, Bottom):
            continue
        intervals = _intervals(renamed_ir)
        if len(intervals) != 1:
            # Two rollups of different widths in one query would need two bucket grids;
            # recorded rather than compared on the wrong one.
            continue
        # The tags go out in the query's own spelling and are mapped back when the
        # response is read; `dd_spellings` says what that spelling is. This was a regex
        # over the query text until the eleventh pass, blind to the hyphen, which is
        # how shape 262 came back `unmatched`.
        dd_keys = dd_spellings(rec["text"])
        out.append({
            "interval": intervals[0],
            "index": i, "occurrences": rec["n"], "datadog_original": rec["text"],
            "datadog": dd_text, "promql": em.promql,
            "caveats": sorted(c.name for c in em.caveats),
            "group_keys": list(inst.group_keys),
            # original spelling -> the normalised name PromQL uses
            "dd_group_keys": {dd_keys.get(k, k): k for k in inst.group_keys},
            "series": [{"metric": metric_name(rename[sp.metric]), "dd_metric": rename[sp.metric],
                        "labels": sp.labels,
                        "dd_labels": {dd_keys.get(k, k): v for k, v in sp.labels.items()},
                        "kind": _kind_for(renamed_ir, metric_name(rename[sp.metric]))}
                       for sp in inst.series],
        })
    return out


def _rename_ir(e, mapping):
    if isinstance(e, ir.Select):
        return replace(e, metric=mapping.get(e.metric, e.metric))
    if isinstance(e, ir.Rollup):
        return replace(e, expr=_rename_ir(e.expr, mapping))
    if isinstance(e, ir.Agg):
        return replace(e, expr=_rename_ir(e.expr, mapping))
    if isinstance(e, ir.Scale):
        return replace(e, expr=_rename_ir(e.expr, mapping))
    if isinstance(e, ir.BinOp):
        return replace(e, lhs=_rename_ir(e.lhs, mapping), rhs=_rename_ir(e.rhs, mapping))
    return e


def _intervals(e) -> list[float]:
    """Every rollup width in the query, defaulting where the query states none.

    The bucket grid and the PromQL evaluation instants must both use the interval the
    *query* asked for. Comparing a `.rollup(sum, 300)` against a 60 s grid lines up
    Datadog's buckets with PromQL windows covering a different span, which looks exactly
    like a translation error and is not one.
    """
    out = []

    def go(x):
        if isinstance(x, ir.Rollup):
            out.append(float(x.interval) if x.interval is not None else ROLLUP)
        for f in ("expr", "lhs", "rhs", "arg"):
            c = getattr(x, f, None)
            if c is not None and not isinstance(c, (str, float, int, tuple)):
                go(c)
    go(e)
    return sorted(set(out))


def _kind_for(e, metric) -> str:
    found = []

    def go(x):
        if isinstance(x, ir.Select) and x.metric == metric:
            found.append(x.kind)
        for f in ("expr", "lhs", "rhs", "arg"):
            c = getattr(x, f, None)
            if c is not None and not isinstance(c, (str, float, int, tuple)):
                go(c)
    go(e)
    return found[0] if found else "unknown"


# ------------------------------------------------------------------ data

def make_values(kind: str, n: int, seed: int) -> np.ndarray:
    """Per-scrape increments for a counter, or levels for anything else."""
    rng = np.random.default_rng(seed)
    steps = rng.integers(0, 40, size=n).astype(float)
    return np.cumsum(steps) if kind == "counter" else (50.0 + rng.normal(0, 10, n))


def build_data(reps: list[dict], t0: float) -> dict:
    grid = np.arange(t0, t0 + WINDOW + SCRAPE / 2, SCRAPE)
    data = {}
    for r in reps:
        for j, spec in enumerate(r["series"]):
            key = (spec["dd_metric"], tuple(sorted(spec["labels"].items())))
            if key in data:
                continue
            data[key] = {"spec": spec, "ts": grid,
                         "vs": make_values(spec["kind"], len(grid), seed=_seed(key))}
    return data


def _seed(key: tuple) -> int:
    r"""A stable seed for one synthetic series, from its own name.

    This was `hash(key) % 2**31`. Python salts the hash of a string per process, so
    the same series was built from different pseudo-random values on every run and the
    corpus-scale differential -- 330 shapes, 22,692 bucket comparisons, the agreement
    this paper reports at floating-point precision -- was not re-runnable. The cache is
    committed, so no printed figure was ever wrong; but a measurement nobody can repeat
    is a measurement on the author's word, and the trial window that produced it has
    closed. SHA-256 of the key's own text does not move between processes.
    """
    import hashlib

    return int.from_bytes(
        hashlib.sha256(repr(key).encode()).digest()[:4], "big") % 2**31


def write_prometheus(workdir: Path, data: dict) -> str:
    lines = []
    by_metric = collections.defaultdict(list)
    for (_, _), d in data.items():
        by_metric[d["spec"]["metric"]].append(d)
    for metric, entries in by_metric.items():
        # Declared gauge regardless of the metric's role. OpenMetrics requires a counter's
        # samples to carry a `_total` suffix, and the translator emits the bare name the
        # Datadog query used, so declaring the truth here would mean querying a name the
        # translation never produces. The TYPE line is metadata: `x - x offset 60s` and
        # `increase(x[60s])` evaluate the same either way, and the data written is the
        # cumulative series in both cases.
        lines.append(f"# TYPE {metric} gauge")
        name = metric
        for d in entries:
            lbl = ",".join(f'{k}="{v}"' for k, v in sorted(d["spec"]["labels"].items()))
            sel = f"{name}{{{lbl}}}" if lbl else name
            for t, v in zip(d["ts"], d["vs"]):
                lines.append(f"{sel} {v:.17g} {t:.3f}")
    lines.append("# EOF")
    (workdir / "in").mkdir(parents=True, exist_ok=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "in" / "data.om").write_text("\n".join(lines) + "\n")
    (workdir / "prometheus.yml").write_text("global:\n  scrape_interval: 15s\n")
    run("docker", "run", "--rm", "--user", _ids(), "-v", f"{workdir}:/w",
        "--entrypoint", "/bin/promtool", IMAGE,
        "tsdb", "create-blocks-from", "openmetrics", "/w/in/data.om", "/w/data")
    out = run("docker", "run", "-d", "--user", _ids(), "-v", f"{workdir}:/w",
              "-p", f"{PORT}:9090", IMAGE,
              "--config.file=/w/prometheus.yml", "--storage.tsdb.path=/w/data",
              "--enable-feature=promql-extended-range-selectors",
              "--storage.tsdb.retention.time=3650d")
    return out.stdout.strip()


def write_datadog(creds, data: dict, run_tag: str) -> int:
    sent = 0
    for (_, _), d in data.items():
        spec = d["spec"]
        tags = ([f"run:{run_tag}"]
                + [f"{k}:{v}" for k, v in sorted(spec["dd_labels"].items())])
        if spec["kind"] == "counter":
            deltas = np.diff(d["vs"], prepend=d["vs"][0])
            pts = [(float(t), float(v)) for t, v in zip(d["ts"], deltas)][1:]
            mtype = COUNT
        else:
            pts = [(float(t), float(v)) for t, v in zip(d["ts"], d["vs"])]
            mtype = GAUGE
        for i in range(0, len(pts), 200):
            submit(creds, count_payload(spec["dd_metric"], pts[i:i + 200],
                                        tags=tags, metric_type=mtype))
            sent += 1
    return sent


# ------------------------------------------------------------------ comparison

def wait_ready(base_url: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/-/ready", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError("prometheus not ready")


def prom_range(base_url: str, expr: str, start: float, end: float, step: float) -> dict:
    body = urllib.parse.urlencode({"query": expr, "start": f"{start:.3f}",
                                   "end": f"{end:.3f}", "step": f"{step:g}"}).encode()
    req = urllib.request.Request(f"{base_url}/api/v1/query_range", data=body)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise ValueError(exc.read()[:200].decode(errors="replace")) from None
    if d.get("status") != "success":
        raise ValueError(d.get("error", "query failed"))
    out = {}
    for s in d["data"]["result"]:
        labels = {k: v for k, v in s["metric"].items() if k != "__name__"}
        key = tuple(sorted(labels.items()))
        out[key] = {round(float(t), 3): float(v) for t, v in s["values"]}
    return out


def dd_series(raw: dict, keymap: dict) -> dict:
    """Each returned series keyed by its grouping tag values, in PromQL's spelling.

    `keymap` maps each grouping tag's Datadog spelling to the normalised label name the
    PromQL side uses, so a tag key containing dots lines up with its counterpart.

    `tag_set` carries the grouping tags; `scope` carries the query's filter terms as
    well, so a key that is both filtered and grouped appears twice there -- once with
    the glob the query asked for (`kube_stateful_set:*`) and once with the concrete
    value. Reading the glob as the group value makes every such series fail to match its
    PromQL counterpart, which is 32 shapes of this corpus.
    """
    out = {}
    for s in raw.get("series") or []:
        tags = {}
        terms = s.get("tag_set") or []
        if not terms:
            terms = [x.strip() for x in str(s.get("scope", "")).split(",")]
        for term in terms:
            k, sep, v = str(term).partition(":")
            if sep and k in keymap and "*" not in v:
                tags[keymap[k]] = v
        key = tuple(sorted(tags.items()))
        out[key] = {round(float(t) / 1000.0, 3): (None if v is None else float(v))
                    for t, v in (s.get("pointlist") or [])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if not shutil.which("docker"):
        print("docker not found", file=sys.stderr)
        return 2
    creds = credentials()

    # Floor to a multiple of 300 s, which is a multiple of every rollup interval this
    # corpus states below 1800 s (15, 20, 60, 300). An arbitrary whole second leaves the
    # scrape grid off every bucket boundary, and off a boundary the left-closed Datadog
    # bucket and a right-closed PromQL window hold the same samples -- so the evaluation
    # instant could be wrong by up to half a scrape and still agree on every bucket. The
    # convention this rig is built on was wrong once; an aligned grid is what tests it.
    t0 = float(int(time.time() - SETTLE - WINDOW) // 300 * 300)
    run_tag = f"r{int(t0)}"
    print("building representatives ...")
    reps = representatives(run_tag)
    if args.limit:
        reps = reps[:args.limit]
    data = build_data(reps, t0)
    print(f"  {len(reps)} shapes, {len(data)} series")

    print("submitting to Datadog ...")
    n_req = write_datadog(creds, data, run_tag)
    print(f"  {n_req} requests")

    workdir = Path(tempfile.mkdtemp(prefix="qlport-shapes-"))
    container, rows = "", []
    try:
        print("backfilling Prometheus ...")
        container = write_prometheus(workdir, data)
        base = f"http://localhost:{PORT}"
        wait_ready(base)
        print(f"waiting for Datadog ingestion ...")
        deadline = time.time() + INGEST_WAIT
        probe = reps[0]
        while time.time() < deadline:
            raw = query(creds, probe["datadog"], t0 - 60, t0 + WINDOW + 60)
            if raw.get("series"):
                break
            time.sleep(20)


        def compare_one(r):
            """One representative against both systems; returns a row."""
            try:
                D = r["interval"]
                buckets = dd.bucket_starts(t0 + D, t0 + WINDOW - D, D)
                if len(buckets) < 2:
                    return {"index": r["index"], "status": "window-too-short",
                            "detail": f"interval {D:g}s over a {WINDOW:g}s window",
                            "occurrences": r["occurrences"]}
                raw = query(creds, r["datadog"], time.time() - RANGE, time.time())
                ddv = dd_series(raw, r["dd_group_keys"])
                pmv = prom_range(base, r["promql"], dd.eval_instant(buckets[0], D),
                                 dd.eval_instant(buckets[-1], D), D)
            except Exception as exc:
                return {"index": r["index"], "status": "error",
                        "detail": str(exc)[:200], "occurrences": r["occurrences"]}
            if not ddv or not pmv:
                return {"index": r["index"], "status": "no-data",
                        "detail": f"datadog series={len(ddv)} promql series={len(pmv)}",
                        "occurrences": r["occurrences"]}
            diffs, compared = [], 0
            for gkey, dpoints in ddv.items():
                ppoints = pmv.get(gkey)
                if ppoints is None and len(pmv) == 1:
                    ppoints = next(iter(pmv.values()))
                if ppoints is None:
                    continue
                for b, dv in dpoints.items():
                    if dv is None:
                        continue
                    u = round(dd.eval_instant(b, r["interval"]), 3)
                    pv = ppoints.get(u)
                    if pv is None:
                        continue
                    compared += 1
                    diffs.append(reldiff(dv, pv))
            tol = (CANCELLATION_TOLERANCE
                   if any(op in r["promql"] for op in (" - ", " + ")) else TOLERANCE)
            if not compared:
                return {"index": r["index"], "status": "unmatched",
                        "detail": f"dd groups={list(ddv)[:2]} pq groups={list(pmv)[:2]}",
                        "occurrences": r["occurrences"]}
            return {"index": r["index"], "status": "compared",
                    "occurrences": r["occurrences"], "buckets": compared,
                    "tolerance": tol,
                    "exact": sum(1 for d in diffs if d <= tol) / compared,
                    "bit_identical": sum(1 for d in diffs if d == 0) / compared,
                    "median_d": float(np.median(diffs)),
                    "max_d": float(np.max(diffs)),
                    "caveats": r["caveats"],
                    "datadog": r["datadog_original"], "promql": r["promql"]}

        # A newly created custom metric becomes queryable on its own schedule, so a
        # first pass finds stragglers rather than failures. Retry only those.
        pending, results = list(reps), {}
        for attempt in range(1, 7):
            still = []
            for k, r in enumerate(pending, 1):
                if k % 50 == 0:
                    print(f"    pass {attempt}: {k}/{len(pending)}")
                row = compare_one(r)
                # A newly created metric is not queryable yet (no-data), and a run this
                # long occasionally draws a 429 or a 503 from the API. Neither is a
                # statement about the translation, so both are retried; recording a
                # rate limit as a shape that could not be compared would be the harness
                # masquerading as a result, which is the failure mode this rig has had
                # four times already.
                transient = row["status"] == "no-data" or (
                    row["status"] == "error"
                    and any(code in str(row.get("detail", ""))
                            for code in ("429", "503", "Resource Exhausted", "timed out")))
                if transient and attempt < 6:
                    still.append(r)
                else:
                    results[r["index"]] = row
            print(f"  pass {attempt}: {len(pending) - len(still)} settled, "
                  f"{len(still)} awaiting data")
            pending = still
            if not pending:
                break
            time.sleep(60)
        for r in pending:
            results[r["index"]] = compare_one(r)
        rows = [results[r["index"]] for r in reps]
    finally:
        if container:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)

    by_status = collections.Counter(r["status"] for r in rows)
    cmp_rows = [r for r in rows if r["status"] == "compared"]
    agree = [r for r in cmp_rows if r["exact"] == 1.0]
    payload = {
        "provenance": {
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run": run_tag, "shapes": len(reps), "series": len(data),
            "rollup": ROLLUP, "range_seconds": RANGE, "site": creds.site,
        },
        "status_counts": dict(by_status),
        "shapes_compared": len(cmp_rows),
        "shapes_exact": len(agree),
        "occurrences_compared": sum(r["occurrences"] for r in cmp_rows),
        "occurrences_exact": sum(r["occurrences"] for r in agree),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"  status: {dict(by_status)}")
    if cmp_rows:
        print(f"  shapes compared : {len(cmp_rows)}  agreeing on every bucket: {len(agree)}")
        print(f"  occurrences     : {payload['occurrences_compared']:,}  "
              f"exact: {payload['occurrences_exact']:,}")
        bad = sorted((r for r in cmp_rows if r["exact"] < 1.0),
                     key=lambda r: -r["occurrences"])[:8]
        for r in bad:
            print(f"\n  MISMATCH ({r['occurrences']} occ, agree {r['exact']:.0%}, "
                  f"max d {r['max_d']:.3g})")
            print(f"    dd : {r['datadog'][:92]}")
            print(f"    pq : {r['promql'][:92]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

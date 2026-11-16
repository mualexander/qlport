#!/usr/bin/env python3
"""Replay a real request log into Prometheus and Datadog, and compare the rollup rules.

This is the dual-write model of paper §3.1 made real. One stream of observations reaches
both systems: Prometheus stores the cumulative counter, Datadog stores the per-interval
deltas that are its image under φ. Every rollup candidate is then read back from each
system and compared bucket by bucket, so the measurement is against Datadog rather than
against a model of Datadog -- the gap §9 names as the one that would change the standing
of everything else.

It also answers §8's other threat. The agreement figures of Table 2 come from synthetic
workloads whose shape was chosen; this replays an hour of a real web server instead,
where the per-minute request count swings seventeen-fold without being asked to.

Source: the NASA Kennedy Space Center HTTP logs of July 1995 (Internet Traffic Archive),
1.9M requests over 28 days. Its publishers ask that it not be analysed beyond general
traffic patterns, and `harness/replay.py` enforces that by construction: the parser keeps
a timestamp and an HTTP status class and discards the host and request line, so what is
replayed is a count of requests per interval per status class.

Timestamps are shifted to the present because Datadog's intake refuses points much older
than an hour. The shape is real; the absolute times are not.

Needs Docker and DD_API_KEY / DD_APP_KEY / DD_SITE in .env.

  python scripts/replay_dualwrite.py <path-to-NASA_access_log_Jul95.gz>
  python scripts/replay_dualwrite.py <path> --window 1800 --scrape 15 --rollup 60
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport.harness import prometheus as pm                      # noqa: E402
from qlport.harness.datadog import (                             # noqa: E402
    COUNT, count_payload, credentials, pointlist, query, submit,
)
from qlport.harness.replay import (                              # noqa: E402
    busiest_window, events_to_counter, parse_events, shift_to,
)
from qlport.metrics import reldiff                               # noqa: E402
from qlport.semantics import datadog_eval as dd                  # noqa: E402
from qlport.semantics import promql_eval as pq                   # noqa: E402
from qlport.semantics.ingest import phi_series                   # noqa: E402

IMAGE = "prom/prometheus:v3.15.0"
PORT = 19093                      # the other verification scripts hold 19090-19092
OUT_DIR = ROOT / "data" / "raw" / "replay"
DD_METRIC = "qlport.replay.requests"
PM_METRIC = "qlport_replay_requests"
SETTLE = 180.0                    # land the last sample this far before now
INGEST_WAIT = 600.0               # a brand-new custom metric takes minutes to appear


def run(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _ids() -> str:
    import os
    return f"{os.getuid()}:{os.getgid()}"


# ---------------------------------------------------------------- the workload

def load_window(path: Path, *, width: float, scrape: float,
                align_to: float | None = None) -> tuple[dict, dict]:
    """The busiest `width`-second window of the trace, as one counter per status class."""
    with gzip.open(path, "rt", encoding="latin-1", errors="replace") as f:
        events = list(parse_events(f))
    times = np.array([e for e, _ in events], dtype=float)
    classes = np.array([c for _, c in events])
    start, end = busiest_window(times, width=width)

    # Land the last sample SETTLE seconds ago, so every point is inside Datadog's
    # intake window and the newest is old enough to have been ingested. A whole second,
    # because the intake takes integer timestamps -- and because the trace's own events
    # are at second resolution, an integer shift keeps every sample on the scrape grid
    # rather than scattering it a fraction of a second off.
    new_start = float(int(time.time() - SETTLE - width))
    if align_to:
        # Snap the scrape grid onto the rollup grid, so samples land exactly on bucket
        # boundaries. That is the only case in which the closure convention is
        # observable at all: off the boundary, [b, b+D) and (b, b+D] read the same
        # samples and the two readings agree whichever is right. Real scrapes are
        # jittered and rarely aligned, which is how the wrong convention survived
        # being assumed for so long.
        new_start = float(int(new_start // align_to) * int(align_to))
    series, counts = {}, {}
    for cls in sorted({str(c) for c in classes[(times >= start) & (times < end)]}):
        sel = times[(times >= start) & (times < end) & (classes == cls)]
        s = events_to_counter(sel, interval=scrape, start=start, end=end,
                              labels={"class": cls})
        series[cls] = pm.quantise(shift_to(s, new_start))
        counts[cls] = int(len(sel))
    meta = {"trace_window_start": start, "trace_window_end": end,
            "replayed_from": new_start, "events_in_window": counts,
            "events_parsed": len(events)}
    return series, meta


# ---------------------------------------------------------------- the two writes

def write_datadog(creds, series: dict, run_tag: str) -> list[dict]:
    """Submit φ(s) -- the per-interval deltas -- as COUNT points, as the agent would."""
    acks = []
    for cls, s in series.items():
        deltas = phi_series(s, "zero")
        points = [(float(t), float(v)) for t, v in zip(deltas.ts, deltas.vs)]
        tags = ["rig:replay", f"run:{run_tag}", f"class:{cls}"]
        for i in range(0, len(points), 100):
            acks.append(submit(creds, count_payload(DD_METRIC, points[i:i + 100],
                                                    tags=tags, metric_type=COUNT)))
    return acks


def write_prometheus(workdir: Path, series: dict) -> str:
    """Backfill the cumulative counters and start a server on them."""
    lines = [f"# TYPE {PM_METRIC} counter"]
    for cls, s in series.items():
        for t, v in zip(s.ts, s.vs):
            lines.append(f'{PM_METRIC}_total{{class="{cls}"}} {v:.17g} {t:.3f}')
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
              # Required, not optional: on 3.15.0 `anchored` still answers
              # "anchored modifier is experimental and not enabled" without it.
              "--enable-feature=promql-extended-range-selectors",
              "--storage.tsdb.retention.time=3650d")
    return out.stdout.strip()


def wait_ready(base_url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/-/ready", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{base_url} not ready after {timeout}s")


# ---------------------------------------------------------------- the comparison

CANDIDATES = {
    "x - x offset D": lambda m, cls, D: f'{PM_METRIC}_total{{class="{cls}"}} - '
                                        f'{PM_METRIC}_total{{class="{cls}"}} offset {D:.0f}s',
    "increase(x[D] anchored)": lambda m, cls, D: f'increase({PM_METRIC}_total{{class="{cls}"}}[{D:.0f}s] anchored)',
    "rate(x[D]) * D": lambda m, cls, D: f'rate({PM_METRIC}_total{{class="{cls}"}}[{D:.0f}s]) * {D:.0f}',
}


def promql_at(base_url: str, expr: str, at: float) -> float:
    import urllib.parse
    body = urllib.parse.urlencode({"query": expr, "time": f"{at:.3f}"}).encode()
    req = urllib.request.Request(f"{base_url}/api/v1/query", data=body)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        # The body carries the parse error; without it a 400 says nothing about which
        # of several candidate expressions the server rejected.
        raise ValueError(f"{expr}: {exc.read()[:300].decode(errors='replace')}") from None
    if d.get("status") != "success":
        raise ValueError(d.get("error", "query failed"))
    res = d["data"]["result"]
    return float(res[0]["value"][1]) if res else float("nan")


def await_datadog(creds, q: str, start: float, end: float) -> dict:
    deadline = time.time() + INGEST_WAIT
    raw = {}
    while True:
        raw = query(creds, q, start, end)
        if any(v for v in pointlist(raw).values()):
            return raw
        if time.time() >= deadline:
            return raw
        time.sleep(15.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--window", type=float, default=1800.0)
    ap.add_argument("--scrape", type=float, default=15.0)
    ap.add_argument("--rollup", type=float, default=60.0)
    ap.add_argument("--align", action="store_true",
                    help="snap the scrape grid onto the rollup grid, so samples land on "
                         "bucket boundaries and the closure convention becomes visible")
    args = ap.parse_args(argv)

    if not shutil.which("docker"):
        print("docker not found", file=sys.stderr)
        return 2
    creds = credentials()
    D = args.rollup

    print(f"reading {args.trace.name} ...")
    series, meta = load_window(args.trace, width=args.window, scrape=args.scrape,
                               align_to=D if args.align else None)
    print(f"  window holds {meta['events_in_window']}, replaying from "
          f"{datetime.fromtimestamp(meta['replayed_from'], timezone.utc):%H:%M:%SZ}")

    run_tag = f"r{int(meta['replayed_from'])}"
    print("submitting to Datadog ...")
    acks = write_datadog(creds, series, run_tag)

    workdir = Path(tempfile.mkdtemp(prefix="qlport-replay-"))
    container = ""
    rows: list[dict] = []
    try:
        print("backfilling Prometheus ...")
        container = write_prometheus(workdir, series)
        base = f"http://localhost:{PORT}"
        wait_ready(base)
        version = json.loads(urllib.request.urlopen(
            f"{base}/api/v1/status/buildinfo", timeout=10).read())["data"]["version"]

        first = min(float(s.ts[0]) for s in series.values())
        last = max(float(s.ts[-1]) for s in series.values())
        buckets = dd.bucket_starts(first + D, last - D, D)
        print(f"comparing {len(buckets)} buckets of {D:.0f}s across {len(series)} series ...")

        for cls in series:
            q = (f"sum:{DD_METRIC}{{run:{run_tag},class:{cls}}}"
                 f".as_count().rollup(sum, {D:.0f})")
            raw = await_datadog(creds, q, first - D, last + D)
            got = {t: v for t, v in pointlist(raw).items() if v is not None}
            for b in buckets:
                if b not in got:
                    continue
                # Two evaluation instants per bucket. `eval_instant` is the measured
                # convention (§3.1): Datadog's bucket is [b, b+D) and a PromQL window is
                # (u-D, u], so the instant that reads the same samples is strictly
                # inside the bucket. `b + D` is the bucket's right edge -- the obvious
                # choice, and the one an earlier draft of this paper would have used.
                # Recording both turns "the correction matters" into a number.
                for label, u in (("inside", dd.eval_instant(b, D)),
                                 ("on the edge", b + D)):
                    for name, build in CANDIDATES.items():
                        value = promql_at(base, build(PM_METRIC, cls, D), u)
                        rows.append({"class": cls, "bucket": float(b),
                                     "candidate": name, "instant": label,
                                     "datadog": got[b], "promql": value,
                                     "d": reldiff(got[b], value)})
    finally:
        if container:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["instant"], r["candidate"])].append(r["d"])
    summary = {f"{inst} | {cand}": {"buckets": len(v),
                                    "exact": sum(1 for x in v if x == 0) / len(v),
                                    "median_d": float(np.median(v)),
                                    "max_d": float(np.max(v))}
               for (inst, cand), v in by.items() if v}

    payload = {
        "provenance": {
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "NASA-HTTP Jul 1995 (Internet Traffic Archive)",
            "trace_file": args.trace.name, "run": run_tag,
            "window_seconds": args.window, "scrape_seconds": args.scrape,
            "rollup_seconds": D, "site": creds.site, "aligned": bool(args.align),
            "prometheus_version": version, "image": IMAGE,
            **meta,
        },
        "acks": acks, "summary": summary, "rows": rows,
    }
    # Two runs, two files: the aligned one is the only case in which the closure
    # convention is observable, and the unaligned one is the realistic scrape. Keeping
    # both is the comparison.
    out = OUT_DIR / ("dualwrite_aligned.json" if args.align else "dualwrite_unaligned.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    for name, s in summary.items():
        print(f"  {name:40} {s['buckets']:4} buckets  exact {s['exact']:6.1%}  "
              f"median d {s['median_d']:.3g}  max d {s['max_d']:.3g}")
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())

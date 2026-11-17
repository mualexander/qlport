#!/usr/bin/env python3
"""E7: what METRIC_TYPE_UNKNOWN costs when it fails (paper §5, §7).

Every translation in the corpus carries METRIC_TYPE_UNKNOWN: the emitted expression is
exact if the metric is a GAUGE and wrong if it is a COUNT, and a dashboard does not say
which. §5 calls the wrong case "not approximately right but unrelated". That is an
assertion about a magnitude, and this measures it.

The construction is the dual-write model of §3.1 with the assumption deliberately
violated. One stream of counter observations: the per-interval deltas go to Datadog as a
COUNT metric, the cumulative totals go to Prometheus. A query with no method --- the
commonest shape in the corpus, and the one that leaves the type unstated --- is then read
from both sides:

    Datadog    avg:m{*}.rollup(avg, D)      the mean of the stored deltas
    PromQL     avg_over_time(m[Ds])         the mean of the running totals

Both are what each system is asked for. They are unrelated quantities, and the point of
measuring rather than asserting it is that the gap is not a fixed factor: the mean of the
totals grows with the counter's age while the mean of the deltas does not, so the error
depends on how long the counter has been running and is unbounded in it.

A GAUGE control is submitted alongside, carrying the same stored values. Datadog cannot
tell the two apart --- the stored points are identical -- so the control reads the same
from Datadog and agrees with Prometheus, which is the whole content of the precondition.

Writes data/raw/datadog/e7.json.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                    # noqa: E402

from qlport.harness.datadog import (                                  # noqa: E402
    COUNT, GAUGE, count_payload, credentials, query, submit,
)
from qlport.semantics import datadog_eval as dd                       # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data/raw/datadog/e7.json"

PROM_IMAGE = "prom/prometheus:v3.15.0"
PROM_NAME = "qlport-e7-prom"
PROM_PORT = 19095

STEP = 10.0          # submission grid
INTERVAL = 30.0      # rollup bucket under test
SPAN = 1800.0        # half an hour of points
DELTA = 10.0         # per-interval increment: every bucket holds the same three
INGEST_WAIT = 150.0


def run(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def start_prometheus(workdir: Path, lines: list[str]) -> None:
    (workdir / "in").mkdir(parents=True, exist_ok=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "in" / "data.om").write_text("\n".join(lines) + "\n")
    (workdir / "prometheus.yml").write_text("global:\n  scrape_interval: 15s\n")
    ids = f"{os.getuid()}:{os.getgid()}"
    run("docker", "run", "--rm", "--user", ids, "-v", f"{workdir}:/w",
        "--entrypoint", "/bin/promtool", PROM_IMAGE,
        "tsdb", "create-blocks-from", "openmetrics", "/w/in/data.om", "/w/data")
    run("docker", "run", "-d", "--name", PROM_NAME, "--user", ids,
        "-v", f"{workdir}:/w", "-p", f"{PROM_PORT}:9090", PROM_IMAGE,
        "--config.file=/w/prometheus.yml", "--storage.tsdb.path=/w/data",
        "--storage.tsdb.retention.time=3650d")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{PROM_PORT}/-/ready",
                                        timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError("prometheus not ready")


def prom_at(expr: str, at: float) -> float | None:
    body = urllib.parse.urlencode({"query": expr, "time": f"{at:.3f}"}).encode()
    req = urllib.request.Request(f"http://localhost:{PROM_PORT}/api/v1/query", data=body)
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    res = d["data"]["result"]
    return float(res[0]["value"][1]) if res else None


def dd_buckets(creds, metric: str, start: float, end: float) -> dict:
    raw = query(creds, f"avg:{metric}{{*}}.rollup(avg, {int(INTERVAL)})", start, end)
    out = {}
    for s in raw.get("series") or []:
        for t_ms, v in s.get("pointlist") or []:
            if v is not None:
                out[round(float(t_ms) / 1000.0, 3)] = float(v)
    return out


def main() -> int:
    creds = credentials()
    t0 = float(int(time.time() - SPAN - 240.0))
    t0 = math.floor(t0 / INTERVAL) * INTERVAL
    run_tag = f"r{int(t0)}"
    m_count = f"qlport.e7.{run_tag}.counter"
    m_gauge = f"qlport.e7.{run_tag}.gauge"

    n = int(SPAN // STEP)
    ts = [t0 + k * STEP for k in range(n)]
    deltas = [DELTA] * n
    cumulative = list(np.cumsum(deltas))

    print(f"{run_tag}: {n} points, delta {DELTA:.0f} every {STEP:.0f}s, "
          f"bucket {INTERVAL:.0f}s")
    print(f"  cumulative climbs {cumulative[0]:.0f} -> {cumulative[-1]:.0f}")

    # Datadog receives the deltas, as the agent would send them, under both types.
    for metric, mtype in ((m_count, COUNT), (m_gauge, GAUGE)):
        pts = [(t, d) for t, d in zip(ts, deltas)]
        for i in range(0, len(pts), 200):
            submit(creds, count_payload(metric, pts[i:i + 200], tags=["rig:e7"],
                                        metric_type=mtype))

    # Prometheus receives the cumulative totals for the counter, and the same stored
    # values as the gauge -- which is exactly the difference the type makes.
    lines = ["# TYPE qlport_e7_counter gauge"]
    lines += [f'qlport_e7_counter {v:.17g} {t:.3f}' for t, v in zip(ts, cumulative)]
    lines += ["# TYPE qlport_e7_gauge gauge"]
    lines += [f'qlport_e7_gauge {v:.17g} {t:.3f}' for t, v in zip(ts, deltas)]
    lines.append("# EOF")

    workdir = Path(tempfile.mkdtemp(prefix="qlport-e7-"))
    subprocess.run(["docker", "rm", "-f", PROM_NAME], capture_output=True)
    try:
        start_prometheus(workdir, lines)
        print("  prometheus up")
        print(f"  waiting {INGEST_WAIT:.0f}s for Datadog ingestion")
        time.sleep(INGEST_WAIT)

        qs, qe = ts[0] - INTERVAL, ts[-1] + 2 * INTERVAL
        dd_c = dd_buckets(creds, m_count, qs, qe)
        dd_g = dd_buckets(creds, m_gauge, qs, qe)
        buckets = [float(b) for b in dd.bucket_starts(ts[0], ts[-1], INTERVAL)
                   if b >= ts[0] and b + INTERVAL <= ts[-1]]

        rows = []
        for b in buckets:
            u = dd.eval_instant(b, INTERVAL)
            row = {
                "bucket": b,
                "age_s": b - ts[0],
                "datadog_counter": dd_c.get(round(b, 3)),
                "datadog_gauge": dd_g.get(round(b, 3)),
                "promql_counter": prom_at(
                    f"avg_over_time(qlport_e7_counter[{int(INTERVAL)}s])", u),
                "promql_gauge": prom_at(
                    f"avg_over_time(qlport_e7_gauge[{int(INTERVAL)}s])", u),
            }
            for kind in ("counter", "gauge"):
                a, p = row[f"datadog_{kind}"], row[f"promql_{kind}"]
                row[f"d_{kind}"] = (None if a is None or p is None
                                   else abs(a - p) / max(abs(a), abs(p), 1e-12))
            rows.append(row)

        scored = [r for r in rows if r["d_counter"] is not None]
        payload = {
            "provenance": {
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "site": creds.site, "prometheus_image": PROM_IMAGE,
                "step": STEP, "interval": INTERVAL, "delta": DELTA,
                "points": n, "run": run_tag,
            },
            "rows": rows,
            "summary": {
                "buckets": len(scored),
                "gauge_max_d": max((r["d_gauge"] for r in scored
                                    if r["d_gauge"] is not None), default=None),
                "counter_min_d": min(r["d_counter"] for r in scored),
                "counter_max_d": max(r["d_counter"] for r in scored),
                "counter_first": scored[0]["d_counter"] if scored else None,
                "counter_last": scored[-1]["d_counter"] if scored else None,
                "promql_counter_first": scored[0]["promql_counter"] if scored else None,
                "promql_counter_last": scored[-1]["promql_counter"] if scored else None,
                "datadog_counter_value": scored[0]["datadog_counter"] if scored else None,
            },
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {OUT}")
        s = payload["summary"]
        print(f"  gauge  (assumption holds): max d = {s['gauge_max_d']}")
        print(f"  counter(assumption fails): d from {s['counter_first']:.4f} "
              f"to {s['counter_last']:.4f} over {s['buckets']} buckets")
        print(f"    datadog reads {s['datadog_counter_value']}, promql reads "
              f"{s['promql_counter_first']} -> {s['promql_counter_last']}")
        return 0
    finally:
        subprocess.run(["docker", "rm", "-f", PROM_NAME], capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())

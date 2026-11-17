#!/usr/bin/env python3
"""E9: what `.weighted()` computes, and whether PromQL can say it (paper §7).

`.weighted()` is $70$ of the $175$ occurrences §7 calls inexpressible --- the largest
single entry in the column the abstract's $2.0\\%$ comes from --- and the warrant is one
sentence: "a time-weighted space average, which no PromQL operator computes". That is an
assertion, and by Remark rem:refusal's own standard it is not evidence. Two things make
it worth testing rather than repeating:

  * $54$ of the $56$ distinct corpus forms apply it to `sum:`, not to `avg:`, so
    whatever it does is probably not an average at all.
  * Datadog documents it for gauges reported at irregular intervals, which suggests a
    per-point weight rather than a per-series one --- and a per-point weight is
    `count_over_time`, which PromQL has.

Two series of one gauge metric with different reporting densities, so that any weighting
by how much of the interval a series covers is visible:

  dense   value 10, every 10 s   -- three points in a 30 s bucket
  sparse  value 100, every 30 s  -- one

The same two series go to Prometheus, and the candidate expressions are read back from
it. Writes data/raw/datadog/e9.json.
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

from qlport.harness.datadog import (                                  # noqa: E402
    COUNT, GAUGE, RATE, count_payload, credentials, query, submit,
)

OUT = Path(__file__).resolve().parents[1] / "data/raw/datadog/e9.json"
PROM_IMAGE = "prom/prometheus:v3.15.0"
PROM_NAME = "qlport-e9-prom"
PROM_PORT = 19099

INTERVAL = 60.0
SPAN = 1200.0
INGEST_WAIT = 150.0

# The discriminator is spacing *within* a series, not density between series. A first
# version varied how often two series reported and `.weighted()` changed nothing,
# because each series was internally regular and a per-point weight is then uniform.
# Here one series reports at 0 s, 50 s and 55 s of every bucket: a plain mean of its
# three points is 66.7, and a mean weighted by the interval each point covers is 16.7.
IRREGULAR = [(0.0, 0.0), (50.0, 100.0), (55.0, 100.0)]
REGULAR = [(o, 40.0) for o in (0.0, 15.0, 30.0, 45.0)]      # control: 40 either way


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


def prom(expr: str, at: float):
    body = urllib.parse.urlencode({"query": expr, "time": f"{at:.3f}"}).encode()
    req = urllib.request.Request(f"http://localhost:{PROM_PORT}/api/v1/query", data=body)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return {"__error__": exc.read()[:200].decode(errors="replace")}
    return {x["metric"].get("d", "<none>"): round(float(x["value"][1]), 4)
            for x in d["data"]["result"]}


def dd(creds, q: str, start: float, end: float):
    try:
        raw = query(creds, q, start, end)
    except Exception as exc:                                          # noqa: BLE001
        return {"__error__": str(exc)[:200]}
    out = {}
    for s in raw.get("series") or []:
        vals = [round(float(v), 4) for _, v in (s.get("pointlist") or [])
                if v is not None]
        # the interior buckets, away from the partially covered ends
        out[str(s.get("scope", ""))] = sorted(set(vals[2:-2] or vals))[:6]
    return out


def main() -> int:
    creds = credentials()
    t0 = math.floor((time.time() - SPAN - 240.0) / INTERVAL) * INTERVAL
    tag = f"r{int(t0)}"
    metric = f"qlport.e9.{tag}.g"

    series = {}
    for name, offsets in (("irregular", IRREGULAR), ("regular", REGULAR)):
        pts = [(t0 + b * INTERVAL + off, val)
               for b in range(int(SPAN // INTERVAL)) for off, val in offsets]
        series[name] = pts
        for i in range(0, len(pts), 200):
            submit(creds, count_payload(metric, pts[i:i + 200],
                                        tags=[f"d:{name}", "rig:e9"],
                                        metric_type=GAUGE))

    # The same irregular shape under the other two metric types. `.weighted()` doing
    # nothing to a gauge would be no surprise after E6; whether it does nothing to a
    # COUNT or a RATE as well is what decides how the 70 occurrences are classified.
    typed = {}
    for suffix, mtype in (("c", COUNT), ("r", RATE)):
        name = f"qlport.e9.{tag}.{suffix}"
        typed[suffix] = name
        pts = [(t0 + b * INTERVAL + off, val)
               for b in range(int(SPAN // INTERVAL)) for off, val in IRREGULAR]
        for i in range(0, len(pts), 200):
            submit(creds, count_payload(name, pts[i:i + 200], tags=["rig:e9"],
                                        interval=int(INTERVAL), metric_type=mtype))
    print(f"{tag}: irregular at {[o for o, _ in IRREGULAR]}s of each "
          f"{INTERVAL:.0f}s bucket; regular control")

    lines = ["# TYPE qlport_e9_g gauge"]
    for name, pts in series.items():
        for t, v in pts:
            lines.append(f'qlport_e9_g{{d="{name}"}} {v:.17g} {t:.3f}')
    lines.append("# EOF")

    R = int(INTERVAL)
    dq = {
        "by_plain": f"sum:{metric}{{*}} by {{d}}",
        "by_weighted": f"sum:{metric}{{*}} by {{d}}.weighted()",
        "by_rollup_avg": f"sum:{metric}{{*}} by {{d}}.rollup(avg, {R})",
        "by_weighted_rollup_avg":
            f"sum:{metric}{{*}} by {{d}}.weighted().rollup(avg, {R})",
        "by_rollup_sum": f"sum:{metric}{{*}} by {{d}}.rollup(sum, {R})",
        "by_weighted_rollup_sum":
            f"sum:{metric}{{*}} by {{d}}.weighted().rollup(sum, {R})",
        "by_rollup_count": f"sum:{metric}{{*}} by {{d}}.rollup(count, {R})",
        "sum_plain": f"sum:{metric}{{*}}",
        "sum_weighted": f"sum:{metric}{{*}}.weighted()",
        "count_type_plain": f"sum:{typed['c']}{{*}}.rollup(avg, {R})",
        "count_type_weighted": f"sum:{typed['c']}{{*}}.weighted().rollup(avg, {R})",
        "rate_type_plain": f"sum:{typed['r']}{{*}}.rollup(avg, {R})",
        "rate_type_weighted": f"sum:{typed['r']}{{*}}.weighted().rollup(avg, {R})",
    }
    pq = {
        # the candidate: weight each series by how many points it contributed
        "candidate_weighted_mean":
            f"sum(avg_over_time(qlport_e9_g[{R}s]) * count_over_time(qlport_e9_g[{R}s]))"
            f" / sum(count_over_time(qlport_e9_g[{R}s]))",
        "candidate_weighted_sum":
            f"sum(avg_over_time(qlport_e9_g[{R}s]) * count_over_time(qlport_e9_g[{R}s]))",
        "plain_sum_of_means": f"sum(avg_over_time(qlport_e9_g[{R}s]))",
        "plain_mean_of_means": f"avg(avg_over_time(qlport_e9_g[{R}s]))",
        "per_series_mean": f"avg_over_time(qlport_e9_g[{R}s])",
        "per_series_count": f"count_over_time(qlport_e9_g[{R}s])",
        "per_series_sum": f"sum_over_time(qlport_e9_g[{R}s])",
    }

    workdir = Path(tempfile.mkdtemp(prefix="qlport-e9-"))
    subprocess.run(["docker", "rm", "-f", PROM_NAME], capture_output=True)
    try:
        start_prometheus(workdir, lines)
        print("  prometheus up")
        print(f"  waiting {INGEST_WAIT:.0f}s for Datadog ingestion")
        time.sleep(INGEST_WAIT)

        import qlport.semantics.datadog_eval as de
        mid = t0 + SPAN / 2
        at = de.eval_instant(math.floor(mid / INTERVAL) * INTERVAL, INTERVAL)
        qs, qe = t0 - INTERVAL, t0 + SPAN + 2 * INTERVAL

        out_dd = {k: dd(creds, v, qs, qe) for k, v in dq.items()}
        out_pq = {k: prom(v, at) for k, v in pq.items()}
        print("\n  Datadog")
        for k in dq:
            print(f"    {k:26s} {out_dd[k]}")
        print("\n  PromQL")
        for k in pq:
            print(f"    {k:26s} {out_pq[k]}")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "provenance": {
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "site": creds.site, "prometheus_image": PROM_IMAGE, "run": tag,
                "interval": INTERVAL, "irregular": IRREGULAR, "regular": REGULAR,
            },
            "datadog_queries": dq, "promql_queries": pq,
            "datadog": out_dd, "promql": out_pq,
        }, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {OUT}")
        return 0
    finally:
        subprocess.run(["docker", "rm", "-f", PROM_NAME], capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())

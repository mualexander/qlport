#!/usr/bin/env python3
"""Verify the PromQL models in semantics/promql_eval.py against a real Prometheus.

`increase(x[D] anchored)` was implemented from the text of Prometheus proposal 0052,
so notebook 02's claim about it was a claim about our own model. This script answers
the same questions to a real server on the same samples.

How it runs. Each workload from notebook 02 is rendered as OpenMetrics, backfilled
into a TSDB with `promtool tsdb create-blocks-from openmetrics`, and served by a
Prometheus started on that TSDB with no scrape targets. Every candidate is then read
back with one range query per candidate, evaluated one instant per epoch-aligned
Datadog bucket -- strictly inside the bucket, for the reason `bucket_grid` gives. Raw
responses land in data/raw/anchored/ and the
notebook reads only those.

Needs Docker. The image tag is pinned below, and the digest it resolved to is recorded
in every output file.

  python scripts/verify_anchored.py                 # all workloads
  python scripts/verify_anchored.py --only bursty   # one of them
  python scripts/verify_anchored.py --keep          # leave the container running
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport.semantics.datadog_eval import EVAL_EPSILON
from qlport.harness.prometheus import (  # noqa: E402
    openmetrics, quantise, query_range, series_from_response, write_raw,
)
from qlport.harness.workload import bursty, poisson, scrape, steady  # noqa: E402

IMAGE = "prom/prometheus:v3.15.0"   # >= 3.7.0, which is where `anchored` appeared
PORT = 19090
METRIC = "qlport_requests"
HOUR = 3600
RAW = ROOT / "data" / "raw" / "anchored"

# The same four workloads and two intervals as notebook 02, so the two tables are
# directly comparable: one computed from the models, one read from a real server.
RESTARTS = tuple(float(x) for x in np.arange(1800, 6 * HOUR, HOUR))
WORKLOADS = {
    "steady": dict(rate_fn=steady(10), jitter=0),
    "poisson": dict(rate_fn=poisson(10), jitter=2),
    "bursty": dict(rate_fn=bursty(10, 1800, 120, 10), jitter=2),
    "bursty + restarts": dict(rate_fn=bursty(10, 1800, 120, 10), jitter=2,
                              restarts=RESTARTS),
}
INTERVALS = (60.0, 300.0)
WARMUP = 600.0   # skip the first 10 min, as notebook 02 does: no left boundary before it

# Each candidate is a PromQL expression template; D is the rollup interval in seconds.
# These are exactly the three candidates of notebook 02, as a real server reads them.
CANDIDATES = {
    "rate(x[D])": "rate({m}[{d:g}s])",
    "(x - x offset D) / D": "({m} - {m} offset {d:g}s) / {d:g}",
    "increase(x[D] anchored) / D": "increase({m}[{d:g}s] anchored) / {d:g}",
}


def run(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def image_digest() -> str:
    out = run("docker", "image", "inspect", IMAGE, "--format", "{{index .RepoDigests 0}}")
    return out.stdout.strip()


def prometheus_version(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/api/v1/status/buildinfo", timeout=30) as r:
        return json.loads(r.read())["data"]


def backfill(workdir: Path, text: str) -> None:
    (workdir / "in").mkdir(parents=True, exist_ok=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "in" / "data.om").write_text(text)
    (workdir / "prometheus.yml").write_text("global:\n  scrape_interval: 15s\n")
    run("docker", "run", "--rm", "--user", f"{_uid()}:{_gid()}",
        "-v", f"{workdir}:/w", "--entrypoint", "/bin/promtool", IMAGE,
        "tsdb", "create-blocks-from", "openmetrics", "/w/in/data.om", "/w/data")


def _uid() -> int:
    import os
    return os.getuid()


def _gid() -> int:
    import os
    return os.getgid()


def start(workdir: Path) -> str:
    """Start Prometheus on the backfilled TSDB and return the container id."""
    out = run("docker", "run", "-d", "--user", f"{_uid()}:{_gid()}",
              "-v", f"{workdir}:/w", "-p", f"{PORT}:9090", IMAGE,
              "--config.file=/w/prometheus.yml", "--storage.tsdb.path=/w/data",
              # Required, not harmless. A 3.15.0 server without this answers
              # "anchored modifier is experimental and not enabled": the modifier is
              # still gated, so every claim about it carries that precondition.
              "--enable-feature=promql-extended-range-selectors",
              f"--storage.tsdb.retention.time={30}d")
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


def bucket_grid(t0: float, first: float, last: float, interval: float) -> tuple[float, float]:
    """First and last evaluation instant, one per epoch-aligned Datadog bucket.

    Datadog's bucket reported at `b` covers `[b, b + interval)`
    (`scripts/trial_checks.py e2`, measured). A PromQL range selector covers
    `(u - interval, u]`, so the instant that reads the same samples is strictly inside
    the bucket, not on its right edge: at `b + interval` the server would count a sample
    sitting exactly on the boundary that Datadog assigns to the next bucket, which on an
    aligned scrape is every bucket. `EVAL_EPSILON` is 1 ms, and Prometheus timestamps are
    whole milliseconds, so the shifted grid stays representable.

    `first` and `last` bound the bucket *starts*, matching notebook 02's modelled table
    (`dd.bucket_starts(WARMUP, ts[-1] - interval, interval)`). Reading them as evaluation
    instants instead put the first bucket at `[first - interval, first)` -- one bucket
    earlier than the model, and for a 300 s rollup wholly inside the declared warm-up --
    so the two tables were summarising different bucket sets while the comment below
    claimed they matched.
    """
    first_bucket = np.ceil((t0 + first) / interval) * interval
    last_bucket = np.floor((t0 + last) / interval) * interval
    return (float(first_bucket + interval - EVAL_EPSILON),
            float(last_bucket + interval - EVAL_EPSILON))


def collect(name: str, interval: float, base_url: str, t0: float, s, meta: dict) -> Path:
    start, end = bucket_grid(t0, WARMUP, float(s.ts[-1]) - interval, interval)
    payload = {
        "workload": name, "interval": interval, "t0": t0, "metric": METRIC,
        "grid": {"start": start, "end": end, "step": interval,
                 "eps": EVAL_EPSILON,
                 "convention": "bucket [b, b+interval); evaluated at b+interval-eps"},
        "warmup": WARMUP,
        "series": {"ts": [float(t) for t in s.ts], "vs": [float(v) for v in s.vs]},
        "candidates": {}, **meta,
    }
    for label, template in CANDIDATES.items():
        expr = template.format(m=f"{METRIC}_total", d=interval)
        raw = query_range(base_url, expr, start, end, interval)
        n = len(series_from_response(raw))      # raises if the query failed
        payload["candidates"][label] = {"expr": expr, "response": raw}
        print(f"    {label:30s} {n:4d} points")
    path = RAW / f"{name.replace(' + ', '_')}_{int(interval)}.json"
    write_raw(path, payload)
    print(f"    wrote {path.relative_to(ROOT)}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", choices=sorted(WORKLOADS),
                    help="verify only these workloads")
    ap.add_argument("--keep", action="store_true",
                    help="leave the last container running for manual queries")
    args = ap.parse_args()

    if not shutil.which("docker"):
        print("docker not found; see the module docstring", file=sys.stderr)
        return 2

    names = args.only or list(WORKLOADS)
    base_url = f"http://localhost:{PORT}"
    digest = image_digest()
    now = int(time.time())

    for name in names:
        print(f"== {name}")
        s = quantise(scrape(start_value=5000, interval=15, duration=6 * HOUR, seed=681,
                            **WORKLOADS[name]))
        # Place the series in the recent past so it falls inside retention, on a whole
        # hour. The hour matters: it is a multiple of every interval in INTERVALS, so
        # the server's epoch-aligned grid lands on exactly the bucket ends the model
        # table uses in series-relative time. Without that, the two tables would be
        # evaluating different buckets, and rate()'s error -- the one candidate whose
        # error depends on where the window falls -- would not be comparable.
        t0 = float((now - int(s.ts[-1]) - 600) // HOUR * HOUR)
        workdir = Path(tempfile.mkdtemp(prefix="qlport-prom-"))
        container = None
        try:
            backfill(workdir, openmetrics(s, METRIC, t0))
            container = start(workdir)
            wait_ready(base_url)
            build = prometheus_version(base_url)
            meta = {"image": IMAGE, "image_digest": digest,
                    "prometheus_version": build.get("version"),
                    "prometheus_revision": build.get("revision"),
                    "collected_unix": now}
            print(f"  prometheus {build.get('version')} on {digest.split('@')[-1][:19]}…")
            for interval in INTERVALS:
                print(f"  interval {interval:g}s")
                collect(name, interval, base_url, t0, s, meta)
        finally:
            if container and not args.keep:
                subprocess.run(["docker", "rm", "-f", container],
                               capture_output=True, text=True)
            if not args.keep:
                shutil.rmtree(workdir, ignore_errors=True)
    print("\ndone; notebook 02 reads data/raw/anchored/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

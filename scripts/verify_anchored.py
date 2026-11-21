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
REGISTRY = ROOT / "paper" / "quoted_figures.json"

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



# --- the case no workload above reaches ------------------------------------------------
# prop:anchored splits on whether a sample lies at or before u - D. Every workload above
# starts its series at t = 0 and reads buckets from WARMUP = 600 s on, so an anchor always
# exists and the second branch -- the series whose first sample falls *inside* the bucket
# being read, which is the case C_D excludes and the reason the proposition exists -- was
# argued from the proposal's text and from def:phi and never read off a server.
#
# This reads it. The series begins 20 s into the bucket at series-relative 0, scraped
# every 15 s, so that bucket holds samples at 20, 35 and 50 s and nothing before them.
# phi_zero runs from the second sample, so Sigma_60(0) counts two deltas and not the
# first sample's own value, which is 5000 and deliberately enormous: the three outcomes
# the measurement has to tell apart are 20 if the server agrees with the proof, 5020 if
# it reads the first sample as a delta, and about 40 if it extrapolates a half-full
# window to the width of the selector. Plain `increase` is queried beside the anchored
# form for exactly that reason -- it is the control that shows what extrapolation does to
# this series, so "no extrapolation" is a reading rather than a restatement of the
# proposal.
NO_ANCHOR = {
    "interval": 60.0,
    "scrape": 15.0,
    "offset": 20.0,       # where the first sample sits inside the first bucket
    "step": 10.0,         # increment per scrape
    "start_value": 5000.0,
    "duration": 600.0,
}
NO_ANCHOR_CANDIDATES = {
    "increase(x[D] anchored)": "increase({m}[{d:g}s] anchored)",
    "increase(x[D])": "increase({m}[{d:g}s])",
}


def no_anchor_series():
    """A counter whose first sample falls strictly inside the first aligned bucket."""
    from qlport.semantics.model import Series

    c = NO_ANCHOR
    ts = np.arange(c["offset"], c["duration"], c["scrape"], dtype=float)
    vs = c["start_value"] + c["step"] * np.arange(len(ts), dtype=float)
    return quantise(Series({}, ts, vs, kind="counter"))


def bucket_sums(s, interval: float) -> dict[float, float]:
    """Sigma_interval(b) at rho = zero, per aligned bucket, from this paper's model."""
    from qlport.semantics.ingest import phi_series

    d = phi_series(s, "zero")
    out: dict[float, float] = {}
    for t, v in zip(d.ts, d.vs):
        b = float(np.floor(t / interval) * interval)
        out[b] = out.get(b, 0.0) + float(v)
    return out


def collect_no_anchor(base_url: str, t0: float, s, meta: dict) -> Path:
    interval = NO_ANCHOR["interval"]
    sums = bucket_sums(s, interval)
    buckets = sorted(b for b in sums if b + interval <= float(s.ts[-1]))
    start = t0 + buckets[0] + interval - EVAL_EPSILON
    end = t0 + buckets[-1] + interval - EVAL_EPSILON

    payload = {
        "workload": "no anchor", "interval": interval, "t0": t0, "metric": METRIC,
        "config": NO_ANCHOR,
        "grid": {"start": start, "end": end, "step": interval, "eps": EVAL_EPSILON,
                 "convention": "bucket [b, b+interval); evaluated at b+interval-eps"},
        "model_bucket_sums": {f"{b:g}": sums[b] for b in buckets},
        "first_sample": float(s.ts[0]),
        "series": {"ts": [float(t) for t in s.ts], "vs": [float(v) for v in s.vs]},
        "candidates": {}, **meta,
    }
    for label, template in NO_ANCHOR_CANDIDATES.items():
        expr = template.format(m=f"{METRIC}_total", d=interval)
        raw = query_range(base_url, expr, start, end, interval)
        got = series_from_response(raw)
        payload["candidates"][label] = {"expr": expr, "response": raw}
        print(f"    {label:28s} {len(got):3d} point(s)")
    path = RAW / "no_anchor_60.json"
    write_raw(path, payload)
    print(f"    wrote {path.relative_to(ROOT)}")

    # The comparison, printed and registered rather than left for a notebook: this file
    # is the only evidence for prop:anchored's second branch, so the figures the proof
    # quotes come straight off it.
    read = {label: {round(t - t0 - interval + EVAL_EPSILON): v
                    for t, v in series_from_response(c["response"]).items()}
            for label, c in payload["candidates"].items()}
    anchored = read["increase(x[D] anchored)"]
    plain = read["increase(x[D])"]
    exact = [b for b in buckets if abs(anchored.get(b, 1e30) - sums[b]) < 1e-9]
    print(f"\n    {'bucket':>7} {'anchor?':>8} {'model':>7} {'anchored':>10} "
          f"{'increase':>10}")
    for b in buckets:
        print(f"    {b:>7g} {'no' if b == buckets[0] else 'yes':>8} {sums[b]:>7g} "
              f"{anchored.get(b, float('nan')):>10.6g} "
              f"{plain.get(b, float('nan')):>10.6g}")
    b0 = buckets[0]
    fig = {
        "na.buckets": len(buckets),
        "na.buckets_exact": len(exact),
        "na.no_anchor_bucket_sum": round(sums[b0], 6),
        "na.no_anchor_anchored": round(anchored[b0], 6),
        "na.no_anchor_plain": round(plain[b0], 1),
        "na.first_value": round(float(s.vs[0]), 6),
    }
    print()
    for k, v in fig.items():
        print(f"    {k:<28} {v:g}")
    if "--write-registry" in sys.argv:
        reg = json.loads(REGISTRY.read_text())
        reg.update(fig)
        REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
        print(f"\n    merged {len(fig)} keys into {REGISTRY.relative_to(ROOT)}")
    return path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", choices=sorted(WORKLOADS),
                    help="verify only these workloads")
    ap.add_argument("--keep", action="store_true",
                    help="leave the last container running for manual queries")
    ap.add_argument("--no-anchor", action="store_true",
                    help="verify prop:anchored's branch for a series that begins "
                         "inside the bucket being read")
    ap.add_argument("--write-registry", action="store_true",
                    help="merge the no-anchor figures into paper/quoted_figures.json")
    args = ap.parse_args()

    if not shutil.which("docker"):
        print("docker not found; see the module docstring", file=sys.stderr)
        return 2

    names = [] if args.no_anchor else (args.only or list(WORKLOADS))
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
    if args.no_anchor:
        print("== no anchor")
        s_na = no_anchor_series()
        # A whole hour, as above, and for a sharper reason here: 60 divides 3600, so the
        # series-relative bucket [0, 60) is an epoch-aligned bucket of the server's own
        # grid. The first sample sits at t0 + 20, inside it, with nothing before it
        # anywhere in the series -- which is what makes the first branch of
        # prop:anchored unavailable and the second one the thing under test.
        t0 = float((now - int(s_na.ts[-1]) - 600) // HOUR * HOUR)
        workdir = Path(tempfile.mkdtemp(prefix="qlport-prom-"))
        container = None
        try:
            backfill(workdir, openmetrics(s_na, METRIC, t0))
            container = start(workdir)
            wait_ready(base_url)
            build = prometheus_version(base_url)
            meta = {"image": IMAGE, "image_digest": digest,
                    "prometheus_version": build.get("version"),
                    "prometheus_revision": build.get("revision"),
                    "collected_unix": now}
            print(f"  prometheus {build.get('version')} on "
                  f"{digest.split('@')[-1][:19]}…")
            print(f"  first sample at t0+{s_na.ts[0]:g}s, inside the bucket "
                  f"[t0, t0+{NO_ANCHOR['interval']:g}s)")
            collect_no_anchor(base_url, t0, s_na, meta)
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

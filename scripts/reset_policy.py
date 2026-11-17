#!/usr/bin/env python3
"""E1: settle the reset policy rho end to end, through the agent (paper §3.1, §7).

Rig A of `docs/trial_account_checks.md`. A local endpoint exposes counters whose values
follow a script; a real Datadog Agent runs an OpenMetrics check against it and a real
Prometheus scrapes the same endpoint. One of the counters resets mid-run. Reading back
what each system stored over the bucket containing the reset separates the four candidate
policies, whose predictions were committed to in advance and are regression-tested in
`tests/test_reset.py`.

Why this cannot be done with the direct API, which every other check here uses: rho is
the *agent's* conversion of a cumulative counter into the per-interval deltas Datadog
stores. Submitting deltas over `/api/v2/series` chooses them, so it cannot measure how
they would have been chosen. The agent has to be in the loop, which means the run is in
real time and takes about twenty minutes.

Three series are exposed, not one:

  to100   resets from 1000 to 100 --- the discriminating case
  to0     resets from 1000 to 0   --- catches a special case at zero, which `zero` and
                                      `clamp` would otherwise be conflated by
  none     never resets           --- the control: its buckets must read exactly four
                                      increments of one, which is what shows the rig is
                                      reading the right buckets at all

Writes data/raw/datadog/e1.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                    # noqa: E402

from qlport.harness.datadog import credentials, query                 # noqa: E402
from qlport.harness.reset import (                                    # noqa: E402
    POLICIES, Schedule, bucket_table, consistent_policies, deltas_under, value_at,
)
from qlport.semantics import datadog_eval as dd                       # noqa: E402
from qlport.semantics.model import Series                             # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data/raw/datadog/e1.json"

EXPO_PORT = 19480                 # the exposition endpoint
PROM_PORT = 19094                 # 19090-19093 are held by the other scripts
PROM_IMAGE = "prom/prometheus:v3.15.0"
AGENT_IMAGE = "datadog/agent:7"
AGENT_NAME = "qlport-e1-agent"
PROM_NAME = "qlport-e1-prom"

STEP = 15.0                       # exposition grid, equal to both scrape intervals
# The agent flushes every 15 s whatever the check's collection interval, so a --step
# below that puts several samples in one flush. Remark rem:flush says phi stops modelling
# the agent there, and `--step 5` is how that is put to the system rather than read off
# its source.
_STEP = STEP
INTERVAL = 60.0                   # the rollup bucket under test
WARMUP = 660.0                    # >=10 min, per the spec: the agent omits its first sample
TAIL_STEPS = 16                   # four more buckets after the reset
INGEST_WAIT = 240.0

METRIC = "qlport_reset"           # exposed as qlport_reset_total
DD_METRIC = "qlport.e1.reset.count"

_LOG: list[dict] = []
_LOG_LOCK = threading.Lock()
_SCHEDULES: dict[str, Schedule] = {}


# ----------------------------------------------------------------- the exposition

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):                                     # noqa: N802
        if self.path.rstrip("/") not in ("/metrics", ""):
            self.send_error(404)
            return
        now = time.time()
        ua = self.headers.get("User-Agent", "")
        lines = [f"# TYPE {METRIC}_total counter"]
        served = {}
        for case, sched in _SCHEDULES.items():
            v = value_at(max(now, sched.t0), sched)
            served[case] = v
            lines.append(f'{METRIC}_total{{case="{case}"}} {v}')
        body = ("\n".join(lines) + "\n").encode()
        with _LOG_LOCK:
            _LOG.append({"t": now, "ua": ua, "values": served})
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                            # noqa: N802
        return                                            # the handler keeps its own log


def serve() -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("0.0.0.0", EXPO_PORT), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# ----------------------------------------------------------------- the containers

def run(*cmd: str, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _ids() -> str:
    return f"{os.getuid()}:{os.getgid()}"


def start_agent(workdir: Path, creds) -> str:
    conf = workdir / "conf.d" / "openmetrics.d"
    conf.mkdir(parents=True, exist_ok=True)
    # An explicit mapping rather than a wildcard, so the Datadog metric name is known in
    # advance instead of discovered: OpenMetrics v2 strips a counter's `_total`, so the
    # name matched here is the bare one.
    #
    # No `type:` on the mapping. A Prometheus counter is already sent as a monotonic
    # count -- which is the conversion under test -- and naming a type here asks for a
    # transformer instead; `monotonic_count` is not one of the transformer types the
    # check accepts, and the first attempt failed with "unknown type" rather than
    # silently measuring something else.
    (conf / "conf.yaml").write_text(
        "init_config:\n"
        "instances:\n"
        f"  - openmetrics_endpoint: http://localhost:{EXPO_PORT}/metrics\n"
        "    namespace: qlport.e1\n"
        "    metrics:\n"
        f"      - {METRIC}: reset\n"
        f"    min_collection_interval: {int(STEP)}\n"
        "    tags:\n"
        "      - rig:e1\n"
    )
    out = run("docker", "run", "-d", "--name", AGENT_NAME, "--network", "host",
              "-e", f"DD_API_KEY={creds.api_key}",
              "-e", f"DD_SITE={creds.site}",
              "-e", "DD_HOSTNAME=qlport-e1",
              # Nothing but the one check: no APM, no logs, no process or container
              # collection. The run should put one metric into the account.
              "-e", "DD_APM_ENABLED=false",
              "-e", "DD_LOGS_ENABLED=false",
              "-e", "DD_PROCESS_AGENT_ENABLED=false",
              "-e", "DD_PROCESS_CONFIG_PROCESS_COLLECTION_ENABLED=false",
              "-e", "DD_CONTAINER_EXCLUDE=name:.*",
              "-e", "DD_ENABLE_METADATA_COLLECTION=false",
              "-v", f"{workdir / 'conf.d'}:/conf.d:ro",
              AGENT_IMAGE)
    return out.stdout.strip()


def start_prometheus(workdir: Path) -> str:
    (workdir / "prometheus.yml").write_text(
        "global:\n"
        f"  scrape_interval: {int(STEP)}s\n"
        "scrape_configs:\n"
        "  - job_name: qlport_e1\n"
        "    static_configs:\n"
        f"      - targets: ['localhost:{EXPO_PORT}']\n"
    )
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    out = run("docker", "run", "-d", "--name", PROM_NAME, "--network", "host",
              "--user", _ids(), "-v", f"{workdir}:/w", PROM_IMAGE,
              "--config.file=/w/prometheus.yml", "--storage.tsdb.path=/w/data",
              f"--web.listen-address=:{PROM_PORT}",
              # `anchored` is still gated behind this flag on 3.15.0.
              "--enable-feature=promql-extended-range-selectors")
    return out.stdout.strip()


def stop(*names: str) -> None:
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, text=True)


def wait_ready(url: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{url} not ready after {timeout}s")


# ----------------------------------------------------------------- reading back

def dd_buckets(creds, case: str, start: float, end: float, agg: str) -> dict:
    q = (f"sum:{DD_METRIC}{{case:{case},rig:e1}}"
         f".rollup({agg}, {int(INTERVAL)})")
    raw = query(creds, q, start, end)
    out = {}
    for s in raw.get("series") or []:
        for t_ms, v in s.get("pointlist") or []:
            if v is not None:
                out[round(float(t_ms) / 1000.0, 3)] = float(v)
    return out


def prom_instant(expr: str, at: float) -> dict:
    body = urllib.parse.urlencode({"query": expr, "time": f"{at:.3f}"}).encode()
    req = urllib.request.Request(f"http://localhost:{PROM_PORT}/api/v1/query", data=body)
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    if d.get("status") != "success":
        raise ValueError(d.get("error", "query failed"))
    return {tuple(sorted(x["metric"].items())): float(x["value"][1])
            for x in d["data"]["result"]}


def agent_scrapes(case: str) -> tuple[np.ndarray, np.ndarray]:
    """The exposition as the *agent* actually read it.

    Prometheus and the agent both scrape, so the log holds both; the agent's requests are
    the ones that matter for rho and are told apart by user agent. Scoring against this
    rather than against the ideal schedule is what makes the reading robust to the
    agent's own scrape phase and drift.
    """
    with _LOG_LOCK:
        rows = [r for r in _LOG if "datadog" in r["ua"].lower()]
    ts = np.array([r["t"] for r in rows], dtype=float)
    vs = np.array([r["values"][case] for r in rows], dtype=float)
    return ts, vs


def main(argv=None) -> int:
    global STEP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmup", type=float, default=WARMUP)
    ap.add_argument("--step", type=float, default=_STEP,
                    help="exposition grid and check interval; below the agent's 15 s "
                         "flush this tests rem:flush's hypothesis")
    ap.add_argument("--tail-steps", type=int, default=TAIL_STEPS,
                    help="exposition steps to serve after the reset")
    ap.add_argument("--ingest-wait", type=float, default=INGEST_WAIT)
    ap.add_argument("--no-prometheus", action="store_true")
    args = ap.parse_args(argv)
    STEP = args.step

    creds = credentials()
    now = time.time()
    t0 = math.floor(now / INTERVAL) * INTERVAL          # grid aligned to the buckets
    # The step whose increment carries the reset. Chosen >= warmup away and congruent to
    # 1 mod 4, which puts the reset in the second quarter of its bucket -- where the spec
    # puts it. The bucket arithmetic does not actually depend on the position, since a
    # sum and a count over four scrapes do not care which one reset; the position is held
    # to the spec so the run is scored against the table it was committed to.
    k_min = math.ceil((now - t0 + args.warmup) / STEP)
    k_reset = k_min + ((1 - k_min) % 4)
    reset_at = t0 + k_reset * STEP
    end_at = t0 + (k_reset + args.tail_steps) * STEP

    _SCHEDULES.update({
        "to100": Schedule(t0=t0, step=STEP, reset_at=reset_at, reset_to=100),
        "to0": Schedule(t0=t0, step=STEP, reset_at=reset_at, reset_to=0),
        "none": Schedule(t0=t0, step=STEP, reset_at=None, reset_to=0),
    })

    print(f"exposition on :{EXPO_PORT}, grid {STEP:.0f}s anchored at {t0:.0f}")
    print(f"  reset at step {k_reset} = {reset_at:.0f} "
          f"({reset_at - now:.0f}s from now), serving until {end_at:.0f}")
    httpd = serve()
    workdir = Path(tempfile.mkdtemp(prefix="qlport-e1-"))
    stop(AGENT_NAME, PROM_NAME)
    try:
        if not args.no_prometheus:
            start_prometheus(workdir)
            wait_ready(f"http://localhost:{PROM_PORT}/-/ready")
            print("  prometheus up")
        start_agent(workdir, creds)
        print("  agent started; waiting for its first scrape")
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if any("datadog" in r["ua"].lower() for r in list(_LOG)):
                break
            time.sleep(2.0)
        else:
            print("  the agent never scraped the endpoint; its status follows",
                  file=sys.stderr)
            print(subprocess.run(["docker", "exec", AGENT_NAME, "agent", "status"],
                                 capture_output=True, text=True).stdout[-3000:],
                  file=sys.stderr)
            return 2
        print(f"  first agent scrape seen after "
              f"{time.time() - now:.0f}s; running to the reset and past it")

        last = 0.0
        while time.time() < end_at:
            time.sleep(5.0)
            if time.time() - last > 120:
                last = time.time()
                n = sum(1 for r in list(_LOG) if "datadog" in r["ua"].lower())
                print(f"    {end_at - time.time():.0f}s left, {n} agent scrapes")

        # Let the last bucket fill before reading either side. Without this the bucket
        # containing `end_at` is read while it is still short of its four scrapes, which
        # shows up as Prometheus disagreeing with Datadog about a bucket neither of them
        # had finished.
        time.sleep(2 * STEP)
        ts_a, vs_a = agent_scrapes("to100")
        print(f"  exposition closed: {len(ts_a)} agent scrapes")

        # Prometheus is read before the containers go away; Datadog needs the wait.
        # Only buckets the exposition covers end to end are evidence. The first is short
        # by the sample the agent omits on startup and the last by wherever the run
        # stopped, so comparing either would be comparing two partial readings.
        complete = [float(b) for b in dd.bucket_starts(ts_a[0], ts_a[-1], INTERVAL)
                    if b >= ts_a[0] and b + INTERVAL <= ts_a[-1]]
        prom = {}
        if not args.no_prometheus:
            for case in _SCHEDULES:
                row = {}
                for b in complete:
                    u = dd.eval_instant(b, INTERVAL)
                    got = prom_instant(
                        f'increase({METRIC}_total{{case="{case}"}}[{int(INTERVAL)}s] '
                        f'anchored)', u)
                    for v in got.values():
                        row[b] = v
                prom[case] = row
            print(f"  prometheus read over {len(complete)} complete buckets")

        print(f"  waiting {args.ingest_wait:.0f}s for the agent's points to land")
        time.sleep(args.ingest_wait)

        qstart, qend = ts_a[0] - INTERVAL, ts_a[-1] + 2 * INTERVAL
        result = {}
        for case in _SCHEDULES:
            sums = dd_buckets(creds, case, qstart, qend, "sum")
            counts = dd_buckets(creds, case, qstart, qend, "count")
            observed = {b: (sums[b], int(counts[b])) for b in sorted(sums)
                        if b in counts}
            ts_c, vs_c = agent_scrapes(case)
            verdict = consistent_policies(ts_c, vs_c, observed, INTERVAL)
            reset_bucket = (math.floor(reset_at / INTERVAL) * INTERVAL
                            if _SCHEDULES[case].reset_at else None)
            pm = prom.get(case) or {}
            agree = {b: (observed[b][0], pm[b]) for b in complete
                     if b in observed and b in pm}
            result[case] = {
                "observed_buckets": {str(k): v for k, v in observed.items()},
                "complete_buckets": complete,
                "prometheus_agrees_on": sorted(
                    b for b, (d_, p_) in agree.items() if abs(d_ - p_) < 1e-6),
                "prometheus_differs_on": {
                    str(b): {"datadog": d_, "prometheus": p_}
                    for b, (d_, p_) in agree.items() if abs(d_ - p_) >= 1e-6},
                "policies": verdict,
                "consistent_with": [p for p in POLICIES if verdict[p]["consistent"]],
                "reset_bucket": reset_bucket,
                "reset_bucket_observed": observed.get(reset_bucket),
                "reset_bucket_predicted": {
                    p: bucket_table(
                        *(lambda d: (d.ts, d.vs))(
                            deltas_under(Series({}, ts_c, vs_c, kind="counter"), p)),
                        INTERVAL, float(ts_c[0]), float(ts_c[-1]) + INTERVAL,
                    ).get(reset_bucket)
                    for p in POLICIES} if reset_bucket else None,
                "prometheus_anchored": {str(k): v
                                        for k, v in (prom.get(case) or {}).items()},
                "agent_scrapes": len(ts_c),
            }
            print(f"\n  {case}: consistent with {result[case]['consistent_with']}")
            print(f"    prometheus agrees on "
                  f"{len(result[case]['prometheus_agrees_on'])}/{len(agree)} "
                  f"complete buckets; differs on "
                  f"{list(result[case]['prometheus_differs_on'])}")
            if reset_bucket:
                print(f"    reset bucket {reset_bucket:.0f}: "
                      f"observed {result[case]['reset_bucket_observed']}, "
                      f"predicted {result[case]['reset_bucket_predicted']}")

        payload = {
            "provenance": {
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "site": creds.site, "agent_image": AGENT_IMAGE,
                "prometheus_image": PROM_IMAGE if not args.no_prometheus else None,
                "step": STEP, "interval": INTERVAL, "warmup": args.warmup,
                "samples_per_flush": round(15.0 / STEP, 3),
                "t0": t0, "reset_at": reset_at, "reset_step": k_reset,
            },
            "exposition": [
                {"t": r["t"], "agent": "datadog" in r["ua"].lower(),
                 "values": r["values"]}
                for r in list(_LOG)],
            "cases": result,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {OUT}")
        return 0
    finally:
        httpd.shutdown()
        stop(AGENT_NAME, PROM_NAME)


if __name__ == "__main__":
    raise SystemExit(main())

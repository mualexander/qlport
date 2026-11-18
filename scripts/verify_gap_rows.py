#!/usr/bin/env python3
"""Put the two argued rows of the gap catalogue to a real server.

`PHI_QUANTILE` has Corollary~2 and `SCOPE_EMPTY_VALUE` has a measured table. The other
two rows of the ``beyond reach'' column --- `FILL_INTERPOLATION` (36 occurrences) and
`LABEL_NAME_GLOB` (31) --- rested on prose: "no PromQL term introduces a sample where
the database holds none", and "label names are not queryable in PromQL at all". Between
them they are 67 of the 106 occurrences the paper calls inexpressible, which is more
than the two rows that carry warrants, so they are the rows most in need of one.

Two experiments, both against Prometheus 3.15.0 in Docker with the image digest
recorded, both writing raw responses to data/raw/gap_rows/.

  E10 (fill). A gauge with two series, one of which stops halfway. Datadog's
  `.fill(zero)` reports 0 for the stopped series in every later bucket. Each candidate
  PromQL expression is read back across the whole span, and what matters is whether the
  stopped series is present, and what value it carries, after it stops. The paper
  considered `or vector(0)` and rejected it correctly; it did not consider holding the
  series alive through `last_over_time` and multiplying by zero, which is the candidate
  this experiment exists to settle.

  E11 (label-name glob). A Datadog scope term with no key, `*-zone`, globs over the
  whole `key:value` string, so expressing it needs a pattern in a matcher's name
  position. The claim is that PromQL has no such syntax. 3.15.0 is the right server to
  ask, because quoted UTF-8 label names are recent and the obvious worry is that they
  opened the name position to more than a literal. Each attempt is sent and whatever
  comes back, answer or error, is recorded.

Needs Docker; the image is the same tag the other server experiments pin.

  python scripts/verify_gap_rows.py
  python scripts/verify_gap_rows.py --keep    # leave the container running
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IMAGE = "prom/prometheus:v3.15.0"
PORT = 19091
RAW = ROOT / "data" / "raw" / "gap_rows"
UA = "qlport-harness/0.1 (academic query-language study)"

METRIC = "qlport_fill"
T0 = 1_700_000_000           # an arbitrary but fixed epoch, so the run is deterministic
STEP = 60                    # one sample a minute
SPAN = 60                    # minutes of data
STOP = 30                    # svc="b" stops after this many minutes

# The candidates, in the order the argument considers them. The question each answers is
# the same: after svc="b" stops, is there a series carrying svc="b", and what is its
# value?
FILL_CANDIDATES = {
    "bare selector":
        f"{METRIC}",
    "or vector(0)":
        f"{METRIC} or vector(0)",
    "or on() vector(0)":
        f"{METRIC} or on() vector(0)",
    "absent() * 0":
        f"{METRIC} or (absent({METRIC}) * 0)",
    "last_over_time, held":
        f"last_over_time({METRIC}[20m])",
    "last_over_time * 0, unioned":
        f"{METRIC} or (last_over_time({METRIC}[20m]) * 0)",
}

# Each is an attempt to put a pattern where a matcher's label name goes. The last two
# are controls: a quoted UTF-8 name, which 3.x does accept, and a value-position regex,
# which has always worked. If the name position were open to patterns, one of the first
# three would be the way to say it.
LABEL_NAME_ATTEMPTS = {
    "bare pattern in name position": '{=~".*-zone"}',
    "pattern as a name, with a value": '{~".*-zone"="aws"}',
    "quoted pattern as a name": '{".*-zone"="aws"}',
    "control: quoted literal name": '{"svc"="a"}',
    "control: regex in value position": '{svc=~"a.*"}',
}


def run(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _uid() -> int:
    import os
    return os.getuid()


def _gid() -> int:
    import os
    return os.getgid()


def image_digest() -> str:
    out = run("docker", "image", "inspect", IMAGE, "--format", "{{index .RepoDigests 0}}")
    return out.stdout.strip()


def exposition() -> str:
    """Two gauge series, one of which stops at `STOP` minutes."""
    lines = [f"# TYPE {METRIC} gauge"]
    for minute in range(SPAN + 1):
        ts = T0 + minute * STEP
        lines.append(f'{METRIC}{{svc="a"}} {10 + minute} {ts}.000')
        if minute <= STOP:
            lines.append(f'{METRIC}{{svc="b"}} {100 + minute} {ts}.000')
    lines.append("# EOF")
    return "\n".join(lines) + "\n"


def backfill(workdir: Path) -> None:
    (workdir / "in").mkdir(parents=True, exist_ok=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "in" / "data.om").write_text(exposition())
    (workdir / "prometheus.yml").write_text("global:\n  scrape_interval: 60s\n")
    run("docker", "run", "--rm", "--user", f"{_uid()}:{_gid()}",
        "-v", f"{workdir}:/w", "--entrypoint", "/bin/promtool", IMAGE,
        "tsdb", "create-blocks-from", "openmetrics", "/w/in/data.om", "/w/data")


def start(workdir: Path) -> str:
    # No feature flags. These claims are about a server as it ships, which is the
    # point: a reading that needed an experimental flag would be a different claim.
    out = run("docker", "run", "-d", "--user", f"{_uid()}:{_gid()}",
              "-v", f"{workdir}:/w", "-p", f"{PORT}:9090", IMAGE,
              "--config.file=/w/prometheus.yml", "--storage.tsdb.path=/w/data",
              "--storage.tsdb.retention.time=3650d")
    return out.stdout.strip()


def wait_ready(base: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/-/ready", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{base} not ready after {timeout}s")


def ask(base: str, path: str, params: dict) -> dict:
    """One API call, returning the decoded body whether it is an answer or an error."""
    req = urllib.request.Request(
        f"{base}{path}", data=urllib.parse.urlencode(params).encode(),
        headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return {"status_code": r.status, "body": json.loads(r.read())}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"raw": raw}
        return {"status_code": exc.code, "body": body}


# Prometheus carries the last sample forward for 5 minutes (the staleness lookback), so
# a series does not vanish the instant it stops. The question "did the expression supply
# a value where the database holds none" is therefore asked strictly outside that
# window: at STOP + 10 minutes, where any value present is the expression's doing.
LOOKBACK_MIN = 5
PROBE = STOP + 2 * LOOKBACK_MIN


def series_present(body: dict) -> dict:
    """Condense a range response to what each label set does past the lookback."""
    out = {}
    for s in body.get("data", {}).get("result", []):
        labels = {k: v for k, v in s["metric"].items()}
        after = [(int(ts), v) for ts, v in s["values"] if int(ts) >= T0 + PROBE * STEP]
        out[json.dumps(labels, sort_keys=True)] = {
            "points_past_lookback": len(after),
            "first_value": after[0][1] if after else None,
            "last_value": after[-1][1] if after else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if not shutil.which("docker"):
        print("docker not found; see the module docstring", file=sys.stderr)
        return 2

    RAW.mkdir(parents=True, exist_ok=True)
    digest = image_digest()
    base = f"http://127.0.0.1:{PORT}"
    container = None
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        try:
            backfill(workdir)
            container = start(workdir)
            wait_ready(base)
            with urllib.request.urlopen(f"{base}/api/v1/status/buildinfo", timeout=30) as r:
                build = json.loads(r.read())["data"]

            fill = {}
            for name, expr in FILL_CANDIDATES.items():
                got = ask(base, "/api/v1/query_range", {
                    "query": expr,
                    "start": f"{T0}", "end": f"{T0 + SPAN * STEP}", "step": f"{STEP}"})
                fill[name] = {"expr": expr, "status_code": got["status_code"],
                              "summary": series_present(got["body"]),
                              "response": got["body"]}
                shown = series_present(got["body"])
                b = [v for k, v in shown.items() if '"svc": "b"' in k]
                live = b and b[0]["points_past_lookback"] > 0
                extra = sorted(k for k in shown if '"svc"' not in k)
                print(f"  {name:28} svc=b past lookback: {'yes' if live else 'no ':3}"
                      + (f" value {b[0]['first_value']}" if live else "")
                      + (f"   [also {extra[0]}]" if extra else ""))

            labels = {}
            for name, expr in LABEL_NAME_ATTEMPTS.items():
                got = ask(base, "/api/v1/query", {"query": expr, "time": f"{T0 + 600}"})
                err = got["body"].get("error") if isinstance(got["body"], dict) else None
                res = (got["body"].get("data", {}) or {}).get("result", []) \
                    if isinstance(got["body"], dict) else []
                labels[name] = {"expr": expr, "status_code": got["status_code"],
                                "error": err, "series_returned": len(res),
                                "response": got["body"]}
                print(f"  {name:32} HTTP {got['status_code']}"
                      + (f"  {err[:58]}" if err
                         else f"  accepted, {len(res)} series"))

            meta = {"image": IMAGE, "image_digest": digest, "build": build,
                    "metric": METRIC, "t0": T0, "step_s": STEP,
                    "span_min": SPAN, "stop_min": STOP,
                    "exposition": exposition()}
            (RAW / "e10_fill.json").write_text(
                json.dumps({"meta": meta, "candidates": fill}, indent=1) + "\n")
            (RAW / "e11_label_name.json").write_text(
                json.dumps({"meta": meta, "attempts": labels}, indent=1) + "\n")
            print(f"\nwrote {RAW.relative_to(ROOT)}/e10_fill.json and e11_label_name.json")
        finally:
            if container and not args.keep:
                subprocess.run(["docker", "rm", "-f", container],
                               capture_output=True, text=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

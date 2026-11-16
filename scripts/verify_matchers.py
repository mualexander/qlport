#!/usr/bin/env python3
"""Verify what a PromQL label matcher actually matches, against a real Prometheus.

`lower/datadog.py` translates a single-label Datadog disjunction into one regex
matcher: `{k IN (aws,gcp)}` and `{k:aws OR k:gcp}` both become `k=~"aws|gcp"`. That is
only sound if Prometheus anchors the *whole* regex -- `^(?:aws|gcp)$` -- rather than
each side of the alternation separately. Under `^aws|gcp$` the same matcher would also
select `awsx`, and 121 of the corpus's refusals would have been translated wrongly
rather than refused.

That is a claim about Prometheus, not about our code, so it is answered by a server.
Five series are backfilled -- four values of `k` chosen so that a prefix/suffix match
is distinguishable from a full match, plus one series carrying no `k` at all -- and
each matcher is read back. Raw responses land in data/raw/matchers/ and
tests/test_harness.py checks them, so the test suite stays offline.

A second phase then takes every distinct `=~` regex the translator emits over the
corpus and submits each one to the server, so that `re.escape` and `glob_to_regex`
output is known to be regexes RE2 accepts rather than regexes Python would. Our PromQL
grammar does not validate regex syntax, so nothing else would catch a rejection.

Needs Docker. The image tag is pinned below and the digest is recorded in the output.

  python scripts/verify_matchers.py
  python scripts/verify_matchers.py --keep     # leave the container up
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport.harness.prometheus import query_range, write_raw  # noqa: E402

IMAGE = "prom/prometheus:v3.15.0"
PORT = 19091                      # not 19090: verify_anchored.py may be running
METRIC = "qlport_matchers"
RAW = ROOT / "data" / "raw" / "matchers"

# `k` values chosen to separate full anchoring from partial: "awsx" has "aws" as a
# proper prefix, and "xgcp" has "gcp" as a proper suffix. One series has no `k` label,
# which is what distinguishes `=~".+"` (present) from `=~".*"` (present or absent).
SERIES = [
    {"k": "aws"},
    {"k": "azure"},
    {"k": "gcp"},
    {"k": "awsx"},
    {"k": "xgcp"},
    {"other": "none"},
]

# Each matcher, with what lower/datadog.py assumes it selects. The script does not
# check these -- it records what the server says and the test compares; keeping the
# expectation here is documentation of the assumption being tested.
MATCHERS = {
    'k=~"aws|gcp"':          "single-label disjunction: exactly the two values",
    'k!~"aws|gcp"':          "its negation, including the series with no k",
    'k=~"aws"':              "one value, not its extensions",
    'k=~".*ws.*|.*cp.*"':    "glob disjunction after glob_to_regex",
    'k=~".+"':               "tag present",
    'k=~".*"':               "tag present or absent",
}


def run(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _ids() -> str:
    import os
    return f"{os.getuid()}:{os.getgid()}"


def exposition(t0: float, n: int = 3, step: float = 15.0) -> str:
    """OpenMetrics for every series in SERIES, `n` samples each.

    harness.prometheus.openmetrics() writes one series; this needs several under one
    TYPE line, and plain integers are enough -- the values are irrelevant here, only
    which series come back.
    """
    lines = [f"# TYPE {METRIC} counter"]
    for i, labels in enumerate(SERIES):
        sel = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        for j in range(n):
            lines.append(f"{METRIC}_total{{{sel}}} {100 * i + j} {t0 + j * step:.3f}")
    lines.append("# EOF")
    return "\n".join(lines) + "\n"


def backfill(workdir: Path, text: str) -> None:
    (workdir / "in").mkdir(parents=True, exist_ok=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "in" / "data.om").write_text(text)
    (workdir / "prometheus.yml").write_text("global:\n  scrape_interval: 15s\n")
    run("docker", "run", "--rm", "--user", _ids(), "-v", f"{workdir}:/w",
        "--entrypoint", "/bin/promtool", IMAGE,
        "tsdb", "create-blocks-from", "openmetrics", "/w/in/data.om", "/w/data")


def start(workdir: Path) -> str:
    out = run("docker", "run", "-d", "--user", _ids(), "-v", f"{workdir}:/w",
              "-p", f"{PORT}:9090", IMAGE,
              "--config.file=/w/prometheus.yml", "--storage.tsdb.path=/w/data",
              "--storage.tsdb.retention.time=30d")
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


def corpus_regexes() -> list[str]:
    """Every distinct `=~` regex the translator emits over the Datadog corpus."""
    import re as _re

    from qlport import corpus as c
    from qlport.diagnostics import Bottom
    from qlport.emit.promql import emit_detail
    from qlport.lower.datadog import lower_all
    from qlport.parse import parse

    out: set[str] = set()
    seen: set[str] = set()
    for q in c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl"):
        if q.lang != "datadog":
            continue
        text = c.normalise("datadog", q.text)
        if text in seen or not c.parse_outcome("datadog", text).ok:
            continue
        seen.add(text)
        for e in lower_all(parse("datadog", text)):
            got = e if isinstance(e, Bottom) else emit_detail(e, default_rollup=60.0)
            if not isinstance(got, Bottom):
                out.update(_re.findall(r'=~\s*"((?:[^"\\]|\\.)*)"', got.promql))
    return sorted(out)


def check_accepted(base_url: str, t0: float, regexes: list[str]) -> dict:
    """Submit each regex as a matcher and record whether the server accepted it."""
    out = {}
    for rx in regexes:
        expr = f'{METRIC}_total{{k=~"{rx}"}}'
        try:
            raw = query_range(base_url, expr, t0, t0 + 15.0, 15.0)
            ok, err = raw.get("status") == "success", raw.get("error", "")
        except urllib.error.HTTPError as e:
            ok, err = False, json.loads(e.read()).get("error", str(e))
        out[rx] = {"accepted": bool(ok), "error": err}
        if not ok:
            print(f"  REJECTED {rx!r}: {err}")
    return out


def selected(raw: dict) -> list[dict]:
    """The label sets a range response returned, minus __name__, sorted."""
    if raw.get("status") != "success":
        raise ValueError(f"query failed: {raw.get('errorType')}: {raw.get('error')}")
    out = [{k: v for k, v in r["metric"].items() if k != "__name__"}
           for r in raw["data"]["result"]]
    return sorted(out, key=lambda d: sorted(d.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if not shutil.which("docker"):
        print("docker not found; see the module docstring", file=sys.stderr)
        return 2

    base_url = f"http://localhost:{PORT}"
    digest = run("docker", "image", "inspect", IMAGE,
                 "--format", "{{index .RepoDigests 0}}").stdout.strip()
    now = int(time.time())
    t0 = float(now - 3600)                      # recent past, inside retention

    workdir = Path(tempfile.mkdtemp(prefix="qlport-match-"))
    container = None
    try:
        backfill(workdir, exposition(t0))
        container = start(workdir)
        wait_ready(base_url)
        with urllib.request.urlopen(f"{base_url}/api/v1/status/buildinfo", timeout=30) as r:
            build = json.loads(r.read())["data"]
        print(f"prometheus {build.get('version')} on {digest.split('@')[-1][:19]}…")

        payload = {
            "metric": METRIC, "t0": t0, "series": SERIES,
            "image": IMAGE, "image_digest": digest,
            "prometheus_version": build.get("version"),
            "prometheus_revision": build.get("revision"),
            "collected_unix": now, "matchers": {},
        }
        for matcher, note in MATCHERS.items():
            expr = f"{METRIC}_total{{{matcher}}}"
            raw = query_range(base_url, expr, t0, t0 + 30.0, 15.0)
            got = selected(raw)
            payload["matchers"][matcher] = {"expr": expr, "note": note,
                                            "selected": got, "response": raw}
            print(f"  {matcher:22s} -> {[d.get('k', '(no k)') for d in got]}")

        regexes = corpus_regexes()
        print(f"submitting {len(regexes)} distinct regexes the translator emits")
        payload["corpus_regexes"] = check_accepted(base_url, t0, regexes)
        bad = [r for r, v in payload["corpus_regexes"].items() if not v["accepted"]]
        print(f"  {len(regexes) - len(bad)}/{len(regexes)} accepted by RE2")

        path = write_raw(RAW / "label_matchers.json", payload)
        print(f"wrote {path.relative_to(ROOT)}")
    finally:
        if container and not args.keep:
            subprocess.run(["docker", "rm", "-f", container],
                           capture_output=True, text=True)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        elif container:
            print(f"container {container[:12]} left running on {base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

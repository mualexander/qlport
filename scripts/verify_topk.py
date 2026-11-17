#!/usr/bin/env python3
"""Can PromQL denote Datadog's `top()`? Asked of a real Prometheus.

Datadog's `top(q, n, 'BY', 'DIR')` ranks series by an aggregation over **all values in
the queried range** --- `max`, `mean`, `min`, `sum`, `last`, `l2norm`, `area` --- and
returns those series' whole trajectories~[rank docs]. PromQL's `topk(n, v)` ranks at each
evaluation instant independently, so the two are different operators and a corpus refusal
of `top()` could plausibly be a language gap rather than unfinished work. §8's headline
rests on which it is: `top()` is 139 occurrences, and moving it from "not implemented" to
"inexpressible" would invert the comparison the paper draws.

The candidate is

    q and on(<labels>) topk(n, <agg>_over_time(q[RANGE] @ END))

where `@` pins the ranking to one evaluation so that it does not drift across the graph,
which is what makes the selection constant the way Datadog's is. `bottomk` stands in for
`topk` where the Datadog query ranks ascending, which 5 of the 139 occurrences do.

The ranking functions probed are the five `harness/ranking.py` models. The corpus uses
four of them -- `mean` 111 times, `last` 34, `max` 15 and `sum` 3 -- and `min` is here
because the candidate has to hold for it, not because a dashboard asks for it.

The workload is five series chosen so the ranking over the window disagrees with the
ranking at individual instants --- a flat high series, a riser that ends highest, a flat
runner-up, a flat low series, and one with a single early spike. Any candidate that
merely tracks instantaneous values gets a different answer at the start of the range than
at the end, and a different answer from Datadog's model.

Needs Docker. Writes data/raw/topk/ranking.json; tests/test_harness.py reads it.

  python scripts/verify_topk.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IMAGE = "prom/prometheus:v3.15.0"
PORT = 19092                       # 19090 and 19091 belong to the other two scripts
OUT = ROOT / "data" / "raw" / "topk" / "ranking.json"

T0 = 1_700_000_000
METRIC = "q"

# The workload and the ranking model live in `harness/ranking.py`, because
# `scripts/datadog_top.py` asks the vendor the same question about the same series and a
# comparison between the two halves is only valid if the series are identical.
from qlport.harness.ranking import (                                  # noqa: E402
    N, PROMQL_OVER_TIME as AGGS, SERIES, STEP, model_rank,
)


def run(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _ids() -> str:
    import os
    return f"{os.getuid()}:{os.getgid()}"


def exposition() -> str:
    lines = [f"# TYPE {METRIC} gauge"]
    for name, f in SERIES.items():
        for i in range(N):
            lines.append(f'{METRIC}{{svc="{name}"}} {f(i):.17g} {T0 + i * STEP:.3f}')
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


def query(base_url: str, expr: str, at: float) -> dict:
    body = urllib.parse.urlencode({"query": expr, "time": f"{at:.3f}"}).encode()
    req = urllib.request.Request(f"{base_url}/api/v1/query", data=body)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def selected(raw: dict) -> list[str]:
    """Which series a result names, by the one label that distinguishes them."""
    if raw.get("status") != "success":
        raise ValueError(raw.get("error", "query failed"))
    return sorted(x["metric"].get("svc", "?") for x in raw["data"]["result"])


def main() -> int:
    if not shutil.which("docker"):
        print("docker not found; see the module docstring", file=sys.stderr)
        return 2

    digest = run("docker", "image", "inspect", IMAGE,
                 "--format", "{{index .RepoDigests 0}}").stdout.strip()
    end = T0 + (N - 1) * STEP
    # The window must cover [T0, end]; a range selector is left-open, so a width of
    # exactly (N-1)*STEP drops the first sample -- and the first sample is where the
    # spike that distinguishes `max` from `mean` lives.
    width = N * STEP
    probe_at = {"near start": T0 + 2 * STEP, "midway": T0 + (N // 2) * STEP,
                "at end": end}

    import tempfile
    workdir = Path(tempfile.mkdtemp(prefix="qlport-topk-"))
    container = ""
    payload: dict = {
        "provenance": {
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "image": IMAGE, "image_digest": digest,
            "t0": T0, "step": STEP, "n_points": N, "range_seconds": width,
            "anchor": end,
            "series": {k: [f(i) for i in range(N)] for k, f in SERIES.items()},
        },
        "cases": [], "naive": [],
    }
    try:
        backfill(workdir, exposition())
        container = start(workdir)
        base = f"http://localhost:{PORT}"
        wait_ready(base)
        payload["provenance"]["prometheus_version"] = json.loads(
            urllib.request.urlopen(f"{base}/api/v1/status/buildinfo",
                                   timeout=10).read())["data"]["version"]

        for by, fn in AGGS.items():
            for n in (2, 3):
                # The corpus uses both directions -- 5 of its 139 occurrences rank
                # ascending -- and `topk` covers only one of them. `bottomk` is the
                # counterpart, and asking for it is the difference between a claim about
                # 134 occurrences and a claim about 139.
                for direction, selector in (("desc", "topk"), ("asc", "bottomk")):
                    expr = (f'{METRIC} and on(svc) '
                            f'{selector}({n}, {fn}({METRIC}[{width}s] @ {end}))')
                    want = model_rank(by, n, direction)
                    got = {label: selected(query(base, expr, at))
                           for label, at in probe_at.items()}
                    payload["cases"].append({
                        "by": by, "n": n, "direction": direction, "selector": selector,
                        "expr": expr, "model": want,
                        "selected": got,
                        "constant": len({tuple(v) for v in got.values()}) == 1,
                        "matches_model": all(v == want for v in got.values()),
                    })

        # The translation anybody writes first, recorded so the difference is on file.
        for n in (2, 3):
            expr = f"topk({n}, {METRIC})"
            got = {label: selected(query(base, expr, at)) for label, at in probe_at.items()}
            payload["naive"].append({
                "n": n, "expr": expr, "selected": got,
                "constant": len({tuple(v) for v in got.values()}) == 1,
            })
    finally:
        if container:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    ok = sum(c["matches_model"] for c in payload["cases"])
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {ok}/{len(payload['cases'])} ranking cases reproduce Datadog's top()")
    for c in payload["cases"]:
        mark = "ok " if c["matches_model"] else "NO "
        print(f"  {mark} by={c['by']:<5} n={c['n']} {c['direction']:<4} "
              f"model={c['model']}  "
              f"got={sorted({tuple(v) for v in c['selected'].values()})}")
    for c in payload["naive"]:
        print(f"  naive topk({c['n']}, q): constant across range = {c['constant']}  "
              f"{c['selected']}")
    return 0 if ok == len(payload["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

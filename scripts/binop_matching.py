#!/usr/bin/env python3
"""E8: how the two languages pair series across a binary operator (paper §5).

Theorem thm:kernel covers a space aggregation of a rollup of a selector. Outside it the
corpus does one thing: arithmetic between two such queries. Proving that case needs
Datadog's matching rule as a premise, and the rule is not documented.

The question that decides it is what happens when the two sides do not produce the same
groups. PromQL's default matching is a one-to-one join on the whole label set, so a group
present on one side and absent on the other contributes nothing -- an inner join. If
Datadog instead reports those groups, the emitted expression returns a strictly smaller
result and the translation is wrong on every dashboard whose two series sets differ,
which is the ordinary case rather than a corner.

The corpus-scale test of §7 cannot see this: it instantiates both operands from the same
filters, so both sides always carry the same groups. Hence a rig built to make them
differ.

  a{g:x} = 10   a{g:y} = 20          groups x and y
  b{g:x} = 2    b{g:z} = 5           groups x and z

The same four series go to Prometheus, and both sides answer the same five questions.

Writes data/raw/datadog/e8.json.
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
    GAUGE, count_payload, credentials, query, submit,
)

OUT = Path(__file__).resolve().parents[1] / "data/raw/datadog/e8.json"

PROM_IMAGE = "prom/prometheus:v3.15.0"
PROM_NAME = "qlport-e8-prom"
PROM_PORT = 19098

STEP = 10.0
INTERVAL = 60.0
SPAN = 1200.0
INGEST_WAIT = 150.0

# Each series carries two tags, so the operands can be grouped on *different* keys --
# the one case the emitter refuses, and the one no corpus query reaches, so whether the
# refusal is a gap or merely unimplemented has to be asked rather than assumed.
SERIES = {("a", "x", "p"): 10.0, ("a", "y", "q"): 20.0,
          ("b", "x", "p"): 2.0, ("b", "z", "r"): 5.0}


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


def prom(expr: str, at: float) -> dict:
    body = urllib.parse.urlencode({"query": expr, "time": f"{at:.3f}"}).encode()
    req = urllib.request.Request(f"http://localhost:{PROM_PORT}/api/v1/query", data=body)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return {"__error__": exc.read()[:200].decode(errors="replace")}
    return {",".join(f"{k}={v}" for k, v in sorted(x["metric"].items())) or "<none>":
            round(float(x["value"][1]), 6) for x in d["data"]["result"]}


def dd(creds, q: str, start: float, end: float) -> dict:
    try:
        raw = query(creds, q, start, end)
    except Exception as exc:                                          # noqa: BLE001
        return {"__error__": str(exc)[:200]}
    out = {}
    for s in raw.get("series") or []:
        scope = str(s.get("scope", ""))
        got = []
        for term in scope.split(","):
            k, sep, v = term.strip().partition(":")
            if sep and k in ("g", "h"):
                got.append(f"{k}={v}")
        g = ",".join(sorted(got)) or "<none>"
        vals = sorted({round(float(v), 6)
                       for _, v in (s.get("pointlist") or []) if v is not None})
        out[g] = vals
    return out


def main() -> int:
    creds = credentials()
    t0 = math.floor((time.time() - SPAN - 240.0) / INTERVAL) * INTERVAL
    tag = f"r{int(t0)}"
    name = {m: f"qlport.e8.{tag}.{m}" for m in ("a", "b")}

    n = int(SPAN // STEP)
    ts = [t0 + k * STEP for k in range(n)]
    print(f"{tag}: four constant series, groups a:(x,y) b:(x,z)")
    for (metric, g, h), v in SERIES.items():
        pts = [(t, v) for t in ts]
        for i in range(0, len(pts), 200):
            submit(creds, count_payload(name[metric], pts[i:i + 200],
                                        tags=[f"g:{g}", f"h:{h}", "rig:e8"],
                                        metric_type=GAUGE))

    lines = []
    for metric in ("a", "b"):
        lines.append(f"# TYPE qlport_e8_{metric} gauge")
        for (m, g, h), v in SERIES.items():
            if m != metric:
                continue
            for t in ts:
                lines.append(
                    f'qlport_e8_{metric}{{g="{g}",h="{h}"}} {v:.17g} {t:.3f}')
    lines.append("# EOF")

    a, b = name["a"], name["b"]
    pa, pb = "qlport_e8_a", "qlport_e8_b"
    cases = {
        "both_grouped_div": (
            f"sum:{a}{{*}} by {{g}} / sum:{b}{{*}} by {{g}}",
            f"sum by (g) (avg_over_time({pa}[{int(INTERVAL)}s]))"
            f" / sum by (g) (avg_over_time({pb}[{int(INTERVAL)}s]))"),
        "both_grouped_add": (
            f"sum:{a}{{*}} by {{g}} + sum:{b}{{*}} by {{g}}",
            f"sum by (g) (avg_over_time({pa}[{int(INTERVAL)}s]))"
            f" + sum by (g) (avg_over_time({pb}[{int(INTERVAL)}s]))"),
        "rhs_ungrouped": (
            f"sum:{a}{{*}} by {{g}} / sum:{b}{{*}}",
            f"sum by (g) (avg_over_time({pa}[{int(INTERVAL)}s]))"
            f" / on() group_left() sum(avg_over_time({pb}[{int(INTERVAL)}s]))"),
        "lhs_ungrouped": (
            f"sum:{a}{{*}} / sum:{b}{{*}} by {{g}}",
            f"sum(avg_over_time({pa}[{int(INTERVAL)}s]))"
            f" / on() group_right() sum by (g) (avg_over_time({pb}[{int(INTERVAL)}s]))"),
        # what PromQL's default matching would do for the broadcast case, for contrast
        # Datadog's own error message for the disjoint case names the rule: group keys
        # must match "or where one set of group keys is a subset of the other". The
        # ungrouped cases above are the subset case with the empty set; these are the
        # general one, which the emitter refuses and PromQL can in fact express.
        "subset_rhs_finer": (
            f"sum:{a}{{*}} by {{g}} / sum:{b}{{*}} by {{g,h}}",
            f"sum by (g) (avg_over_time({pa}[{int(INTERVAL)}s]))"
            f" / on(g) group_right() sum by (g, h) (avg_over_time({pb}[{int(INTERVAL)}s]))"),
        "subset_lhs_finer": (
            f"sum:{a}{{*}} by {{g,h}} / sum:{b}{{*}} by {{g}}",
            f"sum by (g, h) (avg_over_time({pa}[{int(INTERVAL)}s]))"
            f" / on(g) group_left() sum by (g) (avg_over_time({pb}[{int(INTERVAL)}s]))"),
        # grouped on different keys: refused by the emitter, so the PromQL column is
        # what default matching happens to do rather than what we would emit
        "different_keys": (
            f"sum:{a}{{*}} by {{g}} / sum:{b}{{*}} by {{h}}",
            f"sum by (g) (avg_over_time({pa}[{int(INTERVAL)}s]))"
            f" / sum by (h) (avg_over_time({pb}[{int(INTERVAL)}s]))"),
        "rhs_ungrouped_default_matching": (
            f"sum:{a}{{*}} by {{g}} / sum:{b}{{*}}",
            f"sum by (g) (avg_over_time({pa}[{int(INTERVAL)}s]))"
            f" / sum(avg_over_time({pb}[{int(INTERVAL)}s]))"),
    }

    workdir = Path(tempfile.mkdtemp(prefix="qlport-e8-"))
    subprocess.run(["docker", "rm", "-f", PROM_NAME], capture_output=True)
    try:
        start_prometheus(workdir, lines)
        print("  prometheus up")
        print(f"  waiting {INGEST_WAIT:.0f}s for Datadog ingestion")
        time.sleep(INGEST_WAIT)

        at = ts[len(ts) // 2]
        qs, qe = ts[0] - INTERVAL, ts[-1] + 2 * INTERVAL
        results = {}
        for key, (dq, pq) in cases.items():
            results[key] = {"datadog_query": dq, "promql_query": pq,
                            "datadog": dd(creds, dq, qs, qe), "promql": prom(pq, at)}
            print(f"\n  {key}")
            print(f"    datadog: {results[key]['datadog']}")
            print(f"    promql : {results[key]['promql']}")

        payload = {
            "provenance": {
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "site": creds.site, "prometheus_image": PROM_IMAGE,
                "run": tag, "interval": INTERVAL, "step": STEP,
                "series": {f"{m}:g={g},h={h}": v
                           for (m, g, h), v in SERIES.items()},
            },
            "cases": results,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {OUT}")
        return 0
    finally:
        subprocess.run(["docker", "rm", "-f", PROM_NAME], capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download the query corpus into data/corpus/ and extract every query string.

Sources
  kube-prometheus   DataDog -> PromQL target side, curated: the *compiled* output of
                    kubernetes-mixin and the node-exporter / prometheus / alertmanager
                    mixins. kubernetes-mixin itself ships jsonnet, which needs jsonnet
                    and jsonnet-bundler to render; kube-prometheus checks the rendered
                    dashboards and PrometheusRule objects into manifests/, so it gives
                    the same queries without a jsonnet toolchain.
  grafana           PromQL in the wild: the most-downloaded community dashboards whose
                    datasource is Prometheus (grafana.com/api/dashboards).
  integrations-core DataDog/integrations-core */assets/dashboards/*.json: the source
                    side, one vendor-authored dashboard per integration.

Layout
  data/corpus/sources/<name>/   downloaded trees and dashboards (gitignored, large)
  data/corpus/SOURCES.md        what was downloaded, at which commit, on which date
  data/corpus/queries.jsonl     the extracted queries (committed; notebook 00 reads this)

Usage
  .venv/bin/python scripts/fetch_corpus.py                 # download, then extract
  .venv/bin/python scripts/fetch_corpus.py --extract-only  # re-extract what is cached
  .venv/bin/python scripts/fetch_corpus.py --grafana-top 400
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from qlport import corpus  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
SOURCES = CORPUS / "sources"
UA = "qlport-corpus/0.1 (academic query-language study)"

REPOS = {
    "kube-prometheus": dict(repo="prometheus-operator/kube-prometheus", ref="main",
                            keep=("manifests/",), suffix=".yaml"),
    "integrations-core": dict(repo="DataDog/integrations-core", ref="master",
                              keep=("assets/dashboards/",), suffix=".json"),
}
GRAFANA_LIST = ("https://grafana.com/api/dashboards"
                "?orderBy=downloads&direction=desc&dataSourceSlugIn=prometheus"
                "&page={page}&pageSize={size}")
GRAFANA_DOWNLOAD = "https://grafana.com/api/dashboards/{id}/revisions/{rev}/download"


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------
def _open(url: str, *, tries: int = 3, timeout: int = 120):
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout)
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def _json(url: str):
    with _open(url) as r:
        return json.load(r)


# ---------------------------------------------------------------------------
# github repositories
# ---------------------------------------------------------------------------
def fetch_repo(name: str, spec: dict) -> dict:
    """Stream the repository tarball and keep only the files we analyse.

    Streaming matters: integrations-core is ~220 MB packed and the dashboards are
    ~37 MB of it. Extraction uses tarfile's "data" filter, so no member can escape
    the destination directory.
    """
    sha = _json(f"https://api.github.com/repos/{spec['repo']}/commits/{spec['ref']}")["sha"]
    dest = SOURCES / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    url = f"https://codeload.github.com/{spec['repo']}/tar.gz/{sha}"
    kept = 0
    with _open(url, timeout=600) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(spec["suffix"]):
                continue
            inner = member.name.partition("/")[2]          # strip <repo>-<sha>/
            if not any(k in inner for k in spec["keep"]):
                continue
            member.name = inner
            tar.extract(member, dest, filter="data")
            kept += 1
    meta = dict(name=name, repo=spec["repo"], ref=spec["ref"], commit=sha,
                files=kept, fetched=dt.date.today().isoformat())
    (dest / ".fetch.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {name}: {kept} files at {sha[:12]}")
    return meta


# ---------------------------------------------------------------------------
# grafana community dashboards
# ---------------------------------------------------------------------------
def fetch_grafana(top: int, *, pause: float = 0.3) -> dict:
    dest = SOURCES / "grafana"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    listing, page, size = [], 1, 100
    total = None
    while len(listing) < top:
        got = _json(GRAFANA_LIST.format(page=page, size=min(size, top - len(listing))))
        total = got.get("total")
        items = got.get("items") or []
        if not items:
            break
        listing.extend(items)
        page += 1
        time.sleep(pause)
    listing = listing[:top]

    index, ok = [], 0
    for it in listing:
        row = {k: it.get(k) for k in ("id", "slug", "name", "revision", "downloads", "orgSlug")}
        try:
            body = _json(GRAFANA_DOWNLOAD.format(id=it["id"], rev=it["revision"]))
        except RuntimeError as e:
            row["error"] = str(e)
            index.append(row)
            continue
        (dest / f"{it['id']}.json").write_text(json.dumps(body), encoding="utf-8")
        row["file"] = f"{it['id']}.json"
        index.append(row)
        ok += 1
        time.sleep(pause)

    meta = dict(name="grafana", api="grafana.com/api/dashboards",
                filter="dataSourceSlugIn=prometheus, orderBy=downloads desc",
                requested=top, downloaded=ok, available=total,
                fetched=dt.date.today().isoformat())
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (dest / ".fetch.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  grafana: {ok}/{len(listing)} dashboards of {total} available")
    return meta


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def _k8s_objects(text: str):
    """Yield every object in a multi-document manifest, flattening *List kinds."""
    for doc in yaml.safe_load_all(text):
        if not isinstance(doc, dict):
            continue
        if str(doc.get("kind", "")).endswith("List"):
            yield from (o for o in (doc.get("items") or []) if isinstance(o, dict))
        else:
            yield doc


def extract_kube_prometheus() -> tuple[list[corpus.Query], dict]:
    root = SOURCES / "kube-prometheus"
    out: list[corpus.Query] = []
    dashboards = rules = 0
    for path in sorted(root.rglob("manifests/*.yaml")):
        rel = str(path.relative_to(root))
        for obj in _k8s_objects(path.read_text(encoding="utf-8")):
            if obj.get("kind") == "PrometheusRule":
                got = corpus.prometheus_rule_queries(obj, source="kube-prometheus", origin=rel)
                rules += len(got)
                out.extend(got)
            elif obj.get("kind") == "ConfigMap":
                for key, body in (obj.get("data") or {}).items():
                    if not key.endswith(".json"):
                        continue
                    try:
                        doc = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    if "panels" not in doc and "rows" not in doc:
                        continue
                    dashboards += 1
                    out.extend(corpus.grafana_dashboard_queries(
                        doc, source="kube-prometheus", origin=f"{rel}#{key}"))
    return out, dict(dashboards=dashboards, rule_queries=rules)


def extract_grafana() -> tuple[list[corpus.Query], dict]:
    root = SOURCES / "grafana"
    index = json.loads((root / "index.json").read_text())
    out: list[corpus.Query] = []
    used = 0
    for row in index:
        if not row.get("file"):
            continue
        try:
            doc = json.loads((root / row["file"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        got = corpus.grafana_dashboard_queries(
            doc, source="grafana", origin=f"{row['id']} {row.get('slug', '')}".strip())
        if got:
            used += 1
        out.extend(got)
    return out, dict(dashboards=len(index), dashboards_with_promql=used)


def extract_integrations_core() -> tuple[list[corpus.Query], dict]:
    root = SOURCES / "integrations-core"
    out: list[corpus.Query] = []
    files = unreadable = 0
    for path in sorted(root.rglob("assets/dashboards/*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            unreadable += 1          # ddev scaffolding templates are not valid JSON
            continue
        files += 1
        out.extend(corpus.datadog_dashboard_queries(
            doc, source="integrations-core", origin=str(path.relative_to(root))))
    return out, dict(dashboards=files, unreadable=unreadable)


EXTRACTORS = {
    "kube-prometheus": extract_kube_prometheus,
    "grafana": extract_grafana,
    "integrations-core": extract_integrations_core,
}


def write_sources_md(stats: dict[str, dict]) -> None:
    lines = [
        "# Corpus sources",
        "",
        "Written by `scripts/fetch_corpus.py`. Each entry pins the exact commit or",
        "API snapshot the queries in `queries.jsonl` were extracted from.",
        "",
    ]
    for name, st in stats.items():
        meta_path = SOURCES / name / ".fetch.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        lines.append(f"## {name}")
        lines.append("")
        for k, v in {**meta, **st}.items():
            if k != "name":
                lines.append(f"- {k}: `{v}`")
        lines.append("")
    (CORPUS / "SOURCES.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract-only", action="store_true",
                    help="skip the downloads and re-extract data/corpus/sources/")
    ap.add_argument("--grafana-top", type=int, default=200,
                    help="how many community dashboards to take, by downloads")
    ap.add_argument("--only", nargs="*", choices=sorted(EXTRACTORS),
                    help="restrict to these sources")
    args = ap.parse_args()
    names = args.only or list(EXTRACTORS)

    if not args.extract_only:
        print("downloading")
        for name in names:
            if name in REPOS:
                fetch_repo(name, REPOS[name])
            elif name == "grafana":
                fetch_grafana(args.grafana_top)

    print("extracting")
    queries: list[corpus.Query] = []
    stats: dict[str, dict] = {}
    for name in names:
        if not (SOURCES / name).exists():
            print(f"  {name}: not downloaded, skipped")
            continue
        got, st = EXTRACTORS[name]()
        by_lang: dict[str, int] = {}
        for q in got:
            by_lang[q.lang] = by_lang.get(q.lang, 0) + 1
        stats[name] = {**st, **{f"queries_{k}": v for k, v in sorted(by_lang.items())}}
        print(f"  {name}: {len(got)} queries {by_lang}")
        queries.extend(got)

    n = corpus.write_queries(CORPUS / "queries.jsonl", queries)
    write_sources_md(stats)
    print(f"wrote {n} queries to {CORPUS / 'queries.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

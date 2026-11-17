#!/usr/bin/env python3
"""Fetch the declared metric type of every metric `integrations-core` documents.

Paper §5's Remark rem:gauge says a dashboard never records whether a metric was
submitted as COUNT or GAUGE, so the translator infers the kind from the query text and
the emitter reports METRIC_TYPE_UNKNOWN on every translation. That is true of the
dashboards -- and the *same repository* the dashboards come from ships, beside each
integration's `assets/dashboards/`, a `metadata.csv` with a `metric_type` column.

So the assumption the paper carries on 100% of its output is checkable after all, for
the metrics this corpus names. This script fetches those CSVs; `type_declared.py`
does the comparison.

Reproducibility: the commit is read from the corpus's own `.fetch.json` rather than
resolved from `master`, so the declarations can never come from a different tree than
the queries. Re-running `fetch_corpus.py` would move the corpus; this script cannot.

  python scripts/fetch_metric_types.py

Writes the CSVs and the extracted mapping under
data/corpus/sources/integrations-core-metadata/ (gitignored, like every other
download). `type_declared.py` reads it and commits the part the paper uses: all 37,070
declarations is 2 MB, most of it metrics this corpus never names.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "corpus" / "sources"
CORPUS_TREE = SOURCES / "integrations-core"
DEST = SOURCES / "integrations-core-metadata"
CACHE = DEST / "declared.json"
REPO = "DataDog/integrations-core"
UA = "qlport-corpus/0.1 (academic query-language study)"


def pinned_commit() -> str:
    """The commit the corpus was extracted from, not whatever master is today."""
    fetched = CORPUS_TREE / ".fetch.json"
    if not fetched.exists():
        sys.exit(f"{fetched} missing; run scripts/fetch_corpus.py first")
    meta = json.loads(fetched.read_text())
    if meta.get("repo") != REPO:
        sys.exit(f"{fetched} is for {meta.get('repo')}, not {REPO}")
    return meta["commit"]


def fetch(sha: str) -> int:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    url = f"https://codeload.github.com/{REPO}/tar.gz/{sha}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    kept = 0
    with urllib.request.urlopen(req, timeout=600) as resp, \
            tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith("/metadata.csv"):
                continue
            inner = member.name.partition("/")[2]          # strip <repo>-<sha>/
            member.name = inner
            tar.extract(member, DEST, filter="data")       # no member can escape DEST
            kept += 1
    return kept


def read_declarations() -> dict[str, str]:
    """metric name -> declared `metric_type`, lowercased.

    A name declared twice with different types is recorded as "conflict": the point of
    this experiment is what the repository states, and a contradiction is a statement.
    """
    out: dict[str, str] = {}
    for path in sorted(DEST.rglob("metadata.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("metric_name") or "").strip()
                kind = (row.get("metric_type") or "").strip().lower()
                if not name or not kind:
                    continue
                if name in out and out[name] not in (kind, "conflict"):
                    out[name] = "conflict"
                else:
                    out.setdefault(name, kind)
    return out


def main() -> int:
    sha = pinned_commit()
    print(f"integrations-core at {sha[:12]} (pinned by the corpus)")
    files = fetch(sha)
    declared = read_declarations()
    counts: dict[str, int] = {}
    for kind in declared.values():
        counts[kind] = counts.get(kind, 0) + 1
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({
        "declared": declared,
        "type_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "provenance": {
            "repo": REPO, "commit": sha, "metadata_files": files,
            "metrics_declared": len(declared),
            "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }, indent=1, sort_keys=True) + "\n")
    print(f"  {files} metadata.csv files, {len(declared):,} metric names")
    for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {kind:14} {n:>6,}")
    print(f"wrote {CACHE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

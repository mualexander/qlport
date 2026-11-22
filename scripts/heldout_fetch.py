#!/usr/bin/env python3
"""Fetch a held-out query set: dashboards nobody involved in this work has read.

Section 7's threats subsection concedes that the grammar and the implementation order
were fitted to `data/corpus/queries.jsonl`, and names the remedy: "a disjoint set of
dashboards, or a later commit of the same repository, with the grammar and the
translator frozen". This script fetches both of those, into `data/heldout/` so that
nothing it writes can touch the training corpus.

  grafana           The corpus took the 200 most-downloaded Prometheus dashboards of
                    5,878 available. The rest were never listed, downloaded or read.
                    We take the next ranks by download count and drop, by id, any
                    dashboard the corpus already holds.
  integrations-core The same repository at a later commit. Only query strings absent
                    from the committed corpus are held out; the dashboards that did
                    not change between the two commits *are* the training data.
  integrations-extras DataDog/integrations-extras: the community- and partner-maintained
                    sibling of integrations-core, same `*/assets/dashboards/*.json`
                    layout, different authors, never read here. This is the disjoint
                    Datadog set, and the only one of the three with enough queries in
                    it to estimate a rate.

Frozen means frozen: nothing downstream of this script may change `src/qlport/`,
and the point of the exercise is lost if it does.

Usage
  .venv/bin/python scripts/heldout_fetch.py --grafana-to 700
  .venv/bin/python scripts/heldout_fetch.py --integrations-core
  .venv/bin/python scripts/heldout_fetch.py --integrations-extras
  .venv/bin/python scripts/heldout_fetch.py --all
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport import corpus  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
HELDOUT = ROOT / "data" / "heldout"
SOURCES = HELDOUT / "sources"

# Not in fetch_corpus.REPOS, deliberately: the training fetcher must not be able to
# reach this repository, or the held-out set stops being held out.
EXTRAS = dict(repo="DataDog/integrations-extras", ref="master",
              keep=("assets/dashboards/",), suffix=".json")


def _fetch_corpus_module():
    """Load scripts/fetch_corpus.py and point its SOURCES at data/heldout/sources.

    Reusing it rather than reimplementing it keeps one copy of the tarball streaming,
    the retry policy and the dashboard extractors, so a held-out query is extracted by
    exactly the code that extracted a training query.
    """
    spec = importlib.util.spec_from_file_location(
        "fetch_corpus", ROOT / "scripts" / "fetch_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SOURCES = SOURCES
    return mod


# ---------------------------------------------------------------------------
# what the training corpus already holds
# ---------------------------------------------------------------------------
def seen() -> tuple[set[str], set[str]]:
    """Grafana dashboard ids, and normalised Datadog query strings, already in use."""
    ids, dd = set(), set()
    for q in corpus.read_queries(CORPUS / "queries.jsonl"):
        if q.source == "grafana":
            ids.add(q.origin.split()[0])
        elif q.source == "integrations-core":
            dd.add(corpus.normalise("datadog", q.text))
    return ids, dd


# ---------------------------------------------------------------------------
# grafana: ranks past the ones the corpus took
# ---------------------------------------------------------------------------
def fetch_grafana_unseen(fc, upto: int, *, pause: float = 0.3) -> dict:
    ids, _ = seen()
    dest = SOURCES / "grafana"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    listing, page, total = [], 1, None
    while len(listing) < upto:
        got = fc._json(fc.GRAFANA_LIST.format(page=page, size=100))
        total = got.get("total")
        items = got.get("items") or []
        if not items:
            break
        listing.extend(items)
        page += 1
        time.sleep(pause)
    listing = listing[:upto]

    ranked = [(i + 1, it) for i, it in enumerate(listing)]
    fresh = [(r, it) for r, it in ranked if str(it["id"]) not in ids]
    overlap = len(ranked) - len(fresh)
    print(f"  grafana: {len(ranked)} listed, {overlap} already in the corpus, "
          f"{len(fresh)} held out")

    index, ok = [], 0
    for rank, it in fresh:
        row = {k: it.get(k) for k in ("id", "slug", "name", "revision", "downloads")}
        row["rank"] = rank
        try:
            body = fc._json(fc.GRAFANA_DOWNLOAD.format(id=it["id"], rev=it["revision"]))
        except RuntimeError as e:
            row["error"] = str(e)
            index.append(row)
            continue
        (dest / f"{it['id']}.json").write_text(json.dumps(body), encoding="utf-8")
        row["file"] = f"{it['id']}.json"
        index.append(row)
        ok += 1
        if ok % 25 == 0:
            print(f"    {ok}/{len(fresh)}")
        time.sleep(pause)

    meta = dict(name="grafana", api="grafana.com/api/dashboards",
                filter="dataSourceSlugIn=prometheus, orderBy=downloads desc",
                listed_to_rank=len(ranked), overlap_with_corpus=overlap,
                requested=len(fresh), downloaded=ok, available=total,
                fetched=dt.date.today().isoformat())
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (dest / ".fetch.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  grafana: {ok}/{len(fresh)} downloaded of {total} available")
    return meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grafana-to", type=int, default=0,
                    help="list Prometheus dashboards down to this download rank")
    ap.add_argument("--integrations-core", action="store_true")
    ap.add_argument("--integrations-extras", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.grafana_to = args.grafana_to or 700
        args.integrations_core = True
        args.integrations_extras = True
    if not (args.grafana_to or args.integrations_core or args.integrations_extras):
        ap.error("nothing to do: pass --grafana-to N, --integrations-core, "
                 "--integrations-extras or --all")

    HELDOUT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    fc = _fetch_corpus_module()
    stats = {}
    manifest_path = HELDOUT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if args.grafana_to:
        print("grafana (unseen ranks)")
        manifest["grafana"] = fetch_grafana_unseen(fc, args.grafana_to)
        rows, st = fc.extract_grafana()
        stats["grafana"] = st
        manifest["grafana"].update(st)
        manifest["grafana"]["queries_promql"] = len(rows)
        _write(rows, "grafana")

    if args.integrations_core:
        print("integrations-core (later commit)")
        meta = fc.fetch_repo("integrations-core", fc.REPOS["integrations-core"])
        rows, st = fc.extract_integrations_core()
        _, already = seen()
        fresh = [q for q in rows
                 if corpus.normalise("datadog", q.text) not in already]
        meta.update(st)
        meta["queries_datadog_at_commit"] = len(rows)
        meta["queries_datadog_held_out"] = len(fresh)
        meta["distinct_held_out"] = len({corpus.normalise("datadog", q.text)
                                         for q in fresh})
        meta["corpus_commit"] = _corpus_commit()
        manifest["integrations-core"] = meta
        print(f"  {len(rows):,} Datadog queries at {meta['commit'][:12]}, "
              f"{len(fresh):,} absent from the corpus "
              f"({meta['distinct_held_out']:,} distinct)")
        _write(fresh, "integrations-core")

    if args.integrations_extras:
        print("integrations-extras (a disjoint repository)")
        meta = fc.fetch_repo("integrations-extras", EXTRAS)
        rows, st = fc.extract_integrations_core("integrations-extras")
        _, already = seen()
        fresh = [q for q in rows
                 if corpus.normalise("datadog", q.text) not in already]
        meta.update(st)
        # Counted per language. A widget query and the formula that combines several of
        # them are different populations in this paper, and `tab:parse-coverage` keeps
        # them in separate rows, so the manifest does not add them together.
        for lang in ("datadog", "datadog_formula"):
            at = [q for q in rows if q.lang == lang]
            out = [q for q in fresh if q.lang == lang]
            tag = "" if lang == "datadog" else "_formula"
            meta[f"queries{tag}_at_commit"] = len(at)
            meta[f"queries{tag}_held_out"] = len(out)
            meta[f"distinct{tag}_at_commit"] = len({corpus.normalise(lang, q.text)
                                                    for q in at})
            meta[f"distinct{tag}_held_out"] = len({corpus.normalise(lang, q.text)
                                                   for q in out})
            meta[f"shared{tag}_with_corpus"] = (meta[f"distinct{tag}_at_commit"]
                                                - meta[f"distinct{tag}_held_out"])
            meta[f"dashboards{tag}_held_out"] = len({q.origin for q in out})
        manifest["integrations-extras"] = meta
        print(f"  {meta['queries_at_commit']:,} Datadog widget queries in "
              f"{st['dashboards']} dashboards at {meta['commit'][:12]}, "
              f"{meta['queries_held_out']:,} absent from the corpus "
              f"({meta['distinct_held_out']:,} distinct, "
              f"{meta['shared_with_corpus']} shared) across "
              f"{meta['dashboards_held_out']} dashboards")
        _write(fresh, "integrations-extras")

    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    return 0


def _corpus_commit() -> str:
    """The integrations-core commit the training corpus was extracted from.

    Read out of its own section of SOURCES.md rather than by taking the last commit
    line in the file, which would silently become kube-prometheus's if the sections
    were ever reordered.
    """
    text = (CORPUS / "SOURCES.md").read_text()
    section = text.split("## integrations-core", 1)[1]
    return re.search(r"- commit: `([0-9a-f]+)`", section).group(1)


def _write(rows, source: str) -> None:
    """Merge one source's held-out queries into data/heldout/queries.jsonl."""
    path = HELDOUT / "queries.jsonl"
    kept = []
    if path.exists():
        kept = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        kept = [r for r in kept if r.get("source") != source]
    kept += [dict(kind=q.kind, lang=q.lang, origin=q.origin, source=q.source,
                  text=q.text, unit=q.unit) for q in rows]
    with path.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}: {len(kept):,} queries")


if __name__ == "__main__":
    raise SystemExit(main())

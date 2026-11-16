#!/usr/bin/env python3
"""Check §6's syntax-directed regex -> glob proposer against the complete procedure.

`regular.regex_to_globs` proposes a glob from a regular expression's syntax and verifies
the proposal by deciding language equivalence, so what it returns is exact. Its
*refusals* are a different matter: a refusal looks like a fact about the two languages
and is only a fact about the proposer until something independent agrees. §7 records what
happens when that distinction is skipped.

`regular.single_glob_by_search` is the complete procedure of §6's decidability
proposition: it finds the shortest word of the language by breadth-first search over the
DFA and tries every way of inserting stars into it. Where a single glob exists it finds
one; where none exists it says so. This script runs it over every concrete regular
expression in the corpus and records whether the two ever disagree.

Slow by nature --- each candidate costs a DFA equivalence --- so it is a script with a
cached result rather than part of the test suite, exactly as `verify_anchored.py` is.
`tests/test_regular.py` reads the cache and fails if a disagreement ever appears.

  python scripts/crosscheck_globs.py                # all of them
  python scripts/crosscheck_globs.py --budget 300   # stop after five minutes
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport import corpus as c                                       # noqa: E402
from qlport.regular import (                                         # noqa: E402
    STAR_MATCHES_EMPTY, MAX_UNION, SearchTooLarge, regex_to_globs,
    single_glob_by_search,
)

OUT = ROOT / "data" / "raw" / "globs" / "crosscheck.json"
MATCHER = re.compile(r'(\w+)\s*(=~|!~)\s*"((?:[^"\\]|\\.)*)"')


def concrete_regexes() -> dict[str, int]:
    """Every concrete `=~`/`!~` pattern in the PromQL corpus, by occurrence count."""
    occ: collections.Counter[str] = collections.Counter()
    for q in c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl"):
        if q.lang == "promql":
            for _label, _op, rx in MATCHER.findall(q.text):
                occ[rx] += 1
    return {r: n for r, n in occ.items() if "$" not in r}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=3600.0,
                    help="seconds to spend before recording the rest as not reached")
    args = ap.parse_args(argv)

    regexes = concrete_regexes()
    tally: collections.Counter[str] = collections.Counter()
    disagree, reached = [], {}
    started = time.time()

    # Shortest regexes first: the cheap ones are done even on a small budget.
    for rx in sorted(regexes, key=len):
        proposed = regex_to_globs(rx)
        structural = proposed.globs[0] if proposed.kind == "single" else None
        if time.time() - started > args.budget:
            tally["not-reached"] += 1
            continue
        try:
            found = single_glob_by_search(rx)
        except SearchTooLarge as exc:
            tally["search-too-large"] += 1
            reached[rx] = {"outcome": "search-too-large", "detail": str(exc)}
            continue
        reached[rx] = {"outcome": proposed.kind, "structural": structural,
                       "search": found}
        if found == structural:
            tally[f"agree:{proposed.kind}"] += 1
        else:
            tally["disagree"] += 1
            disagree.append({"regex": rx, "kind": proposed.kind,
                             "structural": structural, "search": found,
                             "reason": proposed.reason})

    payload = {
        "provenance": {
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.time() - started, 1),
            "budget_seconds": args.budget,
            "star_matches_empty": STAR_MATCHES_EMPTY,
            "max_union": MAX_UNION,
            "n_concrete_regexes": len(regexes),
        },
        "tally": dict(tally),
        "disagreements": disagree,
        "per_regex": reached,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)}")
    for k, n in sorted(tally.items()):
        print(f"  {k:22} {n}")
    return 1 if disagree else 0


if __name__ == "__main__":
    raise SystemExit(main())

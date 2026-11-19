"""The corpus-scale differential cache, pinned to the translator it verified.

`data/raw/shapes/differential.json` is the record behind §7's claim that κ measures
translations checked against Datadog rather than merely emitted. Every other vendor
cache in this project is regression-tested; this one was not, and it is the largest.

Two things could rot silently without it. The figures the paper quotes -- 332 shapes,
329 compared, 22,608 bucket comparisons, 8,334 occurrences -- could drift from the
table that reports them. And, worse, a change to the lowering or the emitter could
alter what those shapes translate to, which would detach the cached verification from
the code it verified while leaving every other test green: the cache would still say
"329 shapes agreed with Datadog" about expressions the translator no longer produces.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlport.diagnostics import Bottom
from qlport.emit.promql import emit_detail
from qlport.ir import shape
from qlport.lower.datadog import lower_all
from qlport.parse import parse

CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "shapes" / "differential.json"
ROLLUP = 60.0


def _with_queries(run: dict) -> list[dict]:
    """The rows that record a representative query.

    Four of the 333 do not: two whose rollup interval the run window cannot hold
    twice, one whose filters no synthetic series satisfies, and the shape the
    normaliser fix added, which came back `unmatched` because `dd_keys` was blind to a
    hyphen. Each records only why it could not be compared, so there is nothing to
    re-translate. The fourth is expected to compare once the differential is re-run
    with the parse-tree mapping, which will make this 330.
    """
    rows = [r for r in run["rows"] if "datadog" in r]
    assert len(rows) == 329, len(rows)
    return rows


@pytest.fixture(scope="module")
def run() -> dict:
    if not CACHE.exists():                       # pragma: no cover - cache is committed
        pytest.skip(f"{CACHE} missing; run scripts/shape_differential.py")
    return json.loads(CACHE.read_text())


def test_the_run_matches_the_figures_the_paper_reports(run):
    """These are the numbers in tab:differential and in §7's prose.

    The second run of the experiment, on the corpus after `corpus._TMPL` was fixed and
    with a fresh set of synthetic series. It reproduces the first run's 329 of 329 over
    the same 22,608 buckets -- the independent replication §7.6 said it did not have --
    and carries one more shape and one more uncompared status than the first did.
    """
    assert len(run["rows"]) == 333
    assert run["status_counts"] == {
        "compared": 329, "no-data": 1, "unmatched": 1, "window-too-short": 2}
    assert run["shapes_compared"] == 329 and run["shapes_exact"] == 329
    assert run["occurrences_compared"] == 8364 and run["occurrences_exact"] == 8364
    assert sum(r.get("buckets", 0) for r in run["rows"]) == 22608


def test_every_compared_shape_agreed(run):
    """`shapes_exact == shapes_compared` is the headline; check it row by row too."""
    disagreed = [r["index"] for r in run["rows"]
                 if r["status"] == "compared" and not r["exact"]]
    assert disagreed == [], disagreed


def test_the_verified_queries_still_translate(run):
    """A cached verification is about a translator. This pins which one.

    If any representative stops translating, the row's agreement no longer says
    anything about today's emitter.
    """
    broken = []
    for r in _with_queries(run):
        lowered = lower_all(parse("datadog", r["datadog"], "v2"))
        if len(lowered) != 1 or isinstance(lowered[0], Bottom):
            broken.append((r["index"], r["datadog"], "lowering"))
            continue
        if isinstance(emit_detail(lowered[0], default_rollup=ROLLUP), Bottom):
            broken.append((r["index"], r["datadog"], "emission"))
    assert broken == [], broken


def test_the_rows_are_still_one_per_shape(run):
    """§7.8's reduction is sound only if a representative stands for its whole class.

    The run asserts that when it builds the rows; nothing re-checked it afterwards, so
    a change to `shape()` or to the lowering could collapse or split classes and leave
    the cache describing a partition that no longer exists.
    """
    shapes = []
    for r in _with_queries(run):
        lowered = lower_all(parse("datadog", r["datadog"], "v2"))
        if len(lowered) == 1 and not isinstance(lowered[0], Bottom):
            shapes.append(shape(lowered[0]))
    assert len(shapes) == 329, len(shapes)
    assert len(set(shapes)) == 329, f"{len(shapes) - len(set(shapes))} collisions"


# --- the declared-metric-type join --------------------------------------------------
# `scripts/type_declared.py` checks the one precondition the paper carries on 100% of
# its output: the metric type the front end infers from the query text. The repository
# the dashboards come from declares each metric's type in metadata.csv, so for the
# metrics this corpus names the assumption is decidable. The figures it produces are
# quoted in four sections, and the verdict for a given query is a property of today's
# inference rules -- so both are pinned.

TYPES = CACHE.parent.parent / "datadog" / "type_declared.json"


@pytest.fixture(scope="module")
def types() -> dict:
    if not TYPES.exists():                       # pragma: no cover - cache is committed
        pytest.skip(f"{TYPES} missing; run scripts/type_declared.py")
    return json.loads(TYPES.read_text())


def test_the_type_join_is_internally_consistent(types):
    occ = types["occurrences"]
    # 8,444 until corpus._TMPL stopped truncating a template variable at a hyphen,
    # which returned 31 velero occurrences to the translated set.
    assert sum(occ.values()) == types["translated_occurrences"] == 8475
    decidable = occ["agree"] + occ["wrong-gauge"] + occ["wrong-counter"]
    # 7,557 before the same fix; the 31 velero metrics are declared, so they joined
    # the decidable set too and both derived shares held at 89.5% and 6.5%.
    assert decidable == types["decidable_occurrences"] == 7588
    assert occ["wrong-gauge"] + occ["wrong-counter"] == types["wrong_occurrences"] == 491
    assert types["wrong_share_of_decidable_pct"] == 6.5
    assert types["decidable_share_of_translated_pct"] == 89.5


def test_the_join_is_pinned_to_the_corpus_commit(types):
    """The declarations must come from the tree the queries came from, not from master.

    `fetch_metric_types.py` reads the commit out of the corpus's own `.fetch.json`, so
    this pins that it did and that the commit is still the one the corpus records.
    """
    assert types["provenance"]["commit"].startswith("452093828d")
    assert types["provenance"]["declarations_in_population"] == 37070
    assert types["declared_population"] == {
        "gauge": 23597, "count": 13140, "rate": 330, "conflict": 3,
    }


def test_todays_inference_still_gives_the_recorded_verdicts(types):
    """A verdict is a claim about the inference rules, so it is re-derived here.

    Changing what counts as evidence of a counter -- adding a method, dropping one --
    moves these without moving any other number in the paper.
    """
    import sys

    sys.path.insert(0, str(CACHE.parents[3] / "scripts"))
    import type_declared as td

    declared_raw = types["declared_for_corpus_metrics"]
    by_mangled: dict[str, str] = {}
    for name, kind in declared_raw.items():
        key = td.metric_name(name)
        if key in by_mangled and by_mangled[key] != kind:
            by_mangled[key] = "ambiguous"
        else:
            by_mangled.setdefault(key, kind)

    for expected, queries in types["examples"].items():
        for q in queries:
            kinds = []
            for e in lower_all(parse("datadog", q, "v2")):
                if isinstance(e, Bottom):
                    continue
                if isinstance(emit_detail(e, default_rollup=ROLLUP), Bottom):
                    continue
                kinds += [td.verdict(s.kind, by_mangled.get(s.metric))
                          for s in td.selects(e)]
            worst = next(r for r in ("wrong-gauge", "wrong-counter", "ambiguous",
                                     "rate", "undeclared", "agree") if r in kinds)
            assert worst == expected, (q, worst, expected)

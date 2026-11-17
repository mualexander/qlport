"""The corpus-scale differential cache, pinned to the translator it verified.

`data/raw/shapes/differential.json` is the record behind §7's claim that κ measures
translations checked against Datadog rather than merely emitted. Every other vendor
cache in this project is regression-tested; this one was not, and it is the largest.

Two things could rot silently without it. The figures the paper quotes -- 325 shapes,
323 compared, 21,962 bucket comparisons, 8,337 occurrences -- could drift from the
table that reports them. And, worse, a change to the lowering or the emitter could
alter what those shapes translate to, which would detach the cached verification from
the code it verified while leaving every other test green: the cache would still say
"323 shapes agreed with Datadog" about expressions the translator no longer produces.
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

    Two of the 325 do not: the shape whose 1800 s rollup the run window cannot hold
    twice, and the one whose filters no synthetic series satisfies. Both record only
    why they could not be compared, so there is nothing to re-translate.
    """
    rows = [r for r in run["rows"] if "datadog" in r]
    assert len(rows) == 323, len(rows)
    return rows


@pytest.fixture(scope="module")
def run() -> dict:
    if not CACHE.exists():                       # pragma: no cover - cache is committed
        pytest.skip(f"{CACHE} missing; run scripts/shape_differential.py")
    return json.loads(CACHE.read_text())


def test_the_run_matches_the_figures_the_paper_reports(run):
    """These are the numbers in tab:differential and in §7's prose."""
    assert len(run["rows"]) == 325
    assert run["status_counts"] == {"compared": 323, "no-data": 1, "window-too-short": 1}
    assert run["shapes_compared"] == 323 and run["shapes_exact"] == 323
    assert run["occurrences_compared"] == 8337 and run["occurrences_exact"] == 8337
    assert sum(r.get("buckets", 0) for r in run["rows"]) == 21962


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
    assert len(shapes) == 323, len(shapes)
    assert len(set(shapes)) == 323, f"{len(shapes) - len(set(shapes))} collisions"

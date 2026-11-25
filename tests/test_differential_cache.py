"""The corpus-scale differential cache, pinned to the translator it verified.

`data/raw/shapes/differential.json` is the record behind §7's claim that κ measures
translations checked against Datadog rather than merely emitted. Every other vendor
cache in this project is regression-tested; this one was not, and it is the largest.

Two things could rot silently without it. The figures the paper quotes -- 333 shapes,
330 compared, 22,692 bucket comparisons, 8,365 occurrences -- could drift from the
table that reports them. And, worse, a change to the lowering or the emitter could
alter what those shapes translate to, which would detach the cached verification from
the code it verified while leaving every other test green: the cache would still say
"329 shapes agreed with Datadog" about expressions the translator no longer produces.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from texsource import builds, flat, table

from qlport.diagnostics import Bottom
from qlport.emit.promql import emit_detail
from qlport.ir import shape
from qlport.lower.datadog import lower_all
from qlport.parse import parse

CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "shapes" / "differential.json"
ROOT = CACHE.parents[3]
REGISTRY = ROOT / "paper" / "quoted_figures.json"
TABLE = ROOT / "paper" / "tables" / "differential.tex"
PROSE = ROOT / "paper" / "sections" / "07_evaluation.tex"
ROLLUP = 60.0


def _with_queries(run: dict) -> list[dict]:
    """The rows that record a representative query.

    Three of the 333 do not: two whose rollup interval the run window cannot hold
    twice, and one whose filters no synthetic series satisfies. Each records only why
    it could not be compared, so there is nothing to re-translate.

    It was four, and the fourth is why `dd_keys` now reads the parse tree. That shape
    came back `unmatched` -- both systems returned data and no bucket could be paired,
    because the harness submitted the synthetic tags under their normalised spelling
    while asking Datadog to group by the query's own, which it mapped for dotted keys
    and not for hyphenated ones. With the mapping taken off the parse tree it compares
    and agrees, which is what this count rising to 330 records.
    """
    rows = [r for r in run["rows"] if "datadog" in r]
    assert len(rows) == 330, len(rows)
    return rows


@pytest.fixture(scope="module")
def run() -> dict:
    if not CACHE.exists():                       # pragma: no cover - cache is committed
        pytest.skip(f"{CACHE} missing; run scripts/shape_differential.py")
    return json.loads(CACHE.read_text())


def test_the_run_matches_the_figures_the_paper_reports(run):
    """These are the numbers in tab:differential and in §7's prose.

    The third run of the experiment, and the one with nothing left unexplained: the
    `unmatched` status is gone, because the harness now maps group keys off the parse
    tree rather than a hyphen-blind regex. 330 shapes compared and 330 agreed, over
    22,692 buckets, standing for 8,365 of the 8,370 translated single-query
    occurrences; the five that remain are the three shapes below.

    Two earlier runs agreed on every shape they reached, so the agreement itself is
    replicated three times over on three sets of synthetic series. What this run adds
    is that no shape in the corpus is now uncompared for a reason internal to the rig.
    """
    assert len(run["rows"]) == 333
    assert run["status_counts"] == {
        "compared": 330, "no-data": 1, "window-too-short": 2}
    assert run["shapes_compared"] == 330 and run["shapes_exact"] == 330
    assert run["occurrences_compared"] == 8365 and run["occurrences_exact"] == 8365
    assert sum(r.get("buckets", 0) for r in run["rows"]) == 22692


def test_the_paper_states_the_figures_this_cache_derives(run):
    """The join the test above cannot make: from the cache to the paper, by derivation.

    These figures are written down twice -- as literals in the test above, and as
    literals in the prose and the generated table -- and until now nothing connected the
    two. That is not hypothetical: the third run landed in the cache, this file was
    updated to its counts, and §7.4 went on reporting the second run's $329$ of $22{,}608$
    for a further two commits, with every test green. Each copy was self-consistent and
    nothing compared them.

    So the figures are derived from the rows and then looked for where the paper puts
    them: the registry notebook 05 writes, the table §7.4 inputs, and the sentences that
    state the ones no table carries.
    """
    rows = run["rows"]
    compared = [r for r in rows if r["status"] == "compared"]
    agreed = [r for r in compared if r["exact"] == 1.0]
    buckets = sum(int(r["buckets"]) for r in compared)
    occ_cmp = sum(int(r["occurrences"]) for r in compared)
    occ_ok = sum(int(r["occurrences"]) for r in agreed)
    bit_identical = sum(1 for r in compared
                        if r.get("bit_identical") in (1.0, True))
    worst = max(r["max_d"] for r in compared)

    reg = json.loads(REGISTRY.read_text())
    assert reg["nb05.shapes"] == len(rows) == run["provenance"]["shapes"]
    assert reg["nb05.bucket_comparisons"] == buckets
    assert reg["nb05.bit_identical_shapes"] == bit_identical
    assert reg["nb05.synthetic_series"] == run["provenance"]["series"]

    # The table carries the shape and occurrence counts; `tabular` writes them with a
    # thousands comma, which is also how the prose spells a four-digit figure.
    printed = table(TABLE)
    for value in (len(compared), len(agreed), occ_cmp, occ_ok):
        assert f"{value:,}" in printed, (value, printed)
    assert f"& {len(compared) - len(agreed)} & {occ_cmp - occ_ok} " in printed, printed

    # And the two figures no table holds: the occurrence weight of the agreement, and
    # the largest deviation among the shapes that agreed. One significant figure in
    # units of 1e-14 is how §7.4 writes the second.
    for prose in builds(PROSE):
        assert f"${occ_ok:,}$".replace(",", "{,}") in prose, occ_ok
        assert f"${worst / 1e-14:.1f}" + r"\times10^{-14}$" in prose, worst


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
    assert len(shapes) == 330, len(shapes)
    assert len(set(shapes)) == 330, f"{len(shapes) - len(set(shapes))} collisions"


# --- the three runs, and the one figure nothing could fail on -------------------------
# §7.6 compares three runs and the working tree holds one. For the other two the prose
# was the only record, so nothing could fail on them -- and a sentence about the second
# run went wrong in the way that state invites: it divided the FIRST run's shape count
# into the second's compared count, got three uncompared where the second run printed
# four, and settled the discrepancy by asserting that run's cache no longer existed. It
# exists, in the commit that replaced it. `scripts/differential_run_history.py` recovers
# all three from the commits that cached them, names the shapes each could not reach by
# re-deriving the shape list at that commit, and writes the result here.

HISTORY = CACHE.parent / "run_history.json"

WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


@pytest.fixture(scope="module")
def history() -> dict:
    if not HISTORY.exists():                     # pragma: no cover - cache is committed
        pytest.skip(f"{HISTORY} missing; run scripts/differential_run_history.py")
    return json.loads(HISTORY.read_text())


def test_the_run_history_ends_at_the_run_the_working_tree_holds(history, run):
    """The history's newest record and the live cache have to be the same run.

    Without this the history is a museum: a fourth run could land in
    `differential.json`, every figure above could move with it, and these records would
    go on describing a sequence the paper no longer reports as current.
    """
    runs = history["runs"]
    assert [r["ordinal"] for r in runs] == ["first", "second", "third"]
    newest = runs[-1]
    assert newest["run_tag"] == run["provenance"]["run"]
    assert newest["shapes"] == len(run["rows"]) == run["provenance"]["shapes"]
    assert newest["compared"] == run["shapes_compared"] == newest["agreed"]
    assert newest["occurrences_compared"] == run["occurrences_compared"]
    assert newest["bucket_comparisons"] == sum(
        int(r["buckets"]) for r in run["rows"] if r["status"] == "compared")
    assert len(newest["uncompared"]) == len(run["rows"]) - run["shapes_compared"]


def test_the_two_runs_the_paper_quotes_are_what_their_caches_say(history):
    """The first two runs' figures, from the cache to the registry to the sentence.

    The pair that was confusable is the whole point: the first two runs printed the same
    compared count over the same bucket count, because the second reached every shape the
    first did and the one shape it added came back `unmatched`. That is why reading a
    shape count off one of them and a compared count off the other produced an arithmetic
    that looked sound. So the identity is asserted, and so is the shape count that
    distinguishes them -- which the census licensed from `rs.span_declared_wrong`, an
    unrelated rollup-span figure of the same value, until `dr.first_shapes` existed.
    """
    first, second, third = history["runs"]
    reg = json.loads(REGISTRY.read_text())

    assert first["compared"] == second["compared"] == first["agreed"] == second["agreed"]
    assert first["bucket_comparisons"] == second["bucket_comparisons"]
    assert first["shapes"] + 1 == second["shapes"] == third["shapes"]
    assert reg["dr.first_shapes"] == first["shapes"]
    assert reg["dr.first_compared"] == first["compared"]
    assert reg["dr.shared_buckets"] == first["bucket_comparisons"]
    # The bound tightened between them, and §7.6 states the factor in words rather than
    # printing either deviation. It said "an order of magnitude" until this test asked:
    # the two bounds are 1.4e-13 and 2.1e-14, which is a decade by exponent and a factor
    # of 6.6 in fact, and the second is what the sentence now says.
    ratio = first["max_deviation"] / second["max_deviation"]
    assert ratio >= 6

    for prose in builds(PROSE):
        said = flat(prose)
        assert f"over ${first['shapes']}$ shapes" in said, first["shapes"]
        assert f"${first['compared']}$ of ${first['compared']}$" in said
        assert f"the same ${first['bucket_comparisons']:,}$".replace(",", "{,}") in said
        assert f"a bound {WORD[int(ratio)]} times tighter" in said, ratio


def test_the_second_run_could_not_compare_four_and_the_fourth_is_named(history):
    """The sentence that was wrong, checked against the run that printed it.

    Three claims, each of which the broken version got wrong or could not have known:
    that the second run's uncompared count is one more than the first's, that the extra
    one is the shape `corpus._TMPL` returned to the corpus rather than one of the first
    run's three, and that the commit the prose names for that cache is the commit this
    history read it from.
    """
    first, second, third = history["runs"]

    assert len(second["uncompared"]) == len(first["uncompared"]) + 1 == 4
    added = second["shapes_added_since_previous"]
    assert len(added) == 1 and not second["shapes_removed_since_previous"]
    extra = [u for u in second["uncompared"] if u["query"] == added[0]]
    assert len(extra) == 1 and extra[0]["status"] == "unmatched", second["uncompared"]
    # The other three are the first run's, unchanged.
    assert ({u["query"] for u in second["uncompared"]} - set(added)
            == {u["query"] for u in first["uncompared"]})

    for prose in builds(PROSE):
        said = flat(prose)
        assert f"could not compare {WORD[len(second['uncompared'])]}" in said
        assert second["commit"] in said, second["commit"]


def test_the_uncompared_shapes_are_the_same_three_in_the_outer_two_runs(history):
    """`all three were in the first run too` -- by query, because the indices moved.

    The no-data row is index 299 in the first run and 300 in the other two, because the
    shape `corpus._TMPL` returned sorts above it. Counting by index would make that look
    like a different shape and by count would make any three look like these three, so
    the comparison is on the query each row stood for, which is why the history re-derives
    the shape list at each commit instead of recording the indices it found.
    """
    first, second, third = history["runs"]
    assert {u["query"] for u in first["uncompared"]} == {
        u["query"] for u in third["uncompared"]}
    assert [u["index"] for u in first["uncompared"]] != [
        u["index"] for u in third["uncompared"]]
    assert {u["status"] for u in third["uncompared"]} == {
        "window-too-short", "no-data"}

    for prose in builds(PROSE):
        said = flat(prose)
        assert (f"{WORD[len(third['uncompared'])].capitalize()} shapes of the "
                f"${third['shapes']}$ are not compared") in said
        assert "the same three queries the first run could not compare" in said


def test_one_shape_changed_status_in_the_whole_history(history, run):
    """§7.6's strongest claim about the sequence, and the one with the most ways to fail.

    Three runs on three independent sets of synthetic series: if the agreement were an
    artefact of the series rather than of the translation, a shape would have moved.
    None did except the one the harness fix was for, and it moved out of the uncompared
    column rather than into it.

    The claim is about statuses, so a shape being *added* is not one -- which is what
    separates the second run from the first. Both halves are asserted here, because the
    sentence is only true if the addition is not counted and only honest if the addition
    is stated, and §7.6 states it one paragraph earlier.
    """
    runs = history["runs"]
    changes = [c for r in runs for c in r["status_changes_since_previous"]]
    assert len(changes) == 1, changes
    (change,) = changes
    assert change["from"] == "unmatched" and change["to"] == "compared"
    assert change["query"] == runs[1]["shapes_added_since_previous"][0]
    assert all(not r["shapes_removed_since_previous"] for r in runs)
    assert sum(len(r["shapes_added_since_previous"]) for r in runs) == 1

    # And what it compares to, now that it compares: the row §7.6 quotes.
    index = next(u["index"] for u in runs[1]["uncompared"]
                 if u["query"] == change["query"])
    row = next(r for r in run["rows"] if r["index"] == index)
    assert row["status"] == "compared" and row["max_d"] == 0.0
    for prose in builds(PROSE):
        assert f"${int(row['buckets'])}$ buckets, every one of them" in flat(prose)


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

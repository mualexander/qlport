"""The generated tables, against each other and against the registry.

`test_paper_figures.py` asks whether the prose still agrees with a table. Nothing asked
whether the tables agree with each other, and they are written by five notebooks over
three populations, so they can disagree without any of them being internally wrong.
That is not hypothetical: §4 quoted a v1 parse rate of $92.6\\%$ for as long as
`parse_coverage.tex` said $92.9\\%$, and the two sat four pages apart.

Re-running the notebooks does not catch it. A notebook reproduces its own table exactly,
including a share taken over the wrong population --- which is the error that once put
$87\\%$ in §5 meaning the implementation-order figure. So the checks here are the ones a
notebook cannot make about itself: that a column sums to the total another table reports,
that a share has an integer numerator over the count beside it, and that two tables
built from different cells of the same data say the same thing where they overlap.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from texsource import flat, published

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "paper" / "tables"
REGISTRY = ROOT / "paper" / "quoted_figures.json"
HELDOUT = ROOT / "paper" / "sections" / "moved" / "eval_heldout.tex"

# Rounding a difference and differencing two rounded figures are not the same
# operation, and at one decimal place they disagree about the Datadog fall: the
# rates give 1.4538, the printed 7.0 and 5.6 give 1.4. The paper reports the
# first, so it has to tell the reader that subtracting its own two figures is
# not how the third was obtained.
RECONCILED = "the difference of the unrounded rates and not of the two figures printed"


@pytest.fixture(scope="module")
def reg() -> dict:
    return json.loads(REGISTRY.read_text())


def rows(name: str) -> list[list[str]]:
    """A generated `tabular` as a list of cell lists, rules and preamble dropped."""
    text = published(TABLES / f"{name}.tex")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("\\") and "&" not in line:
            continue
        line = re.sub(r"\\(?:midrule|toprule|bottomrule|addlinespace)\b", "", line)
        line = line.removesuffix("\\\\").strip()
        if "&" not in line:
            continue
        out.append([c.strip() for c in line.split("&")])
    return out


def num(cell: str) -> float | None:
    """The number in a cell, or None if it holds no plain number."""
    cell = re.sub(r"\\(?:texttt|emph|quad|textbackslash|midrule)\b", "", cell)
    cell = cell.replace("{", "").replace("}", "").replace("\\%", "").replace(",", "")
    cell = cell.replace("$", "").strip()
    try:
        return float(cell)
    except ValueError:
        return None


def column(name: str, index: int, skip_header: bool = True) -> list[float]:
    body = rows(name)[1:] if skip_header else rows(name)
    return [v for v in (num(r[index]) for r in body if len(r) > index) if v is not None]


# --- the three populations, each summing to a total another artifact reports ----------

def test_the_translation_outcomes_exhaust_the_corpus(reg):
    occ, dist = column("translation_coverage", 1), column("translation_coverage", 3)
    assert len(occ) == len(dist) == 3, (occ, dist)
    assert sum(occ) == reg["nb03.corpus_occurrences"]
    assert sum(dist) == reg["nb03.distinct_after_instantiation"]
    assert occ[0] == reg["nb03.kappa_occurrences"]
    assert dist[0] == reg["nb03.kappa_distinct"]


def test_the_gap_catalogue_exhausts_the_refusals(reg):
    occ, dist = column("gap_catalogue", 1), column("gap_catalogue", 2)
    assert sum(occ) == reg["nb03.not_implemented_occurrences"]
    # And the distinct column against the row translation_coverage reports for bottom.
    assert sum(dist) == column("translation_coverage", 3)[1]


def test_the_differential_exhausts_the_shapes(reg):
    # Row 0 is the compared total and row 1 its agreeing part, so the partition is
    # rows 0, 2 and 3 -- the compared, the no-data and the window-too-short.
    shapes, occ = column("differential", 1), column("differential", 2)
    # The last row of each column is the agreement percentage, not a count.
    assert shapes[-1] == occ[-1] == 100.0, (shapes, occ)
    assert shapes[0] + shapes[3] + shapes[4] == reg["nb05.shapes"]
    assert occ[0] + occ[3] + occ[4] == reg["nb03.shaped_occurrences"]
    # The agreeing row cannot exceed the compared row, and the disagreeing row is the
    # difference -- which is the one place this table could contradict itself.
    assert shapes[1] + shapes[2] == shapes[0]
    assert occ[1] + occ[2] == occ[0]


# --- shares, each over the count beside it -------------------------------------------

def _integer_numerator(share: float, total: float, places: int = 1) -> bool:
    """Is `share` the rounding of some whole number over `total`?"""
    for k in range(int(total) + 1):
        if round(k / total * 100, places) == share:
            return True
    return False


def test_every_share_in_the_feature_table_has_an_integer_numerator():
    """A share with no whole numerator is a share over the wrong denominator.

    `gap_by_feature`'s rows read "queries using this feature" against "translated", so
    the share must be some whole number of those queries. The check is weak per row and
    strong over twelve of them.
    """
    bad = []
    for row in rows("gap_by_feature")[1:]:
        occ, share = num(row[2]), num(row[3])
        if occ is None or share is None:
            continue
        if not _integer_numerator(share, occ):
            bad.append(f"  {row[0]}: {share}% of {occ:g} is no whole number")
    assert not bad, "\n".join(bad)


def test_the_caveat_shares_are_over_what_was_translated(reg):
    bad = []
    for row in rows("translation_caveats")[1:]:
        occ, share = num(row[1]), num(row[3])
        if occ is None or share is None:
            continue
        want = round(occ / reg["nb03.kappa_occurrences"] * 100, 1)
        if want != share:
            bad.append(f"  {row[0]}: {share} stated, {want} over kappa")
    assert not bad, "\n".join(bad)


def test_the_baseline_shares_are_over_the_population_it_names(reg):
    bad = []
    for row in rows("baseline")[1:]:
        ours, ours_share = num(row[1]), num(row[2])
        theirs, theirs_share = num(row[3]), num(row[4])
        for count, share in ((ours, ours_share), (theirs, theirs_share)):
            if count is None or share is None:
                continue
            want = round(count / reg["oo.emits"] * 100, 1)
            if want != share:
                bad.append(f"  {row[0]}: {count:g} is {want}, stated {share}")
    assert not bad, "\n".join(bad)


# --- two tables over the same data, where they overlap --------------------------------

def test_the_parse_table_and_the_translation_table_agree_on_what_does_not_parse(reg):
    datadog = next(r for r in rows("parse_coverage") if r[0] == "datadog")
    assert num(datadog[1]) == reg["nb03.corpus_occurrences"]
    # The residual is the distinct queries v2 fails, which is the unparsed row of
    # translation_coverage -- the same six queries counted by two notebooks.
    assert num(datadog[6]) == column("translation_coverage", 3)[2]
    # And the v2 rate is that residual against the distinct column.
    distinct = num(datadog[2])
    assert round((distinct - num(datadog[6])) / distinct * 100, 1) == num(datadog[5])


def test_the_two_tables_that_count_feature_uses_agree(reg):
    """`implementation_order` and `gap_by_feature` are built from different cells.

    The first reports uses pending at each step and the second uses in total, so for a
    feature appearing in both the counts must match -- and the second's share must be
    the first's completions over them. They disagreed by construction in an earlier
    draft, where one counted occurrences and the other distinct queries.
    """
    order = {r[1]: r for r in rows("implementation_order")[1:]}
    overlap = [r for r in rows("gap_by_feature")[1:] if r[0] in order]
    assert len(overlap) >= 8, f"only {len(overlap)} features in both tables"
    bad = []
    for row in rows("gap_by_feature")[1:]:
        if row[0] not in order:
            continue
        step = order[row[0]]
        if num(step[3]) != num(row[2]):
            bad.append(f"  {row[0]}: {num(step[3]):g} pending vs {num(row[2]):g} uses")
    assert not bad, "\n".join(bad)


def test_the_staircase_column_is_the_registry_staircase(reg):
    """`implementation_order`'s cumulative column is the figure §7.3 argues from.

    It is written by the same notebook that registers `nb00.ceiling_N_features_pct`, and
    the two have to be the same numbers in the same order or the table and the plot are
    about different curves.
    """
    cumulative = column("implementation_order", 5)
    for step, got in enumerate(cumulative, start=1):
        key = f"nb00.ceiling_{step}_features_pct"
        assert key in reg, f"{key} missing for step {step}"
        assert reg[key] == got, (step, reg[key], got)


def test_the_sources_table_accounts_for_the_whole_corpus(reg):
    """Its three rows plus the formulas held aside are the corpus the abstract names."""
    queries = sum(column("corpus_sources", 4))
    assert queries + reg["nb00.formulas"] == 22894
    # The Datadog row is the population kappa is measured over.
    datadog = next(r for r in rows("corpus_sources") if r[1] == "Datadog")
    assert num(datadog[4]) == reg["nb03.corpus_occurrences"]


# --- the rollup-span join ---------------------------------------------------------------
# E18's lookup and the corpus's declared spans were both on disk for some time before
# anything joined them, which is how RQ4 came to be answered for one of its two defaults
# and not the other. These checks are the ones the join script cannot make about itself:
# that its two populations exhaust the code it partitions, that its factors are the
# lookup's own arithmetic, and that the count it starts from is the one the translation
# table reports.

def _derived(name: str) -> dict:
    """A script's cached output, insisting it is there and saying what makes it.

    `data/processed/` is gitignored because everything in it is derived, so a fresh
    checkout has none of it and these tests would otherwise fail with a
    FileNotFoundError that names a path and no remedy. `make data` runs the scripts;
    they read what the notebooks write, which is why both are named here.
    """
    path = ROOT / "data" / "processed" / name
    assert path.exists(), (
        f"{path.relative_to(ROOT)} is missing --- run `make notebooks && make data`"
    )
    return json.loads(path.read_text())


def _spans() -> dict:
    return _derived("rollup_spans.json")


def test_the_declared_and_undeclared_spans_exhaust_rollup_default(reg):
    """The partition is the whole code, not a sample of it."""
    s = _spans()["figures"]
    assert s["rs.span_declared_occurrences"] + s["rs.span_undeclared_occurrences"] == \
        s["rs.rollup_default_occurrences"]
    assert s["rs.span_declared_right"] + s["rs.span_declared_wrong"] == \
        s["rs.span_declared_occurrences"]
    for key in ("rs.rollup_default_occurrences", "rs.span_declared_wrong",
                "rs.span_declared_occurrences"):
        assert reg[key] == s[key], key


def test_the_rollup_default_count_is_the_one_the_translation_table_reports():
    """8,217 has to be the weight of the code in `translations.csv`, not a re-count."""
    import csv

    path = ROOT / "data" / "processed" / "translations.csv"
    weight = sum(
        int(row["n"]) for row in csv.DictReader(path.open())
        if row["outcome"] == "translated"
        and "ROLLUP_DEFAULT" in (row["caveats"] or "").split("|"))
    assert weight == _spans()["figures"]["rs.rollup_default_occurrences"]


def test_only_four_hours_takes_the_emitted_interval():
    """The whole point of the join: 60 s is one row of E18's lookup and no other.

    If a second span ever reads 60 s, the paragraph in §7.6 that calls the default
    "wrong for two in three" is measuring something else.
    """
    spans = _spans()
    lookup = spans["lookup"]
    sixty = sorted(k for k, v in lookup.items() if v == 60)
    assert sixty == ["4h"], sixty
    for span, row in spans["by_span"].items():
        assert row["interval_seconds"] == lookup[span], span
        assert row["factor"] == round(60.0 / lookup[span], 3), span
    right = spans["by_span"]["4h"]["occurrences"]
    assert right == spans["figures"]["rs.span_declared_right"]


def test_the_extrapolation_is_labelled_and_is_an_upper_estimate(reg):
    """It is the measured share carried across, and the prose must not call it measured."""
    s = _spans()["figures"]
    share = s["rs.span_declared_wrong"] / s["rs.span_declared_occurrences"]
    assert s["rs.extrapolated_wrong"] == round(s["rs.rollup_default_occurrences"] * share)
    body = published(ROOT / "paper" / "sections" / "07_evaluation.tex")
    assert "extrapolation and should not be read as a second measurement" in body
    # And it must exceed what was measured, or it is not the thing being warned about.
    assert s["rs.extrapolated_wrong"] > s["rs.span_declared_wrong"]


def test_the_unimplemented_column_decomposes_to_its_total():
    """The five codes of the refused column, against the 287 the prose quotes.

    The appendix described the column as "this row's 208, those four, and the 36" for as
    long as it held three codes; the last two retractions moved their occurrences in and
    the sentence was not updated, so the description summed to 248 against a total of
    287. Pinned here because the error was arithmetic in prose, which no notebook reads.
    """
    import collections
    import csv

    rows = list(csv.DictReader(
        (ROOT / "data" / "processed" / "translations.csv").open()))
    by_code = collections.Counter()
    for row in rows:
        if row["outcome"] == "bottom":
            by_code[row["gap"]] += int(row["n"])

    assert dict(by_code) == {
        "UNSUPPORTED_FUNCTION": 208, "FILL_INTERPOLATION": 36,
        "PHI_QUANTILE": 34, "SCOPE_EMPTY_VALUE": 5, "SCOPE_DISJUNCTION": 4,
    }, dict(by_code)
    assert sum(by_code.values()) == 287
    body = published(ROOT / "paper" / "sections" / "moved" / "eval_unsupported.tex")
    assert "five codes and $287$ occurrences" in body


# --- the two parsers, compared --------------------------------------------------------
# §4's grammar answers "is this PromQL?" everywhere in the paper, and until the
# fifteenth pass it answered alone. `scripts/promtool_parse.py` puts the same question
# to Prometheus 3.15.0's own parser over every expression either translator emits. These
# tests hold the published numbers to that cache; the agreement itself is asserted
# inside the script, where a disagreement stops the measurement rather than being
# reported by it.
def _promtool() -> dict:
    return _derived("promtool.json")


def test_the_two_parsers_agree_on_every_expression():
    """The claim §7.4 and rem:baseline now make. Zero is the only acceptable count."""
    fig = _promtool()["figures"]
    assert fig["pt.disagreements"] == 0
    assert fig["pt.ours_rejected_exprs"] == 0
    assert fig["pt.ours_rejected_occ"] == 0


def test_the_checked_set_is_both_translators_whole_output(reg):
    """A parse check over part of the output would license none of the sentences."""
    fig = _promtool()["figures"]
    assert fig["pt.exprs"] == fig["pt.ours_exprs"] + fig["pt.theirs_exprs"]
    # Ours is the whole corpus: κ's own occurrence count, not the held-out subset.
    assert fig["pt.ours_occ"] == reg["nb03.kappa_occurrences"]
    # Theirs is the set the baseline table reports, key for key.
    assert fig["pt.theirs_occ"] == reg["oo.emits"]


def test_prometheus_rejects_exactly_what_the_baseline_table_calls_invalid(reg):
    """`oo.invalid` was measured with this paper's grammar; 457 is now both verdicts.

    If these ever diverge, §7.4's "not an artefact of measuring PromQL with our own
    ruler" is false and the sentence has to go rather than be softened.
    """
    assert _promtool()["figures"]["pt.theirs_rejected_occ"] == reg["oo.invalid"]


def test_the_match_all_scope_is_the_largest_single_reason():
    """§7.4 attributes the bulk to `{*}` reaching a language where `*` multiplies."""
    reasons = _promtool()["reasons"]
    top, count = max(reasons.items(), key=lambda kv: kv[1])
    assert "inside braces" in top and "'*'" in top
    assert count == max(reasons.values()) and count > sum(reasons.values()) / 2


def test_the_measurement_records_the_build_it_came_off():
    """A parse verdict is a verdict of one parser version; the paper names 3.15.0."""
    prov = _promtool()["provenance"]
    assert prov["image"] == "prom/prometheus:v3.15.0"
    assert prov["command"] == "promtool check rules"


# --- another tool's figures, held to a cache ------------------------------------------
def test_the_dash0_claim_is_read_from_the_cached_readme(reg):
    """§2 quotes two figures this paper did not measure.

    They are the converter's own, so nothing here can derive them; what can be checked
    is that they are the ones the cached README states, and that the cache is the file
    the manifest describes. A force-push to that branch must not be able to change what
    the paper attributes to it, and a hand-edited cache must not pass.
    """
    import importlib.util

    path = ROOT / "scripts" / "dash0_claims.py"
    spec = importlib.util.spec_from_file_location("dash0_claims", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fig = mod.figures()          # raises if the sha256 or the byte count disagrees
    assert fig["dd0.queries_and_monitors"] == reg["dd0.queries_and_monitors"] == 14172
    assert fig["dd0.translate_pct"] == reg["dd0.translate_pct"] == 92.4

    man = json.loads((ROOT / "data" / "raw" / "dash0" / "MANIFEST.json").read_text())
    assert man["branch"] and len(man["commit"]) == 40
    assert man["retrieved"] == "2026-11-21"


# --- the held-out split ---------------------------------------------------------------
def _cv() -> dict:
    return _derived("heldout_cv.json")


def _ho() -> dict:
    return _derived("heldout_eval.json")


def test_every_held_out_figure_is_the_one_the_run_produced(reg):
    """The registry's `ho.*` keys, recomputed from the cached run and compared.

    Everything above this pins the prose to `paper/quoted_figures.json`. Nothing
    pinned that file to anything. Thirteen of its values --- the margins, the two
    shrinkages, every standard error --- could be edited by hand and the whole suite
    stayed green, which is how four deliberate corruptions survived a review. The
    registry is written by `scripts/heldout_eval.py`, so the check is to call the
    function that writes it on the run it was written from and compare key by key.

    This replaces per-figure data-side tests rather than adding to them: a new `ho.*`
    key is covered the moment the script emits it, with nothing to remember to add
    here.
    """
    import importlib.util

    path = ROOT / "scripts" / "heldout_eval.py"
    spec = importlib.util.spec_from_file_location("heldout_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    want = mod.figures(_ho())
    assert len(want) > 100, "the run produced almost no figures; it is not the run"
    drifted = [f"  {k}: registry {reg.get(k, '(absent)')} vs run {v}"
               for k, v in sorted(want.items()) if reg.get(k) != v]
    assert not drifted, (
        "the registry no longer says what the run says:\n" + "\n".join(drifted))
    orphans = sorted(k for k in reg if k.startswith("ho.") and k not in want)
    assert not orphans, (
        f"registry keys under `ho.` that the run does not produce: {orphans}")


def test_the_cross_validated_greedy_is_the_staircase_of_notebook_00(reg):
    """The split experiment must refit the paper's instrument, not a near neighbour.

    `scripts/heldout_split.py` reimplements notebook 00's greedy maximum-coverage order,
    because it has to run it hundreds of times on subsamples and the notebook runs it
    once on everything. A reimplementation that broke ties differently would still
    produce a plausible staircase and would silently measure the optimism of a build
    order nobody followed. So its order, run on the whole corpus, has to reproduce every
    ceiling the notebook recorded.
    """
    full = _cv()["full_staircase"]["datadog"]
    assert len(full) == 29, f"28 Datadog features and a zero point, got {len(full)}"
    for k, got in enumerate(full):
        want = reg[f"nb00.ceiling_{k}_features_pct"]
        assert abs(got - want) < 0.051, f"at {k} features: {got} vs notebook 00's {want}"


def test_the_vocabulary_ceiling_saturates_well_inside_the_corpus(reg):
    """§7's second fitted channel, refitted: it costs almost nothing out of sample.

    If the ceiling were a fit to this corpus, a vocabulary read off a sample of the
    dashboards would miss features the rest use, and coverage of the rest would fall.
    It does not fall far even from a tenth of the dashboards, and past a point no
    replicate misses anything at all --- which is the claim the prose makes.
    """
    rows = {r["fraction"]: r for r in _cv()["datadog"]["vocabulary"]}
    tenth, half = rows[0.1], rows[0.5]
    assert tenth["occ_mean"] == pytest.approx(reg["cv.dd_tenth_coverage_pct"], abs=0.05)
    assert half["occ_mean"] == pytest.approx(reg["cv.dd_half_coverage_pct"], abs=0.05)
    assert tenth["occ_mean"] > 95, "a tenth of the dashboards should already nearly do it"
    assert half["occ_mean"] > tenth["occ_mean"], "more dashboards cannot read worse"

    complete = reg["cv.dd_complete_at_pct"] / 100
    assert rows[complete]["novel_max"] == 0, (
        "cv.dd_complete_at_pct names a sample size at which no replicate missed a "
        "feature, and this one did")
    below = [f for f in rows if f < complete]
    assert below and max(rows[f]["novel_max"] for f in below) > 0, (
        "if nothing is missed at any smaller size either, the figure is not the "
        "saturation point it is quoted as")


def test_the_staircase_survives_being_scored_on_dashboards_it_was_not_built_from(reg):
    """§8 calls the staircase the paper's practical result, so it must not be a fit."""
    s = _cv()["datadog"]["staircase"]
    train, test = s["train"][16], s["test"][16]
    assert train == pytest.approx(reg["cv.dd_stair_train_16_pct"], abs=0.05)
    assert test == pytest.approx(reg["cv.dd_stair_test_16_pct"], abs=0.05)
    assert train - test == pytest.approx(reg["cv.dd_stair_gap_16_pct"], abs=0.05)
    assert 0 <= train - test < 5, (
        "a build order chosen on one half should not be worth much less on the other")
    assert s["train"][0] < s["train"][8] < s["train"][16], "the staircase should rise"


def test_the_held_out_set_is_disjoint_from_the_corpus():
    """The premise of the whole exercise, checked against the two committed files.

    Both `data/corpus/queries.jsonl` and `data/heldout/queries.jsonl` are committed, so
    this needs no network and no `make data`: it is the one claim that cannot be taken
    on trust, because every rate in the held-out table is worthless if a query appears
    on both sides.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from qlport import corpus as c

    train = c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl")
    held = c.read_queries(ROOT / "data" / "heldout" / "queries.jsonl")
    assert held, "nothing is held out"

    seen_ids = {q.origin.split()[0] for q in train if q.source == "grafana"}
    held_ids = {q.origin.split()[0] for q in held if q.source == "grafana"}
    assert held_ids and not (held_ids & seen_ids), "a held-out dashboard is in the corpus"

    seen_dd = {c.normalise("datadog", q.text)
               for q in train if q.source == "integrations-core"}
    held_dd = [q for q in held if q.source == "integrations-core"]
    assert held_dd
    leaked = [q.text for q in held_dd
              if c.normalise("datadog", q.text) in seen_dd]
    assert not leaked, f"{len(leaked)} held-out Datadog queries are in the corpus"


def test_the_held_out_rate_is_reported_on_the_strings_the_corpus_does_not_hold(reg):
    """Dashboard-level disjointness is not string-level disjointness.

    A popular PromQL query recurs across dashboards, so some strings in a held-out
    dashboard are strings the corpus also holds, and those are not held out in any sense
    that bears on a grammar. The table carries both rates and the prose quotes the
    narrower one; this checks the two populations decompose and that the distinction is
    not vacuous.
    """
    g = _ho()["grafana"]
    assert g["shared_distinct"] + g["novel_distinct"] == g["distinct"]
    assert g["shared_distinct"] > 0, (
        "no overlap at all would mean the stricter population is the same as the "
        "looser one, and the paper should stop distinguishing them")
    assert g["novel_distinct"] == reg["ho.gf_novel_distinct"]
    assert g["novel_v2_pct"] == reg["ho.gf_novel_v2_pct"]
    assert g["novel_v2_pct"] <= g["v2_pct"], (
        "the strings the corpus also holds cannot be the harder ones")
    # The Datadog side is built by excluding corpus strings, so it is all novel.
    assert _ho()["integrations-core"]["shared_distinct"] == 0


def test_the_held_out_table_says_what_the_registry_says(reg):
    """The generated table and the figures the prose quotes, against each other."""
    body = rows("heldout_coverage")[1:]           # drop the header row
    by_label = {r[0]: r for r in body}
    # "corpus's" alone now matches two rows, and only insertion order kept this on
    # the Grafana one; a reordered table would have failed somewhere else entirely.
    insample = next(r for lab, r in by_label.items()
                    if lab.startswith("Grafana, the corpus's"))
    assert num(insample[1]) == reg["ho.gf_in_sample_queries"]
    assert num(insample[5]) == reg["ho.gf_in_sample_v2_pct"]
    assert num(insample[6]) == reg["ho.gf_in_sample_residual"]
    gf = next(r for lab, r in by_label.items() if "never saw" in lab)
    assert num(gf[1]) == reg["ho.gf_queries"]
    assert num(gf[2]) == reg["ho.gf_distinct"]
    assert num(gf[4]) == reg["ho.gf_v1_pct"]
    assert num(gf[5]) == reg["ho.gf_v2_pct"]
    assert num(gf[6]) == reg["ho.gf_residual"]
    sub = next(r for lab, r in by_label.items() if "the corpus lacks" in lab)
    assert num(sub[2]) == reg["ho.gf_novel_distinct"]
    assert num(sub[5]) == reg["ho.gf_novel_v2_pct"]
    assert num(insample[5]) < num(sub[5]), (
        "the text says the frozen grammar does no worse out of sample; if that stops "
        "being true the paragraph has to change, not this assertion")
    ddin = next(r for lab, r in by_label.items()
                if "integrations-core" in lab and "the corpus's" in lab)
    assert num(ddin[1]) == reg["ho.dd_in_sample_queries"]
    assert num(ddin[2]) == reg["ho.dd_in_sample_distinct"]
    assert num(ddin[4]) == reg["ho.dd_in_sample_v1_pct"]
    assert num(ddin[5]) == reg["ho.dd_in_sample_v2_pct"]
    dd = next(r for lab, r in by_label.items() if "at a later commit" in lab)
    assert num(dd[1]) == reg["ho.dd_queries"]
    assert num(dd[2]) == reg["ho.dd_distinct"]
    ex = next(r for lab, r in by_label.items() if "integrations-extras" in lab)
    assert num(ex[1]) == reg["ho.ex_queries"]
    assert num(ex[2]) == reg["ho.ex_distinct"]
    assert num(ex[3]) == reg["ho.ex_raw_pct"]
    assert num(ex[4]) == reg["ho.ex_v1_pct"]
    assert num(ex[5]) == reg["ho.ex_v2_pct"]
    assert num(ex[6]) == reg["ho.ex_residual"]


def test_the_in_sample_datadog_row_reproduces_the_parse_coverage_table():
    """The held-out table's comparator is the paper's own pipeline, or it is nothing.

    Section 4's table is written by notebook 00 over the corpus; this row is written by
    `scripts/heldout_eval.py` over the same queries. They are two programs reading one
    population, so every column has to agree --- and the residual of six is the tightest
    of them, because it is a count and not a rate rounded to a decimal place. If this
    fails, the held-out numbers beside it were produced by something that is not the
    pipeline the paper describes.
    """
    want = next(r for r in rows("parse_coverage")[1:] if r[0] == "datadog")
    got = next(r for r in rows("heldout_coverage")[1:]
               if "integrations-core" in r[0] and "the corpus's" in r[0])
    for col, what in ((1, "queries"), (2, "distinct"), (3, "raw"),
                      (4, "v1"), (5, "v2"), (6, "residual")):
        assert num(got[col]) == num(want[col]), (
            f"{what}: the held-out table says {got[col]}, "
            f"tab:parse-coverage says {want[col]}")


def test_the_disjoint_datadog_holdout_is_disjoint_and_large_enough(reg):
    """The extras holdout is the paper's only well-powered held-out kappa.

    Two things make it one, and neither may be assumed. It has to be disjoint: the
    fetcher drops any string whose instantiated form the corpus holds, so the evaluated
    population must be all-novel, and that is recomputed here from the two committed
    jsonl files rather than read back out of the manifest. And it has to be big enough
    that 96.7% means something at one decimal place, which the week-of-upstream-change
    holdout is not.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from qlport import corpus as c
    train = {c.normalise("datadog", q.text)
             for q in c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl")
             if q.lang == "datadog"}
    held = [q for q in c.read_queries(ROOT / "data" / "heldout" / "queries.jsonl")
            if q.source == "integrations-extras" and q.lang == "datadog"]
    leaked = [q for q in held if c.normalise("datadog", q.text) in train]
    assert not leaked, f"{len(leaked)} of the extras queries are in the training corpus"
    assert len(held) == reg["ho.ex_queries"] == reg["ho.ex_kappa_occurrences"]
    assert len({q.origin for q in held}) == reg["ho.ex_dashboards"]
    # Not `shared_distinct == 0 and novel_distinct == distinct`: in measure() those
    # are the same proposition written twice, and both read the cache this script
    # wrote. The leak check above is the one that recomputes.
    assert reg["ho.ex_shared_distinct"] == 0
    assert reg["ho.ex_kappa_occurrences"] > 1000, (
        "below about a thousand occurrences the prose cannot call this a rate; it "
        "would have to be reported the way the later-commit holdout is")


def test_the_held_out_refusals_are_all_catalogued_families(reg):
    """The claim that out-of-sample failure finds no new mode, as an assertion.

    This is the one thing a held-out kappa can establish that an in-sample one cannot,
    and section 7 leans on it: the shortfall is structural rather than specific to this
    corpus. The evidence is containment --- every gap code the disjoint repository
    raises is one the corpus already raises --- so if a fifth family ever appears here,
    the paragraph is wrong and has to be rewritten rather than re-rounded.
    """
    ho = _ho()
    out = set(ho["kappa_extras"]["gaps"])
    inside = set(ho["kappa_in_sample"]["gaps"])
    assert out <= inside, f"gap codes seen only out of sample: {sorted(out - inside)}"
    assert len(out) == 4, f"the prose names four: {sorted(out)}"
    # The prose says exactly which family the corpus raises and this repository does
    # not. `out <= inside` would still hold if a second one went unseen, or if a sixth
    # family appeared in sample, and the sentence would be wrong either way.
    assert inside - out == {"SCOPE_DISJUNCTION"}, (
        "the prose names one family the corpus raises and this repository does not: "
        f"{sorted(inside - out)}")
    k = ho["kappa_extras"]
    for name, n in k["gaps"].items():
        assert reg[f"ho.ex_gap_{name.lower()}"] == n
        assert reg[f"ho.ex_gap_occ_{name.lower()}"] == k["gaps_occ"][name]
    # The breakdown has to exhaust the refusals, in whichever unit the prose uses.
    # This is the assertion the outcome-sum line used to stand in for, and unlike that
    # one it is not an identity: `gaps` is keyed off the gap code and the outcome
    # counts off the outcome, so a refusal reaching the table with no code breaks it.
    assert sum(k["gaps"].values()) == k["by_outcome_distinct"]["bottom"] == 93
    assert sum(k["gaps_occ"].values()) == k["by_outcome_occ"]["bottom"] == 98
    # The mixture claim: which family leads, in sample and out.
    assert max(inside, key=ho["kappa_in_sample"]["gaps"].get) == "UNSUPPORTED_FUNCTION"
    assert max(k["gaps_occ"], key=k["gaps_occ"].get) == "PHI_QUANTILE"


def test_the_held_out_kappa_is_indistinguishable_from_the_in_sample_one(reg):
    """The headline comparison, with the in-sample side checked against notebook 03.

    `heldout_eval.py` recomputes kappa over the corpus with the same classify() the
    notebook uses, which makes it a control: if it does not return the notebook's
    96.6% the held-out figure beside it was produced by a different translator.

    What the paper claims about the two rates is that they cannot be told apart, so
    that is what is asserted. An earlier version of this test asserted the held-out
    rate was the higher one, which is a different claim and a worse one: the script's
    own docstring says a figure worse than the in-sample one is the finding, and a
    guard that reddens the build when the finding arrives is a guard against learning
    anything. It also encoded something the paper does not say, since by distinct
    string the held-out rate is already the lower of the two.
    """
    assert reg["ho.kappa_in_sample_pct"] == reg["nb03.kappa_pct"]
    assert reg["ho.kappa_in_sample_occurrences"] == reg["nb03.corpus_occurrences"]
    assert abs(reg["ho.ex_kappa_pct"] - reg["ho.kappa_in_sample_pct"]) <= 0.5, (
        "the abstract, the introduction, section 7 and section 9 all call the two "
        "rates indistinguishable; at this distance they are not, and those four "
        "sentences have to be rewritten rather than re-rounded")
    assert abs(reg["ho.ex_kappa_distinct_pct"]
               - reg["ho.kappa_in_sample_distinct_pct"]) <= 0.5
    # And the reason the paper may call a tenth of a point nothing: it is smaller than
    # the spread one dashboard produces. If the held-out population ever becomes tight
    # enough that this stops being true, "indistinguishable" needs re-arguing.
    lo, hi = reg["ho.ex_jackknife_low_pct"], reg["ho.ex_jackknife_high_pct"]
    ilo, ihi = (reg["ho.kappa_in_sample_jackknife_low_pct"],
                reg["ho.kappa_in_sample_jackknife_high_pct"])
    assert (hi - lo) > abs(reg["ho.ex_kappa_pct"] - reg["ho.kappa_in_sample_pct"]), (
        "the held-out rate no longer moves more under dropping one dashboard than it "
        "differs from the in-sample rate, so 'inside that spread' stops being true")
    assert max(lo, ilo) <= min(hi, ihi), (
        "the two jackknife spreads no longer overlap, which is a stronger separation "
        "than 'indistinguishable' survives")

    # The fit the Datadog grammar made to the corpus, in sample and out of it, each
    # taken from counts rather than from two already-rounded percentages.
    assert 0 < reg["ho.ex_gain_points"] < reg["ho.dd_in_sample_gain_points"]
    assert reg["ho.ex_v2_pct"] == reg["ho.dd_in_sample_v2_pct"], (
        "the prose says v2 does not fall out of sample, which is what makes the "
        "shrinking margin a fact about v1 rather than about overfitting")


def test_every_cluster_statistic_is_a_function_of_the_stored_tally(reg):
    """The whole jackknife block, recomputed from the one list it keeps.

    Everything the subsection says about what a tenth of a point is worth rests on
    eight cached scalars: a cluster-robust error, a binomial one, their ratio as a
    design effect, an effective cluster count, the share of the dispersion in the
    largest dashboard, the drop-one range, and a bootstrap standard deviation. The
    registry-to-run pin holds the paper to those scalars. Nothing held the scalars to
    anything, so a consistent edit of the cache and the prose was green, and two of
    them -- the exact errors -- were written by the script and read by nobody.

    So the per-dashboard tally is stored and all eight are derived from it here. The
    exact errors stop being dead weight: the design effect is their ratio squared, and
    asserting that against the registry's rounded pair is the assertion that had to be
    dropped last pass because 11.6 is not (1.1/0.3)^2.
    """
    j = _ho()["kappa_extras"]["jackknife"]
    clusters = [tuple(c) for c in j["clusters"]]
    m = len(clusters)
    assert m == j["dashboards"] == reg["ho.ex_dashboards"]
    n = sum(a for a, _ in clusters)
    t = sum(b for _, b in clusters)
    assert n == reg["ho.ex_queries"] and round(t / n * 100, 1) == reg["ho.ex_kappa_pct"]

    share = t / n
    resid = [b - share * a for a, b in clusters]
    s2 = sum(r * r for r in resid)
    s4 = sum(r ** 4 for r in resid)
    cluster_se = math.sqrt(s2 * m / (m - 1)) / n * 100
    srs_se = math.sqrt(share * (1 - share) / n) * 100
    assert cluster_se == pytest.approx(j["cluster_se_exact"])
    assert srs_se == pytest.approx(j["srs_se_exact"])
    assert round(cluster_se, 1) == reg["ho.ex_kappa_se_points"]
    assert round(srs_se, 1) == reg["ho.ex_kappa_srs_se_points"]
    # The design effect is the ratio of the *exact* errors, which is why they are kept.
    assert round((j["cluster_se_exact"] / j["srs_se_exact"]) ** 2, 1) == \
        reg["ho.ex_kappa_design_effect"]
    assert round(s2 * s2 / s4, 1) == reg["ho.ex_kappa_effective_dashboards"]
    assert round(max(r * r for r in resid) / s2 * 100, 1) == \
        reg["ho.ex_kappa_top_dashboard_pct"]

    # The drop-one range, which is the quantity the prose must not call the error:
    # the widest single-dashboard move is a point, and the error is 1.1.
    drop = [(t - b) / (n - a) * 100 for a, b in clusters if n - a]
    assert round(min(drop), 1) == j["low_pct"] and round(max(drop), 1) == j["high_pct"]
    assert max(abs(d - t / n * 100) for d in drop) < j["cluster_se_exact"], (
        "the largest drop-one move now exceeds the cluster-robust error, so the two "
        "are no longer different enough for the distinction the prose draws")

    # And the bootstrap, drawn from this same list in this same order.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "heldout_eval", ROOT / "scripts" / "heldout_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert j["bootstrap_draws"] == mod._BOOTSTRAP_DRAWS
    assert j["bootstrap_seed"] == mod._BOOTSTRAP_SEED
    boot = mod._cluster_bootstrap([list(c) for c in clusters])
    assert boot == pytest.approx(j["bootstrap_se_exact"])
    # The claim the comment below rests on: resampling and linearising agree here, so
    # the linearisation is doing no work and nothing need be said against it.
    assert abs(boot - j["cluster_se_exact"]) < 0.1 * j["cluster_se_exact"]


def test_the_fit_decompositions_are_reported_with_their_own_spread(reg):
    """Both fit decompositions are inside their own noise, and the paper must say so.

    For three drafts this subsection split the PromQL margin into a third that was
    corpus-specific and two thirds that were not, and called the Datadog one four
    fifths. Those are differences between two rates measured over strings that arrive
    a dashboard at a time, and nobody had asked what a dashboard-level jackknife puts
    on them. It puts 0.4 points on a 0.4-point difference and 1.9 on a 1.5-point one.
    Neither fraction was ever resolvable; both sentences were arithmetic on noise.

    So the claim the paper now makes is the negative one these data support: the
    margin falls out of sample by less than the spread either figure carries across
    dashboards, so this corpus does not separate the fit from the rest. If a later
    corpus makes a shrinkage larger than its spread, this test fails and the prose
    gets to say something it currently cannot.
    """
    run = _ho()
    # Every key spelled out. An earlier version built three of these six names by
    # slicing another one, which reads a neighbouring key rather than failing when a
    # name moves, and on this table would have read `ho.dd_shrink_points` --- a key
    # that does not exist --- had the slice not happened to be discarded.
    for lang, shrink, a, b, points_key, se_key, in_sample, held, held_eff, dashboards in (
            ("promql", "shrink_gf", "gain_gf_in_sample", "gain_gf_held",
             "ho.gf_shrink_points", "ho.gf_shrink_se_points",
             "ho.gf_in_sample_gain_points",
             "ho.gf_novel_gain_points", "ho.gf_novel_gain_effective_dashboards",
             "ho.gf_dashboards"),
            ("datadog", "shrink_dd", "gain_dd_in_sample", "gain_ex_held",
             "ho.ex_shrink_points", "ho.ex_shrink_se_points",
             "ho.dd_in_sample_gain_points",
             "ho.ex_gain_points", "ho.ex_gain_effective_dashboards",
             "ho.ex_dashboards")):
        # The difference of the two unrounded margins, not of the two the paper
        # prints. On the Datadog pair those differ: 7.0137 less 5.5599 is 1.4538,
        # which rounds to 1.5, while the printed 7.0 less the printed 5.6 is 1.4.
        # An earlier version of this test asserted the second and so agreed with the
        # error it was added to catch; the version after that asserted only the first
        # and stopped noticing that a reader who subtracts the paper's own two
        # figures gets a third number. Both are checked now, and where they disagree
        # the prose has to say which one it means.
        exact = run[a]["gain"] - run[b]["gain"]
        points, se = reg[points_key], reg[se_key]
        assert points == round(exact, 1), (
            f"{lang}: the registry reports a shrinkage of {points} points where the "
            f"run's two margins differ by {exact:.4f}")
        # The cache stores the shrinkage beside the pair it is a difference of, and
        # `figures()` copies both into the registry untouched, so the registry-to-run
        # pin cannot see them disagree with each other. Both are re-derived here: the
        # spread is the two margins' spreads in quadrature, which is what makes it
        # larger than either, and that is the step the sentences about it rest on.
        assert run[shrink]["points"] == round(exact, 1), (
            f"{lang}: the run stores a shrinkage of {run[shrink]['points']} beside two "
            f"margins differing by {exact:.4f}")
        quad = math.hypot(run[a]["gain_se"], run[b]["gain_se"])
        assert run[shrink]["se_points"] == round(quad, 1), (
            f"{lang}: the run stores {run[shrink]['se_points']} points of spread on the "
            f"shrinkage where its two margins give {quad:.4f} in quadrature")
        naive = round(reg[in_sample] - reg[held], 1)
        if naive != points:
            assert RECONCILED in flat(published(HELDOUT)), (
                f"{lang}: the paper prints {reg[in_sample]} and {reg[held]}, whose "
                f"difference is {naive}, and calls the fall {points}. One of those "
                f"has to be reconciled where the reader meets them")
        assert abs(points) <= se, (
            f"{lang}: the margin now shrinks by {points} points against {se} of "
            f"spread, which these data may be able to separate. The subsection says "
            f"the shrinkage is smaller than the spread either margin carries, and "
            f"that sentence is now wrong")
        # What is NOT asserted here, and the reason: that the held-out margin
        # clears twice its spread. An earlier version did, and it was a test of an
        # impossible null twice over. v2's productions are a superset of v1's, so
        # the margin cannot be negative and no population could have rejected zero.
        # And both populations are censuses of their repositories, so the margin is a
        # fact about those repositories and needs no interval at all. What the spread
        # is for is the negative claim above.
        #
        # An earlier version of this comment gave a third reason -- that the spread
        # rests on an effective two or three dashboards, where a two-sigma reading is
        # not a reading. A 20,000-draw cluster bootstrap of the held-out kappa gives a
        # standard deviation of 1.091 against the linearised 1.095 --- the test above
        # redraws it --- so the normal interval is close to calibrated and that reason
        # blamed a defect these data do not show. The two above do not need it.
        assert reg[held] > 0, f"{lang}: v2 no longer parses more than v1 out of sample"
        assert reg[held_eff] < reg[dashboards] / 10, (
            f"{lang}: the dashboard-level spread now rests on {reg[held_eff]} of "
            f"{reg[dashboards]} dashboards in effect, which is no longer the "
            f"concentration the subsection describes. If the dispersion has spread "
            f"out, the spread is worth more than the prose currently says it is")


def test_the_two_kappas_are_closer_than_the_spread_that_separates_them(reg):
    """Why the paper may call the held-out and in-sample rates indistinguishable.

    This test used to be called one about the third digit, which was wrong on its
    face: in 96.7 the first decimal is the third digit, so the name denied of itself
    what the body asserted. What the body actually checks is narrower and worth
    keeping --- that the gap between the two rates is smaller than the spread a
    dashboard-level resampling puts on them, in both units.

    Occurrences repeat one query shape across a dozen metrics inside one dashboard,
    so a binomial error over 3,065 of them describes a sample this paper does not
    have. The clustered figure is the one to read, and the two rates sit well inside
    it.
    """
    srs, se = reg["ho.ex_kappa_srs_se_points"], reg["ho.ex_kappa_se_points"]
    assert se > srs, (
        "clustering by dashboard no longer widens the error, which would mean the "
        "occurrences are independent after all and the paragraph arguing they are "
        "not has to go")
    assert se > abs(reg["ho.ex_kappa_pct"] - reg["ho.kappa_in_sample_pct"]), (
        "the two rates are further apart than one clustered standard error, so "
        "'indistinguishable' is no longer what the arithmetic says")
    assert se > abs(reg["ho.ex_kappa_distinct_pct"]
                    - reg["ho.kappa_in_sample_distinct_pct"]), (
        "the same, over distinct forms, where the gap runs the other way")
    # The in-sample side clusters too, which is why section 7 may generalise the
    # remark to every kappa in the paper rather than to this one. `> 0` was what
    # this said, which a standard error always is.
    assert reg["ho.kappa_in_sample_se_points"] >= 0.5, (
        "the in-sample kappa's clustered error has collapsed, and section 7's "
        "remark about every kappa in this paper rests on this one being like the "
        "other")
    # And what that spread rests on, which is the reason the subsection reports it
    # as a range the figure moves over rather than as an interval. Both populations
    # were read whole --- no dashboard in integrations-extras was unreadable --- so
    # there is no sampling error to estimate; the dispersion across dashboards is a
    # stability statement, and it is concentrated.
    assert reg["ho.ex_unreadable"] == 0, (
        "a dashboard in integrations-extras could not be read, so the population is "
        "no longer the census the subsection says it is")
    assert reg["ho.ex_kappa_effective_dashboards"] < 5, (
        "the held-out dispersion is no longer concentrated in a handful of "
        "dashboards, and the sentence saying it is has to go")
    assert (reg["ho.ex_kappa_effective_dashboards"]
            < reg["ho.kappa_in_sample_effective_dashboards"]), (
        "the held-out side is supposed to be the thinner of the two; if the corpus "
        "side is now thinner, the comparison between them reads backwards")
    assert reg["ho.ex_kappa_top_dashboard_pct"] > 50, (
        "no single dashboard carries most of the held-out dispersion any more")


def test_the_two_vocabularies_are_not_quoted_under_one_name(reg):
    """The corpus's 28 features and the translator's 25 are different ceilings.

    The subsection once said "one feature beyond the 25 ... so the 25 features reach
    99.9%". Both halves were measured against the corpus's 28. Against the 25 that
    actually bound kappa the figure is 97.3% with four features outside, so the
    sentence overstated the reachable ceiling by 2.6 points. Each ceiling is now
    registered under the vocabulary it was measured against, and the in-sample one is
    a control: it has to come back as notebook 03's.
    """
    assert reg["ho.dd_in_sample_sstar_size"] == reg["nb03.feature_set_size"] == 25
    assert reg["ho.dd_in_sample_sstar_pct"] == reg["nb03.feature_ceiling_pct"], (
        "the script's in-sample S* ceiling must reproduce notebook 03's, or the "
        "out-of-sample figure beside it was measured against a different set")
    assert reg["ho.ex_vocabulary_size"] > reg["ho.ex_sstar_size"], (
        "S* is a subset of what the corpus exhibits; if they coincide the two "
        "ceilings stop being different quantities and the paragraph can be cut")
    assert reg["ho.ex_sstar_pct"] < reg["ho.ex_ceiling_pct"]
    assert reg["ho.ex_sstar_novel_features"] > reg["ho.ex_novel_features"]
    # Occurrences outside a vocabulary, counted once per string. The per-feature
    # counts in `novel_features` do not sum to this and must not be quoted as if
    # they did: against S* they total 102 for 82 strings' worth of occurrences.
    e = _ho()["integrations-extras"]
    assert (e["translator_vocabulary"]["novel_occurrences"]
            == reg["ho.ex_sstar_outside_occurrences"] == 82)
    assert (e["corpus_vocabulary"]["novel_occurrences"]
            == reg["ho.ex_ceiling_outside_occurrences"] == 2)
    assert sum(e["translator_vocabulary"]["novel_features"].values()) > 82, (
        "if these ever agree the double-counting this guard exists for has gone, "
        "and the comment above can go with it")


def test_the_one_out_of_sample_parse_failure_is_the_construct_the_prose_names(reg):
    """"stddev: as a space aggregator" --- the identity, not just the count.

    `ho.ex_novel_features == 1` pins how many features the corpus vocabulary lacks
    and says nothing about which, so the sentence naming the construct could go stale
    while the count stayed right.
    """
    e = _ho()["integrations-extras"]
    assert list(e["novel_features"]) == ["AGG_OTHER"], (
        "the prose names the one out-of-sample feature; this is not it")
    assert e["blockers"] == {"COLON": 2}, (
        "the prose says the two unparsed strings are one construct")
    assert reg["ho.ex_residual"] == 2 and reg["ho.ex_unparsed"] == 2, (
        "these count different things --- distinct strings v2 cannot parse, and "
        "refused occurrences --- and agree only while each stddev: string occurs "
        "once; asserting they are equal to each other would be a coincidence")
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from qlport import corpus as c
    corpus_stddev = [q.text for q in c.read_queries(ROOT / "data" / "corpus"
                                                    / "queries.jsonl")
                     if q.lang == "datadog" and q.text.lstrip().startswith("stddev:")]
    assert not corpus_stddev, (
        "the corpus now uses stddev: as an aggregator, so it is no longer the "
        f"construct the holdout found: {corpus_stddev[:2]}")


def test_the_extras_holdout_names_its_commit_and_its_own_size(reg):
    """What integrations-core's temporal holdout gets, the disjoint one gets too.

    `data/heldout/sources/` is gitignored, so the shipped-dashboard count, the
    at-commit occurrences and the overlap that was dropped are recorded in the
    manifest and nowhere else. Without this they can drift from the prose in silence.
    """
    man = json.loads((ROOT / "data" / "heldout" / "MANIFEST.json").read_text())
    xm = man["integrations-extras"]
    assert xm["repo"] == "DataDog/integrations-extras"
    assert len(xm["commit"]) == 40
    assert xm["commit"] != man["integrations-core"]["commit"]
    # Read and shipped are separate keys now, and they are equal here only because
    # nothing in this repository is unreadable. The corpus side, checked below, is
    # where they come apart.
    assert xm["dashboards"] == reg["ho.ex_dashboards_read"]
    assert xm["files"] == reg["ho.ex_dashboards_at_commit"]
    assert xm["files"] - xm["dashboards"] == reg["ho.ex_unreadable"] == 0
    cm = man["integrations-core"]
    assert cm["dashboards"] == reg["ho.dd_corpus_dashboards_read"]
    assert cm["files"] == reg["ho.dd_corpus_dashboards_at_commit"]
    assert cm["files"] - cm["dashboards"] == reg["ho.dd_corpus_unreadable"] == 9
    assert xm["queries_at_commit"] == reg["ho.ex_queries_at_commit"]
    assert xm["queries_held_out"] == reg["ho.ex_queries"]
    assert xm["shared_with_corpus"] == reg["ho.ex_shared_forms"]
    # "3,073 occurrences, of which 3,065 ... seven such forms, eight occurrences".
    assert reg["ho.ex_queries_at_commit"] - reg["ho.ex_queries"] == 8
    assert reg["ho.ex_dashboards"] <= reg["ho.ex_dashboards_read"]
    # The formula layer is carried in the committed file and scored nowhere; the
    # prose says so, and these are the numbers it says it with.
    assert xm["queries_formula_held_out"] == reg["ho.ex_formula_queries"] == 265
    assert xm["distinct_formula_held_out"] == reg["ho.ex_formula_distinct"] == 166


def test_the_template_share_is_measured_on_both_sides(reg):
    """Section 7's house-style caveat, now with a number against it.

    The in-sample side is a control: the script counts template variables by notebook
    00's test, so it has to return notebook 00's figure.
    """
    assert reg["ho.dd_in_sample_template_pct"] == reg["nb00.datadog_with_variable_pct"]
    assert reg["ho.ex_template_pct"] < reg["ho.dd_in_sample_template_pct"], (
        "section 7 says the corpus is the more heavily templated of the two, which "
        "is the direction its own caveat feared")


def test_the_grafana_comparator_reproduces_the_promql_residual(reg):
    """The claim the Datadog check is said to be a stricter version of.

    The subsection cites the Grafana comparator's reproduction of the PromQL residual
    as precedent, and nothing checked it.
    """
    promql = next(r for r in rows("parse_coverage")[1:] if r[0] == "promql")
    assert reg["ho.gf_in_sample_residual"] == num(promql[6]) == 58
    assert reg["ho.gf_in_sample_distinct"] < num(promql[2]), (
        "the Grafana comparator is a restriction of that row, not the row; if it "
        "ever equals it, the paragraph explaining why they differ is wrong")


def test_the_staircase_figure_the_ledger_cannot_see(reg):
    """The queries the first six Datadog features complete, recomputed.

    This is 45, and since the held-out run so is ho.ex_gap_phi_quantile. The figure
    ledger accounts for a prose numeral by magnitude, so that unrelated key now
    licenses this one, and ALLOWED_IN's scope is consulted only for figures the
    registry cannot account for -- which this no longer is. The table that generates
    the staircase is the real source, so the check belongs here.
    """
    body = rows("implementation_order")[1:]
    assert sum(num(r[4]) for r in body[:6]) == 45, (
        "eval_staircase.tex says the first six features complete 45 queries")


def test_the_temporal_holdout_names_both_commits_and_its_own_size(reg):
    """A later commit of the same repository is only a holdout if it is a later commit.

    One week of upstream change is what it gave, and the resulting set is small enough
    that the paper reports it as a check rather than an estimate. Both facts have to
    stay attached to the number: the two commits that bracket it, and the count.
    """
    man = json.loads((ROOT / "data" / "heldout" / "MANIFEST.json").read_text())
    ic = man["integrations-core"]
    assert len(ic["commit"]) == 40 and len(ic["corpus_commit"]) == 40
    assert ic["commit"] != ic["corpus_commit"], "that is the same commit"

    corpus_md = (ROOT / "data" / "corpus" / "SOURCES.md").read_text()
    section = corpus_md.split("## integrations-core", 1)[1]
    assert ic["corpus_commit"] in section, (
        "the held-out manifest names a base commit the corpus was not extracted from")

    assert reg["ho.dd_kappa_occurrences"] == reg["ho.dd_queries"] == ic[
        "queries_datadog_held_out"], (
        "the manifest counts the formula layer too, so these agree only while the week "
        "of upstream change added no formula queries; if it ever does, the evaluation "
        "and the manifest are counting different populations")
    assert reg["ho.dd_kappa_occurrences"] < 1000, (
        "if the temporal holdout ever gets big enough to estimate kappa, the prose "
        "that calls it underpowered has to be rewritten")

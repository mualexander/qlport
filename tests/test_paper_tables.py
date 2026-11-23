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

from texsource import builds, flat, rendered, table

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "paper" / "tables"
REGISTRY = ROOT / "paper" / "quoted_figures.json"
HELDOUT = ROOT / "paper" / "sections" / "moved" / "eval_heldout.tex"
GRAMMARS = ROOT / "paper" / "sections" / "04_grammars.tex"

# Rounding a difference and differencing two rounded figures are not the same
# operation, and at one decimal place they disagree about three of the six trios the
# paper prints: the Datadog fall (the rates give 1.4538, the printed 7.0 and 5.6 give
# 1.4), the Datadog held-out margin (5.5599 against 99.9 less 94.4) and the design
# effect (11.644 against (1.1/0.3)^2 = 13.4). The paper reports the first of each
# pair, so it has to say once, in each build, that its figures are taken from the
# counts and not from each other.
#
# Two sentences rather than one because there are two PDFs and the appendix is in a
# different place in each; neither is the other's words, which `paper/body.tex`'s
# convention requires of a summary. Each is required only while some trio actually
# fails to reconcile, and each must say enough to be the statement: a previous version
# of this check was a single literal in one file, and setting it to the word
# "difference" left the suite green.
STAIRCASE = ROOT / "paper" / "sections" / "moved" / "eval_staircase.tex"
RECONCILED = (
    (GRAMMARS, "taken from the counts and not from each other"),
    (HELDOUT, "quotient of counts and every margin a difference of two of them, "
              "taken before either is rounded"),
    (STAIRCASE, "each a mean over $1{,}000$ replicates rather than a quotient of "
                "counts, with the gap between them taken from the means and not from "
                "these two printed figures"),
)


@pytest.fixture(scope="module")
def reg() -> dict:
    return json.loads(REGISTRY.read_text())


def rows(name: str) -> list[list[str]]:
    """A generated `tabular` as a list of cell lists, rules and preamble dropped."""
    text = table(TABLES / f"{name}.tex")
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
    for body in builds(ROOT / "paper" / "sections" / "07_evaluation.tex"):
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
    for body in builds(ROOT / "paper" / "sections" / "moved" / "eval_unsupported.tex"):
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
    """§8 calls the staircase the paper's practical result, so it must not be a fit.

    The three figures are taken from the exact means and not from the two curves after
    rounding, and the gap is pooled over five seeds. At one seed it read 1.0; the five
    per-seed means run 1.00 to 1.23 and the shipped one was the lowest of them, so the
    printed tenth of a point was a seed artefact of the same kind, and at the same
    scale, as the bootstrap figure corrected in the pass before this one.

    The spread stored beside it is the Monte-Carlo error of the pooled mean, 0.08, and
    not the range of the per-seed means: a range over a handful of seeds bounds the
    next seed by luck, which is how last pass's bootstrap spread came to be quoted as
    an uncertainty on something it was not an uncertainty on. The range is kept as a
    tripwire and asserted against what the replicate spread predicts for it.
    """
    s = _cv()["datadog"]["staircase"]
    assert s["at"] == 16 and s["reps"] == 1000 and len(s["seeds"]) == 5
    assert reg["cv.dd_stair_train_16_pct"] == round(s["train_at"], 1)
    assert reg["cv.dd_stair_test_16_pct"] == round(s["test_at"], 1)
    assert reg["cv.dd_stair_gap_16_pct"] == round(s["gap_at"], 1)
    assert s["gap_at"] == pytest.approx(s["train_at"] - s["test_at"], abs=1e-9)
    assert 0 <= s["gap_at"] < 5, (
        "a build order chosen on one half should not be worth much less on the other")
    assert s["train"][0] < s["train"][8] < s["train"][16], "the staircase should rise"

    # The Monte-Carlo error of the mean, from the replicate spread and the count.
    assert reg["cv.dd_stair_gap_16_mc_se"] == round(
        s["gap_at_replicate_sd"] / s["reps"] ** 0.5, 2)
    assert reg["cv.dd_stair_gap_16_replicate_sd"] == round(s["gap_at_replicate_sd"], 1)
    # Each seed's own mean is 200 replicates, so its error is sqrt(5) times the pooled
    # one and the range of five draws should sit inside a few of those. Wider than that
    # and the seeds are not interchangeable, which is the thing this pooling assumes.
    per_seed_se = s["gap_at_replicate_sd"] / (s["reps"] / len(s["seeds"])) ** 0.5
    assert s["gap_at_seed_range"] <= 4 * per_seed_se, (
        f"the five per-seed means span {s['gap_at_seed_range']:.3f} where one seed's "
        f"own Monte-Carlo error is {per_seed_se:.3f}")
    assert reg["cv.dd_stair_gap_16_seed_range"] == round(s["gap_at_seed_range"], 2)
    # And the figure the prose has to carry: a third of individual half-splits show no
    # gap at all, which is why the mean is not a statement about one reader's split.
    assert reg["cv.dd_stair_gap_16_share_not_positive_pct"] == round(
        s["gap_at_share_not_positive"] * 100)
    assert 0.2 < s["gap_at_share_not_positive"] < 0.5, (
        "the share of replicates with no gap has moved far enough that the sentence "
        "about what one split would show needs rewriting")


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


def _require_the_rounding_note(because: str) -> None:
    """Both builds must carry the note, in their own words, and say enough to be it.

    This said "both builds" and read the comment-stripped source, which is neither of
    them. Parking the note behind `\\iffalse` and printing "are each read off the
    other" in its place left the note in no PDF, the paper asserting the opposite, and
    nothing but the build tripwire failing. The guard added to close a defect class was
    an instance of it; `builds` is now the only way to ask what a document says.
    """
    for path, want in RECONCILED:
        assert len(want) > 40 and "counts" in want, (
            "RECONCILED has been weakened to something that is not the statement")
        for text in builds(path):
            assert want in flat(text), (
                f"{because}, and {path.name} does not say its figures come from the "
                f"counts rather than from each other")


def test_every_printed_trio_either_reconciles_or_is_accounted_for(reg):
    """Three figures, two of which the reader will difference to get the third.

    The paper prints six such trios and three of them do not reconcile at one decimal
    place. That is ordinary --- a difference of counts is not the difference of two
    rounded quotients --- and it is also exactly the shape of a defect this suite has
    had twice: a guard asserted the printed subtraction and agreed with the error it
    was added to catch, and the guard that replaced it asserted only the exact
    arithmetic and stopped noticing that the reader gets a third number.

    So every trio is enumerated here and the disagreement is counted rather than
    tolerated silently. While any of them disagrees, the note has to be in the paper.
    If a future run makes all six reconcile, this fails and the note comes out.
    """
    diffs = (
        ("the Datadog in-sample margin", "ho.dd_in_sample_v2_pct",
         "ho.dd_in_sample_v1_pct", "ho.dd_in_sample_gain_points"),
        ("the Datadog held-out margin", "ho.ex_v2_pct", "ho.ex_v1_pct",
         "ho.ex_gain_points"),
        ("the Datadog fall", "ho.dd_in_sample_gain_points", "ho.ex_gain_points",
         "ho.ex_shrink_points"),
        ("the PromQL in-sample margin", "ho.gf_in_sample_v2_pct",
         "ho.gf_in_sample_v1_pct", "ho.gf_in_sample_gain_points"),
        ("the PromQL held-out margin", "ho.gf_v2_pct", "ho.gf_v1_pct",
         "ho.gf_novel_gain_points"),
        ("the PromQL fall", "ho.gf_in_sample_gain_points", "ho.gf_novel_gain_points",
         "ho.gf_shrink_points"),
        ("the ceiling overstatement", "ho.ex_ceiling_pct", "ho.ex_sstar_pct",
         "ho.ex_ceiling_overstatement_points"),
        ("the cross-validated staircase gap", "cv.dd_stair_train_16_pct",
         "cv.dd_stair_test_16_pct", "cv.dd_stair_gap_16_pct"),
    )
    unreconciled = [name for name, a, b, c in diffs
                    if a in reg and b in reg and round(reg[a] - reg[b], 1) != reg[c]]
    missing = [name for name, a, b, c in diffs if not (a in reg and b in reg)]
    assert not missing, f"these trios name keys the registry does not have: {missing}"
    # The one that is a ratio rather than a difference.
    deff = round((reg["ho.ex_kappa_se_points"]
                  / reg["ho.ex_kappa_srs_se_points"]) ** 2, 1)
    if deff != reg["ho.ex_kappa_design_effect"]:
        unreconciled.append("the design effect")
    # And the two that are sums in quadrature.
    for name, a, b, c in (
            ("the Datadog fall's spread", "ho.ex_gain_se_points",
             "ho.dd_in_sample_gain_se_points", "ho.ex_shrink_se_points"),
            ("the PromQL fall's spread", "ho.gf_novel_gain_se_points",
             "ho.gf_in_sample_gain_se_points", "ho.gf_shrink_se_points")):
        if round(math.hypot(reg[a], reg[b]), 1) != reg[c]:
            unreconciled.append(name)
    assert unreconciled, (
        "every printed trio now reconciles, so the rounding note in section 4 and in "
        "eval_heldout is telling the reader about a problem the paper no longer has")
    _require_the_rounding_note("; ".join(unreconciled) + " do not reconcile")


def test_every_cluster_statistic_is_a_function_of_the_stored_tally(reg):
    """Both jackknife blocks, recomputed from the one list each of them keeps.

    Everything the subsection says about what a tenth of a point is worth rests on
    eight cached scalars per side: a cluster-robust error, a binomial one, their ratio
    as a design effect, an effective cluster count, the share of the dispersion in the
    largest dashboard, the drop-one range, and a bootstrap standard deviation. The
    registry-to-run pin holds the paper to those scalars. Nothing held the scalars to
    anything, so a consistent edit of the cache and the prose was green, and two of
    them -- the exact errors -- were written by the script and read by nobody.

    So the per-dashboard tally is stored and all eight are derived from it here. The
    exact errors stop being dead weight: the design effect is their ratio squared, and
    asserting that against the registry's rounded pair is the assertion that had to be
    dropped two passes ago because 11.6 is not (1.1/0.3)^2.

    Both sides, because the first version of this test read `kappa_extras` only. The
    in-sample block stores a tally too, and the paper quotes its scalars -- the 0.6,
    the 9.6 effective dashboards that the held-out 2.0 is compared against, the
    96.6-to-97.0 range -- so moving a refusal between two in-sample dashboards changed
    the dispersion the paper describes and passed. Guarding one of two halves is the
    defect this file keeps finding, one level down.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "heldout_eval", ROOT / "scripts" / "heldout_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = _ho()

    for block, dashboards, occurrences, pct, se, srs, deff, eff, top in (
            ("kappa_extras", "ho.ex_dashboards", "ho.ex_queries", "ho.ex_kappa_pct",
             "ho.ex_kappa_se_points", "ho.ex_kappa_srs_se_points",
             "ho.ex_kappa_design_effect", "ho.ex_kappa_effective_dashboards",
             "ho.ex_kappa_top_dashboard_pct"),
            ("kappa_in_sample", "ho.dd_in_sample_dashboards",
             "ho.kappa_in_sample_occurrences", "ho.kappa_in_sample_pct",
             "ho.kappa_in_sample_se_points", None, None,
             "ho.kappa_in_sample_effective_dashboards", None)):
        j = run[block]["jackknife"]
        clusters = [tuple(c) for c in j["clusters"]]
        m = len(clusters)
        assert m == j["dashboards"] == reg[dashboards], block
        n = sum(a for a, _ in clusters)
        t = sum(b for _, b in clusters)
        assert n == reg[occurrences], block
        assert round(t / n * 100, 1) == reg[pct], block

        share = t / n
        resid = [b - share * a for a, b in clusters]
        s2 = sum(r * r for r in resid)
        s4 = sum(r ** 4 for r in resid)
        cluster_se = math.sqrt(s2 * m / (m - 1)) / n * 100
        srs_se = math.sqrt(share * (1 - share) / n) * 100
        assert cluster_se == pytest.approx(j["cluster_se_exact"]), block
        assert srs_se == pytest.approx(j["srs_se_exact"]), block
        assert round(cluster_se, 1) == reg[se], block
        assert round(srs_se, 1) == j["srs_se_pct"], block
        # The design effect is the ratio of the *exact* errors, which is why they are
        # kept: at one decimal place the printed pair gives 13.4 against 11.6.
        assert round((j["cluster_se_exact"] / j["srs_se_exact"]) ** 2, 1) == \
            j["design_effect"], block
        assert round(s2 * s2 / s4, 1) == reg[eff], block
        assert round(max(r * r for r in resid) / s2 * 100, 1) == \
            j["top_cluster_share_pct"], block
        if srs:
            assert j["srs_se_pct"] == reg[srs]
        if deff:
            assert j["design_effect"] == reg[deff]
        if top:
            assert j["top_cluster_share_pct"] == reg[top]

        # The drop-one range, which is the quantity the prose must not call the error.
        drop = [(t - b) / (n - a) * 100 for a, b in clusters if n - a]
        assert round(min(drop), 1) == j["low_pct"], block
        assert round(max(drop), 1) == j["high_pct"], block
        assert max(abs(d - share * 100) for d in drop) < j["cluster_se_exact"], (
            f"{block}: the largest drop-one move now exceeds the cluster-robust "
            "error, so the two are no longer different enough for the distinction "
            "the prose draws between them")
        # `worst_dashboard_refusals` is recoverable from the tally; the path beside it
        # is not, is in no registry key and is quoted nowhere, so it is not asserted.
        assert max(a - b for a, b in clusters) == j["worst_dashboard_refusals"], block

        # And the bootstrap, drawn from this same list in this same order.
        assert j["bootstrap_draws"] == mod._BOOTSTRAP_DRAWS, block
        assert j["bootstrap_seeds"] == list(mod._BOOTSTRAP_SEEDS), block
        boot = mod._cluster_bootstrap([list(c) for c in clusters])
        assert boot["se_exact"] == pytest.approx(j["bootstrap_se_exact"]), block
        assert boot["se_spread"] == pytest.approx(j["bootstrap_se_spread"]), block
        # A single seed's third decimal is not a measurement. The stored figure is the
        # mean over five, and the spread across them has to stay small enough that the
        # mean is worth a decimal place at all -- which is the check that the version
        # of this quoting one seed's 1.0908 against a linearised 1.0950, as evidence
        # that the two agreed, could not make. That seed was the highest of eight.
        assert j["bootstrap_se_spread"] < 0.05 * j["bootstrap_se_exact"], block
        # Magnitude: resampling and linearising agree, which is all `jackknife`'s
        # comment now claims of them.
        assert abs(boot["se_exact"] - j["cluster_se_exact"]) \
            < 0.1 * j["cluster_se_exact"], block

    # Shape: they do not agree, and that is the half the comment got wrong twice. The
    # replicate distribution is left-skewed on both sides, and its interval is
    # displaced from the normal one further where the dispersion is more concentrated
    # --- which is what an effective count of two predicts and what the retracted
    # worry was about. Asserted as a comparison between the two sides, because that is
    # the form the claim takes and neither number alone means anything.
    def displacement(block: str) -> float:
        j = run[block]["jackknife"]
        cl = j["clusters"]
        n = sum(a for a, _ in cl)
        point = sum(b for _, b in cl) / n * 100
        se = j["cluster_se_exact"]
        lo, hi = point - 1.96 * se, point + 1.96 * se
        return max(abs(j["bootstrap_pct_2_5"] - lo),
                   abs(j["bootstrap_pct_97_5"] - hi)) / se

    # Measured today: skew -0.61 against -0.31, and the furthest edge displaced by
    # 0.33 of a standard error against 0.19, a ratio of 1.8.
    assert run["kappa_extras"]["jackknife"]["bootstrap_skew"] < -0.4
    assert run["kappa_in_sample"]["jackknife"]["bootstrap_skew"] < 0
    assert displacement("kappa_extras") > 1.5 * displacement("kappa_in_sample"), (
        "the held-out interval is no longer displaced further than the in-sample one, "
        "so the concentration costs nothing in shape and `jackknife`'s comment about "
        "the tails is now the wrong way round")


def test_every_margin_spread_is_a_function_of_its_stored_replicates(reg):
    """The four gain standard errors, re-derived from the jackknife behind each.

    `jackknife` stores its per-dashboard tally with a comment saying that a cached
    number nothing can recompute is a number on the author's word, and the eight
    kappa scalars are derived from it next door. `gain_spread`, four hundred lines
    above it in the same file, returned bare scalars: the four margins' standard
    errors were written by the script and read by nothing but a `math.hypot` of two of
    them. Setting the in-sample Datadog one to zero against a printed 0.9 passed every
    statistics guard in this suite.

    Those four, and the two quadratures they make, carry the central hedge of section
    4, of section 7 and of this appendix -- that neither fall is larger than the
    spread it inherits. So the replicates are stored and everything is derived here.
    """
    run = _ho()
    for key in ("gain_dd_in_sample", "gain_ex_held",
                "gain_gf_in_sample", "gain_gf_held"):
        block = run[key]
        reps = block["replicates"]
        n = len(reps)
        assert n == block["dashboards"], (
            f"{key} stores {n} replicates for {block['dashboards']} dashboards")
        assert reps == sorted(reps), f"{key}'s replicates are not in sorted order"
        mean = sum(reps) / n
        # The delete-one-cluster jackknife, as `gain_spread` computes it: each
        # replicate drops one dashboard's exclusively-held strings and recomputes.
        var = (n - 1) / n * sum((r - mean) ** 2 for r in reps)
        se = math.sqrt(var)
        assert block["gain_se"] == pytest.approx(se, abs=1e-9), (
            f"{key} stores a standard error of {block['gain_se']} where its own "
            f"{n} replicates give {se:.6f}")
        assert block["gain_se_points"] == round(se, 1), (
            f"{key} prints {block['gain_se_points']} where its replicates give "
            f"{se:.4f}")
        # And the concentration, the same quantity the kappa block reports: how many
        # dashboards the spread is really resting on.
        dev = [r - mean for r in reps]
        s2 = sum(d * d for d in dev)
        s4 = sum(d ** 4 for d in dev)
        assert block["effective_clusters"] == round(s2 * s2 / s4, 1), (
            f"{key} stores an effective count of {block['effective_clusters']} "
            f"against {s2 * s2 / s4:.3f} from its replicates")

    # The two quadratures, from the re-derived errors rather than from the cache, so
    # that an edit to a stored `gain_se` cannot survive by being consistent with them.
    for a, b, shrink, se_key in (
            ("gain_dd_in_sample", "gain_ex_held", "shrink_dd", "ho.ex_shrink_se_points"),
            ("gain_gf_in_sample", "gain_gf_held", "shrink_gf",
             "ho.gf_shrink_se_points")):
        quad = math.hypot(*(_jackknife_se(run[k]["replicates"]) for k in (a, b)))
        assert run[shrink]["se_points"] == round(quad, 1) == reg[se_key], (
            f"{shrink}: the run stores {run[shrink]['se_points']} and the registry "
            f"{reg[se_key]} where the two sets of replicates give {quad:.4f}")


def _jackknife_se(reps: list[float]) -> float:
    """A delete-one-cluster standard error, from the replicates alone."""
    n = len(reps)
    mean = sum(reps) / n
    return math.sqrt((n - 1) / n * sum((r - mean) ** 2 for r in reps))


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
            _require_the_rounding_note(
                f"{lang}: the paper prints {reg[in_sample]} and {reg[held]}, whose "
                f"difference is {naive}, and calls the fall {points}")
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
        # not a reading. It was then retracted on the strength of a bootstrap standard
        # deviation agreeing with the linearised one, which was the wrong evidence:
        # two standard deviations agreeing says nothing about the tails. Measured
        # properly, the width agrees to about a per cent and the 95 per cent interval
        # is displaced from the normal one by a third of a standard error, against a
        # sixth in sample, with twice the skew. So the original worry was about the
        # right thing. It is still not a reason this test needs, because the two above
        # do not depend on reading any interval at all -- which is the point: a reason
        # retracted for the wrong reason is not a reason restored.
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
    # The corpus side's census comes from the corpus's own provenance note, not from
    # this manifest, whose integrations-core entry is the TEMPORAL HOLDOUT at a later
    # commit. They agree today and the test below says why that is not good enough.
    corpus_md = (ROOT / "data" / "corpus" / "SOURCES.md").read_text()
    section = corpus_md.split("## integrations-core", 1)[1].split("\n## ", 1)[0]
    for field, key in (("files", "ho.dd_corpus_dashboards_at_commit"),
                       ("dashboards", "ho.dd_corpus_dashboards_read"),
                       ("unreadable", "ho.dd_corpus_unreadable")):
        want = int(re.search(rf"- {field}: `(\d+)`", section).group(1))
        assert want == reg[key], f"{key} is {reg[key]} where SOURCES.md says {want}"
    assert (reg["ho.dd_corpus_dashboards_at_commit"]
            - reg["ho.dd_corpus_dashboards_read"]) == reg["ho.dd_corpus_unreadable"]
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


def test_the_corpus_census_is_not_read_off_the_holdout(reg):
    """The two integrations-core commits agree on their file counts, by luck.

    The corpus was extracted at one commit of that repository and the temporal holdout
    fetched at a later one, and this subsection's whole argument is that those are two
    populations. Both ship 534 dashboard files with the same 9 unreadable, so for a
    commit the corpus side's census was read off the holdout's manifest entry and
    agreed digit for digit. The agreement is contingent --- four of the 534 files
    differ in content between the commits --- so this asserts the two commits really
    are different, which is what makes reading one for the other an error rather than
    a shortcut, and leaves the figures themselves to SOURCES.md next door.
    """
    man = json.loads((ROOT / "data" / "heldout" / "MANIFEST.json").read_text())
    cm = man["integrations-core"]
    corpus_md = (ROOT / "data" / "corpus" / "SOURCES.md").read_text()
    section = corpus_md.split("## integrations-core", 1)[1].split("\n## ", 1)[0]
    corpus_commit = re.search(r"- commit: `([0-9a-f]+)`", section).group(1)
    assert cm["commit"] != corpus_commit, (
        "the holdout is now at the corpus's own commit, so it holds out nothing")
    assert cm["corpus_commit"] == corpus_commit, (
        "the manifest records a different corpus commit from SOURCES.md")
    # And the coincidence that hid the error, stated so that losing it is loud rather
    # than silent: if a dashboard is ever added upstream, these stop matching and the
    # docstring above stops being the explanation.
    assert cm["files"] == reg["ho.dd_corpus_dashboards_at_commit"], (
        "the two commits no longer ship the same number of dashboard files -- which "
        "is fine, but the corpus census must now be read from SOURCES.md alone and "
        "this assertion should go")


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


# --- the generated tables, held to the data -------------------------------------------
# Every guard in this suite joined prose to the registry or the registry to the data.
# Nothing joined a TABLE to anything. A table cell only ever licensed a prose figure, so
# an adversary changed the LALR conflict count from two to four, a floating-point
# agreement from 8.6e-14 to 8.6e-2, the `.rollup` count by a hundred, the no-glob
# columns so they summed to 27 against a registry that says 7, and a provenance date by
# thirteen months -- five tables, five false claims in print, 795 passed each time.
#
# These are the authorities the suite can reach. What a table says and what the thing it
# reports says are now one assertion each.

def test_the_grammar_table_is_the_table_the_parsers_build():
    """Section 4's answer to RQ1, re-derived rather than read off a generated file.

    The conflict count is the result: "Datadog is conflict-free LALR(1)" and "PromQL's
    table carries 2 conflicts" are the section's two findings, and both are cells in
    this table. `lalr_conflict_report` computes them in process and this suite already
    calls it -- `test_promql_has_exactly_one_lalr_conflict` asserts the parser -- but
    nothing compared the answer to the paper. Changing the cell to four, and the
    remark beside it to match, was green.
    """
    from qlport.parse import get_parser, lalr_conflict_report

    printed = table(TABLES / "grammar_stats.tex")
    rows = [[c.strip() for c in ln.split("&")]
            for ln in printed.split("\n") if "&" in ln and "Language" not in ln]
    seen = {}
    for lang, version, terminals, rules, states, conflicts in rows:
        seen[(lang, version)] = (terminals, rules, states,
                                 conflicts.removesuffix(r"\\").strip())
    assert len(seen) == 4, sorted(seen)
    for (lang, version), (terminals, rules, states, cell) in sorted(seen.items()):
        p = get_parser(lang, version)
        report = lalr_conflict_report(lang, version)
        assert terminals == str(len(p.terminals)), (lang, version, terminals)
        assert rules == str(len(p.rules)), (lang, version, rules)
        assert states == str(len(p.parser.parser._parse_table.states)), (lang, version)
        if not report:
            want = "none"
        else:
            kinds = sorted({(c["kind"], c["terminal"]) for c in report})
            assert len(kinds) == 1, kinds
            want = (f"{len(report)}" + r"$\times$ \texttt{"
                    + f"{kinds[0][0]} on {kinds[0][1]}" + "}")
        assert cell == want, (
            f"{lang} {version}: the table says {cell!r}, the parser gives {want!r}")


def test_the_conflict_count_the_prose_states_is_the_one_the_parser_reports():
    """And in both builds, because the count is stated in a condensed body paragraph
    in one and in a deferred remark in the other."""
    from qlport.parse import lalr_conflict_report

    import re as _re

    n = len(lalr_conflict_report("promql", "v2"))
    assert n == 2, f"the PromQL table now carries {n} conflicts, not 2"
    sections = ROOT / "paper" / "sections"
    for full in (False, True):
        said = []
        for path in sorted(sections.rglob("*.tex")):
            said += _re.findall(r"\$(\d+)\$ conflicts", flat(rendered(path, full)))
        build = "main_full" if full else "main"
        assert said, f"{build} states no conflict count at all"
        assert set(said) == {str(n)}, (
            f"{build} states conflict counts {sorted(set(said))} where the parser "
            f"reports {n}")


def test_the_model_verification_table_says_floating_point_and_means_it():
    """Section 3b calls this agreement "floating-point precision", which is a claim
    about a magnitude, so the magnitude is what gets asserted.

    An adversary moved the largest deviation from 8.60e-14 to 8.60e-02 -- twelve orders
    of magnitude, across 1,680 buckets per candidate -- and nothing failed, because the
    table was generated and therefore trusted and therefore unchecked.
    """
    printed = table(TABLES / "model_verification.tex")
    rows = [[c.strip() for c in ln.split("&")]
            for ln in printed.split("\n") if "&" in ln and "candidate" not in ln]
    assert len(rows) == 3, rows
    for candidate, buckets, worst, nan_mismatch in rows:
        assert int(buckets.replace(",", "")) >= 1000, (candidate, buckets)
        assert float(worst) < 1e-10, (
            f"{candidate} deviates by {worst} from the server, which is not what "
            f"section 3b means by floating-point precision")
        assert nan_mismatch.removesuffix(r"\\").strip() == "0", (candidate, nan_mismatch)


def test_the_no_glob_table_columns_are_the_registry_totals(reg):
    """Two rows whose columns have to add to the figures section 6 quotes.

    The three-way split of section 6 rests on these two numbers and on the glob counts
    beside them. Adding ten to each distinct count made the column sum to 27 against a
    registry that says 7, and the suite had nothing to say about it.
    """
    printed = table(TABLES / "regex_glob_reasons.tex")
    rows = [[c.strip() for c in ln.split("&")]
            for ln in printed.split("\n") if "&" in ln and "Construct" not in ln]
    assert len(rows) == 2, rows
    assert sum(int(r[1]) for r in rows) == reg["nb04.no_glob_distinct"]
    assert sum(int(r[2]) for r in rows) == reg["nb04.no_glob_occurrences"]
    # And each row's example really is a pattern with no glob, in the cache that says so.
    import csv
    refused = {r["regex"] for r in csv.DictReader(
        (ROOT / "data" / "processed" / "regex_glob.csv").open()) if r["reason"]}
    for row in rows:
        example = row[3].removesuffix(r"\\").strip()
        example = example.removeprefix(r"\texttt{").removesuffix("}")
        assert example.replace(r"\_", "_") in refused, (
            f"{example} is the table's example of a construct with no glob, and the "
            f"cache does not record it as one")


def test_the_operator_table_is_a_recount_of_the_corpus():
    """Thirty-two cells of corpus frequency that section 4 argues from.

    `.rollup` moved from 331 to 431 with nothing to stop it. The counters the notebook
    used are importable, so the table is recounted here rather than trusted.
    """
    import collections
    import json

    from qlport import corpus as c

    texts: dict[str, list[str]] = collections.defaultdict(list)
    with (ROOT / "data" / "corpus" / "queries.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            # The same exclusion notebook 00 makes and states: some community
            # dashboards under the Prometheus datasource carry Loki panels, and LogQL
            # is no evidence about PromQL. Leaving it out here is what makes this a
            # recount of the table's population rather than of a different one -- it
            # is three occurrences of `sum` on the PromQL side.
            if row["lang"] == "promql" and not c.is_promql(row["text"]):
                continue
            texts[row["lang"]].append(row["text"])

    counted: dict[tuple[str, str, str], int] = {}
    for lang in ("datadog", "promql"):
        funcs: collections.Counter = collections.Counter()
        aggs: collections.Counter = collections.Counter()
        for t in texts[lang]:
            funcs += c.function_names(lang, t)
            aggs += c.aggregation_ops(lang, t)
        for kind, tally in (("aggregator", aggs), ("function", funcs)):
            for name, n in tally.items():
                counted[(lang, kind, name)] = n

    printed = table(TABLES / "operator_frequency.tex")
    rows = [[col.strip() for col in ln.split("&")]
            for ln in printed.split("\n") if "&" in ln and "Language" not in ln]
    assert len(rows) == 32, len(rows)
    for lang, kind, name, occurrences in rows:
        plain = (name.removeprefix(r"\texttt{").removesuffix("}")
                 .replace(r"\_", "_"))
        got = int(occurrences.removesuffix(r"\\").strip().replace(",", ""))
        assert counted.get((lang, kind, plain)) == got, (
            f"the table says {lang} {kind} {plain} occurs {got} times; the corpus "
            f"gives {counted.get((lang, kind, plain))}")


def test_the_provenance_table_dates_are_the_dates_the_caches_carry():
    """The table a reader checks the vendor experiments against.

    A helper that collects exactly these timestamps was written, carries its own
    assertion, and was called by nothing -- so it read like the check behind this table
    while the table was unchecked, and a date moved thirteen months with the suite
    green. Given which trial window these experiments ran in, a provenance table
    nobody checks is worth more than its size suggests.
    """
    import json
    import re as _re
    from datetime import date, datetime

    carried = set()
    raw = ROOT / "data" / "raw" / "datadog"
    for path in sorted(raw.glob("*.json")):
        doc = json.loads(path.read_text())
        prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
        at = prov.get("collected_at") or doc.get("collected_at")
        if at:
            carried.add(datetime.fromisoformat(at).date())
    assert carried, "no vendor cache carries a timestamp"

    printed = table(TABLES / "vendor_provenance.tex")
    rows = [[col.strip() for col in ln.split("&")]
            for ln in printed.split("\n") if "&" in ln and "Experiment" not in ln]
    assert len(rows) >= 5, rows
    seen = set()
    for experiment, taken, _site, _what in rows:
        # One row gives a run reference rather than a date, for a probe taken on a rig
        # whose cache records the run and not the day. That is the only other form the
        # column may take, and saying so here is what keeps it from becoming the form
        # a wrong date can hide in.
        if _re.fullmatch(r"run r\d+", taken):
            continue
        assert _re.fullmatch(r"\d{1,2} \w{3} \d{4}", taken), (
            f"{experiment} has an unparseable date {taken!r}")
        when = datetime.strptime(taken, "%d %b %Y").date()
        assert when in carried, (
            f"{experiment} is dated {taken}, which no cache under data/raw/datadog "
            f"records; the dates the caches carry are {sorted(carried)}")
        seen.add(when)
    assert len(seen) >= 4 and seen <= carried, sorted(seen)
    assert isinstance(next(iter(carried)), date)

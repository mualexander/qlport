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
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "paper" / "tables"
REGISTRY = ROOT / "paper" / "quoted_figures.json"


@pytest.fixture(scope="module")
def reg() -> dict:
    return json.loads(REGISTRY.read_text())


def rows(name: str) -> list[list[str]]:
    """A generated `tabular` as a list of cell lists, rules and preamble dropped."""
    text = (TABLES / f"{name}.tex").read_text()
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
    body = (ROOT / "paper" / "sections" / "07_evaluation.tex").read_text()
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
    body = (ROOT / "paper" / "sections" / "moved" / "eval_unsupported.tex").read_text()
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
    insample = next(r for lab, r in by_label.items() if "corpus's" in lab)
    assert num(insample[1]) == reg["ho.gf_in_sample_queries"]
    assert num(insample[5]) == reg["ho.gf_in_sample_v2_pct"]
    assert num(insample[6]) == reg["ho.gf_in_sample_residual"]
    gf = next(r for lab, r in by_label.items() if "never saw" in lab)
    assert num(gf[1]) == reg["ho.gf_queries"]
    assert num(gf[2]) == reg["ho.gf_distinct"]
    assert num(gf[4]) == reg["ho.gf_v1_pct"]
    assert num(gf[5]) == reg["ho.gf_v2_pct"]
    assert num(gf[6]) == reg["ho.gf_residual"]
    sub = next(r for lab, r in by_label.items() if lab.startswith(r"\quad"))
    assert num(sub[2]) == reg["ho.gf_novel_distinct"]
    assert num(sub[5]) == reg["ho.gf_novel_v2_pct"]
    assert num(insample[5]) < num(sub[5]), (
        "the text says the frozen grammar does no worse out of sample; if that stops "
        "being true the paragraph has to change, not this assertion")
    dd = next(r for lab, r in by_label.items() if "integrations-core" in lab)
    assert num(dd[1]) == reg["ho.dd_queries"]
    assert num(dd[2]) == reg["ho.dd_distinct"]


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

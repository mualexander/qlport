"""Every figure the paper states in prose must come from somewhere checkable.

Three numbers went stale in one afternoon before this existed: the abstract kept
$\\kappa$ at 95.6% after it moved to 95.5%, and §7 kept a translated count and an
emitted-expression count that were each one too high. None was visible in any table,
and nothing recomputed them, so nothing could notice.

The rule here is not that every numeral must match something. It is that every numeral
must be *accounted for*: either it appears in a generated table, or in the figures
registry a notebook writes, or in the allowlist below with a reason. A new unexplained
number fails, which forces the decision rather than letting it pass.

What this catches is arithmetic drift, which was eight of the eleven problems the last
full read-through found.

It used to say here that a number which is right and means the wrong thing could not be
caught -- §5 once quoted 87% for the kernel's reach when 87% was the implementation-order
figure -- and that nothing mechanical would. That was too pessimistic. `BOUND` at the
foot of this file ties named figures to the registry key they came from, so a section
quoting the wrong quantity fails even when the quantity it quoted is itself correct
somewhere else. What survives is narrower: a figure nobody thought to bind, and a
sentence whose prose misdescribes a number that is accurate. Both still need a reader.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# rglob, not glob: sections/moved/ holds the blocks the 20-page build defers to an
# appendix and the full build keeps inline. They are the same prose either way, so
# every check here has to see them.
SECTIONS = sorted((ROOT / "paper" / "sections").rglob("*.tex"))


def _section_path(name: str) -> Path:
    """A bound figure names its file, which may be a section or an apparatus file.

    Apparatus lives in sections/moved/ so that one copy of the prose serves both the
    20-page build and the full one, and a binding has to follow it there.
    """
    for candidate in SECTIONS:
        if candidate.name == name:
            return candidate
    raise AssertionError(f"{name} is bound but no such file exists")
# The abstract moved into body.tex when the paper gained a 20-page and a full
# build; it is the most-read part of the document and the one place a stale figure
# survived two passes, so it is scanned like any section.
MAIN = ROOT / "paper" / "body.tex"
TABLES = sorted((ROOT / "paper" / "tables").glob("*.tex"))
REGISTRY = ROOT / "paper" / "quoted_figures.json"

_NUM = re.compile(r"(\d[\d{},]*(?:\.\d+)?)\s*(\\%)?")


def _norm(tok: str, pct: str) -> tuple[str, float] | None:
    try:
        return ("pct" if pct else "num", round(float(tok.replace("{,}", "")
                                                   .replace(",", "")), 4))
    except ValueError:
        return None


def _math(text: str) -> list[str]:
    """The `$...$` spans, with LaTeX comments and escaped dollars removed first.

    `\\%` is not a comment and `\\$65M` is currency rather than math; both bit an
    earlier version of this extractor and silently emptied its input.
    """
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = text.replace(r"\$", "\x00")
    return re.findall(r"\$([^$]*)\$", text)


def _math_text(path: Path) -> str:
    """Every `$...$` span of one file, joined. Prose outside math is not a figure."""
    return "\x01".join(_math(path.read_text()))


def _prose_figures() -> dict[tuple[str, float], set[str]]:
    out: dict[tuple[str, float], set[str]] = {}
    for path in [*SECTIONS, MAIN]:
        for span in _math(path.read_text()):
            for tok, pct in _NUM.findall(span):
                key = _norm(tok, pct)
                if key:
                    out.setdefault(key, set()).add(path.name)
    return out


def _accounted() -> set[tuple[str, float]]:
    """What the generated artifacts state, as counts and as shares separately.

    Kind is part of the claim. A table cell is a percentage when it carries `\\%` and a
    count otherwise; a registry value is a percentage when its name says so. An earlier
    version of this function cross-licensed the two, so any table printing a bare `2`
    licensed a `$2.0\\%$` anywhere in the prose -- which is how §3b kept a stale
    expressiveness share through two full read-throughs.
    """
    ok: set[tuple[str, float]] = set()
    for path in TABLES:
        for tok, pct in _NUM.findall(re.sub(r"(?<!\\)%.*", "", path.read_text())):
            key = _norm(tok, pct)
            if key:
                ok.add(key)
    if REGISTRY.exists():
        for name, value in json.loads(REGISTRY.read_text()).items():
            if isinstance(value, (int, float)):
                share = "pct" in name.rsplit(".", 1)[-1].split("_")
                ok.add(("pct" if share else "num", round(float(value), 4)))
    return ok


# Figures that are neither tabulated nor registry-backed, each with the reason it is
# not. "Historical" means the prose deliberately quotes a superseded value to describe
# how a figure changed; those must not be recomputed. Entries marked "unregistered"
# are live figures that no notebook records yet -- they are the next ones to migrate,
# and listing them here is a statement that they are unchecked, not that they are fine.
ALLOWED: dict[float, str] = {
    # --- constants of the apparatus -------------------------------------------------
    3.15: "Prometheus version the measurements were taken against",
    15: "scrape interval, seconds",
    60: "rollup interval (seconds); also the first share in the discussion's "
        "worked illustration of conjunctive coverage, which is hypothetical "
        "and not measured",
    20: "automatic rollup interval over an hour (seconds); also the upper end of "
        "rate's bucket disagreement in the replay, and the proverb's 20% of the work",
    300: "automatic rollup interval over a day, seconds",
    1800: "the one rollup width the differential window cannot hold two of",
    8587: "regex matchers in the PromQL corpus; the table beside it reports the split",
    0.628: "cor:rate: what rate() reads on one of two databases with equal phi images",
    887: "rem:type-measured: translated occurrences whose declared type is not decidable",
    98.8: "superseded: the ceiling before the empty-value scopes were refused, named in "
          "rem:empty-tag as the figure this correction moves",
    0.667: "and on the other, which is why rate is inadmissible on C_warm too",
    325: "shapes the differential had before the abstraction kept rollup intervals",
    335: "and after that, before the empty-value scopes left the translated set",
    3600: "the second rollup width the differential window cannot hold two of",
    114: "matcher occurrences in single-quoted and backtick literals, which the "
         "double-quoted-only scanner of an earlier pass did not see",
    22903: "query strings extraction yields",
    22894: "of those, what remains once the LogQL panels leave",
    7867: "PromQL panel queries, against 139 alerting and 96 recording rules",
    84: "buckets compared per row of the dual-write replay table",
    236.0: "the aligned replay bucket where the two engines differ in the last bit",
    235.999: "and what PromQL read for it, truncated -- the float64 noise a "
             "bit-identity test counted as a disagreement",
    0: "zero", 1: "one", 2: "two", 3: "three", 6: "six",
    10: "ten", 64: "the glob-union bound B", 100: "a percentage scale factor",
    80: "the proverbial 80% of queries, which this corpus does not support",
    # --- the declared metric types, from data/raw/datadog/metric_types.json ----------
    37070: "metric names integrations-core declares at the pinned commit",
    23597: "of those, declared gauge",
    13140: "of those, declared count",
    330: "of those, declared rate",
    # --- E9: what `.weighted()` does, and what it would have done if weighted --------
    66.7: "E9: the plain mean of the irregular series, with and without the modifier",
    16.7: "E9: what a mean weighted by each point's interval would have read",
    0.8: "points of kappa that implementing .weighted() closed",
    # --- E1, E7, E8: values from cached experiment records ---------------------------
    1000: "E1: the counter value just before the scripted reset",
    103: "E1: what Datadog stored over the reset bucket",
    897: "E1: what a raw signed difference would have stored",
    111: "E1 fast run: the reset bucket at three samples per flush",
    110: "E1 fast run: what losing the pre-reset increment would have given",
    25: "series-start check: the bucket sum for a series older than the window",
    1760: "E7: avg_over_time of the running totals at the end of the run",
    0.994: "E7: the relative difference after half an hour",
    0.5: "E7: the relative difference at the start of the run",
    # --- corpus figures from notebooks 00, 01, 04, 05: unregistered -------------------
    5389: "unregistered (test_corpus): distinct Datadog metric names",
    723: "unregistered (test_corpus): distinct grouping keys",
    388: "unregistered (nb 04): refused patterns",
    378: "of those, the two constructs that are finite unions exceeding the bound "
         "(238 + 140 from the reasons table), leaving 10 refused outright",
    344: "unregistered (nb 04): patterns needing a union",
    238: "unregistered (nb 04): refusals that are .+ or contain it",
    221: "unregistered (nb 01): PromQL queries the first six features complete",
    184: "distinct concrete regexes the complete procedure re-decides, all reached",
    168: "superseded: how many it reached under the 2,400 s budget, named in §6 as the "
         "figure this run replaces",
    183: "superseded: the population that budgeted run was over, one pattern smaller",
    2400: "the time budget that stopped an earlier cross-check at 168 of 183, seconds",
    4725: "what the unbudgeted cross-check took, seconds",
    176: "the slowest single pattern in it, seconds, timed uncontended",
    91: "and the fastest of that same four-pattern family",
    139: "three different quantities of this magnitude: nb00.top_occurrences, which\n         is registered and bound in \u00a77 and concl_evidence; the PromQL alerting\n         rules of \u00a77.1; and the \u00a74 grammar-gap count. Only the first is a\n         registry figure, so the other two ride here",
    133: "unregistered (nb 04): occurrences of .+",
    96: "unregistered (nb 00): PromQL recording rules",
    93: "unregistered (nb 01): queries needing a bare metric name",
    70: "unregistered (nb 03): .weighted() occurrences; also the second share "
        "in the discussion's hypothetical illustration",
    49: "unregistered (nb 04): occurrences of the most elaborate regex",
    36: "unregistered (nb 03): FILL_INTERPOLATION occurrences",
    31: "unregistered (nb 03): LABEL_NAME_GLOB occurrences",
    22: "unregistered (nb 00): per_second() occurrences",
    9: "unregistered (nb 04): occurrences of .*",
    8: "unregistered (nb 03): unparsed occurrences",
    8102: "the PromQL population after the nine LogQL panels are dropped. It has no\n          registry key, so it is pinned instead to the generated parse-coverage table\n          by tests/test_paper_builds.py -- which is what the 8,096 this replaces\n          escaped, having been allowlisted here as though it were the live value",
    5231: "unregistered (nb 00): PromQL function occurrences, recomputed as\n          sum(corpus.function_names) over the 8,102 kept expressions. The 5,227 this\n          replaces was the pre-correction corpus; the 69% rate/irate share derived\n          from it is 69.3% either way and did not move",
    99.9: "unregistered (nb 01): Datadog instantiated parse rate, v2",
    99.1: "unregistered (nb 01): PromQL instantiated parse rate",
    95: "unregistered (nb 00): share of Datadog space aggregators that are sum/avg;\n        also the coverage level \u00a7C.2's shape reduction reports 107 shapes reaching,\n        and a round level in \u00a74 -- three uses, one magnitude",
    92.6: "unregistered (nb 01): Datadog instantiated parse rate, v1",
    71.2: "unregistered (nb 01): PromQL raw parse rate",
    69: "unregistered (nb 00): share of PromQL functions that are rate/irate",
    34: "unregistered: percentile-aggregator occurrences (nb 03), and the union share (nb 04)",
    27.4: "unregistered (replay): aligned-grid agreement on the bucket edge",
    23: "unregistered (nb 03): shapes covering 80% -- also an allowed small integer",
    19: "unregistered (replay): lower end of rate's bucket disagreement, percent",
    30: "the rollup width E6 separates the two divisor placements at (seconds); also the "
        "uniform error the differential's shared-metric-name fault produced, an "
        "observation of a broken run rather than a computed figure",
    11.8: "unregistered (nb 01): Datadog raw parse rate",
    6.6: "unregistered (nb 00): as_count to as_rate ratio",
    4.5: "the glob refusals against every matcher rather than the concrete ones: "
         "388 of nb04.matchers = 8,587, the denominator the forward direction would "
         "have used. Stated as a derivation because the ratio has no registry key "
         "of its own, and the 4.6 this replaces was 388/8,472 -- the matcher count "
         "from before the scan learnt PromQL's other two string forms",
    3.6: "unregistered (nb 03): the bottom share",
    2.7: "unregistered (nb 01): PromQL queries the first six features complete",
    1.68: "unregistered (nb 02): worst relative error under a reset",
    1.4: "unregistered (nb 03): points of kappa the disjunction rule closed",
    1.1: "unregistered (nb 00): PromQL subquery share",
    0.9: "unregistered (nb 03): the 90% line on the coverage figure",
    90: "the same line quoted as a threshold in the staircase prose, and in the shape\n"
        "reduction; also, in \u00a72, the coverage Chronosphere's converter reports for\n"
        "itself -- a quotation of another tool's self-reported figure, not one of ours,\n"
        "and the paper says in the same sentence that its derivation is not stated",
    0.3: "unregistered (nb 00): PromQL offset share",
}


# Values above that are counts, versions, seconds or bare integers. They license a plain
# numeral and must never license a *percentage* of the same magnitude.
#
# §3b carried "$2.0\\%$" for the expressiveness gap through two full review passes after
# the figure moved to 1.2%, because the integer 2 is allowlisted here and the match was
# made on magnitude alone. A share and a count are different claims; a checker that
# cannot tell them apart launders one into the other.
SCALAR_ONLY: set[float] = {
    8587, 84, 236.0, 235.999, 3.15, 15, 25, 64, 300, 1000, 1800, 0, 1, 2, 3, 6, 10, 49, 103, 110, 111, 133, 139, 221, 238, 344, 388, 723, 897, 1760, 8102, 5231, 5389, 325, 3600, 184, 2400, 4725, 168, 183, 176, 91, 0.628, 0.667, 887, 335, 1.4,
}


# Historical figures: a value the prose quotes deliberately to describe how something
# changed. These are scoped to the sections entitled to say them, because an unscoped
# entry would license the stale value *anywhere* -- and a stale kappa in the abstract is
# exactly the bug this file exists to catch. The value and the place are both the claim.
ALLOWED_IN: dict[float, tuple[str, set[str]]] = {
    542: ("the four misclassified episodes, less the overlap between two of them",
          {"08_discussion.tex"}),
    348: ("refusals under the commutation argument",
          {"01_introduction.tex", "05_translation.tex", "07_evaluation.tex",
           "08_discussion.tex"}),
    124: ("disjunctive scopes before prop:disjunction, and separately what the\n"
           "empty-string reading of a glob star is worth",
          {"01_introduction.tex", "06_regular.tex", "07_evaluation.tex",
           "08_discussion.tex", "eval_rows_fill_glob.tex", "reg_star_empty.tex"}),
    120: ("of those, the single-key ones",
          {"01_introduction.tex", "07_evaluation.tex", "08_discussion.tex"}),
    116: ("translations once thought to carry ROLLUP_ALIGNMENT alone",
          {"07_evaluation.tex", "eval_type_caveat.tex"}),
    94.2: ("kappa before the disjunction rule",
           {"07_evaluation.tex", "eval_rows_fill_glob.tex"}),
    95.6: ("kappa after it and before the counter-average fix",
           {"07_evaluation.tex", "eval_rows_fill_glob.tex"}),
    11628: ("the formula count an earlier draft reported, which was the metric ones\n            plus 5,606 over log, event and security-signal requests; the live figure\n            is nb00.formulas = 6,022 and Table 6's caption said 11,628 until the\n            tenth pass, where it broke the corpus arithmetic on its own page",
           {"eval_held_aside.tex"}),
    54: ("E9: the superseded count of forms attaching .weighted() to sum:, now 52",
         {"eval_weighted.tex"}),
    56: ("E9: the superseded count of distinct forms carrying .weighted(), now 52 ---\n         and, separately, the share of Datadog queries using by {...}",
         {"eval_weighted.tex", "eval_language_use.tex"}),
    45: ("the queries the first six Datadog features complete, an nb-00 staircase\n         quantity. The reason recorded here until the tenth pass described nb 04's\n         refusals, which the cross-check now reports as all 50 of 184",
         {"eval_staircase.tex"}),
    76.2: ("METRIC_TYPE_UNKNOWN before E6", {"tr_gauge.tex"}),
    6381: ("the occurrences that share carried", {"tr_gauge.tex"}),
    87: ("the implementation-order figure contrasted with the kernel's reach, and\n"
         "separately the scope-only template share it was once confused with",
         {"07_evaluation.tex", "gr_template_shares.tex", "tr_two_figures.tex"}),
}


@pytest.fixture(scope="module")
def unexplained():
    ok = _accounted()
    allowed = {round(float(v), 4) for v in ALLOWED}
    scalar = {round(float(v), 4) for v in SCALAR_ONLY}
    out = {}
    for key, files in _prose_figures().items():
        kind, value = key
        if key in ok:
            continue
        if value in allowed and not (kind == "pct" and value in scalar):
            continue
        scoped = ALLOWED_IN.get(value)
        if scoped and set(files) <= scoped[1]:
            continue
        out[key] = sorted(files)
    return out


def test_no_scalar_allowance_can_license_a_percentage():
    """The hole that hid a stale figure for two review passes, kept shut.

    A percentage is a different claim from a count of the same magnitude, so the
    allowlist entry for one must not stand in for the other. Pinned by construction
    rather than by the absence of a failure, since the absence is what let it through.
    """
    assert SCALAR_ONLY <= {round(float(v), 4) for v in ALLOWED}, \
        "SCALAR_ONLY names a value that is not allowlisted at all"
    assert not ({0, 1, 2, 3, 6, 10} - SCALAR_ONLY), \
        "a bare small integer must never license a share"


def test_the_registry_was_written():
    assert REGISTRY.exists(), f"{REGISTRY.name} missing; run `make notebooks`"


def test_every_prose_figure_is_accounted_for(unexplained):
    """A numeral in the prose must be in a table, in the registry, or allowlisted.

    If this fails on a number you just added, the fix is usually to record it in a
    notebook's figures block rather than to add it below: an allowlist entry says the
    number is unchecked, which is the state that let the last three drift.
    """
    assert not unexplained, "\n".join(
        f"  {kind} {value:g} in {', '.join(files)}"
        for (kind, value), files in sorted(unexplained.items()))


def test_the_allowlist_has_no_duplicate_keys():
    """Two reasons under one key means one of them was silently discarded.

    `34` and `34.0` are the same dict key, and so are two separate `56` entries, so the
    file was losing a documented justification per collision without saying so -- in a
    file whose whole premise is that every unchecked number carries its reason. Only
    the source text shows this; the dict it builds no longer can.
    """
    import ast
    import collections

    src = Path(__file__).read_text()
    for node in ast.walk(ast.parse(src)):
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if getattr(target, "id", None) not in {"ALLOWED", "ALLOWED_IN"}:
            continue
        keys = [ast.literal_eval(k) for k in node.value.keys]
        dupes = sorted(k for k, n in collections.Counter(keys).items() if n > 1)
        assert not dupes, f"{target.id} states two reasons for: {dupes}"


def test_the_allowlist_has_no_dead_entries():
    """An allowlist that outlives its numbers stops being a record of anything."""
    prose = {k[1] for k in _prose_figures()}
    dead = sorted(v for v in {**ALLOWED, **ALLOWED_IN}
                  if round(float(v), 4) not in prose)
    assert not dead, f"allowlisted but no longer in the paper: {dead}"


def test_a_historical_figure_is_not_licensed_outside_its_section():
    """The scoping is the point, so it is checked rather than trusted.

    Putting the superseded kappa in the abstract -- the exact drift that prompted this
    file -- must fail even though §7 is entitled to quote the same number.
    """
    abstract = MAIN.read_text()
    assert "95.6" not in abstract, "stale kappa in the abstract"
    stale = {k: v for k, v in _prose_figures().items() if k[1] == 95.6}
    assert stale, "the historical kappa should still be quoted somewhere"
    assert all(set(f) <= ALLOWED_IN[95.6][1] for f in stale.values()), stale


def test_the_registry_agrees_with_the_tables_on_kappa():
    """The two independent paths to the same figure must not diverge.

    The tables are written by one notebook cell and the registry by another; if a
    future edit recomputed either differently, every prose check would still pass
    while the paper contradicted itself.
    """
    reg = json.loads(REGISTRY.read_text())
    coverage = (ROOT / "paper" / "tables" / "translation_coverage.tex").read_text()
    assert f"{reg['nb03.kappa_occurrences']:,}" in coverage
    assert f"{reg['nb03.kappa_distinct']:,}" in coverage
    assert f"{reg['nb03.kappa_pct']}\\%" in coverage


# Prose figures bound to the registry key they came from. The checks above ask whether a
# number is accounted for *somewhere*; these ask whether this number, in this section, is
# still what its own source says. The difference is the whole point: when a figure moves,
# the stale copy in the prose usually still matches something -- another table, another
# quantity of the same magnitude -- so magnitude alone cannot tell drift from coincidence.
#
# Only distinctive values are bound: percentages, and counts of three digits or more. A
# binding on "6 shapes cover half the occurrences" would pass on any section containing a
# six, which is every section, so it would record a check that cannot fail.
#
# A key absent here is not pinned. Adding one is cheap and is how a new quoted figure
# should arrive; the list is a statement about which of the paper's numbers are tied to
# their source by name rather than by luck.
BOUND: tuple[tuple[str, str], ...] = (
    ("nb00.ceiling_8_features_pct", "eval_staircase.tex"),
    ("oo.corpus", "07_evaluation.tex"),
    ("oo.invalid", "07_evaluation.tex"),
    ("oo.invalid_matchall", "07_evaluation.tex"),
    ("oo.dropped_modifier", "07_evaluation.tex"),
    ("oo.drops_total", "07_evaluation.tex"),
    ("oo.dropped_rollup", "07_evaluation.tex"),
    ("oo.dropped_fill", "07_evaluation.tex"),
    ("oo.invented_window", "07_evaluation.tex"),
    ("oo.invented_window_pct", "07_evaluation.tex"),
    ("oo.cosmetic_note_only_pct", "07_evaluation.tex"),
    ("oo.substantive_note_pct", "07_evaluation.tex"),
    ("oo.gap_unsupported_function", "eval_baseline.tex"),
    ("nb03.caveat_counter_reset_pct", "07_evaluation.tex"),
    ("nb03.caveat_metric_type_unknown_occurrences", "05_translation.tex"),
    ("nb03.caveat_metric_type_unknown_pct", "05_translation.tex"),
    ("nb03.caveat_metric_type_unknown_pct", "07_evaluation.tex"),
    ("nb03.caveat_phi_series_start_pct", "07_evaluation.tex"),
    ("nb03.caveat_rollup_default_pct", "01_introduction.tex"),
    ("nb03.caveat_rollup_default_pct", "07_evaluation.tex"),
    ("nb03.caveat_rollup_default_pct", "08_discussion.tex"),
    ("nb03.caveat_rollup_default_pct", "body.tex"),
    ("nb03.ceiling_pct", "01_introduction.tex"),
    ("nb03.ceiling_pct", "eval_beyond_reach.tex"),
    ("nb03.ceiling_pct", "08_discussion.tex"),
    ("nb03.ceiling_pct", "09_conclusion.tex"),
    ("nb03.closure_occurrences", "05_translation.tex"),
    ("nb03.corpus_distinct", "07_evaluation.tex"),
    ("nb03.corpus_occurrences", "01_introduction.tex"),
    ("nb03.corpus_occurrences", "07_evaluation.tex"),
    ("nb03.corpus_occurrences", "08_discussion.tex"),
    ("nb03.corpus_occurrences", "body.tex"),
    ("nb03.emitted_expressions", "07_evaluation.tex"),
    ("nb03.kappa_distinct", "07_evaluation.tex"),
    ("nb03.kappa_occurrences", "05_translation.tex"),
    ("nb03.kappa_pct", "01_introduction.tex"),
    ("nb03.kappa_pct", "05_translation.tex"),
    ("nb03.kappa_pct", "07_evaluation.tex"),
    ("nb03.kappa_pct", "08_discussion.tex"),
    ("nb03.kappa_pct", "09_conclusion.tex"),
    ("nb03.kappa_pct", "body.tex"),
    ("nb03.kernel_occurrences", "tr_two_figures.tex"),
    ("nb03.kernel_pct_corpus", "01_introduction.tex"),
    ("nb03.kernel_pct_corpus", "05_translation.tex"),
    ("nb03.kernel_pct_translated", "05_translation.tex"),
    ("nb03.language_gap_occurrences", "07_evaluation.tex"),
    ("nb03.language_gap_pct", "01_introduction.tex"),
    ("nb03.language_gap_pct", "03b_ingestion.tex"),
    ("nb03.language_gap_pct", "07_evaluation.tex"),
    ("nb03.language_gap_pct", "08_discussion.tex"),
    ("nb03.language_gap_pct", "09_conclusion.tex"),
    ("nb03.language_gap_pct", "body.tex"),
    ("nb03.not_implemented_occurrences", "07_evaluation.tex"),
    ("nb03.not_implemented_pct", "01_introduction.tex"),
    ("nb03.not_implemented_pct", "03b_ingestion.tex"),
    ("nb03.not_implemented_pct", "07_evaluation.tex"),
    ("nb03.not_implemented_pct", "08_discussion.tex"),
    ("nb03.not_implemented_pct", "09_conclusion.tex"),
    ("nb03.not_implemented_pct", "body.tex"),
    ("nb03.shaped_occurrences", "07_evaluation.tex"),
    ("nb03.shapes", "07_evaluation.tex"),
    # The ninth review pass found each of these four quoted at a superseded value,
    # accepted because an allowlist entry called it unregistered. A registry key plus a
    # binding is what an allowlist entry cannot be: a check.
    # nb03.feature_slack_occurrences is not bound: 87 is too common a value to match
    # meaningfully, which test_a_bound_figure_is_distinctive_enough_to_check enforces.
    # It drifted anyway (quoted as 82), so test_paper_builds.py checks it by derivation.
    ("nb04.matchers", "06_regular.tex"),
    ("nb04.template_pct", "06_regular.tex"),
    ("nb04.glob_exists_pct", "06_regular.tex"),
    ("nb03.shapes", "concl_evidence.tex"),
    ("nb03.shapes_for_95pct", "eval_shape_reduction.tex"),
    # `top(...)`'s occurrence count decides which column the largest unimplemented entry
    # belongs in, and the two counts beside it are what the measurement covers: 134 was
    # the reach of the claim before `bottomk` was probed, and 111 is how often the corpus
    # asks for the ranking function the probe leads with.
    ("nb00.top_occurrences", "07_evaluation.tex"),
    ("nb00.top_occurrences", "concl_evidence.tex"),
    ("nb00.top_desc_occurrences", "eval_top.tex"),
    ("nb00.top_rank_mean_calls", "eval_top.tex"),
)


def _spellings(value: float) -> set[str]:
    """How the paper may write a registry value: `8{,}364`, `96.3`, `100` or `100.0`.

    A share of exactly one hundred is written both ways in this paper, so both count.
    """
    out = {f"{value}"}
    if float(value) == int(value):
        out.add(f"{int(value):,}".replace(",", "{,}"))
    return out


def test_a_bound_figure_still_matches_its_source():
    """A pinned figure must equal what its own source now says, in its own section.

    This is the check `test_every_prose_figure_is_accounted_for` cannot make. When
    `.weighted()` moved 70 occurrences out of the gap catalogue, the expressiveness
    share went from 2.0% to 1.2% -- and §3b's stale 2.0% went on matching the integer
    two for two more read-throughs. Bound to `language_gap_pct`, it fails the moment
    the key moves and the prose does not follow.
    """
    reg = json.loads(REGISTRY.read_text())
    stale = []
    for key, section in BOUND:
        assert key in reg, f"{key} is bound but no longer in the registry"
        # A section file, body.tex, or an apparatus file under sections/moved/.
        path = MAIN if section == "body.tex" else _section_path(section)
        text = _math_text(path)
        if not any(s in text for s in _spellings(reg[key])):
            stale.append(f"  {section} should state {key} = {reg[key]}")
    assert not stale, "\n".join(stale)


def test_every_binding_names_a_distinctive_value():
    """A binding on a small count records a check that cannot fail; keep them out."""
    reg = json.loads(REGISTRY.read_text())
    weak = sorted({
        f"{key} = {reg[key]}" for key, _ in BOUND
        if "pct" not in key.rsplit(".", 1)[-1].split("_")
        and float(reg[key]) == int(reg[key]) and int(reg[key]) < 100
    })
    assert not weak, f"bound to a value too common to mean anything: {weak}"

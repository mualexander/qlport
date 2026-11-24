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

from texsource import anywhere, flat, rendered, strip_comments, table

ROOT = Path(__file__).resolve().parent.parent
# rglob, not glob: sections/moved/ holds the blocks the 20-page build defers to an
# appendix and the full build keeps inline. They are the same prose either way, so
# every check here has to see them.
SECTIONS = sorted((ROOT / "paper" / "sections").rglob("*.tex"))


def _section_path(name: str) -> Path:
    """A bound figure names its file, which may be a section or an apparatus file.

    Apparatus lives in sections/moved/ so that one copy of the prose serves both the
    20-page build and the full one, and a binding has to follow it there.

    Exactly one file may answer to a name. This returned the first match in sorted
    order, and `sections/eval_heldout.tex` sorts before `sections/moved/eval_heldout
    .tex`, so a same-named file one directory up silently became the file every
    binding and every pinned phrase was checked against. With the real block left
    saying the opposite, the whole suite stayed green.
    """
    found = [c for c in SECTIONS if c.name == name]
    assert len(found) == 1, (
        f"{name} names {len(found)} files under paper/sections: "
        f"{[str(c.relative_to(ROOT)) for c in found]}")
    return found[0]

# The abstract moved into body.tex when the paper gained a 20-page and a full
# build; it is the most-read part of the document and the one place a stale figure
# survived two passes, so it is scanned like any section.
MAIN = ROOT / "paper" / "body.tex"
TABLES = sorted((ROOT / "paper" / "tables").glob("*.tex"))
ORDER = ROOT / "paper" / "tables" / "implementation_order.tex"
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
    text = strip_comments(text)
    text = text.replace(r"\$", "\x00")
    return re.findall(r"\$([^$]*)\$", text)


def _math_in_both_builds(path: Path) -> tuple[str, str]:
    r"""One file's math spans as each of the two PDFs will have them.

    `published` resolves `%` and not `\iffull`, so it returns the union of two
    documents no reader ever sees together. For a census -- "is this figure accounted
    for anywhere" -- the union is the conservative reading and `_math_text` keeps it.
    For a claim that a file *states* a figure it is not: a figure inside
    `\inlineonly` reaches one PDF and is absent from the other, and the binding
    asserting the file states it is then true of neither document on its own. Both
    bindings and historical pairs go through here, so the claim has to hold in each
    build rather than in their union.
    """
    return tuple("\x01".join(_math(rendered(path, full))) for full in (False, True))


def _prose_figures() -> dict[tuple[str, float], set[str]]:
    out: dict[tuple[str, float], set[str]] = {}
    for path in [*SECTIONS, MAIN]:
        for span in _math(anywhere(path)):
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
        # `table`, not `anywhere`. A generated table is inputted by both builds and
        # must therefore render identically in them, which `table` asserts; reading
        # the union instead meant an `\iffull 7\else 999\fi` in a table licensed both
        # 7 and 999 for any prose figure, in the one path every prose figure is
        # licensed through. Sixteen of the twenty-two tables were read through `table`
        # somewhere else in this suite and the licensing path was not one of them.
        for tok, pct in _NUM.findall(table(path)):
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
#
# Keyed by kind as well as magnitude. It used to be keyed by magnitude alone with a
# second ledger, SCALAR_ONLY, naming the entries that must not licence a percentage:
# §3b carried "$2.0\\%$" for the expressiveness gap through two review passes after the
# figure moved to 1.2%, because the integer 2 was allowlisted and the match was made on
# magnitude. A share and a count are different claims. Two ledgers that have to agree
# about which is which is a worse answer than one key that says so, and SCALAR_ONLY had
# grown to 49 entries, of which only three --- 0.0, 1.4 and 6.0 --- had a per-cent
# spelling anywhere in the paper to forbid, so 46 forbade nothing. The pass that
# retired it said 39, which is a number nobody computed; this one is the count of
# SCALAR_ONLY's magnitudes that appear with a \\% in any `$...$` span of the paper at
# the commit before the ledger left.
#
# An entry that licences nothing is removed, not kept for its prose: see
# `test_no_allowance_licenses_anything_it_is_still_needed_for`. Seventy-seven of these
# were shadowed --- the figure was already in a table, a registry key, or a derivation,
# and the reason recorded here was not the reason it passed. A ledger that describes a
# check it is not performing is worse than no ledger.
ALLOWED: dict[tuple[str, float], str] = {
    # --- constants of the apparatus -------------------------------------------------
    ("num", 1800): "the one rollup width the differential window cannot hold two of",
    ("num", 3600): "the second rollup width the differential window cannot hold two of",
    ("num", 325): "shapes the differential had before the abstraction kept rollup "
                  "intervals",
    ("num", 335): "and after that, before the empty-value scopes left the translated "
                  "set",
    ("num", 114): "matcher occurrences in single-quoted and backtick literals, which "
                  "the double-quoted-only scanner of an earlier pass did not see",
    ("pct", 20): "the upper end of rate's bucket disagreement in the replay, and the "
                 "proverb's 20% of the work",
    ("pct", 30): "the uniform error the differential's shared-metric-name fault "
                 "produced, an observation of a broken run rather than a computed "
                 "figure",
    # --- the trial probes, whose runs are not reproducible on a lapsed account -------
    ("num", 16.7): "E9: what a mean weighted by each point's interval would have read",
    ("num", 4096): "the limit an earlier run of E17 published, off one HTTP 408 in a "
                   "run that had already accepted 5,096. The retry and the "
                   "boundary-contradiction guard that followed are why the current "
                   "run reports a range",
    ("num", 5096): "the rung that run had already accepted when it reported 4,096",
    ("num", 408): "and the status it took for a refusal",
    ("num", 1.68): "unregistered (nb 02): worst relative error under a reset",
    # --- the glob cross-check, and the budgeted run it replaces ----------------------
    ("num", 168): "superseded: how many patterns the budgeted run reached, named in "
                  "§6 as the figure this run replaces",
    ("num", 183): "superseded: the population that run was over, one pattern smaller",
    ("num", 2400): "the time budget that stopped it at 168 of 183, seconds",
    ("num", 176): "the slowest single pattern in the current run, seconds, timed "
                  "uncontended",
    # --- figures no notebook records yet ---------------------------------------------
    ("num", 723): "unregistered (test_corpus): distinct grouping keys, the narrower "
                  "census tr_mangling reports widening from",
    ("num", 221): "unregistered (nb 01): PromQL queries the first six features "
                  "complete",
    ("pct", 2.7): "and what that is as a share of the PromQL corpus",
    ("num", 22608): "historical: the second run's bucket comparisons, quoted beside "
                    "the third run's 22,692 to say the agreement did not move",
}


# Historical figures: a value the prose quotes deliberately to describe how something
# changed. These are scoped to the sections entitled to say them, because an unscoped
# entry would license the stale value *anywhere* -- and a stale kappa in the abstract is
# exactly the bug this file exists to catch. The value, the kind and the place are all
# part of the claim.
#
# Scoping is the weakest thing in this file and it is worth saying why rather than
# discovering it again. `unexplained()` returns at the first accounting it finds, and a
# registry key of the same magnitude is an accounting, so any scope whose value
# collides with a live registry value is inert everywhere --- the scope never runs.
# Three entries here were in that state and have been removed. Where the figure is
# load-bearing, PHRASES below is the answer, because a phrase names the sentence and
# not just the number.
ALLOWED_IN: dict[tuple[str, float], tuple[str, set[str]]] = {
    ("num", 348): ("refusals under the commutation argument",
                   {"01_introduction.tex", "05_translation.tex", "07_evaluation.tex",
                    "08_discussion.tex"}),
    ("num", 124): ("disjunctive scopes before prop:disjunction. It was also, until "
                   "E17's counterfactual was measured, the figure this file recorded "
                   "for what the empty-string reading of a glob star is worth; that "
                   "is 515 now, derived from data/raw/globs/star_reading.json, and "
                   "the prose no longer says 124",
                   {"01_introduction.tex", "06_regular.tex", "07_evaluation.tex",
                    "08_discussion.tex", "eval_rows_fill_glob.tex",
                    "reg_star_empty.tex"}),
    ("pct", 94.2): ("kappa before the disjunction rule",
                    {"07_evaluation.tex", "eval_rows_fill_glob.tex"}),
    ("pct", 95.6): ("kappa after it and before the counter-average fix",
                    {"07_evaluation.tex", "eval_rows_fill_glob.tex"}),
    ("num", 11628): ("the formula count an earlier draft reported, which was the "
                     "metric ones plus 5,606 over log, event and security-signal "
                     "requests; the live figure is nb00.formulas = 6,022 and Table "
                     "6's caption said 11,628 until the tenth pass, where it broke "
                     "the corpus arithmetic on its own page",
                     {"eval_held_aside.tex"}),
    ("pct", 56): ("E9: the superseded share of distinct forms carrying .weighted(), "
                  "now 52 --- and, separately, the share of Datadog queries using "
                  "by {...}", {"eval_weighted.tex", "eval_language_use.tex"}),
    ("num", 0.23): ("the seed spread the staircase cross-validation reported while "
                    "its tie-break still depended on the interpreter's hash salt, "
                    "quoted beside the 0.08 that replaced it",
                    {"eval_heldout.tex"}),
    ("num", 0.28): ("the seed spread the FIXED tie-break gives at the previous draft's "
                    "replicate budget, which is what shows the 0.23 was never the hash "
                    "salt. Reproduce with `staircase_cv(by_dash, 200)` and read "
                    "`gap_at_seed_range`: a counterfactual configuration, so no "
                    "committed cache holds it",
                    {"eval_heldout.tex"}),
    ("pct", 76.2): ("METRIC_TYPE_UNKNOWN before E6", {"tr_gauge.tex"}),
    ("num", 6381): ("the occurrences that share carried", {"tr_gauge.tex"}),
    ("num", 31): ("LABEL_NAME_GLOB's occurrences. The code left the catalogue when "
                  "corpus._TMPL stopped truncating $cluster-location at its hyphen, "
                  "so this is a historical figure: the sections entitled to it are "
                  "the ones telling that story, plus the two the ceiling rewrite "
                  "still has to reach",
                  {"01_introduction.tex", "07_evaluation.tex", "08_discussion.tex",
                   "eval_baseline.tex", "eval_beyond_reach.tex",
                   "eval_differential_fixes.tex", "eval_rows_fill_glob.tex",
                   "reg_trial_probes.tex"}),
    ("num", 87): ("the implementation-order figure contrasted with the kernel's "
                  "reach", {"07_evaluation.tex", "tr_two_figures.tex"}),
    ("pct", 87): ("the scope-only template share it was once confused with",
                  {"07_evaluation.tex", "gr_template_shares.tex",
                   "tr_two_figures.tex"}),
}


def _derived_keys() -> set[tuple[str, float]]:
    """The `(kind, value)` keys the two derivation ledgers license.

    Kind-blind for three rounds, across all three places that rebuilt this set. A
    derivation of an occurrence count licensed the same magnitude printed as a per
    cent, and a derivation of a share licensed it printed as a count; sixty magnitudes
    carried that free pass, and the sentence "$96\\%$ of the refusals and $38.3$
    dashboards" --- a share used as a count and a count used as a share, both
    fabricated --- passed every test in this file. The same re-keying that retired
    SCALAR_ONLY from the allowlist, applied where it was not.

    Every entry names one spelling. One named two for a commit --- $120$ occurrences is
    $1.4$ points of $\\kappa$ over the corpus and $1.4\\%$ of the translated set, the
    same count over two denominators that round alike --- with a paragraph here
    defending the exception. Withholding the second kind turned out to change nothing:
    a generated table already accounts for $1.4\\%$, so the grant was inert and the
    defence was of a decision that had no effect. `test_no_derived_kind_grants_more
    _than_it_has_to` is what would have said so.
    """
    return {(kind, round(float(value), 4))
            for kinds, value, _, _ in (*DERIVED, *CORPUS_DERIVED)
            for kind in kinds.split()}


def _unaccounted(allowed: dict, allowed_in: dict) -> dict:
    """Prose figures that nothing accounts for, under the two ledgers given.

    Parameterised so the reachability test can run it with an entry withheld. The
    order matters and is the thing most likely to be misread: a figure accounted for
    by a table, a registry key or a derivation returns here before either ledger is
    consulted, which is why an ALLOWED_IN scope whose magnitude collides with a live
    registry value never runs.
    """
    ok = _accounted()
    derived = _derived_keys()
    out = {}
    for key, files in _prose_figures().items():
        kind, value = key
        if key in ok or key in derived:
            continue
        if (kind, round(value, 4)) in allowed:
            continue
        scoped = allowed_in.get((kind, round(value, 4)))
        if scoped and set(files) <= scoped[1]:
            continue
        out[key] = sorted(files)
    return out


@pytest.fixture(scope="module")
def unexplained():
    return _unaccounted(_keyed(ALLOWED), _keyed(ALLOWED_IN))


def _keyed(ledger: dict) -> dict:
    return {(k, round(float(v), 4)): r for (k, v), r in ledger.items()}


def test_no_allowance_licenses_anything_it_is_still_needed_for():
    """An allowlist entry that changes nothing when removed has to come out.

    Ninety-eight entries were here and seventy-seven of them licensed nothing: the
    figure was already in a generated table, in the registry, or in a derivation, and
    the sentence recorded beside it was not the reason it passed. Six ALLOWED_IN
    scopes were inert the same way --- one of them said so in its own comment and was
    kept anyway, which is how a ledger stops being read. The pass that removed them
    wrote "three" here and in its commit message, having printed the six a few minutes
    earlier; a file about figures nothing checks is a poor place to put a figure
    nothing checks, so the number is $7.4, 21, 45, 116, 120, 648$ and this test is
    what keeps it honest.

    Nothing about that was visible from reading the file, so it is asserted instead.
    A value that moves out from under an entry fails `unexplained` and the entry comes
    back; a value that moves under one fails here and it goes.
    """
    allowed, scoped = _keyed(ALLOWED), _keyed(ALLOWED_IN)
    base = len(_unaccounted(allowed, scoped))
    dead = []
    for ledger, name in ((allowed, "ALLOWED"), (scoped, "ALLOWED_IN")):
        for key in list(ledger):
            held = {k: v for k, v in ledger.items() if k != key}
            a, b = (held, scoped) if name == "ALLOWED" else (allowed, held)
            if len(_unaccounted(a, b)) == base:
                dead.append(f"  {name}[{key}] licenses nothing")
    assert not dead, (
        "these entries can be deleted without changing what this file checks:\n"
        + "\n".join(dead))


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


def test_a_historical_figure_is_not_licensed_outside_its_section():
    """The scoping is the point, so it is checked rather than trusted.

    Putting the superseded kappa in the abstract -- the exact drift that prompted this
    file -- must fail even though §7 is entitled to quote the same number.
    """
    abstract = anywhere(MAIN)
    assert "95.6" not in abstract, "stale kappa in the abstract"
    stale = {k: v for k, v in _prose_figures().items() if k[1] == 95.6}
    assert stale, "the historical kappa should still be quoted somewhere"
    assert all(set(f) <= ALLOWED_IN[("pct", 95.6)][1] for f in stale.values()), stale


def test_the_registry_agrees_with_the_tables_on_kappa():
    """The two independent paths to the same figure must not diverge.

    The tables are written by one notebook cell and the registry by another; if a
    future edit recomputed either differently, every prose check would still pass
    while the paper contradicted itself.
    """
    reg = json.loads(REGISTRY.read_text())
    coverage = table(ROOT / "paper" / "tables" / "translation_coverage.tex")
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
    # The held-out apparatus. Until these arrived, every ho.* figure was tied to the
    # prose by magnitude alone, and the held-out kappa could drift from the paper in
    # silence: 96.7% also happens to be \S\ref{sec:baseline}'s coverage-by-valid-output,
    # so a stale 96.7 in five files went on being accounted for by a table about
    # something else. These bind it where it means the holdout. 07_evaluation.tex and
    # 09_conclusion.tex are deliberately absent --- the baseline's own 96.7% lives
    # there, so a binding on those files would pass whatever the registry said.
    ("ho.ex_kappa_pct", "eval_heldout.tex"),
    ("ho.ex_kappa_pct", "body.tex"),
    ("ho.ex_kappa_pct", "01_introduction.tex"),
    ("ho.ex_kappa_distinct_pct", "eval_heldout.tex"),
    ("ho.ex_sstar_pct", "eval_heldout.tex"),
    ("ho.ex_template_pct", "07_evaluation.tex"),
    ("ho.ex_queries", "eval_heldout.tex"),
    ("ho.ex_queries", "01_introduction.tex"),
    ("ho.ex_distinct", "eval_heldout.tex"),
    ("ho.ex_distinct", "04_grammars.tex"),
    ("ho.ex_queries_at_commit", "eval_heldout.tex"),
    ("ho.ex_kappa_distinct", "eval_heldout.tex"),
    ("ho.ex_v1_pct", "eval_heldout.tex"),
    ("ho.ex_jackknife_high_pct", "eval_heldout.tex"),
    ("ho.gf_novel_distinct", "eval_heldout.tex"),
    ("nb00.ceiling_7_features_pct", "eval_staircase.tex"),
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
    # §6's three-way split, each cell to its own key. The paragraph stating them kept
    # the figures of the run before the tag alphabet was pinned -- 62.7% where a glob
    # now exists for 91.9%, and a third of concrete occurrences refused where it is one
    # in twelve -- and passed, because a stale figure in the registry's own magnitude
    # range matches something. `nb04.no_glob_occurrences` is 84 and so cannot be bound;
    # it is carried by Table 5 and by the caption that counts it.
    # The edge of the gap the bound sits above. §6.2's justification is that nothing
    # the corpus needs falls between this and the square, so a bound anywhere in
    # between is the same decision -- a claim about a distribution, which is why the
    # edge is a registered figure rather than a remark.
    ("nb04.largest_union_below_the_bound", "06_regular.tex"),
    ("nb04.concrete_occurrences", "06_regular.tex"),
    ("nb04.glob_exists_occurrences", "06_regular.tex"),
    ("nb04.glob_single_occurrences", "06_regular.tex"),
    ("nb04.glob_single_pct", "06_regular.tex"),
    ("nb04.glob_union_occurrences", "06_regular.tex"),
    ("nb04.glob_union_pct", "06_regular.tex"),
    ("nb04.no_glob_pct", "01_introduction.tex"),
    ("nb04.no_glob_pct", "08_discussion.tex"),
    ("nb04.no_glob_pct", "09_conclusion.tex"),
    ("nb03.shapes", "concl_evidence.tex"),
    ("nb03.shapes_for_95pct", "eval_shape_reduction.tex"),
    # The differential run's scale and its agreement bound. §7.4 quoted the second
    # run's counts for two commits after the third run landed in the cache, because
    # the cache test and the prose each held their own literals and nothing compared
    # them. `test_differential_cache.py` now derives these from the rows; this is the
    # other end of that join.
    ("nb05.shapes", "07_evaluation.tex"),
    ("nb05.shapes", "eval_shape_reduction.tex"),
    ("nb05.synthetic_series", "07_evaluation.tex"),
    ("nb05.synthetic_series", "eval_shape_reduction.tex"),
    ("nb05.bucket_comparisons", "07_evaluation.tex"),
    ("nb05.bit_identical_shapes", "07_evaluation.tex"),
    # `top(...)`'s occurrence count decides which column the largest unimplemented entry
    # belongs in, and the two counts beside it are what the measurement covers: 134 was
    # the reach of the claim before `bottomk` was probed, and 111 is how often the corpus
    # asks for the ranking function the probe leads with.
    ("nb00.top_occurrences", "07_evaluation.tex"),
    ("nb00.top_occurrences", "concl_evidence.tex"),
    ("nb00.top_desc_occurrences", "eval_top.tex"),
    ("nb00.top_rank_mean_calls", "eval_top.tex"),
    # Everything below was generated from where the figures stood when the guard was
    # tightened: every registry key the prose quotes, bound to each section that quotes
    # it. Before this, 46 keys were in the registry and in the paper and joined to it
    # only by magnitude -- `test_every_prose_figure_is_accounted_for` would accept
    # `$491$` in §8 because *some* key is 491, not because §8's 491 is that key. The
    # per-section grain is deliberate: §7.4 kept the second differential run's counts
    # for two commits while the rest of the paper moved on, and one binding per key
    # would not have caught it.
    ("nb00.ceiling_0_features_pct", "04_grammars.tex"),
    ("nb00.ceiling_0_features_pct", "eval_language_use.tex"),
    ("nb00.ceiling_0_features_pct", "eval_staircase.tex"),
    ("nb00.ceiling_25_features_pct", "07_evaluation.tex"),
    ("nb00.ceiling_26_features_pct", "07_evaluation.tex"),
    ("nb00.ceiling_28_features_pct", "01_introduction.tex"),
    ("nb00.ceiling_28_features_pct", "05_translation.tex"),
    ("nb00.ceiling_28_features_pct", "07_evaluation.tex"),
    ("nb00.ceiling_28_features_pct", "08_discussion.tex"),
    ("nb00.ceiling_28_features_pct", "eval_beyond_reach.tex"),
    ("nb00.ceiling_28_features_pct", "eval_type_caveat.tex"),
    ("nb00.ceiling_28_features_pct", "ing_replay.tex"),
    ("nb00.ceiling_28_features_pct", "tr_gauge.tex"),
    ("nb00.ceiling_2_features_pct", "07_evaluation.tex"),
    ("nb00.ceiling_2_features_pct", "eval_empty_tag.tex"),
    ("nb00.datadog_with_variable_pct", "01_introduction.tex"),
    ("nb00.datadog_with_variable_pct", "04_grammars.tex"),
    ("nb00.datadog_with_variable_pct", "07_evaluation.tex"),
    ("nb00.datadog_with_variable_pct", "08_discussion.tex"),
    ("nb00.datadog_with_variable_pct", "09_conclusion.tex"),
    ("nb00.datadog_with_variable_pct", "body.tex"),
    ("nb00.formula_arith_pct", "eval_held_aside.tex"),
    ("nb00.formula_bare_ref_pct", "eval_held_aside.tex"),
    ("nb00.formula_const_pct", "eval_held_aside.tex"),
    ("nb00.formula_func_pct", "eval_held_aside.tex"),
    ("nb00.formulas", "07_evaluation.tex"),
    ("nb00.formulas", "eval_held_aside.tex"),
    ("nb00.formulas_distinct", "eval_held_aside.tex"),
    ("nb00.non_metric_formula_bare_ref_pct", "eval_held_aside.tex"),
    ("nb00.non_metric_formulas", "eval_held_aside.tex"),
    ("nb00.promql_parse_range_vars_only_pct", "07_evaluation.tex"),
    ("nb00.promql_window_from_variable", "08_discussion.tex"),
    ("nb00.promql_window_from_variable_pct", "01_introduction.tex"),
    ("nb00.promql_window_from_variable_pct", "08_discussion.tex"),
    ("nb00.promql_window_from_variable_pct", "09_conclusion.tex"),
    ("nb00.promql_with_range", "08_discussion.tex"),
    ("nb00.promql_with_variable_pct", "01_introduction.tex"),
    ("nb00.promql_with_variable_pct", "04_grammars.tex"),
    ("nb00.promql_with_variable_pct", "07_evaluation.tex"),
    ("nb00.promql_with_variable_pct", "08_discussion.tex"),
    ("nb00.promql_with_variable_pct", "09_conclusion.tex"),
    ("nb00.promql_with_variable_pct", "body.tex"),
    ("nb03.caveat_rollup_alignment_occurrences", "05_translation.tex"),
    ("nb03.caveat_rollup_alignment_occurrences", "08_discussion.tex"),
    ("nb03.caveat_rollup_alignment_occurrences", "eval_type_caveat.tex"),
    ("nb03.caveat_rollup_alignment_occurrences", "tr_gauge.tex"),
    ("nb03.caveat_rollup_alignment_occurrences", "tr_type_measured.tex"),
    ("nb03.caveat_rollup_alignment_pct", "01_introduction.tex"),
    ("nb03.caveat_rollup_alignment_pct", "05_translation.tex"),
    ("nb03.caveat_rollup_alignment_pct", "07_evaluation.tex"),
    ("nb03.caveat_rollup_alignment_pct", "08_discussion.tex"),
    ("nb03.caveat_rollup_alignment_pct", "eval_beyond_reach.tex"),
    ("nb03.caveat_rollup_alignment_pct", "eval_type_caveat.tex"),
    ("nb03.caveat_rollup_alignment_pct", "ing_replay.tex"),
    ("nb03.caveat_rollup_alignment_pct", "tr_gauge.tex"),
    ("nb03.caveat_rollup_default_occurrences", "08_discussion.tex"),
    ("nb03.caveat_tag_presence_empty_occurrences", "01_introduction.tex"),
    ("nb03.caveat_tag_presence_empty_occurrences", "07_evaluation.tex"),
    ("nb03.caveat_tag_presence_empty_occurrences", "08_discussion.tex"),
    ("nb03.caveat_tag_presence_empty_occurrences", "eval_empty_tag.tex"),
    ("nb03.caveat_tag_presence_empty_occurrences", "eval_rows_fill_glob.tex"),
    ("nb03.caveat_tag_presence_empty_pct", "07_evaluation.tex"),
    ("nb03.caveat_tag_presence_empty_pct", "eval_empty_tag.tex"),
    ("nb03.closure_pct_corpus", "01_introduction.tex"),
    ("nb03.closure_pct_corpus", "05_translation.tex"),
    ("nb03.closure_pct_corpus", "07_evaluation.tex"),
    ("nb03.closure_pct_corpus", "08_discussion.tex"),
    ("nb03.closure_pct_corpus", "09_conclusion.tex"),
    ("nb03.closure_pct_corpus", "body.tex"),
    ("nb03.closure_pct_corpus", "eval_beyond_reach.tex"),
    ("nb03.closure_pct_corpus", "eval_rows_fill_glob.tex"),
    ("nb03.closure_pct_translated", "01_introduction.tex"),
    ("nb03.closure_pct_translated", "05_translation.tex"),
    ("nb03.closure_pct_translated", "07_evaluation.tex"),
    ("nb03.closure_pct_translated", "08_discussion.tex"),
    ("nb03.closure_pct_translated", "eval_beyond_reach.tex"),
    ("nb03.closure_pct_translated", "eval_type_caveat.tex"),
    ("nb03.closure_pct_translated", "ing_replay.tex"),
    ("nb03.closure_pct_translated", "tr_gauge.tex"),
    ("nb03.distinct_after_instantiation", "07_evaluation.tex"),
    ("nb03.feature_ceiling_pct", "07_evaluation.tex"),
    ("nb03.kernel_per_second_occurrences", "05_translation.tex"),
    ("nb03.type_decidable_occurrences", "05_translation.tex"),
    ("nb03.type_decidable_occurrences", "eval_type_caveat.tex"),
    ("nb03.type_decidable_occurrences", "tr_type_measured.tex"),
    ("nb03.type_decidable_pct", "01_introduction.tex"),
    ("nb03.type_decidable_pct", "05_translation.tex"),
    ("nb03.type_decidable_pct", "07_evaluation.tex"),
    ("nb03.type_decidable_pct", "08_discussion.tex"),
    ("nb03.type_decidable_pct", "concl_evidence.tex"),
    ("nb03.type_decidable_pct", "eval_type_caveat.tex"),
    ("nb03.type_decidable_pct", "tr_type_measured.tex"),
    ("nb03.type_rate_occurrences", "eval_top.tex"),
    ("nb03.type_rate_occurrences", "reg_trial_probes.tex"),
    ("nb03.type_rate_occurrences", "tr_type_measured.tex"),
    ("nb03.type_undeclared_occurrences", "07_evaluation.tex"),
    ("nb03.type_undeclared_occurrences", "09_conclusion.tex"),
    ("nb03.type_undeclared_occurrences", "tr_type_measured.tex"),
    ("nb03.type_wrong_counter_occurrences", "05_translation.tex"),
    ("nb03.type_wrong_counter_occurrences", "eval_type_caveat.tex"),
    ("nb03.type_wrong_counter_occurrences", "tr_type_measured.tex"),
    ("nb03.type_wrong_gauge_occurrences", "05_translation.tex"),
    ("nb03.type_wrong_gauge_occurrences", "eval_type_caveat.tex"),
    ("nb03.type_wrong_gauge_occurrences", "tr_type_measured.tex"),
    ("nb03.type_wrong_occurrences", "01_introduction.tex"),
    ("nb03.type_wrong_occurrences", "05_translation.tex"),
    ("nb03.type_wrong_occurrences", "07_evaluation.tex"),
    ("nb03.type_wrong_occurrences", "08_discussion.tex"),
    ("nb03.type_wrong_occurrences", "09_conclusion.tex"),
    ("nb03.type_wrong_occurrences", "body.tex"),
    ("nb03.type_wrong_occurrences", "concl_evidence.tex"),
    ("nb03.type_wrong_occurrences", "eval_type_caveat.tex"),
    ("nb03.type_wrong_occurrences", "tr_type_measured.tex"),
    ("nb03.type_wrong_pct", "01_introduction.tex"),
    ("nb03.type_wrong_pct", "05_translation.tex"),
    ("nb03.type_wrong_pct", "07_evaluation.tex"),
    ("nb03.type_wrong_pct", "08_discussion.tex"),
    ("nb03.type_wrong_pct", "09_conclusion.tex"),
    ("nb03.type_wrong_pct", "body.tex"),
    ("nb03.type_wrong_pct", "concl_evidence.tex"),
    ("nb03.type_wrong_pct", "eval_type_caveat.tex"),
    ("nb03.type_wrong_pct", "tr_type_measured.tex"),
    ("nb03.type_wrong_pct_of_corpus", "05_translation.tex"),
    ("nb03.type_wrong_pct_of_corpus", "eval_type_caveat.tex"),
    ("nb03.type_wrong_pct_of_corpus", "tr_type_measured.tex"),
    ("nb04.concrete_distinct", "06_regular.tex"),
    ("nb04.concrete_distinct", "07_evaluation.tex"),
    ("nb04.concrete_distinct", "reg_complete_dialect.tex"),
    ("nb04.concrete_distinct", "reg_star_empty.tex"),
    ("nb04.glob_union_distinct", "06_regular.tex"),
    ("nb04.glob_union_distinct", "07_evaluation.tex"),
    ("nb04.glob_union_distinct", "eval_type_caveat.tex"),
    ("oo.emits", "07_evaluation.tex"),
    ("oo.emits_where_we_refuse", "01_introduction.tex"),
    ("oo.emits_where_we_refuse", "07_evaluation.tex"),
    ("oo.emits_where_we_refuse", "eval_baseline.tex"),
    ("oo.held_out_multi", "07_evaluation.tex"),
    ("oo.held_out_multi", "eval_baseline.tex"),
    ("oo.ours_emits", "07_evaluation.tex"),
    ("oo.ours_emits", "eval_shape_reduction.tex"),
    ("oo.ours_parses", "07_evaluation.tex"),
    ("oo.ours_parses", "eval_shape_reduction.tex"),
    ("oo.ours_refuses", "01_introduction.tex"),
    ("oo.ours_refuses", "07_evaluation.tex"),
    ("oo.ours_refuses", "eval_baseline.tex"),
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
        for build, text in zip(("main", "main_full"), _math_in_both_builds(path)):
            if not any(s in text for s in _spellings(reg[key])):
                stale.append(
                    f"  {section} should state {key} = {reg[key]} in {build}.pdf")
    assert not stale, "\n".join(stale)


# --- the sentence a figure lives in ---------------------------------------------------
# BOUND asks whether a file states the registry's value anywhere in it. That is
# containment, and containment is weak exactly where a paper is most likely to drift:
# in a sentence that states several figures of the same magnitude.
#
# The held-out kappa is the case that showed it. `ho.ex_kappa_pct` was bound to three
# files, and every one of them passed with the registry set to 96.6 --- the in-sample
# value, the precise drift the binding was added to catch --- because each of those
# files already states 96.6 as the comparator in the same sentence. A twelfth review
# pass set it to 96.6 and watched the whole suite stay green while five files went on
# saying 96.7. The binding recorded a check that could not fail.
#
# These are stronger, and they are stronger in the only way that matters: a rate is
# pinned beside the population it is a rate of. `96.7\% of those 3{,}065 occurrences`
# is a string nothing else in this paper can produce, where `96.7` alone is produced by
# the baseline's coverage-by-valid-output and by three unrelated tables.
#
# Each entry is a fragment of the paper's own LaTeX with `{registry.key}` spliced in,
# asserted to appear in its file after whitespace is flattened, so line wrapping is free
# to move but words and figures are not. The cost is that rewording one of these
# sentences breaks its test. That is the intended trade and the reason the list is
# short: these are the sentences carrying claims the paper cannot afford to state
# stalely, not every sentence with a number in it.

# What a phrase cannot do, established by perturbation rather than guessed at, so that
# the next pass does not add a ledger for each one on discovering it:
#
#   * It pins numerals and the words between them, never the relation. Extending a
#     phrase past its last numeral --- "sits inside the 1.9 points of spread" rather
#     than "sits inside the spread" --- is the only defence, and it works only as far
#     as the phrase runs. It is also no defence against the relation being false: the
#     sentence pinned here before this one said the fall was smaller than the movement
#     on either margin, which it is not, and the phrase held it in place for a commit.
#   * A negating frame placed outside the span passes. "It would be wrong to say any of
#     what follows." before a pinned sentence leaves the suite green.
#   * It says nothing about numerals spelled out in words. "a tenth of a point" is
#     outside every ledger in this file.
#
# A fourth limit was found the same way and is now closed, which is why it is written
# down here rather than left as a note: the check read one document where the paper
# has two. A sentence inside `\shortonly` reaches `main.pdf` and one inside
# `\inlineonly` reaches `main_full.pdf`, so a pinned claim could be put in one and
# its negation in the other with the suite green. The phrase check now resolves each
# build and asks for the sentence in both. That is the same defect as a pin satisfied
# by a comment, one level up, and it was found in the same week as the comment one.
#
# The three above want a reader, and saying so here is the honest form of that. A guard
# that claimed to cover them would be the third mechanism in three passes to record a
# check it was not making.
PHRASES: tuple[tuple[str, str], ...] = (
    # kappa, out of sample, beside its population and its in-sample comparator.
    ("eval_heldout.tex",
     r"returns $\kappa={ho.ex_kappa_pct}\%$ of those ${ho.ex_queries}$ occurrences "
     r"against ${ho.kappa_in_sample_pct}\%$ in sample, and "
     r"${ho.ex_kappa_distinct_pct}\%$ of the ${ho.ex_kappa_distinct}$ distinct "
     r"instantiated forms against ${ho.kappa_in_sample_distinct_pct}\%$"),
    # And what that tenth of a point is worth, which is the claim, not the rate.
    ("eval_heldout.tex",
     r"Treating the ${ho.ex_queries}$ as independent draws gives a standard error of "
     r"${ho.ex_kappa_srs_se_points}$ points. Clustering them by dashboard instead "
     r"gives "
     r"${ho.ex_kappa_se_points}$, a design effect of "
     r"${ho.ex_kappa_design_effect}$"),
    ("eval_heldout.tex",
     r"$\kappa$ is ${ho.ex_kappa_pct}\%$ out of sample against "
     r"${ho.kappa_in_sample_pct}\%$ in it, on ${ho.ex_queries}$ occurrences from "
     r"${ho.ex_dashboards}$ dashboards"),
    ("01_introduction.tex",
     r"the translator reaches ${ho.ex_kappa_pct}\%$ of ${ho.ex_queries}$ occurrences"),
    ("07_evaluation.tex",
     r"is ${ho.ex_kappa_pct}\%$ of those ${ho.ex_queries}$ occurrences against "
     r"${ho.kappa_in_sample_pct}\%$ here"),
    # body.tex states the rate with no population beside it, so the phrase is the
    # pinning: a registry that said 96.6 would require the abstract to say 96.6.
    # "indistinguishable" came out when the abstract stopped claiming the two rates
    # agree: the held-out figure's spread rests on an effective two dashboards, so
    # what the comparison licenses is that nothing large is hiding, which `eval_heldout`
    # had said at length three levels down and the abstract had not.
    ("body.tex",
     r"by other authors it returns ${ho.ex_kappa_pct}\%$"),
    # 1.1 is a cluster-robust standard error, which a 20,000-draw cluster bootstrap
    # puts at 1.085. It is not a drop-one quantity: the largest move any single
    # dashboard makes is 0.93 and the whole drop-one range is 1.01, which is what the
    # appendix's 96.7-to-97.7 prints. Both this phrase and the one in \S7 pinned the
    # drop-one description for a commit, so both named the wrong quantity in step.
    # Repinned when the conclusion stopped setting the distinct-form gap against this
    # figure: the 1.1 is a spread on the occurrence-weighted rate, and the sentence
    # now says which of the two gaps it bounds. `eval_heldout` carried that
    # correction for a commit and the conclusion did not.
    ("09_conclusion.tex",
     r"against the ${ho.ex_kappa_se_points}$ points of dashboard-clustered standard "
     r"error on that occurrence-weighted figure"),
    # The grammar rates, each beside the population it was measured on.
    ("eval_heldout.tex",
     r"Frozen grammar v2 parses ${ho.ex_v2_pct}\%$ of those ${ho.ex_distinct}$ "
     r"strings"),
    ("eval_heldout.tex",
     r"v1 parses ${ho.ex_v1_pct}\%$ against ${ho.dd_in_sample_v1_pct}\%$"),
    ("eval_heldout.tex",
     r"falls from ${ho.dd_in_sample_gain_points}$ points to ${ho.ex_gain_points}$"),
    # The two fit decompositions, each beside the spread that makes it unresolvable.
    # These are the figures the twelfth pass found quoted as thirds and fifths.
    ("eval_heldout.tex",
     r"so the ${ho.gf_shrink_points}$-point difference between them is no larger than "
     r"the ${ho.gf_shrink_se_points}$ points of spread"),
    # Both numerals, and the word that names the statistic. 0.3 is a delete-one
    # cluster jackknife standard error; the largest move any single dashboard makes on
    # the 0.8 is 0.19 and the whole drop-one range is 0.21, so "movement" --- which
    # this phrase pinned for a commit --- overstated it by 43 per cent. That is the
    # same error, on the same page, as the one the commit was correcting for kappa.
    ("eval_heldout.tex",
     r"a dashboard-level jackknife puts a standard error of "
     r"${ho.gf_novel_gain_se_points}$ points on each"),
    ("eval_heldout.tex",
     r"a standard error of ${ho.ex_gain_se_points}$ points on the held-out margin and "
     r"${ho.dd_in_sample_gain_se_points}$ on the in-sample one, which in quadrature is "
     r"${ho.ex_shrink_se_points}$ points on the ${ho.ex_shrink_points}$-point fall"),
    # The census, which is why the paragraph may decline to report a sampling error.
    # Read and shipped as two keys on either side of the "of", because one key on
    # both sides could not fail. And the bridge from the census to the population
    # the comparison is made over, which a reader otherwise has to work out.
    ("eval_heldout.tex",
     r"${ho.ex_dashboards_read}$ of the ${ho.ex_dashboards_at_commit}$ dashboard "
     r"files \texttt{integrations-extras} ships at its pinned commit were read, "
     r"and ${ho.dd_corpus_dashboards_read}$ of the "
     r"${ho.dd_corpus_dashboards_at_commit}$ \texttt{integrations-core} ships at "
     r"the corpus's own, the other ${ho.dd_corpus_unreadable}$"),
    ("eval_heldout.tex",
     r"Of the ${ho.dd_corpus_dashboards_read}$, the "
     r"${ho.dd_in_sample_dashboards}$ that carry a surviving Datadog query are "
     r"the corpus side of every comparison here"),
    # And what the dashboard-level movement rests on, beside the figure itself.
    ("eval_heldout.tex",
     r"effective number of dashboards behind it is "
     r"${ho.ex_kappa_effective_dashboards}$, one of them carrying "
     r"${ho.ex_kappa_top_dashboard_pct}\%$ of it, against "
     r"${ho.kappa_in_sample_effective_dashboards}$ on the corpus side"),
    # Both numerals, because the claim is that the first is inside the second and a
    # phrase cannot pin a relation. The sentence this replaced was pinned here while
    # it was false: the fall is 1.4538 and the in-sample margin's own spread is 0.8888,
    # so "smaller than the movement either figure carries" was wrong on the figure the
    # reader would check first. Extending a phrase to make it bite harder extended it
    # over a claim that had not been checked.
    # "dashboard-level jackknives" became "delete-one-dashboard jackknives" when the
    # two margin errors a few lines away stopped being called clustered ones: one
    # procedure, one name for it, everywhere in the paper.
    ("04_grammars.tex",
     r"the ${ho.ex_shrink_points}$-point fall is no larger than the "
     r"${ho.ex_shrink_se_points}$ points of spread it inherits from the two "
     r"delete-one-dashboard jackknives behind it"),
    ("07_evaluation.tex",
     r"${ho.gf_shrink_se_points}$ points on the first and "
     r"${ho.ex_shrink_se_points}$ on the second"),
    ("07_evaluation.tex",
     r"its margin over v1 is ${ho.gf_novel_gain_points}$ points against "
     r"${ho.gf_in_sample_gain_points}$ here"),
    ("07_evaluation.tex",
     r"its margin over v1 is ${ho.ex_gain_points}$ points against the seven"),
    ("07_evaluation.tex",
     r"a tenth of a point where the dashboard-clustered standard error on the "
     r"held-out figure is ${ho.ex_kappa_se_points}$"),
    # The two ceilings, which must not be quoted under one name.
    ("eval_heldout.tex",
     r"$S^\star$ covers ${ho.ex_sstar_pct}\%$ of these occurrences"),
    ("eval_heldout.tex",
     r"would overstate the reachable ceiling by "
     r"${ho.ex_ceiling_overstatement_points}$ points"),
    # The caveat the subsection ends on, and the formula layer as it ships.
    ("eval_heldout.tex",
     r"yet ${ho.ex_feature_set_seen_pct}\%$ of its occurrences have a feature set "
     r"that occurs there verbatim"),
    ("eval_heldout.tex",
     r"it is ${ho.ex_formula_queries_at_commit}$ occurrences in "
     r"${ho.ex_formula_forms_at_commit}$ distinct forms, of which "
     r"${ho.ex_formula_shared_forms}$ forms carrying "
     r"${ho.ex_formula_shared_queries}$ occurrences"),
    # The template share, whose two halves §7 used to present as two channels.
    ("07_evaluation.tex",
     r"${ho.dd_in_sample_template_pct}\%$ of corpus occurrences carry a template "
     r"variable against ${ho.ex_template_pct}\%$"),
)

_PHRASE_KEY = re.compile(r"\{([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)\}")


def _spell(value: int | float) -> str:
    """One spelling, the one this paper uses: `3{,}065`, `96.7`, `7.0`, `139`.

    A count and a rate of the same magnitude are written differently and the registry
    already distinguishes them --- `ho.ex_gain_points` is the JSON float 7.0 and the
    paper writes `$7.0$`, while `ho.ex_dashboards` is the integer 139. Collapsing 7.0
    to `7` here would have made three of these phrases unsatisfiable and invited the
    obvious repair, which is to accept both spellings and so to stop distinguishing
    the two kinds of figure.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}".replace(",", "{,}")
    return repr(float(value))


def test_a_pinned_sentence_still_says_what_the_registry_says():
    """A headline figure must appear beside the population it is a figure of.

    This is the check BOUND cannot make. `ho.ex_kappa_pct` was bound to three files
    and the binding passed with the registry set to the in-sample value, because each
    of those files states the in-sample value in the same sentence as the held-out
    one. Containment cannot tell a rate from its own comparator. A phrase can.
    """
    reg = json.loads(REGISTRY.read_text())
    missing = []
    for section, template in PHRASES:
        path = MAIN if section == "body.tex" else _section_path(section)
        for key in _PHRASE_KEY.findall(template):
            assert key in reg, f"{key} is pinned but no longer in the registry"
        want = _PHRASE_KEY.sub(lambda m: _spell(reg[m.group(1)]), template)
        # Both builds, because there are two PDFs and `\inlineonly` reaches only one
        # of them. A sentence pinned here has to be in each; the alternative is a
        # pin satisfied by a condition no reader ever sees resolved, which is the
        # same defect as a pin satisfied by a comment one level up.
        for full in (False, True):
            if flat(want) not in flat(rendered(path, full)):
                build = "main_full" if full else "main"
                missing.append(f"  {section} in {build}.pdf should contain: {flat(want)}")
    assert not missing, "\n".join(missing)


def test_every_pinned_sentence_pins_something():
    """A phrase with no registry key in it records a check that cannot drift."""
    bare = [t for _, t in PHRASES if not _PHRASE_KEY.search(t)]
    assert not bare, f"pinned sentences with no registry key: {bare}"


def test_the_kappa_each_side_quotes_is_the_one_measured_on_that_side():
    """The two kappas are a tenth of a point apart, so only a phrase separates them.

    Setting `ho.ex_kappa_pct` to the in-sample value must fail. Before the phrases
    above it did not: every file bound to it already carried that value as the
    comparator, so the suite stayed green while the paper went on stating the other
    one in five places. This asserts the hole is shut, by checking that no file
    pinning the held-out rate would still pin it were the registry to hold the
    in-sample one.
    """
    reg = json.loads(REGISTRY.read_text())
    swapped = dict(reg, **{"ho.ex_kappa_pct": reg["ho.kappa_in_sample_pct"]})
    assert reg["ho.ex_kappa_pct"] != swapped["ho.ex_kappa_pct"], \
        "the two kappas have converged; this test no longer separates anything"
    caught = []
    for section, template in PHRASES:
        if "ho.ex_kappa_pct" not in template:
            continue
        path = MAIN if section == "body.tex" else _section_path(section)
        want = _PHRASE_KEY.sub(lambda m: _spell(swapped[m.group(1)]), template)
        caught.append(all(flat(want) not in flat(rendered(path, full))
                          for full in (False, True)))
    assert caught and all(caught), \
        "a file pinning the held-out kappa would accept the in-sample one"


def test_every_binding_names_a_distinctive_value():
    """A binding on a small count records a check that cannot fail; keep them out."""
    reg = json.loads(REGISTRY.read_text())
    weak = sorted({
        f"{key} = {reg[key]}" for key, _ in BOUND
        if "pct" not in key.rsplit(".", 1)[-1].split("_")
        and float(reg[key]) == int(reg[key]) and int(reg[key]) < 100
    })
    assert not weak, f"bound to a value too common to mean anything: {weak}"


# --- the registry's own arithmetic ----------------------------------------------------
# Everything above asks whether the paper still says what the registry says. Nothing
# asked whether the registry is self-consistent, and a share is the figure where that
# matters: it is a claim about a numerator and a denominator, and the denominator is the
# half a reader never sees. §5 once quoted $87\%$ for the kernel's reach when $87\%$ was
# the implementation-order figure -- two correct numbers, one of them answering the wrong
# question. A binding catches a stale copy of the answer. Only the question catches this,
# so the triples below record it: each share, what it counts, and what it counts out of.
#
# A share whose numerator is not itself registered cannot appear here, which is a reason
# to register one. `oo.substantive_note` was added for exactly that.

SHARES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("nb00.datadog_with_variable_pct", ("nb00.datadog_with_variable",),
     "nb03.corpus_occurrences"),
    ("nb00.promql_window_from_variable_pct", ("nb00.promql_window_from_variable",),
     "nb00.promql_with_range"),
    # The caveat shares are over what was translated, not over the corpus: a caveat is
    # carried by a translation, so a query that never translated cannot carry one.
    ("nb03.caveat_counter_reset_pct", ("nb03.caveat_counter_reset_occurrences",),
     "nb03.kappa_occurrences"),
    ("nb03.caveat_metric_type_unknown_pct",
     ("nb03.caveat_metric_type_unknown_occurrences",), "nb03.kappa_occurrences"),
    ("nb03.caveat_phi_series_start_pct", ("nb03.caveat_phi_series_start_occurrences",),
     "nb03.kappa_occurrences"),
    ("nb03.caveat_rollup_alignment_pct", ("nb03.caveat_rollup_alignment_occurrences",),
     "nb03.kappa_occurrences"),
    ("nb03.caveat_rollup_default_pct", ("nb03.caveat_rollup_default_occurrences",),
     "nb03.kappa_occurrences"),
    ("nb03.caveat_tag_presence_empty_pct",
     ("nb03.caveat_tag_presence_empty_occurrences",), "nb03.kappa_occurrences"),
    # The same count over both bases, which is the pair §5 confused once already.
    ("nb03.closure_pct_corpus", ("nb03.closure_occurrences",), "nb03.corpus_occurrences"),
    ("nb03.closure_pct_translated", ("nb03.closure_occurrences",),
     "nb03.kappa_occurrences"),
    ("nb03.kernel_pct_corpus", ("nb03.kernel_occurrences",), "nb03.corpus_occurrences"),
    ("nb03.kernel_pct_translated", ("nb03.kernel_occurrences",),
     "nb03.kappa_occurrences"),
    ("nb03.kappa_pct", ("nb03.kappa_occurrences",), "nb03.corpus_occurrences"),
    ("nb03.language_gap_pct", ("nb03.language_gap_occurrences",),
     "nb03.corpus_occurrences"),
    ("nb03.not_implemented_pct", ("nb03.not_implemented_occurrences",),
     "nb03.corpus_occurrences"),
    # The ceiling is what implementing the unimplemented set would reach, so it is the
    # translated and the unimplemented together -- and the distance to 100% is the
    # queries that do not parse, not a language barrier.
    ("nb03.ceiling_pct", ("nb03.kappa_occurrences", "nb03.not_implemented_occurrences"),
     "nb03.corpus_occurrences"),
    ("nb03.type_decidable_pct", ("nb03.type_decidable_occurrences",),
     "nb03.kappa_occurrences"),
    # And the one share in this group taken over the decidable set rather than either
    # of the two obvious bases, which is why it is written down here.
    ("nb03.type_wrong_pct", ("nb03.type_wrong_occurrences",),
     "nb03.type_decidable_occurrences"),
    ("nb03.type_wrong_pct_of_corpus", ("nb03.type_wrong_occurrences",),
     "nb03.corpus_occurrences"),
    ("nb04.glob_exists_pct", ("nb04.glob_exists_occurrences",),
     "nb04.concrete_occurrences"),
    ("nb04.glob_single_pct", ("nb04.glob_single_occurrences",),
     "nb04.concrete_occurrences"),
    ("nb04.glob_union_pct", ("nb04.glob_union_occurrences",),
     "nb04.concrete_occurrences"),
    ("nb04.no_glob_pct", ("nb04.no_glob_occurrences",), "nb04.concrete_occurrences"),
    ("oo.cosmetic_note_only_pct", ("oo.cosmetic_note_only",), "oo.emits"),
    ("oo.invented_window_pct", ("oo.invented_window",), "oo.emits"),
    ("oo.substantive_note_pct", ("oo.substantive_note",), "oo.emits"),
)

# Parts that must exhaust a whole. Each is a partition the notebooks assert while they
# run; the registry is what survives the run, and these say the same thing about it.
PARTITIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("the three-way glob split, by occurrence",
     ("nb04.glob_single_occurrences", "nb04.glob_union_occurrences",
      "nb04.no_glob_occurrences"), "nb04.concrete_occurrences"),
    ("the three-way glob split, by distinct pattern",
     ("nb04.glob_single_distinct", "nb04.glob_union_distinct", "nb04.no_glob_distinct"),
     "nb04.concrete_distinct"),
    ("a glob exists, by occurrence",
     ("nb04.glob_single_occurrences", "nb04.glob_union_occurrences"),
     "nb04.glob_exists_occurrences"),
    ("a glob exists, by distinct pattern",
     ("nb04.glob_single_distinct", "nb04.glob_union_distinct"),
     "nb04.glob_exists_distinct"),
    ("the declared type is wrong in one of two directions",
     ("nb03.type_wrong_gauge_occurrences", "nb03.type_wrong_counter_occurrences"),
     "nb03.type_wrong_occurrences"),
    ("every query the baseline emits for carries a note of one kind or none",
     ("oo.cosmetic_note_only", "oo.substantive_note", "oo.no_note"), "oo.emits"),
)


def test_every_recorded_share_is_its_own_numerator_over_its_own_denominator():
    """A share must equal the count it names, over the base it names.

    This is the check the registry could not make about itself. A notebook computing a
    share against the wrong population writes a number that is correct arithmetic about
    a quantity nobody asked for, and every other test here passes: the prose matches the
    key, the key is in the registry, and the registry was written by the notebook.
    """
    reg = json.loads(REGISTRY.read_text())
    wrong = []
    for share, numerators, denominator in SHARES:
        for key in (share, denominator, *numerators):
            assert key in reg, f"{key} is named by SHARES but not in the registry"
        got = round(sum(reg[k] for k in numerators) / reg[denominator] * 100, 1)
        if got != reg[share]:
            wrong.append(f"  {share} = {reg[share]}, but "
                         f"{' + '.join(numerators)} / {denominator} = {got}")
    assert not wrong, "\n".join(wrong)


def test_the_registry_parts_add_up_to_their_totals():
    """A split that does not exhaust its whole has lost a class or double-counted one."""
    reg = json.loads(REGISTRY.read_text())
    wrong = []
    for name, parts, total in PARTITIONS:
        got = sum(reg[p] for p in parts)
        if got != reg[total]:
            wrong.append(f"  {name}: {' + '.join(parts)} = {got}, "
                         f"but {total} = {reg[total]}")
    assert not wrong, "\n".join(wrong)


def test_the_two_note_shares_are_not_complements():
    """The trap in the pair above, written down so that nobody closes it.

    `oo.cosmetic_note_only` and `oo.substantive_note` both require a note, so the
    queries carrying none are in neither and the two shares sum to $99.9\\%$ rather
    than to $100\\%$. Reading either as `100 - the other` is off by those ten, which is
    the kind of arithmetic that looks like a rounding error and is not one.
    """
    reg = json.loads(REGISTRY.read_text())
    assert reg["oo.no_note"] > 0
    assert reg["oo.cosmetic_note_only_pct"] + reg["oo.substantive_note_pct"] < 100.0


def test_the_baseline_names_no_gap_the_catalogue_has_dropped():
    """A registry key outlives the quantity it measured, because the writers merge.

    `oo.gap_label_name_glob` recorded 31 occurrences of a gap code that left the
    catalogue when `corpus._TMPL` stopped truncating a template variable at a hyphen.
    The script stopped writing it; the key stayed, and went on licensing the numeral
    $31$ in the prose. The registry is append-only in practice, so the figures it holds
    have to be tied to something that shrinks.
    """
    reg = json.loads(REGISTRY.read_text())
    catalogue = table(ROOT / "paper" / "tables" / "gap_catalogue.tex")
    for key in sorted(k for k in reg if k.startswith("oo.gap_")):
        code = key[len("oo.gap_"):].upper().replace(".", "_")
        assert code.replace("_", r"\_") in catalogue, \
            f"{key} names a gap code the catalogue no longer lists"


def test_the_coverage_staircase_is_a_staircase():
    """The 29 cumulative shares §7.3 draws its argument from, as a shape.

    The claim there is not any one of these numbers; it is that the curve is a
    staircase -- flat for six features, then a jump of 28 points, then a long shallow
    climb -- which is why coverage cannot be bought incrementally. Nothing checked the
    shape. A step lost to a renamed key, or a cumulative share that went down, would
    leave 29 individually plausible figures and no staircase, and the only reader who
    would notice is one holding the figure up to the prose.
    """
    reg = json.loads(REGISTRY.read_text())
    steps = [(i, reg[f"nb00.ceiling_{i}_features_pct"]) for i in range(100)
             if f"nb00.ceiling_{i}_features_pct" in reg]
    assert [i for i, _ in steps] == list(range(len(steps))), \
        f"the staircase has a gap: {[i for i, _ in steps]}"
    # One step per feature, plus the empty set at the foot of it.
    assert len(steps) == reg["nb00.datadog_features"] + 1, steps
    falls = [(a, b) for (_, a), (_, b) in zip(steps, steps[1:]) if b < a]
    assert not falls, f"cumulative coverage fell: {falls}"
    assert steps[-1][1] == 100.0, steps[-1]
    # And the jump that makes it a staircase rather than a curve is still there, or
    # §7.3's argument is about a different shape than the one the registry holds.
    assert max(b - a for (_, a), (_, b) in zip(steps, steps[1:])) > 20


# --- allowlisted figures that can be derived after all --------------------------------
# An allowlist entry records why a number is unchecked, and that is the whole of what it
# records. It does not notice when the number stops being true. §8 said the glob
# refusals were $4.5\%$ of every matcher for as long as the entry beside it read "388 of
# nb04.matchers", months after the numerator became 84: the derivation was written down
# in English, where nothing runs it. Three figures were found wrong that way in one
# afternoon, each with an accountant sitting beside it.
#
# So the derivations that *can* run, do. Each entry below is the figure, a sentence, and
# a function of something committed -- the registry, a cache under `data/raw`, a
# generated table, or a module constant. A figure with no committed source stays in
# ALLOWED with its reason, which is the honest state for a superseded run's count or a
# number read off a vendor's documentation; what is not honest is leaving a division
# there in prose.

RAW = ROOT / "data" / "raw"


def _cache(relative: str) -> dict:
    return json.loads((RAW / relative).read_text())


def _reg() -> dict:
    return json.loads(REGISTRY.read_text())


def _star_rows() -> list[dict]:
    return _cache("globs/star_reading.json")["rows"]


def _alphabet() -> str:
    from qlport.regular import TAG_ALPHABET

    return TAG_ALPHABET


def _flat_prefix_completions() -> int:
    """Queries completed by the greedy order before its step, off the order table.

    Read as a prefix length and a sum rather than as a literal, because the step moved
    when `greedy`'s tie-break stopped depending on the interpreter's hash salt and the
    sentence quoting this stayed at six features for a commit.
    """
    body = [r.split("&") for r in table(ORDER).splitlines() if r.count("&") == 5]
    rows = [[c.strip() for c in r] for r in body if r[0].strip().isdigit()]
    step = next(i for i, r in enumerate(rows) if int(r[4].replace(",", "")) > 100)
    return sum(int(r[4].replace(",", "")) for r in rows[:step])


def _replay_per_minute() -> list[float]:
    """Requests a minute in the aligned replay recording, across status classes."""
    from collections import defaultdict

    rec = _cache("replay/dualwrite_aligned.json")
    candidate = rec["rows"][0]["candidate"]
    per_bucket: dict[object, float] = defaultdict(float)
    for row in rec["rows"]:
        if row["candidate"] == candidate and row["instant"] == "inside":
            per_bucket[row["bucket"]] += row["datadog"]
    return sorted(per_bucket.values())


DERIVED: tuple[tuple[str, float, str, object], ...] = (
    ("num", 163, "calls of top() in the corpus, which is the denominator the function "
                 "mix is over rather than the 139 occurrences beside it",
     lambda: sum(_reg()[f"nb00.top_rank_{f}_calls"]
                 for f in ("mean", "last", "max", "sum"))),
    ("num", 81, "the quietest minute of the replayed trace",
     lambda: round(_replay_per_minute()[0])),
    ("num", 402, "and the busiest, which is the factor of five the appendix claims",
     lambda: round(_replay_per_minute()[-1])),
    ("num", 0.04, "the Monte-Carlo error of one seed's own staircase mean",
     lambda: round(_reg()["cv.dd_stair_gap_16_replicate_sd"]
                   / (_reg()["cv.stair_reps"] / 5) ** 0.5, 2)),
    ("num", 44, "queries the flat prefix of the greedy order completes, summed off "
                "the table that generates the staircase",
     _flat_prefix_completions),
    ("num", 7.4, "the eight misclassifications as a share of the corpus",
     lambda: round(648 / _reg()["nb03.corpus_occurrences"] * 100, 1)),
    ("num", 648, "and their sum, with the four disjunctions a residue of the 124",
     lambda: 348 + 124 + 70 + 36 + 31 + 34 + 5),
    ("num", 0.8, "points of kappa that withdrawing .weighted() closed",
     lambda: round(70 / _reg()["nb03.corpus_occurrences"] * 100, 1)),
    ("num", 1.4, "and points the single-key disjunction rule closed",
     lambda: round(120 / _reg()["nb03.corpus_occurrences"] * 100, 1)),
    ("num", 887, "translated occurrences whose declared type is not decidable",
     lambda: _reg()["nb03.kappa_occurrences"]
     - _reg()["nb03.type_decidable_occurrences"]),
    ("num", 7867, "PromQL panel queries: the population less its alerting and recording rules",
     lambda: 8102 - 139 - 96),
    ("num", 41, "the tag alphabet", lambda: len(_alphabet())),
    ("num", 255, "Datadog dashboards a tenth-of-the-corpus vocabulary does not read",
     lambda: _reg()["cv.dd_dashboards"] - _reg()["cv.dd_tenth_dashboards"]),
    ("num", 248, "what the appendix's stale three-code description of the refused column "
          "summed to, against the 287 it was describing",
     lambda: 208 + 4 + 36),
    ("num", 541, "tag keys only ever filtered and never grouped: what widening the census "
          "from the grouping lists to every key a matcher can name added",
     lambda: 1264 - 723),
    ("num", 37070, "declared metric names: the unambiguous ones plus the three two "
            "integrations declare incompatibly",
     lambda: 23597 + 13140 + 330 + 3),
    ("num", 14, "PHI_QUANTILE occurrences whose metric the repository does not declare, so "
         "the corollary's hypothesis is untested rather than shown to fail",
     lambda: 34 - 20),
    ("num", 17128, "both translators' emitted expressions, every one of which has now been "
            "through Prometheus's own parser: ours over the whole corpus, theirs over "
            "the held-out set",
     lambda: _reg()["pt.ours_occ"] + _reg()["pt.theirs_occ"]),
    ("num", 1681, "the branch bound, which is the square of it", lambda: len(_alphabet()) ** 2),
    ("num", 3362, "and twice that, which is what `2..|5..` needs",
     lambda: 2 * len(_alphabet()) ** 2),
    ("num", 3.5, "the differential's largest deviation, in units of 1e-14",
     lambda: round(max(r["max_d"] for r in _cache("shapes/differential.json")["rows"]
                       if r["status"] == "compared") / 1e-14, 1)),
    ("num", 2998, "what the unbudgeted cross-check took, seconds",
     lambda: round(_cache("globs/crosscheck.json")["provenance"]["elapsed_seconds"])),
    ("num", 184, "the concrete patterns it re-decided",
     lambda: _cache("globs/crosscheck.json")["provenance"]["n_concrete_regexes"]),
    ("num", 599, "occurrences refused under the other reading of a glob's star",
     lambda: _cache("globs/star_reading.json")["other"]["none"]["occurrences"]),
    ("num", 515, "and occurrences that change classification between the two readings",
     lambda: _cache("globs/star_reading.json")["moved"]["occurrences"]),
    ("num", 72, "over this many patterns",
     lambda: _cache("globs/star_reading.json")["moved"]["distinct"]),
    ("num", 238, "occurrences of the patterns containing `.+`, which is what the alphabet "
          "question is worth",
     lambda: sum(r["occurrences"] for r in _star_rows() if ".+" in r["regex"])),
    ("num", 21, "and how many patterns those are",
     lambda: sum(1 for r in _star_rows() if ".+" in r["regex"])),
    ("num", 71, "of the 74 the branch bound refuses, the two device-name alternations",
     lambda: sum(r["occurrences"] for r in _star_rows()
                 if r["taken"] == "none" and "mmcblk.p" in r["regex"])),
    # E1: the scripted counter reset, read off the account rather than off the prose.
    ("num", 103, "what Datadog stored over the reset bucket",
     lambda: _cache("datadog/e1.json")["cases"]["to100"]["reset_bucket_observed"][0]),
    ("num", 897, "what a raw signed difference would have stored, in magnitude",
     lambda: -_cache("datadog/e1.json")["cases"]["to100"]
     ["reset_bucket_predicted"]["raw"][0]),
    ("num", 1000, "and so the counter value just before the reset",
     lambda: _cache("datadog/e1.json")["cases"]["to100"]["reset_bucket_observed"][0]
     - _cache("datadog/e1.json")["cases"]["to100"]["reset_bucket_predicted"]["raw"][0]),
    ("num", 111, "the fast run's reset bucket at three samples per flush",
     lambda: _cache("datadog/e1_fast.json")["cases"]["to100"]
     ["reset_bucket_observed"][0]),
    ("num", 110, "and what losing the increment sharing a flush with it would have given",
     lambda: _cache("datadog/e1_fast.json")["cases"]["to100"]
     ["reset_bucket_observed"][0] - 1),
    # E7: whether a counter and a rate read alike over a window.
    ("num", 0.5, "the relative difference at the start of the run",
     lambda: _cache("datadog/e7.json")["summary"]["counter_first"]),
    ("num", 0.994, "and after half an hour",
     lambda: round(_cache("datadog/e7.json")["summary"]["counter_last"], 3)),
    ("num", 1760, "avg_over_time of the running totals at the end of it",
     lambda: _cache("datadog/e7.json")["summary"]["promql_counter_last"]),
    # E9: what .weighted() computes, which is the same as what it does not.
    ("num", 66.7, "the plain mean of the irregular series, with and without the modifier",
     lambda: round(_cache("datadog/e9.json")["promql"]["per_series_mean"]["irregular"],
                   1)),
    # The declared-type census, which §7.3 once stated three too many of, twice.
    ("num", 37070, "metric names integrations-core declares at the pinned commit",
     lambda: sum(_cache("datadog/type_declared.json")["declared_population"].values())),
    ("num", 23597, "of those, declared gauge",
     lambda: _cache("datadog/type_declared.json")["declared_population"]["gauge"]),
    ("num", 13140, "and declared count",
     lambda: _cache("datadog/type_declared.json")["declared_population"]["count"]),
    # The one replay bucket where the two engines differ in the last bit, which is why
    # every exactness column in this paper has a tolerance rather than an equality.
    ("num", 236, "the aligned replay bucket where they differ",
     lambda: next(r["datadog"] for r in _cache("replay/dualwrite_aligned.json")["rows"]
                  if 0 < r["d"] < 1e-12)),
    ("num", 235.999, "and what PromQL read for it, truncated",
     lambda: __import__("math").floor(
         next(r["promql"] for r in _cache("replay/dualwrite_aligned.json")["rows"]
              if 0 < r["d"] < 1e-12) * 1000) / 1000),
)


def test_no_derived_kind_grants_more_than_it_has_to():
    """A derivation that names two kinds must need both of them.

    A `DERIVED` entry does three jobs: its arithmetic is recomputed, its figure must
    still be quoted, and its `(kind, value)` key licenses that spelling in the prose.
    The first two keep an entry honest even where the third licenses nothing, which is
    why a single inert kind is left alone. A *second* kind does only the third job, so
    an inert one is a grant with no reason -- and an over-broad kind is how the free
    pass this ledger just gave up would come back.
    """
    base = _unaccounted(_keyed(ALLOWED), _keyed(ALLOWED_IN))
    real = _derived_keys
    pointless = []
    try:
        for name, ledger in (("DERIVED", DERIVED), ("CORPUS_DERIVED", CORPUS_DERIVED)):
            for kinds, value, note, _ in ledger:
                declared = kinds.split()
                if len(declared) < 2:
                    continue
                for kind in declared:
                    key = (kind, round(float(value), 4))
                    globals()["_derived_keys"] = lambda k=key: real() - {k}
                    if _unaccounted(_keyed(ALLOWED), _keyed(ALLOWED_IN)) == base:
                        pointless.append(f"  {name}: {kind} {value} ({note})")
    finally:
        globals()["_derived_keys"] = real
    assert not pointless, (
        "these derivations name a kind that licenses nothing:\n"
        + "\n".join(pointless) + "\n  (declare the one kind the paper prints)")


def test_every_derivable_figure_still_derives():
    """A figure whose derivation can be executed must equal what it computes.

    This is the check the allowlist cannot make about its own reasons. A reason is
    prose: it can name a numerator that has moved, a denominator that has been
    recounted, or a cache that has been re-recorded, and go on reading correctly.
    """
    wrong = []
    for _, value, note, derive in DERIVED:
        got = derive()
        if round(float(got), 4) != round(float(value), 4):
            wrong.append(f"  {value} ({note}) derives as {got}")
    assert not wrong, "\n".join(wrong)


def test_every_derivable_figure_is_still_quoted():
    """A derivation for a figure the paper no longer states is dead weight."""
    prose = set(_prose_figures())
    missing = [f"  {kind} {v} ({note})" for kinds, v, note, _ in DERIVED
               for kind in kinds.split()
               if (kind, round(float(v), 4)) not in prose]
    assert not missing, "derived but no longer in the paper:\n" + "\n".join(missing)


# --- figures the paper quotes as history ----------------------------------------------
# A historical figure is a claim about two numbers: the one that was and the one that is.
# "$22{,}608$ buckets in the second run" is only worth printing beside the third run's
# $22{,}692$, and it is the live half that moves. The stale half cannot go stale -- it is
# already a record -- so nothing here could ever fail on it, which is exactly why an
# allowlist entry for a historical value is weaker than it looks: it licenses the old
# number for ever and says nothing about whether the sentence around it is still true.
#
# Each entry names the old value, the registry key holding the figure that replaced it,
# and the file where the comparison is made. The test asks that both still appear there
# together. If a fourth differential run moves `nb05.bucket_comparisons`, §7.4's
# "the same $329$ of $329$ over the same $22{,}608$ buckets" becomes a comparison with
# nothing on the other side, and this is what notices.

SUPERSEDED: tuple[tuple[float, str, str], ...] = (
    (0.23, "cv.dd_stair_gap_16_seed_range", "eval_heldout.tex"),
    (22608, "nb05.bucket_comparisons", "07_evaluation.tex"),
    (325, "nb05.shapes", "eval_differential_fixes.tex"),
    (335, "nb05.shapes", "eval_differential_fixes.tex"),
    (332, "nb05.shapes", "eval_differential_fixes.tex"),
    (95.6, "nb03.kappa_pct", "eval_rows_fill_glob.tex"),
    (94.2, "nb03.kappa_pct", "eval_rows_fill_glob.tex"),
    (76.2, "nb03.caveat_metric_type_unknown_pct", "tr_gauge.tex"),
)


def test_every_historical_figure_is_quoted_beside_the_one_that_replaced_it():
    reg = json.loads(REGISTRY.read_text())
    broken = []
    for old, key, section in SUPERSEDED:
        assert key in reg, f"{key} supersedes {old} but is not in the registry"
        path = MAIN if section == "body.tex" else _section_path(section)
        for build, text in zip(("main", "main_full"), _math_in_both_builds(path)):
            if not any(s in text for s in _spellings(old)):
                broken.append(
                    f"  {section} no longer states the historical {old} in {build}.pdf")
            elif not any(s in text for s in _spellings(reg[key])):
                broken.append(f"  {section} states {old} but not {key} = {reg[key]} "
                              f"in {build}.pdf, so the comparison has nothing on the "
                              "other side")
    assert not broken, "\n".join(broken)


def test_a_historical_figure_is_not_the_figure_it_supersedes():
    """A record that equals the live value is not a record of anything.

    It would also pass the test above by accident, which is the failure mode worth
    keeping out: a lineage collapses to one number and the prose still reads as though
    it moved.
    """
    reg = json.loads(REGISTRY.read_text())
    collapsed = [f"  {old} == {key}" for old, key, _ in SUPERSEDED
                 if round(float(reg[key]), 4) == round(float(old), 4)]
    assert not collapsed, "\n".join(collapsed)


# --- figures accounted for by a coincidence -------------------------------------------
# `_accounted` licenses a prose figure if any cell of any generated table carries the
# same magnitude and the same kind. For most figures that cell is the figure's own
# source. For these it is not: §8's $648$ misclassified occurrences were accepted
# because `regex_glob.tex` happens to print $648$ for the occurrences a union of globs
# covers, which is a different quantity in a different section of the paper.
#
# That is not a hole to close -- a table is a legitimate source and the guard cannot
# tell which table a sentence meant -- but it is a hole to *know the size of*. These are
# the figures whose only accountant is a table that is not about them. A new one means
# either that a figure has lost its source or that a coincidence has been mistaken for
# one, and both deserve a decision rather than silence: register the figure in a
# notebook, bind it with BOUND, or add it here.

TABLE_COINCIDENCE: frozenset[tuple[str, float]] = frozenset({
    ("num", 8), ("num", 11), ("num", 17), ("num", 26), ("num", 32),
    ("num", 35), ("num", 55), ("num", 60), ("num", 64), ("num", 74),
    # 95: the upper edge of the vocabulary figure's bootstrap band, which the gap
    # tables also happen to print. It was accounted for by a derivation of the per cent
    # of Datadog space aggregators that are sum or avg until that ledger learnt kinds.
    # 54: E9's superseded count of forms attaching .weighted() to sum:, which this
    # ledger carried a sentence for until the greedy tie-break fix rewrote
    # implementation_order and its pending-uses column started printing a 54. The
    # sentence is the better record and the coincidence is what the guard now sees.
    ("num", 54),
    # 200: the most-downloaded Prometheus dashboards the corpus took, licensed by
    # `cv.reps` until the cross-validation pooled five seeds and that key became 1,000.
    # It was never a replicate count; corpus_sources carries the real one.
    ("num", 200),
    ("num", 91), ("num", 95), ("num", 100), ("num", 160), ("num", 208), ("num", 262),
    ("num", 266), ("num", 300), ("num", 329), ("num", 330), ("num", 1680),
    ("num", 1766), ("num", 7546), ("num", 8102), ("num", 8365),
    # 80: the proverb's 80% of the queries, and one of the four thresholds the shape
    # distribution is summarised at -- the measured figures there are the shape counts
    # beside each threshold, not the thresholds. It was licensed by
    # `cv.pq_complete_at_pct` until pooling five seeds showed PromQL has no saturation
    # point at any sample size and the run stopped emitting that key at all.
    ("pct", 80),
    ("pct", 3.8), ("pct", 5.3), ("pct", 11.8), ("pct", 20.2),
    ("pct", 27.4), ("pct", 71.2), ("pct", 83.6), ("pct", 91.4),
    ("pct", 94.7), ("pct", 98.3),
    ("pct", 99.1),
})


def _unchecked() -> list[tuple[str, float]]:
    """Prose figures with no table, no registry key and no derivation behind them."""
    reg: set[tuple[str, float]] = set()
    for name, value in json.loads(REGISTRY.read_text()).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            share = "pct" in name.rsplit(".", 1)[-1].split("_")
            reg.add(("pct" if share else "num", round(float(value), 4)))
    derived = _derived_keys()
    return sorted(k for k in _prose_figures()
                  if k not in reg and k not in _accounted() and k not in derived)


def _table_only() -> dict[tuple[str, float], tuple[list[str], list[str]]]:
    """Prose figures no registry key accounts for, keyed to the tables that do."""
    reg: set[tuple[str, float]] = set()
    for name, value in json.loads(REGISTRY.read_text()).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            share = "pct" in name.rsplit(".", 1)[-1].split("_")
            reg.add(("pct" if share else "num", round(float(value), 4)))
    tables: dict[tuple[str, float], list[str]] = {}
    for path in TABLES:
        for tok, pct in _NUM.findall(table(path)):
            key = _norm(tok, pct)
            if key:
                tables.setdefault(key, []).append(path.stem)
    derived = _derived_keys()
    out = {}
    for key, files in _prose_figures().items():
        if key not in reg and key in tables and key not in derived:
            out[key] = (sorted(files), sorted(set(tables[key])))
    return out


def test_no_new_figure_rests_on_a_table_that_is_not_about_it():
    new = sorted(k for k in _table_only() if k not in TABLE_COINCIDENCE)
    assert not new, "\n".join(
        f"  {kind} {value:g} in {', '.join(_table_only()[(kind, value)][0])} "
        f"is accounted for only by {', '.join(_table_only()[(kind, value)][1])}"
        for kind, value in new) + (
        "\n  Register it in a notebook, bind it with BOUND, or record the coincidence.")


def test_the_coincidence_list_has_no_dead_entries():
    """A figure that has since gained a source should leave this list, not sit in it."""
    live = set(_table_only())
    dead = sorted(k for k in TABLE_COINCIDENCE if k not in live)
    assert not dead, f"no longer accounted for by a coincidence: {dead}"


# How far the checking actually reaches. The number that matters is the last one: a
# figure whose only accountant is a sentence in a ledger is a figure nothing
# recomputes, and three of those were found wrong in one afternoon. It was 81 before
# the derivations above, 38 after them, and 35 once E17 and E18 put four historical
# figures on the record that cannot be derived. It was 33, and it is 32: the queries
# the greedy order's flat prefix completes moved into DERIVED, where it is summed off
# the table that generates the staircase, and that paid for the one figure this pass
# added -- the 0.23-point seed spread the cross-validation reported while its tie-break
# still depended on the interpreter's hash salt, which is history and cannot be
# derived. The ratchet is against drift, not against putting history on the record: a
# bump needs a reason in this comment, and a reduction needs a derivation.
#
# It is 33 again, and the one that bought the bump is 0.28: the seed spread the fixed
# tie-break gives at the PREVIOUS draft's replicate budget. It is what shows that the
# 0.23 it replaced was never the hash salt, so the paper has to print it to say what
# the correction is a correction of -- and it is a counterfactual configuration, so no
# committed cache holds it and nothing can derive it. The same pass put four figures
# ONTO derivations (the top() call total, the replay's two extremes, and one seed's
# Monte-Carlo error), which is why the bump is one and not five.
UNCHECKED_CEILING = 33


def test_the_unchecked_figures_do_not_multiply():
    """A ratchet, not a target. Lowering it means writing another derivation."""
    unchecked = _unchecked()
    assert len(unchecked) <= UNCHECKED_CEILING, (
        f"{len(unchecked)} figures rest on an allowlist sentence alone, over a ceiling "
        f"of {UNCHECKED_CEILING}:\n" + "\n".join(f"  {k} {v:g}" for k, v in unchecked))
    assert len(unchecked) >= UNCHECKED_CEILING - 2, (
        f"only {len(unchecked)} are unchecked now; lower UNCHECKED_CEILING to "
        f"{len(unchecked)} so the ratchet keeps its grip")


def test_the_unchecked_figures_are_exactly_what_the_ledgers_claim():
    """The two ledgers and the unchecked set have to be the same set.

    This is the invariant the ceiling was standing in for, and it is sharper: a
    ledger entry that is not unchecked is licensing nothing, and an unchecked figure
    that is in no ledger is one nothing has said anything about at all. Both are
    failures and each used to need its own test, one of which asked a weaker question
    and found none of the seventy-seven dead entries.
    """
    ledgers = set(_keyed(ALLOWED)) | set(_keyed(ALLOWED_IN))
    unchecked = {(k, round(v, 4)) for k, v in _unchecked()}
    assert ledgers == unchecked, (
        f"  in a ledger but checked elsewhere: {sorted(ledgers - unchecked)}\n"
        f"  unchecked and in no ledger: {sorted(unchecked - ledgers)}")


# --- figures the corpus itself settles -------------------------------------------------
# The entries these replace said "unregistered (nb 00)" and "unregistered (test_corpus)",
# which locates a figure without checking it: a notebook that no longer computes the
# quantity, or a test that renamed its assertion, leaves the sentence here reading
# exactly as it did. The corpus is committed, so the question is decidable from it.

_CORPUS: list = []


def _queries() -> list:
    if not _CORPUS:
        from qlport import corpus as c

        _CORPUS.extend(c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl"))
    return _CORPUS


def _promql_kept() -> list[str]:
    from qlport import corpus as c

    return [q.text for q in _queries() if q.lang == "promql" and c.is_promql(q.text)]


_SPANS: dict = {}


def _live_spans() -> "object":
    """Every `live_span` the corpus's Datadog dashboard sources declare."""
    import collections

    if not _SPANS:
        counts: collections.Counter = collections.Counter()
        files = set()
        root = ROOT / "data" / "corpus" / "sources" / "integrations-core"

        def walk(obj, path):
            if isinstance(obj, dict):
                if isinstance(obj.get("live_span"), str):
                    counts[obj["live_span"]] += 1
                    files.add(path)
                for value in obj.values():
                    walk(value, path)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value, path)

        for candidate in sorted(root.rglob("*.json")):
            try:
                walk(json.loads(candidate.read_text()), candidate)
            except (ValueError, OSError):            # not a dashboard, or unreadable
                continue
        _SPANS["counts"] = counts
        _SPANS["files"] = len(files)
    return _SPANS["counts"]


def _counts(lang: str, which: str, texts) -> "object":
    import collections

    from qlport import corpus as c

    fn = c.function_names if which == "function" else c.aggregation_ops
    total: collections.Counter = collections.Counter()
    for text in texts:
        total += fn(lang, text)
    return total


def _with_variable(lang: str) -> tuple[int, int]:
    """Occurrences of `lang` carrying a template variable, and how many there are.

    The definition notebook 00 uses, re-run here: `"$" in text`, over that language's
    occurrences, with the nine LogQL panels excluded from the PromQL side exactly as
    the notebook excludes them -- 8,111 occurrences become 8,102, and six of the nine
    carry a variable, which is the whole of the difference between 6,690 and 6,684.

    This exists because the abstract's "Nor is this one vendor's doing" argument rested
    on these two figures and nothing recounted either. `SHARES` related the Datadog
    share to the Datadog count, and the PromQL pair had not even that -- but both sides
    of a `SHARES` relation are registry keys, so moving a count and its share together
    satisfied it. A reviewer moved the PromQL pair to 4,684 / 57.8% across six files
    with the suite green.
    """
    from qlport import corpus as c

    sub = [q for q in _queries() if q.lang == lang
           and (lang != "promql" or c.is_promql(q.text))]
    return sum("$" in q.text for q in sub), len(sub)


CORPUS_DERIVED: tuple[tuple[str, float, str, object], ...] = (
    ("pct", 88.6, "Datadog occurrences carrying a template variable, as a share",
     lambda: round(100 * _with_variable("datadog")[0]
                   / _with_variable("datadog")[1], 1)),
    ("pct", 82.5, "and the PromQL share, which is what the abstract's \"not one "
                  "vendor's doing\" rests on",
     lambda: round(100 * _with_variable("promql")[0]
                   / _with_variable("promql")[1], 1)),
    ("num", 22903, "query strings the extraction yields", lambda: len(_queries())),
    ("num", 22894, "and what remains once the nine LogQL panels leave",
     lambda: len(_queries()) - (sum(1 for q in _queries() if q.lang == "promql")
                                - len(_promql_kept()))),
    ("num", 96, "PromQL recording rules",
     lambda: sum(1 for q in _queries() if q.lang == "promql" and q.kind == "record")),
    ("num", 7867, "and panel queries, once the LogQL ones are gone",
     lambda: sum(1 for q in _queries()
                 if q.lang == "promql" and q.kind == "panel")
     - (sum(1 for q in _queries() if q.lang == "promql") - len(_promql_kept()))),
    ("num", 5231, "PromQL function occurrences over those expressions",
     lambda: sum(_counts("promql", "function", _promql_kept()).values())),
    ("pct", 69, "of which rate and irate together are this per cent",
     lambda: round((lambda f: (f["rate"] + f["irate"]) / sum(f.values()) * 100)(
         _counts("promql", "function", _promql_kept())))),
    ("pct", 0.3, "and offset appears in this share of them",
     lambda: round(sum(1 for t in _promql_kept()
                       if re.search(r"(?<![\w:])offset\s", t))
                   / len(_promql_kept()) * 100, 1)),
    ("pct", 95, "Datadog space aggregators that are sum or avg, per cent",
     lambda: round((lambda a: (a["sum"] + a["avg"]) / sum(a.values()) * 100)(
         _counts("datadog", "aggregator",
                 [q.text for q in _queries() if q.lang == "datadog"])))),
    ("num", 6.6, "and how many times .as_count outnumbers .as_rate",
     lambda: round((lambda f: f[".as_count"] / f[".as_rate"])(
         _counts("datadog", "function",
                 [q.text for q in _queries() if q.lang == "datadog"])), 1)),
    # §5's mangling denominators. `test_corpus.py` asserts both when it measures the
    # collision rate; the derivation is shared rather than copied, so the two files
    # cannot drift into asserting different populations of the same name.
    ("num", 5389, "distinct Datadog metric names before mangling",
     lambda: len(__import__("test_corpus")._corpus_identifiers()[0])),
    ("num", 1264, "and distinct tag keys -- the scopes' and the grouping lists' together, "
           "which is the class lem:scope's injectivity hypothesis is over",
     lambda: len(__import__("test_corpus")._corpus_identifiers()[1])),
    # E17: the scope-length ladder. Every rung and both edges of the boundary come off
    # the cache, including the HTTP statuses, because an earlier run of this experiment
    # published a limit of exactly 4,096 off one 408 and the numbers are the evidence
    # that the limit is the transport's rather than the query language's.
    ("num", 6656, "the longest scope the account accepted",
     lambda: _cache("datadog/e17_scope_length.json")["boundary"]["largest_accepted"]),
    ("num", 7168, "and the shortest it refused",
     lambda: _cache("datadog/e17_scope_length.json")["boundary"]["smallest_refused"]),
    ("num", 8192, "the rung the bisection started from, which is the ladder's first refusal "
           "rather than the smallest, since the bisection then found a smaller one",
     lambda: next(r["n"] for r in _cache("datadog/e17_scope_length.json")["rows"]
                  if r["accepted"] is False)),
    ("num", 6912, "and the one inside the boundary that answered 500",
     lambda: next(r["n"] for r in _cache("datadog/e17_scope_length.json")["rows"]
                  if r["accepted"] is None)),
    # The status is the head of `detail`; `failure` only says the kind of failure it
    # was, which is a distinction worth keeping since a timeout is not a refusal.
    ("num", 431, "the status both refusals carried",
     lambda: int(next(r["detail"] for r in
                      _cache("datadog/e17_scope_length.json")["rows"]
                      if r["accepted"] is False).split()[0])),
    ("num", 500, "and the status of the inconclusive rung",
     lambda: int(next(r["detail"] for r in
                      _cache("datadog/e17_scope_length.json")["rows"]
                      if r["accepted"] is None).split()[0])),
    ("num", 58, "the request URL at the largest accepted rung, kilobytes",
     lambda: round(next(r["url_bytes"] for r in
                        _cache("datadog/e17_scope_length.json")["rows"]
                        if r["n"] == 6656) / 1000)),
    ("num", 62, "and at the smallest refused one",
     lambda: round(next(r["url_bytes"] for r in
                        _cache("datadog/e17_scope_length.json")["rows"]
                        if r["n"] == 7168) / 1000)),
    ("num", 12, "what the bound this paper fixes needs, kilobytes",
     lambda: round(next(r["url_bytes"] for r in
                        _cache("datadog/e17_scope_length.json")["rows"]
                        if r["n"] == 1681) / 1000)),
    # E18: the interval curve. The two cells the body argues from, and the span of the
    # series, which is why the month column is dashed.
    ("num", 14400, "the interval reported over a month-wide range",
     lambda: _cache("datadog/e18.json")["interpretation"]
     ["reported_by_range"]["1mo"]),
    ("num", 3500, "the seconds of points the run submitted",
     lambda: _cache("datadog/e18.json")["provenance"]["span"]),
    # Corollary rate's witness, constructed in tests/test_paper_formulas.py rather than
    # copied: two databases with equal phi images whose `rate` readings differ.
    ("num", 1.5, "the start-of-window extrapolation the clip cuts to, seconds",
     lambda: __import__("test_paper_formulas").rate_clip()),
    ("num", 0.628, "what rate() reads where the durationToZero clip binds",
     lambda: __import__("test_paper_formulas").rate_witness()[0]),
    ("num", 0.667, "and where it does not",
     lambda: __import__("test_paper_formulas").rate_witness()[1]),
    # What the extractor does not reach, as a share of the population it builds.
    ("pct", 0.2, "the fourteen queries outside the corpus, over the PromQL side it would "
          "have joined",
     lambda: round(14 / len(_promql_kept()) * 100, 1)),
    ("num", 3.15, "the Prometheus the measurements were taken against",
     lambda: float(".".join(
         _cache("topk/ranking.json")["provenance"]["prometheus_version"]
         .split(".")[:2]))),
    # The spans the corpus's own Datadog widgets declare, which is what decides how
    # often the emitter's assumed 60 s is the right one. §7's threats section called
    # that default "systematic rather than arbitrary" without asking.
    ("num", 460, "live_span values the corpus's Datadog dashboards declare",
     lambda: sum(_live_spans().values())),
    ("pct", 38.3, "of which one hour, the modal span, is this share",
     lambda: round(_live_spans()["1h"] / sum(_live_spans().values()) * 100, 1)),
    ("pct", 28.3, "and four hours, the only span whose interval is the assumed 60 s",
     lambda: round(_live_spans()["4h"] / sum(_live_spans().values()) * 100, 1)),
)


def test_every_corpus_figure_still_comes_out_of_the_corpus():
    wrong = []
    for _, value, note, derive in CORPUS_DERIVED:
        got = derive()
        if round(float(got), 4) != round(float(value), 4):
            wrong.append(f"  {value} ({note}) derives as {got}")
    assert not wrong, "\n".join(wrong)


def test_the_template_variable_counts_are_the_ones_the_corpus_carries():
    """The two counts behind the shares, which the prose does not print.

    `CORPUS_DERIVED` is for figures the paper quotes, and it quotes only the
    percentages. The counts are registry-only, which is exactly the shape that let the
    PromQL pair move: `SHARES` tied `nb00.promql_with_variable_pct` to
    `nb00.promql_with_variable` and both are registry keys, so a consistent edit of the
    pair satisfied it. Recounting the numerator from the committed corpus is what makes
    the share mean anything.
    """
    reg = _reg()
    for lang in ("datadog", "promql"):
        count, total = _with_variable(lang)
        assert reg[f"nb00.{lang}_with_variable"] == count, (
            f"nb00.{lang}_with_variable is {reg[f'nb00.{lang}_with_variable']} and the "
            f"corpus carries {count}")
        assert reg[f"nb00.{lang}_with_variable_pct"] == round(100 * count / total, 1)


def test_every_corpus_figure_is_still_quoted():
    prose = set(_prose_figures())
    missing = [f"  {kind} {v} ({note})" for kinds, v, note, _ in CORPUS_DERIVED
               for kind in kinds.split()
               if (kind, round(float(v), 4)) not in prose]
    assert not missing, "derived from the corpus but no longer in the paper:\n" + \
        "\n".join(missing)


# --- claims written as words ----------------------------------------------------------

_WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
         "twenty-three": 23, "twenty-four": 24, "twenty-five": 25, "twenty-six": 26,
         "twenty-seven": 27, "twenty-eight": 28}


def test_the_staircase_thresholds_written_as_words_are_the_registry_curve():
    r"""The paper's self-described practical result, spelled out in English.

    `_prose_figures` reads `$...$` spans, so a number written as a word is invisible to
    every binding and every ledger in this file. The staircase's headline thresholds
    are written as words -- "half the corpus needs nine features; $90\%$ needs sixteen
    of the twenty-eight" -- and an adversary changed them to three and four, reversing
    section 7.3 and the paragraph that exists to support it, at 795 passed. Nothing
    could have caught it: the registry still said what it says.

    The Datadog thresholds are checked against `nb00.ceiling_N_features_pct`, which is
    the curve those sentences are claims about. WHAT IS NOT CHECKED, and why: the
    registry carries no such curve for PromQL, so "fourteen features" and "seventeen
    of twenty-four" are held only to the vocabulary size. Reconstructing the PromQL
    curve here was tried and abandoned -- a greedy order written from the paper's own
    description of it reproduced none of the Datadog values, so a guard built on it
    would have been a second implementation agreeing with neither the notebook nor the
    paper. The honest state of this is one language bound and one not.
    """
    reg = _reg()
    ceiling = {n: reg[f"nb00.ceiling_{n}_features_pct"]
               for n in range(reg["cv.dd_vocabulary"] + 1)
               if f"nb00.ceiling_{n}_features_pct" in reg}
    assert len(ceiling) >= 20, sorted(ceiling)
    half = min(n for n, v in sorted(ceiling.items()) if v >= 50)
    ninety = min(n for n, v in sorted(ceiling.items()) if v >= 90)
    dd_vocab, pq_vocab = reg["cv.dd_vocabulary"], reg["cv.pq_vocabulary"]

    # The two sentences are told apart by their own punctuation: the Datadog pair is
    # joined by a semicolon and names "the" vocabulary, the PromQL pair by "and".
    datadog = re.compile(r"Half the corpus needs (\w+(?:-\w+)?) features; "
                         r"\$90\\%\$ needs (\w+(?:-\w+)?) of the (\w+(?:-\w+)?)")
    promql = re.compile(r"half the corpus needs (\w+(?:-\w+)?) features and "
                        r"\$90\\%\$ needs (\w+(?:-\w+)?) of (\w+(?:-\w+)?)")
    caption = re.compile(r"after (\w+(?:-\w+)?) of Datadog's (\w+(?:-\w+)?) features "
                         r"and\s+(\w+(?:-\w+)?) of PromQL's (\w+(?:-\w+)?)")
    found: dict[str, int] = {}
    for name in ("moved/eval_staircase.tex", "07_evaluation.tex"):
        path = ROOT / "paper" / "sections" / name
        for full in (False, True):
            said = flat(rendered(path, full))
            for m in datadog.finditer(said):
                found["datadog"] = found.get("datadog", 0) + 1
                assert _WORD.get(m.group(1)) == half, (
                    f"{name}: half the corpus is said to need {m.group(1)!r}; the "
                    f"registry's curve first passes 50% at {half}")
                assert _WORD.get(m.group(2)) == ninety, (
                    f"{name}: 90% is said to need {m.group(2)!r}; the curve first "
                    f"passes 90% at {ninety}")
                assert _WORD.get(m.group(3)) == dd_vocab, (m.group(3), dd_vocab)
            for m in promql.finditer(said):
                found["promql"] = found.get("promql", 0) + 1
                # Only the vocabulary, for the reason the docstring gives.
                assert _WORD.get(m.group(3)) == pq_vocab, (m.group(3), pq_vocab)
            for m in caption.finditer(said):
                found["caption"] = found.get("caption", 0) + 1
                assert _WORD.get(m.group(1)) == ninety, (m.group(1), ninety)
                assert _WORD.get(m.group(2)) == dd_vocab, (m.group(2), dd_vocab)
                assert _WORD.get(m.group(4)) == pq_vocab, (m.group(4), pq_vocab)
    # Per pattern, not as a total. This was `found >= 3` over the sum of three
    # patterns, which measured 6; rewording the one sentence that binds the
    # half-corpus threshold left the total at 4 and the guard passing with "three
    # features" in print against a curve that says nine. A total is exactly the wrong
    # shape for a completeness check: it lets any one member go missing.
    assert found and all(n >= 2 for n in found.values()), (
        f"a spelled-out staircase claim is no longer matched by the pattern that "
        f"binds it, so it is unbound rather than wrong: {dict(found)} (each should be "
        f"2 -- one per build)")


# The paper quoting its own former claim in order to warn about it is not the paper
# making the claim, so quoted spans come out before the sweep below looks for it.
_QUOTED = re.compile(r"``.*?''", re.S)


def test_the_inexpressible_column_carries_its_qualification_everywhere():
    r"""Section 7 states this as a rule and the paper broke it in seven places.

    The rule, at `07_evaluation.tex`: all 287 refused occurrences are unwritten code,
    "with one qualification this section states where it arises and carries everywhere
    the column is mentioned: 14 of them are refused on a hypothesis this paper tests
    for the declared metrics and cannot test for theirs, so they are undecided rather
    than retracted." Section 8 then states the defensible form in as many words --- the
    column "is empty of anything demonstrated", "which is a weaker claim than
    `empty'" --- and forty-four lines later said "The column is empty now, which is a
    stronger claim than any of the figures it used to hold". The abstract said it
    twice, the introduction twice, the discussion once more and `concl_evidence` once.

    So the rule is enforced rather than stated. Any sentence asserting that the column
    is empty, or has emptied, or holds nothing, has to say in what sense, in both
    builds. The licensed senses are the ones the evidence carries: nothing
    *demonstrated*, nothing this paper can *show*, a *near*-empty column, or the 14
    named as undecided.

    What this does not touch, deliberately: sentences about the column's history ("did
    not start empty"), about what shape would make it empty, and the three that say
    "empty selection", which is a different object. Narrowing the pattern to present
    assertions does that without a ledger of exceptions, and a ledger here would be
    the thing that lets the eighth site in.
    """
    # Plural too. `\bcolumn\b` does not match "columns", which is how
    # `eval_beyond_reach.tex` -- the one block whose paragraph heading IS this
    # question -- kept "The first two columns are empty" through the sweep that fixed
    # the other seven sites.
    #
    # And the verb list is wider, because "stands empty", "contains nothing" and "has
    # no entries" all passed the first one. That list is a pattern and so is not a
    # completeness argument: what makes the paper's current text right is the one-time
    # sweep that found eight sites, and what this does is stop a ninth being written in
    # one of the shapes anyone would reach for. Said plainly because the first version
    # of this docstring claimed the rule was "enforced rather than stated", which was
    # more than a verb list can do.
    #
    # IT ALSO KEYS ON THE WORD. Every branch below needs "column" in the sentence, and
    # the tenth site found was `03b_ingestion`'s "the first two are empty on this
    # corpus", which names the three categories and not the columns they are. A
    # sentence that asserts the claim without the noun is outside this pattern, and
    # widening it to the claim would mean matching "the first two are empty" anywhere.
    # So: this catches the shapes, and the sweep catches the paper.
    claim = re.compile(
        r"[^.]*\bcolumns?\b[^.]*?\b(?:is|are|stands?|sits?|remains?|stays?)\s+"
        r"(?:empty|bare|vacant)\b[^.]*\.|"
        r"[^.]*\bcolumns?\b[^.]*?\b(?:emptied|emptying|empties|"
        r"holds? none|holds? nothing|contains? nothing|has no entries|have no entries|"
        r"with nothing in (?:it|them)|nothing left beyond it)\b[^.]*\.|"
        # SUBJECT FIRST. Both branches above require "column" before the verb, so
        # "Nothing remains in the inexpressible column" -- which asserts exactly what
        # they forbid, with the subject moved to the front -- matched neither. It was
        # added to the discussion and the whole suite stayed green.
        r"[^.]*\b(?:nothing|none|no entry|no entries|not one entry)\b[^.]*?"
        r"\bcolumns?\b[^.]*\.",
        re.I)
    also = re.compile(r"[^.]*\b(?:empty|emptied|bare) columns?\b[^.]*\.", re.I)
    qualified = ("demonstrated", "demonstrate", "near-empty", "undecided",
                 "we can show", "can show", "is shown to", "nothing we can",
                 "nothing this paper can")
    bare = []
    for path in sorted([*SECTIONS, MAIN]):
        for full in (False, True):
            said = _QUOTED.sub("", flat(rendered(path, full)))
            for pattern in (claim, also):
                for m in pattern.finditer(said):
                    if any(q in m.group(0) for q in qualified):
                        continue
                    build = "main_full" if full else "main"
                    bare.append(f"  {path.name} in {build}: {m.group(0).strip()[:160]}")
    assert not bare, (
        "these assert the inexpressible column is empty without the qualification "
        "§7 says it carries everywhere:\n" + "\n".join(sorted(set(bare)))
        + "\n  (say nothing *demonstrated*, or name the 14 undecided)")
    # And the rule itself has to still be in the paper, in both builds, or this test is
    # enforcing a convention the paper no longer announces.
    rule = "carries everywhere the column is mentioned"
    for full in (False, True):
        assert rule in flat(rendered(
                ROOT / "paper" / "sections" / "07_evaluation.tex", full)), (
            "§7 no longer states the qualification as one that travels, so either "
            "the rule is gone or it has moved and this test has to move with it")


_VERBATIM = re.compile(r"\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}", re.S)
_MACRO_ARG = re.compile(
    r"\\(?:label|ref|apref|apin|cite[a-z]*|input|includegraphics|url|href|"
    r"lstinputlisting)\s*\{[^}]*\}")


def test_every_percentage_and_thousand_in_the_prose_is_in_math_mode():
    r"""The whole figure apparatus is built on `$...$`, so a numeral outside one is
    invisible to all of it.

    `_math` extracts math spans; `_prose_figures`, `_accounted`, every ledger in this
    file, `_unchecked` and `UNCHECKED_CEILING` are built on that. Nothing required a
    numeral to be in a span. `16.5\%` typesets indistinguishably from `$16.5\%$` at a
    glance, and an adversary put it in the introduction as the type-inference error
    rate --- two and a half times the $6.5\%$ every other site in the paper states ---
    with the census blind, the ratchet unmoved at 33, and the suite green.

    The commit before this one closed the sibling spelling, a numeral written as a
    word, and called the class handled. The un-dollared digit is the same class and is
    the wider hole, because it needs no rewriting of the sentence.

    WHAT THIS COVERS, exactly: a numeral followed by `\%`, and a numeral carrying the
    paper's thousands separator `{,}`. Those two shapes are figures wherever they
    appear. WHAT IT DOES NOT: bare integers and decimals, of which there are about 460
    outside math mode and almost none a figure --- `LALR(1)`, `\$65M`, `3.15.0`,
    `November~2026`, `numbered above~3`, the section numbers of `\S48.1`. A ledger of
    those would be noise, and noise is what a ledger cannot afford to be. So this is a
    rule about two shapes, and says so, rather than a rule about numerals that would
    need 460 exceptions to hold.
    """
    bad = []
    for path in sorted([*SECTIONS, MAIN]):
        for full in (False, True):
            text = _MACRO_ARG.sub(" ", _VERBATIM.sub(" ", rendered(path, full)))
            text = text.replace(r"\$", "\x00\x00")
            spans = [(m.start(), m.end()) for m in re.finditer(r"\$[^$]*\$", text)]
            for m in re.finditer(r"\d[\d{},]*(?:\.\d+)?", text):
                if any(a <= m.start() < b for a, b in spans):
                    continue
                tail = text[m.end():m.end() + 2]
                shape = ("a percentage" if tail == r"\%"
                         else "a thousands-separated figure"
                         if "{,}" in m.group(0) else None)
                if shape is None:
                    continue
                build = "main_full" if full else "main"
                near = text[max(0, m.start() - 40):m.end() + 6].replace("\n", " ")
                bad.append(f"  {path.name} in {build}: {shape} outside math mode "
                           f"--- ...{near}...")
    assert not bad, (
        "these are figures the census cannot see, because every ledger in this file "
        "reads `$...$` spans:\n" + "\n".join(sorted(set(bad)))
        + "\n  (wrap them in $...$)"
    )


# Claims written as words that carry a result, each with what decides it. A guard that
# binds one sentence pair and reports the class closed is the shape of the last three
# rounds; this is the list, and it is a list rather than a pattern because each of
# these is a different kind of claim.
#
# `(file, regex with one word group, what the word must equal)`. The regex is searched
# in both builds, and a pattern that stops matching fails rather than passing quietly
# -- which is how `found >= 3` over a total let the staircase sentence go unbound.
_SPELLED_FRACTION = {("two", "three"): 2 / 3, ("one", "three"): 1 / 3,
                     ("two", "ten"): 0.2, ("three", "four"): 0.75}


def test_the_claims_written_as_words_are_the_ones_the_data_supports():
    r"""Three sites where a result is spelled out, and nothing read any of them.

    `16.5\%` is caught by the math-mode rule next door; a word is not. An adversary
    changed the abstract's "wrong two times in three" to "one time in three" --- the
    registry says the interval is *right* one time in three --- and §7's to "two times
    in ten" three lines below a printed `$66.9\%$`, both at 812 passing. And
    `concl_evidence` writes §4's LALR result as "§4's two conflicts", which the
    conflict-count guard's `\$(\d+)\$ conflicts` regex cannot reach, so it became "six
    conflicts" against a parser that reports two.
    """
    from qlport.parse import lalr_conflict_report

    reg = _reg()
    wrong = reg["rs.span_declared_wrong_pct"]
    conflicts = len(lalr_conflict_report("promql", "v2"))

    checks = [
        (MAIN, r"is wrong (\w+) times? in (\w+)", "fraction", wrong),
        (ROOT / "paper" / "sections" / "09_conclusion.tex",
         r"is wrong (\w+) times? in (\w+)", "fraction", wrong),
        (ROOT / "paper" / "sections" / "moved" / "concl_evidence.tex",
         r"'s (\w+) conflicts", "count", conflicts),
    ]
    for path, pattern, kind, want in checks:
        seen = 0
        for full in (False, True):
            said = flat(rendered(path, full))
            for m in re.finditer(pattern, said):
                seen += 1
                if kind == "fraction":
                    key = (m.group(1).lower(), m.group(2).lower())
                    assert key in _SPELLED_FRACTION, (
                        f"{path.name} spells a fraction this test cannot read: "
                        f"{m.group(0)!r}")
                    got = _SPELLED_FRACTION[key] * 100
                    assert abs(got - want) < 2.0, (
                        f"{path.name} says the interval is wrong {m.group(0)!r}, "
                        f"which is {got:.1f}% against the registry's {want}%")
                else:
                    assert _WORD.get(m.group(1).lower()) == want, (
                        f"{path.name} states the conflict count as {m.group(1)!r}; "
                        f"the parser reports {want}")
        assert seen == 2, (
            f"{path.name}: the sentence this binds was found {seen} times across the "
            f"two builds, not twice, so it has been reworded and is now unbound "
            f"rather than wrong (pattern {pattern!r})")

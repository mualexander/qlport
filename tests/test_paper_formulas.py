r"""What the paper states, against what the code does.

Everything else in `tests/` checks the code against reality or the prose against a
figure. This file checks the paper's *formal statements* against the implementation,
which is the one correspondence nothing else touches: a proposition could name an
expression form the emitter does not produce, or a theorem could list preconditions the
emitter does not record, and every other test in the suite would still pass. The
notebooks verify the code against real servers; that is the code against the world, not
the paper against the code.

`tr_kernel_pf` ends ``that $C$ contains nothing further is by inspection of the emitter:
the six codes above are the only ones it records''. It was seven, and the proof had
confused the theorem's six numbered hypotheses with the codes they name --- hypothesis 4
names two. This file performs that inspection rather than asserting it.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "paper" / "sections"
SRC = ROOT / "src" / "qlport"


def _gap_calls(path: Path, names: tuple[str, ...]) -> set[str]:
    """Every `Gap.X` passed to one of `names`, read off the module's syntax tree."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if called not in names:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Attribute) and getattr(arg.value, "id", "") == "Gap":
                found.add(arg.attr)
    return found


def _environment(path: Path, kind: str, label: str) -> str:
    """The body of one `\\begin{kind}...\\label{label}...\\end{kind}`."""
    text = path.read_text()
    start = text.index(f"\\label{{{label}}}")
    opened = text.rindex(f"\\begin{{{kind}}}", 0, start)
    closed = text.index(f"\\end{{{kind}}}", start)
    return text[opened:closed]


def _codes_named(body: str) -> set[str]:
    """The gap-code names a statement mentions, as `\\texttt{UPPER_CASE}`."""
    from qlport.diagnostics import Gap

    known = {g.name for g in Gap}
    out = set()
    for token in re.findall(r"\\texttt\{([^}]*)\}", body):
        name = token.replace("\\_", "_").strip()
        if name in known:
            out.add(name)
    return out


# --- Theorem (kernel soundness): its hypotheses against the emitter's caveats ---------

def test_the_kernel_theorem_names_exactly_the_caveats_the_emitter_records():
    """The proof's ``by inspection of the emitter'' performed.

    A precondition the emitter records and the theorem does not name is an unstated
    hypothesis, which makes the theorem false as written. A precondition the theorem
    names and the emitter never records is a hypothesis doing no work -- weaker, but it
    means the theorem is not describing this translator.
    """
    recorded = _gap_calls(SRC / "emit" / "promql.py", ("note",))
    named = _codes_named(_environment(SECTIONS / "05_translation.tex",
                                     "theorem", "thm:kernel"))
    assert recorded == named, {
        "recorded but not named by the theorem": sorted(recorded - named),
        "named but never recorded": sorted(named - recorded),
    }
    # Seven, across six numbered hypotheses: the fourth names two.
    assert len(recorded) == 7, sorted(recorded)


def test_the_caveats_are_recorded_in_one_place_only():
    """The theorem is about the emitter, so a caveat raised elsewhere escapes it."""
    for path in sorted(SRC.rglob("*.py")):
        if path == SRC / "emit" / "promql.py":
            continue
        assert not _gap_calls(path, ("note",)), path


# --- the gap catalogue: every code it reports is one the code can raise ---------------

def test_every_code_in_the_catalogue_is_one_the_translator_can_raise():
    """A catalogue row no code path produces would be a figure with no mechanism."""
    table = (ROOT / "paper" / "tables" / "gap_catalogue.tex").read_text()
    reported = _codes_named(table)
    assert reported, "no gap codes read out of the catalogue"
    raisable = (_gap_calls(SRC / "lower" / "datadog.py", ("_refuse", "Bottom"))
                | _gap_calls(SRC / "emit" / "promql.py", ("_refuse", "Bottom")))
    assert reported <= raisable, sorted(reported - raisable)


def test_the_two_codes_that_both_refuse_and_caution_are_the_two_the_paper_says():
    """`METRIC_TYPE_UNKNOWN` and `ROLLUP_DEFAULT` appear on both paths, deliberately.

    Each refuses only where the caveat would be a lie: `ROLLUP_DEFAULT` when no default
    was supplied at all, which is a caller's choice and never happens in this paper's
    runs, and `METRIC_TYPE_UNKNOWN` when summing over time needs a counter and the type
    is unknown, which is a refusal the corpus never reaches. Both are therefore absent
    from the catalogue while present in the code, and a third joining them would mean a
    caveat had quietly become a refusal somewhere.
    """
    emitter = SRC / "emit" / "promql.py"
    both = _gap_calls(emitter, ("note",)) & _gap_calls(emitter, ("_refuse", "Bottom"))
    assert both == {"METRIC_TYPE_UNKNOWN", "ROLLUP_DEFAULT"}, sorted(both)
    table = (ROOT / "paper" / "tables" / "gap_catalogue.tex").read_text()
    assert not (_codes_named(table) & both), "a dual-role code reached the catalogue"


# --- the two propositions that state an expression form ------------------------------
# `prop:offset` and `prop:anchored` are the only statements in the paper that name a
# PromQL string. If the emitter writes a different one, the propositions are true about
# an expression nobody emits, which every other test in the suite is content with.

COUNTER = "sum:http.requests{*}.as_count().rollup(sum, 60)"


def _emit(query: str, *, anchored: bool = False):
    from qlport.diagnostics import Bottom
    from qlport.emit.promql import emit_detail
    from qlport.lower.datadog import lower_all
    from qlport.parse import parse

    lowered = lower_all(parse("datadog", query, "v2"))
    assert len(lowered) == 1 and not isinstance(lowered[0], Bottom), lowered
    got = emit_detail(lowered[0], default_rollup=60.0, anchored=anchored)
    assert not isinstance(got, Bottom), got
    return got


def test_the_emitter_writes_the_cumulative_difference_the_proposition_states():
    r"""Proposition offset names `x - x offset $\Delta$`; this is that string."""
    got = _emit(COUNTER)
    assert got.promql == "sum(http_requests - http_requests offset 60s)", got.promql
    body = _environment(SECTIONS / "03b_ingestion.tex", "proposition", "prop:offset")
    assert r"\texttt{x - x offset $\Delta$}" in body


def test_the_emitter_writes_the_anchored_increase_the_proposition_states():
    r"""And proposition anchored names `increase(x[$\Delta$] anchored)`."""
    got = _emit(COUNTER, anchored=True)
    assert got.promql == "sum(increase(http_requests[60s] anchored))", got.promql
    body = _environment(SECTIONS / "03b_ingestion.tex", "proposition", "prop:anchored")
    assert r"\texttt{increase(x[$\Delta$] anchored)}" in body


def test_the_two_forms_carry_the_preconditions_the_theorem_pairs_with_them():
    """The kernel theorem's fourth hypothesis is a disjunction, and this is why.

    The cumulative difference is exact where no series resets and every series has a
    sample before the bucket, so it carries `COUNTER_RESET` and `PHI_SERIES_START`. The
    anchored form needs neither and needs a server flag instead, so it carries
    `ANCHORED_VERSION` and drops both. The hypothesis says ``or the anchored form was
    emitted''; these are the two sides of that or, read off the emitter.
    """
    plain = {c.name for c in _emit(COUNTER).caveats}
    anchored = {c.name for c in _emit(COUNTER, anchored=True).caveats}
    assert {"COUNTER_RESET", "PHI_SERIES_START"} <= plain
    assert "ANCHORED_VERSION" not in plain
    assert "ANCHORED_VERSION" in anchored
    assert not ({"COUNTER_RESET", "PHI_SERIES_START"} & anchored)


@pytest.mark.parametrize("scope", ["{service:a OR service:b}", "{service IN (a,b)}"])
def test_a_single_key_disjunction_becomes_one_matcher(scope):
    """Proposition disjunction says such a scope is exactly one alternation matcher.

    Both syntaxes, because §6 reads the same proposition backwards and the reverse
    direction's $\\mathtt{LIST|GET}\\mapsto k \\in (\\dots)$ depends on the two being
    the same claim.
    """
    got = _emit(f"sum:http.requests{scope}")
    assert got.promql == 'sum(avg_over_time(http_requests{service=~"a|b"}[60s]))', \
        got.promql
    assert got.promql.count("=~") == 1


# --- the ingestion mapping -------------------------------------------------------------

def test_phi_is_the_mapping_the_definition_writes():
    """Definition phi, recomputed from the formula and compared to the implementation.

    The formula is $\\delta_\\rho(v_{i-1}, v_i)$ over $2 \\le i \\le n$, which fixes
    three things a reimplementation could get wrong independently: that the first sample
    is dropped, that the timestamp kept is the later of each pair, and what each reset
    policy does with a decrease.
    """
    import numpy as np

    from qlport.semantics.ingest import Series, phi_series

    ts = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    vs = np.array([5.0, 9.0, 3.0, 4.0, 4.0])          # one reset between 9 and 3
    counter = Series({"svc": "a"}, ts, vs, kind="counter")

    def delta(u: float, v: float, rho: str) -> float:
        if v >= u:
            return v - u
        return v if rho == "zero" else 0.0

    for rho in ("zero", "clamp"):
        got = phi_series(counter, rho)
        want = [delta(vs[i - 1], vs[i], rho) for i in range(1, len(vs))]
        assert list(got.ts) == list(ts[1:]), (rho, list(got.ts))
        assert list(got.vs) == want, (rho, list(got.vs), want)
        assert got.kind == "delta"
        assert got.labels == {"svc": "a"}

    # And a series that is not a counter maps by the identity, which is what makes the
    # mapping total on D and lets the propositions quantify over all of it.
    gauge = Series({"svc": "a"}, ts, vs, kind="gauge")
    assert phi_series(gauge, "zero") is gauge


def test_the_definition_names_every_policy_the_paper_argues_about():
    """`drop` is in the code and not in Definition phi, which §8 explains rather than
    hides: it is not a reset policy at all, and E1 ruled it out against the account."""
    import inspect

    from qlport.semantics import ingest

    source = inspect.getsource(ingest.phi_series)
    assert '"zero"' in source and '"clamp"' in source and '"drop"' in source
    body = _environment(SECTIONS / "03b_ingestion.tex", "definition", "def:phi")
    assert r"\mathsf{zero}" in body and r"\mathsf{clamp}" in body
    assert "drop" not in body
    discussion = (SECTIONS / "08_discussion.tex").read_text()
    assert "drop" in discussion, "the policy the definition omits is argued nowhere"


# --- §6's dichotomy and the three things no glob says ---------------------------------
# These are proved in the appendix and then used as the warrant for every refusal §6.4
# reports. The proofs are proofs; what nothing checked is that `regular.py` refuses the
# shapes the corollary rules out and admits the ones it does not, which is the step
# between a theorem and a figure.

def test_a_glob_with_no_star_denotes_one_word_and_one_with_a_star_does_not():
    """Lemma dichotomy, over the alphabet the module decides against.

    A glob language is a singleton or unbounded; there is nothing in between, and that
    is the whole engine of the corollary below.
    """
    import itertools

    from qlport.regular import TAG_ALPHABET, glob_to_regex, shortest_word

    sigma = ["a", "b", "c", "x"]              # four characters is enough to enumerate
    assert all(c in TAG_ALPHABET for c in sigma)

    def language(glob: str, length: int) -> set[str]:
        pattern = re.compile(glob_to_regex(glob))
        return {"".join(w) for w in itertools.product(sigma, repeat=length)
                if pattern.fullmatch("".join(w))}

    # n = 0: a single word, and nothing of any other length.
    assert language("abc", 3) == {"abc"}
    assert language("abc", 2) == set() and language("abc", 4) == set()
    assert shortest_word(glob_to_regex("abc"))[0] == "abc"

    # n >= 1: a word of every length from m(g) upwards, each carrying the literal
    # segments as factors in order.
    m = len("ab") + len("c")
    assert language("ab*c", m - 1) == set()
    for length in range(m, m + 3):
        words = language("ab*c", length)
        assert words, length
        assert all(w.startswith("ab") and w.endswith("c") for w in words), words
    assert len(language("ab*c", m + 1)) == len(sigma)


def test_no_glob_denotes_a_finite_language_of_two_or_more_words():
    """Corollary no-glob, part (1), as the proposer's verdict on three witnesses.

    $\\Sigma$ itself, an alternation of distinct literals, and a bounded repetition are
    all finite with more than one word, so none has a single glob. Each is still a
    finite *union* of globs, which is why §6 changes target rather than stopping here.
    """
    from qlport.regular import MAX_UNION, regex_to_glob, regex_to_globs

    for regex in (".", "LIST|GET", "5."):
        assert regex_to_glob(regex) is None, regex
        got = regex_to_globs(regex)
        assert got.kind == "union" and got.verified, (regex, got)
        assert 2 <= len(got.globs) <= MAX_UNION, (regex, len(got.globs))


def test_no_glob_denotes_the_non_empty_values():
    """Part (2): $\\Sigma^{+}$ has no single glob and is a union of $|\\Sigma|$ of them.

    Both halves matter. The first is the corollary; the second is why §6.4 can say the
    refusal rests on the bound rather than on the corollary, and it is the sentence the
    alphabet question turned on.
    """
    from qlport.regular import TAG_ALPHABET, regex_to_glob, regex_to_globs

    assert regex_to_glob(".+") is None
    got = regex_to_globs(".+")
    assert got.kind == "union" and got.verified and got.exact
    assert set(got.globs) == {f"{c}*" for c in TAG_ALPHABET}


def test_no_finite_union_of_globs_confines_a_star_to_part_of_the_alphabet():
    """Part (3) and Corollary absent-char, which is the unconditional one.

    `5[0-9]+` is infinite and uses no letter, so no union of any size reaches it. The
    refusal `regular.py` gives it has to be the reason-less-by-bound kind, or §6.4's
    separation of ten occurrences from the other seventy-four is not what it says.
    """
    from qlport.regular import regex_to_glob, regex_to_globs

    assert regex_to_glob("5[0-9]+") is None
    got = regex_to_globs("5[0-9]+")
    assert got.kind == "none", got
    assert got.reason == "unbounded-character-class", got.reason
    # And the distinction the corollary draws: a *bounded* run over the same characters
    # is a union, so what refuses here is the unbounded star and not the character set.
    assert regex_to_globs("5[0-9]").kind == "union"


def test_the_reasons_the_corpus_refusals_carry_are_the_two_the_section_names():
    """§6.4 says two constructs account for every refusal; these are their codes.

    Read off the committed cross-check rather than re-decided: a third reason appearing
    would mean the section is enumerating a set the procedure no longer produces.
    """
    import json

    rows = json.loads((ROOT / "data" / "raw" / "globs"
                       / "star_reading.json").read_text())["rows"]
    from qlport.regular import regex_to_globs

    reasons = {regex_to_globs(r["regex"]).reason
               for r in rows if r["taken"] == "none"}
    assert reasons == {"union-too-large", "unbounded-character-class"}, reasons


# --- the corollary with a witness the paper gave as two numbers -----------------------

def rate_clip() -> float:
    r"""The start-of-window extrapolation the clip cuts to, seconds.

    The quantity the witness turns on: `sampled * (v_1 / increase)` at the clipped base,
    which is what the remark states as $1.5$\,s.
    """
    sampled, first, increase = 990.0 - 945.0, 1.0, 30.0
    return round(sampled * (first / increase), 3)


def rate_witness() -> tuple[float, float]:
    r"""Two databases with equal $\phi_\rho$ images and different `rate` readings.

    Corollary rate's negative half, as a construction. The paper gave the result as
    ``$0.628$ against $0.667$ on an ordinary counter'' and gave no witness, so the
    reader had the conclusion and no way to it; this is the counter.

    Four samples $15$\,s apart inside a $60$\,s window, rising by $10$ each time, so
    $\phi_\rho$ is three deltas of $10$ whatever the series starts at. `rate` reads the
    absolute first value through \texttt{durationToZero}: at a base of $1$ the clip
    binds and the extrapolation is cut to $1.5$\,s, at a base of $4$ it does not and the
    full $5$\,s to the window edge is used. Equal images, different readings, so the
    expression is not $\phi_\rho$-invariant -- and the series begins inside the window,
    which is where the corollary says to look.
    """
    import numpy as np

    from qlport.semantics.ingest import Series, phi_series
    from qlport.semantics.promql_eval import rate

    t, width = 1000.0, 60.0
    ts = np.array([945.0, 960.0, 975.0, 990.0])
    readings, images = [], []
    for base in (1.0, 4.0):
        vs = np.array([base + 10.0 * i for i in range(4)])
        readings.append(round(rate(ts, vs, t, width), 3))
        phi = phi_series(Series({"svc": "a"}, ts, vs, kind="counter"), "zero")
        images.append((tuple(phi.ts), tuple(phi.vs)))
    assert images[0] == images[1], images      # the premise: equal phi images
    return readings[0], readings[1]


def test_rate_is_not_phi_invariant_and_here_is_why():
    """The corollary and its appendix have to agree about the warm class.

    They did not, for eleven passes: the corollary asserted invariance on
    $\\mathcal{C}_{\\mathrm{warm}}$ while the appendix three lines later reported the
    witness that refutes it. So this test is not only that the witness reproduces --- it
    is that the corollary denies the warm class rather than granting it, which is the
    half no reader of the appendix alone would have caught.
    """
    clipped, unclipped = rate_witness()
    assert (clipped, unclipped) == (0.628, 0.667), (clipped, unclipped)
    body = _environment(SECTIONS / "03b_ingestion.tex", "corollary", "cor:rate")
    flat = body.replace("\\", "")
    assert "durationToZero" in flat
    # The corollary names the warm class to deny it, not to grant it.
    assert r"\mathcal{C}_{\mathrm{warm}}" in body
    assert "either" in flat, "the corollary must refuse the warm class too"
    # The withdrawn positive half read "It *is* phi-invariant on the class C_warm".
    assert "invariant on the class" not in flat, "the withdrawn positive half is back"
    witness = (SECTIONS / "moved" / "ing_rate_disc.tex").read_text()
    assert "0.628" in witness and "durationToZero" in witness
    assert "no series begins inside the window" in witness


def test_the_clip_the_witness_turns_on_is_in_the_model():
    """`durationToZero` is the mechanism, so the port has to contain it.

    The corollary is explicit that the dependence is in Prometheus's implementation and
    not in its documentation, which makes the ported line the only written statement of
    it this paper controls.
    """
    import inspect

    from qlport.semantics import promql_eval

    source = inspect.getsource(promql_eval.rate)
    assert "to_zero" in source
    assert "sv[0]" in source, "the clip must read the absolute first sample"


# --- the result distance, its three cases and the lemma that it is a metric -----------

def test_the_distance_has_the_three_cases_the_definition_gives():
    r"""Definition d, case by case, including the one the prose had to add.

    The empty-domain case was in `metrics.py` before it was in the definition --- two
    results that both select nothing agree, and a maximum over the empty set would
    otherwise leave $d$ undefined on exactly the pairs §3b's failure modes produce.
    """
    import math

    from qlport.metrics import d

    assert d({}, {}) == 0.0
    assert d({("a",): 1.0}, {("a",): 1.0}) == 0.0
    assert d({("a",): 1.0}, {("a",): 2.0}) == 0.5
    assert math.isinf(d({("a",): 1.0}, {("b",): 1.0}))
    assert math.isinf(d({("a",): 1.0}, {}))
    # delta(0, 0) = 0 is part of the definition, not a division to avoid.
    assert d({("a",): 0.0}, {("a",): 0.0}) == 0.0


def test_delta_is_a_metric_and_d_is_an_extended_one():
    """Lemma metric, numerically. The proof is in the appendix; this is the sanity.

    Not decoration, as the section says: Proposition admissible is a triangle-inequality
    argument, so a delta that failed the inequality would take an argument with it.
    """
    import itertools

    from qlport.metrics import d, reldiff

    values = [-100.0, -3.5, -1.0, 0.0, 0.5, 1.0, 2.0, 7.25, 1e6]
    for a, b in itertools.product(values, repeat=2):
        assert reldiff(a, b) == reldiff(b, a)                      # symmetry
        assert (reldiff(a, b) == 0.0) == (a == b)                   # identity
        assert 0.0 <= reldiff(a, b) <= 2.0
    for a, b, c in itertools.product(values, repeat=3):
        assert reldiff(a, c) <= reldiff(a, b) + reldiff(b, c) + 1e-12, (a, b, c)
    # And d inherits all three on results over a shared domain.
    keys = [("x",), ("y",)]
    results = [{k: v for k, v in zip(keys, pair)}
               for pair in itertools.product(values[:5], repeat=2)]
    for r1, r2 in itertools.product(results, repeat=2):
        assert d(r1, r2) == d(r2, r1)
        assert (d(r1, r2) == 0.0) == (r1 == r2)


def test_an_absent_reading_is_scored_against_the_translation():
    """The NaN convention §3 now states, and the direction it errs in.

    `d` is NaN where a reading is, which is outside Definition d's range -- a result
    maps into the reals. What matters is that every exactness test compares against a
    tolerance, and NaN fails that comparison, so the bucket counts as a disagreement.
    An implementation that scored it as agreement would inflate every exactness figure
    in the paper, and this is the test that it does not.
    """
    import math

    from qlport.metrics import d

    nan = float("nan")
    assert math.isnan(d({("a",): nan}, {("a",): 1.0}))
    assert math.isnan(d({("a",): nan}, {("a",): nan}))
    assert not (d({("a",): nan}, {("a",): 1.0}) <= 1e-12)
    assert not (d({("a",): nan}, {("a",): nan}) <= 1e-12)
    body = _environment(SECTIONS / "03_framework.tex", "definition", "def:d")
    assert r"\mathbb{R}" in body or r"\infty" in body
    # The convention is stated beside the definition rather than left to the code.
    section = (SECTIONS / "03_framework.tex").read_text()
    assert "NaN" in section and "conservative" in section


def test_coverage_is_the_quotient_the_definition_writes():
    """Definition kappa over the corpus, with multiplicity, against the registry.

    The denominator is the multiset: a query on forty dashboards counts forty times,
    which is why kappa is an occurrence share and why §4's parse rates, which are
    deliberately over distinct queries, are a different kind of number.
    """
    import json

    reg = json.loads((ROOT / "paper" / "quoted_figures.json").read_text())
    translated = reg["nb03.kappa_occurrences"]
    total = reg["nb03.corpus_occurrences"]
    assert round(translated / total * 100, 1) == reg["nb03.kappa_pct"]
    # Everything not translated is bottom by the definition, the unparsed included.
    rows = (ROOT / "paper" / "tables" / "translation_coverage.tex").read_text()
    assert str(total - translated) == "295"
    assert "287" in rows and "8" in rows        # 287 refused plus 8 unparsed


# --- the key set, and the two matching lemmas -----------------------------------------
# §5 says of Definition keyset that ``$\chi$ is what the emitter computes, so the case
# analysis below is the one the code performs rather than a description of it''. That
# sentence is a claim about `_match_keys`, and this is where it is held to.

def _chi(term):
    from qlport.emit.promql import _match_keys

    return _match_keys(term)


def test_chi_is_the_function_the_definition_defines():
    r"""Definition keyset, case by case, against `_match_keys`.

    $\star$ for a number, $\overline{k}$ for an aggregation that groups, the operand's
    own keys through a scaling, and for a binary operation: the other side when one is
    $\star$, the larger when one contains the other, and undefined when neither does.
    """
    from qlport.emit.promql import _SCALAR
    from qlport import ir

    def agg(keys):
        return ir.Agg("sum", tuple(keys), "by",
                      ir.Rollup("sum", 60.0, ir.Select("m", (), "gauge"), False))

    # Number is the scalar, which matches anything.
    assert _chi(ir.Number(2.0)) is _SCALAR
    # An aggregation is keyed by its by-list, and an empty list is one unlabelled series.
    assert _chi(agg(["g", "h"])) == ("g", "h")
    assert _chi(agg([])) == ()
    # A scaling carries its operand's keys through unchanged.
    assert _chi(ir.Scale(2.0, agg(["g"]))) == ("g",)

    coarse, fine = agg(["g"]), agg(["g", "h"])
    # One side scalar: the other side's keys.
    assert _chi(ir.BinOp("+", ir.Number(1.0), fine, ())) == ("g", "h")
    assert _chi(ir.BinOp("+", fine, ir.Number(1.0), ())) == ("g", "h")
    # Equal, and one containing the other: the larger either way round.
    assert _chi(ir.BinOp("/", coarse, coarse, ())) == ("g",)
    assert _chi(ir.BinOp("/", fine, coarse, ())) == ("g", "h")
    assert _chi(ir.BinOp("/", coarse, fine, ())) == ("g", "h")
    # Neither containing the other: undefined, which is what Datadog itself rejects.
    assert _chi(ir.BinOp("/", agg(["g"]), agg(["h"]), ())) is None
    # And undefined propagates, so one incomparable pair poisons the whole term.
    assert _chi(ir.BinOp("+", ir.BinOp("/", agg(["g"]), agg(["h"]), ()), coarse, ())) \
        is None


def test_chi_is_undefined_where_the_definition_is_silent():
    """An aggregation that groups by exclusion has no $\\overline{k}$ to be keyed by.

    Definition keyset covers `by` and says nothing about `without`, which is a PromQL
    construct the shared IR carries and no Datadog query produces. The code returns
    undefined rather than guessing, which is the reading that matches the definition's
    silence -- and the one that makes `_binop` refuse instead of emitting a pairing it
    cannot justify.
    """
    from qlport import ir

    without = ir.Agg("sum", ("g",), "without",
                     ir.Rollup("sum", 60.0, ir.Select("m", (), "gauge"), False))
    assert _chi(without) is None
    body = _environment(SECTIONS / "05_translation.tex", "definition", "def:keyset")
    assert r"\mathsf{by}" in body
    assert "without" not in body


def test_the_arithmetic_extension_adds_no_caveat_of_its_own():
    """Theorem arith says $C$ is the union of the subterms' caveats and nothing more.

    Checked as behaviour rather than by reading: the caveats of a quotient are exactly
    the union of the caveats of its two operands. A caveat introduced by the extension
    would be a precondition the theorem does not carry.
    """
    left = _emit("sum:a{*} by {g}.as_count().rollup(sum, 60)")
    right = _emit("sum:b{*} by {g}.as_count().rollup(sum, 60)")
    both = _emit("sum:a{*} by {g}.as_count().rollup(sum, 60) "
                 "/ sum:b{*} by {g}.as_count().rollup(sum, 60)")
    union = {c.name for c in left.caveats} | {c.name for c in right.caveats}
    assert {c.name for c in both.caveats} == union, (
        sorted({c.name for c in both.caveats}), sorted(union))


def test_the_two_lemmas_are_the_two_halves_e8_measured():
    """Lemma dd-matching and lemma pq-matching, against the cached account.

    The lemmas are stated as facts about two systems, and E8 is where both were asked.
    `tests/test_harness_datadog.py` already checks that the emitted form reproduces the
    pairing; what is checked here is the pair of *statements* --- that Datadog keys the
    result by the larger key set and rejects incomparable ones, and that PromQL's
    default matching returns nothing on the case Datadog broadcasts, which is the whole
    reason the emitter writes an `on(...)` clause at all.
    """
    import json

    cases = json.loads((ROOT / "data" / "raw" / "datadog"
                        / "e8.json").read_text())["cases"]

    # Datadog: keyed by the finer side, whichever operand it is.
    for case in ("subset_lhs_finer", "subset_rhs_finer"):
        assert set(cases[case]["datadog"]) == {"g=x,h=p"}, case
    # Datadog: incomparable key sets are rejected, in the lemma's own words.
    assert "subset" in cases["different_keys"]["datadog"]["__error__"]
    # PromQL: default matching pairs identical label sets and drops the rest, so the
    # broadcast case comes back empty while Datadog answers.
    assert cases["rhs_ungrouped_default_matching"]["promql"] == {}
    assert cases["rhs_ungrouped"]["promql"], "the on(...) form must recover it"
    # PromQL: an aggregation drops every label outside its by list, __name__ included,
    # which is why these results are keyed by g alone and not by the metric name.
    assert all(key.startswith("g=") for key in cases["both_grouped_div"]["promql"])

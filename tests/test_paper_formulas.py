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

"""The regular sub-problem (paper §6): glob <-> regex, and the decision procedure.

Two things here are worth more than the rest. First, every glob set `regex_to_globs`
returns is checked to denote the same language as the regex it came from, so "exact"
means decided rather than proposed. Second, the syntax-directed proposer is checked
against the complete procedure of §6's decidability proposition, because a proposer's
refusal looks like
a fact about the languages and is only a fact about the proposer until something
independent agrees.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from qlport.regular import (
    MAX_UNION, STAR_MATCHES_EMPTY, equivalent, glob_to_regex, regex_to_glob,
    regex_to_globs, shortest_word, single_glob_by_search, unanchor,
)

# ---------------------------------------------------------------- glob -> regex


@pytest.mark.parametrize("glob,regex", [
    ("aws", "aws"),
    ("k8s_*", "k8s_.*"),
    ("*pod*", ".*pod.*"),
    ("*", ".*"),
    ("a.b", r"a\.b"),
])
def test_glob_to_regex_is_exact_by_construction(glob, regex):
    assert glob_to_regex(glob) == regex


def test_a_question_mark_in_a_glob_is_a_literal():
    # Datadog documents prefix, suffix and substring wildcards and nothing else, so
    # there is no single-character wildcard to translate `?` into.
    assert glob_to_regex("a?b") == r"a\?b"


@pytest.mark.parametrize("glob,yes,no", [
    ("kube-dns",  ["kube-dns"],            ["kube-dnsx", "kubedns", ""]),
    ("istiod-*",  ["istiod-", "istiod-a"], ["istiod", "xistiod-a"]),
    ("a.b",       ["a.b"],                 ["axb", "ab"]),
    ("x*y",       ["xy", "xaby"],          ["x", "y", "xyz"]),
    ("a+b",       ["a+b"],                 ["ab", "aab"]),
])
def test_the_emission_and_comparison_renderings_denote_one_language(glob, yes, no):
    """Both renderings must accept the same strings, checked by membership.

    This compared `_glob_for_comparison(g)` with itself, which is true of anything.
    The two renderings cannot be compared as strings -- the emitted form carries
    escapes the decider's parser rejects -- so the language is checked directly.
    """
    from qlport.regular import _glob_for_comparison
    for rendering in (glob_to_regex(glob), _glob_for_comparison(glob)):
        for s in yes:
            assert re.fullmatch(rendering, s), (rendering, s)
        for s in no:
            assert not re.fullmatch(rendering, s), (rendering, s)


# ---------------------------------------------------------------- anchors


@pytest.mark.parametrize("regex,body", [
    ("^k8s_.*", "k8s_.*"),
    ("abc$", "abc"),
    ("^abc$", "abc"),
    ("abc", "abc"),
])
def test_redundant_anchors_are_stripped(regex, body):
    assert unanchor(regex) == body


def test_an_interior_anchor_is_refused_rather_than_dropped():
    assert unanchor("a^b") is None
    assert unanchor("a$b") is None
    assert regex_to_globs("a$b").reason == "interior-anchor"


# ---------------------------------------------------------------- the three outcomes


@pytest.mark.parametrize("regex,glob", [
    ("^k8s_.*", "k8s_*"),
    (".*pod.*", "*pod*"),
    (".*", "*"),
    ("5.*", "5*"),
    ("keep(.*)", "keep*"),
    ("abc", "abc"),
    ("a.*b.*c", "a*b*c"),
])
def test_a_single_glob_is_found_and_verified(regex, glob):
    got = regex_to_globs(regex)
    assert got.kind == "single" and got.verified
    assert got.globs == (glob,)
    assert regex_to_glob(regex) == glob


@pytest.mark.parametrize("regex,globs", [
    ("LIST|GET", ("GET", "LIST")),
    ("idle|iowait|steal", ("idle", "iowait", "steal")),
    ("xfs|ext.*", ("ext*", "xfs")),
    ("(Error|Failed)", ("Error", "Failed")),
    ("1(\\.0)?", ("1", "1.0")),
    ("resource|", ("", "resource")),
])
def test_a_union_is_found_where_one_glob_cannot_serve(regex, globs):
    got = regex_to_globs(regex)
    assert got.kind == "union" and got.verified
    assert got.globs == globs
    # A union is not a single glob, and `regex_to_glob` must not pretend otherwise.
    assert regex_to_glob(regex) is None


# Four cases left this list when `TAG_ALPHABET` was pinned. `.` had been refused as a
# `single-character-wildcard` and `.+` as a `nonempty-wildcard`, both on the reading
# that the alphabet was unbounded; over the 41 characters a Datadog tag value may
# contain, each is a union the bound admits, and `test_a_full_class_is_a_finite_union`
# below pins what they return instead. `5..` is still refused, but for its size rather
# than its kind: 41 squared is 1,681 branches against a bound of 64.
@pytest.mark.parametrize("regex,reason", [
    ("5..", "union-too-large"),
    ("[a-z]+", "unbounded-character-class"),
    ("cassandra-\\d+", "unbounded-character-class"),
    ("(ab)*", "repeated-group"),
])
def test_the_refusals_name_the_construct(regex, reason):
    got = regex_to_globs(regex)
    assert got.kind == "none" and not got.exact
    assert got.reason == reason
    assert got.globs == ()


def test_a_full_class_is_a_finite_union_over_the_tag_alphabet():
    """`.` and `.+` are unions, not refusals, once the alphabet is pinned.

    This is the half of §6 that turns on what $\\Sigma$ is. Definition 14 left it
    unstated for eleven passes, and the two readings differ by 218 corpus occurrences:
    over every string `.+` is no finite union of globs, and over the 41 characters a
    Datadog tag value may contain it is the union of `c*`. The equivalence is decided
    over the same alphabet, so what comes back is verified rather than proposed.
    """
    from qlport.regular import TAG_ALPHABET

    assert len(TAG_ALPHABET) == 41
    for regex in (".", ".+"):
        got = regex_to_globs(regex)
        assert got.kind == "union" and got.exact, (regex, got)
        assert len(got.globs) == len(TAG_ALPHABET), (regex, got.globs)
    # `.` is one character from the alphabet; `.+` is one character then anything.
    assert set(regex_to_globs(".").globs) == set(TAG_ALPHABET)
    assert set(regex_to_globs(".+").globs) == {c + "*" for c in TAG_ALPHABET}
    # `.*` stays a single glob: a star already admits every string, E5 having measured
    # that it matches the empty one too.
    assert regex_to_globs(".*").globs == ("*",)


# --------------------------- the dichotomy corollary: three things no glob says


def test_no_glob_denotes_a_finite_language_of_two_words():
    # Dichotomy, part 1. A glob is a singleton or unbounded in length, so `GET|LIST` has no
    # single glob however it is written -- only a union.
    assert regex_to_glob("LIST|GET") is None
    assert single_glob_by_search("LIST|GET") is None


def test_no_glob_denotes_the_non_empty_values():
    # Dichotomy, part 2, and the reading of `*` it depends on.
    assert STAR_MATCHES_EMPTY
    assert regex_to_glob(".+") is None
    assert regex_to_glob(".*") == "*"


def test_no_glob_confines_a_star_to_part_of_the_alphabet():
    # Dichotomy, part 3.
    assert regex_to_glob("[0-9]+") is None
    assert regex_to_glob("[0-9]*") is None


# ------------------------------------- the decision procedure


@pytest.mark.parametrize("regex,word,unique", [
    ("abc", "abc", True),
    ("ab.*cd", "abcd", True),
    ("LIST|GET", "GET", True),      # shortest is unique even though the language is not
    ("5..", "555", False),          # three characters, any two free -> not unique
    ("a|b", "a", False),
])
def test_shortest_word_and_its_uniqueness(regex, word, unique):
    got, is_unique = shortest_word(regex)
    assert is_unique is unique
    if unique:
        # When the shortest word is unique it is a specific string, and the glob
        # construction reads its characters -- so the content is the claim, not the
        # length. Asserting only the length let a mutation returning the wrong
        # character survive.
        assert got == word
    else:
        assert len(got) == len(word)


def test_the_empty_language_has_no_shortest_word():
    assert shortest_word("[]") == (None, False)
    assert single_glob_by_search("[]") is None


@pytest.mark.parametrize("regex", [
    "^k8s_.*", ".*pod.*", ".*", "5.*", "keep(.*)", "abc", "a.*b.*c",
    "5..", "[a-z]+", "LIST|GET", ".+", "xfs|ext.*",
])
def test_the_complete_search_agrees_with_the_structural_proposer(regex):
    """The complete procedure against the fast one, on every outcome."""
    assert single_glob_by_search(regex) == regex_to_glob(regex)


def test_a_non_unique_shortest_word_settles_it_without_searching():
    # The decidability proof's first step: L(g)'s shortest word is unique, so a regex
    # whose shortest word is not cannot be a glob -- no candidate need be built.
    _, unique = shortest_word("5..")
    assert not unique
    assert single_glob_by_search("5..") is None


def test_the_search_finds_a_glob_the_structural_pass_would_also_find():
    # A case with stars in both outer gaps, to exercise the subset pruning.
    assert single_glob_by_search(".*mid.*") == "*mid*"


# ------------------------------------- verification is not optional


def test_every_returned_glob_set_denotes_the_regex_it_came_from():
    from qlport.regular import _globs_for_comparison
    for regex in ("^k8s_.*", "LIST|GET", "xfs|ext.*", "1(\\.0)?", "5[0-9]",
                  "(Error|Failed)", ".*", "abc"):
        got = regex_to_globs(regex)
        assert got.exact and got.verified, regex
        assert equivalent(unanchor(regex), _globs_for_comparison(got.globs)), regex


def test_a_union_wider_than_the_bound_is_refused_with_the_bound_named():
    # The dichotomy does not forbid it; the bound does, because a scope is text someone
    # maintains. `[a-z][a-z]` is 676 literals.
    got = regex_to_globs("[a-z][a-z]")
    assert got.kind == "none"
    assert got.reason in {"union-too-large", "too-many-combinations",
                          "character-class-too-wide"}
    assert MAX_UNION == 64


def test_globs_are_returned_sorted_and_deduplicated():
    # greenery holds alternation branches in a set, so order has to come from here.
    got = regex_to_globs("b|a|b")
    assert got.globs == ("a", "b")


# --------------------- the recorded cross-check against the complete procedure


CROSSCHECK = Path(__file__).resolve().parent.parent / "data" / "raw" / "globs" / "crosscheck.json"


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    if not CROSSCHECK.exists():  # pragma: no cover - run scripts/crosscheck_globs.py
        pytest.skip(f"{CROSSCHECK.name} not recorded; run scripts/crosscheck_globs.py")
    return json.loads(CROSSCHECK.read_text())


def test_the_proposer_and_the_complete_procedure_never_disagreed(crosscheck):
    """The point of the file: §6's refusals are not the proposer's refusals.

    `scripts/crosscheck_globs.py` re-decides every concrete corpus regex with the
    complete procedure. A disagreement would mean either a glob the syntax-directed
    proposer missed -- making a measured refusal an artefact -- or a proposal the search
    cannot confirm. Neither has happened, and this fails if one starts to.
    """
    assert crosscheck["disagreements"] == []
    assert crosscheck["tally"].get("disagree", 0) == 0


def test_the_cross_check_reached_every_pattern(crosscheck):
    """The per-outcome tally is quoted in the paper, so it is pinned figure by figure.

    A floor of 150 with a positive count per outcome would have passed while the
    cross-check silently degraded by eighteen patterns.

    It now reaches all of them. The earlier run stopped at 168 of 183 because it was
    given a 2,400 s budget; three patterns of one family cost 100-170 s each and the
    clock ran out in the middle of them. Unbudgeted it takes 4,725 s and nothing is
    left unreached -- and nothing exceeds the search's own candidate bound either, so
    there is no undecided residue to disclose.
    """
    assert crosscheck["tally"] == {
        "agree:single": 61, "agree:union": 73, "agree:none": 50,
    }, crosscheck["tally"]
    assert crosscheck["tally"].get("not-reached", 0) == 0
    assert crosscheck["tally"].get("search-too-large", 0) == 0
    n = crosscheck["provenance"]["n_concrete_regexes"]
    assert sum(crosscheck["tally"].values()) == n == len(crosscheck["per_regex"]) == 184


def test_the_cross_check_covers_the_same_patterns_the_paper_tabulates(crosscheck):
    """The complete procedure and the structural proposer must share a population.

    Both now read it from `corpus.promql_matcher_values` over the same filtered PromQL
    set; each used to carry its own double-quoted-only scanner, and this one also counted
    the LogQL panels, so §6's "168 of the 183" was over a different 183 than the table
    beside it.
    """
    from qlport import corpus as c

    root = Path(__file__).resolve().parent.parent
    occ: dict[str, int] = {}
    for q in c.read_queries(root / "data" / "corpus" / "queries.jsonl"):
        if q.lang == "promql" and c.is_promql(q.text):
            for rx in c.promql_matcher_values(q.text):
                occ[rx] = occ.get(rx, 0) + 1
    concrete = {r for r in occ if "$" not in r}
    assert set(crosscheck["per_regex"]) == concrete, (
        len(crosscheck["per_regex"]), len(concrete))


def test_the_cross_check_was_run_under_the_reading_the_paper_states(crosscheck):
    # A result recorded under the other reading of `*` would not support §6's numbers.
    prov = crosscheck["provenance"]
    assert prov["star_matches_empty"] is STAR_MATCHES_EMPTY
    assert prov["max_union"] == MAX_UNION


def test_redundant_escapes_are_dropped_before_deciding():
    """`re.escape` output must be decidable, not reported as unparseable.

    A hyphen never needs a backslash, but a regex that has been through `re.escape`
    carries one, and the decision procedure's parser rejects it. The escape changes no
    denotation, so dropping it turns an "unparsed" into an answer.
    """
    from qlport.regular import drop_redundant_escapes
    assert drop_redundant_escapes(r"kube\-system") == "kube-system"
    assert drop_redundant_escapes(r"a\.b") == r"a\.b"        # a dot does need it
    assert drop_redundant_escapes(r"\d+") == r"\d+"          # and so does a class
    assert drop_redundant_escapes(r"a\*b") == r"a\*b"


def test_an_escaped_hyphen_regex_now_decides():
    got = regex_to_globs(r"generic\-router|generic\-device")
    assert got.kind == "union", got.reason
    assert got.globs == ("generic-device", "generic-router")


def test_dropping_escapes_does_not_change_the_language():
    from qlport.regular import drop_redundant_escapes
    for r in (r"kube\-system", r"a\-b|c\-d", r"x\-"):
        assert equivalent(r.replace("\\-", "-"), drop_redundant_escapes(r))


# --- the gate that makes "a returned glob is exact" true ------------------------------
# `regex_to_globs` proposes a glob set from the regex's syntax and then *decides*
# equivalence before returning it. §6 rests on that gate: it is what makes a returned
# glob exact and a refusal a statement about the proposer rather than about the
# languages. Every other test here feeds it regexes the proposer already gets right, so
# the gate itself was dead code as far as the suite was concerned -- disabling it, or
# returning a known-wrong proposal as verified, left all 547 tests passing.


def test_a_wrong_proposal_is_refused_by_the_equivalence_gate(monkeypatch):
    """Force the proposer to lie and check the gate catches it."""
    import qlport.regular as R

    monkeypatch.setattr(R, "_pattern_globs", lambda pattern: ["definitely-not-abc"])
    got = R.regex_to_globs("abc")
    assert not got.verified and not got.exact
    assert got.reason == "proposal-not-equivalent"
    assert got.globs == ()


def test_a_wrong_proposal_is_refused_even_when_it_is_close(monkeypatch):
    """A near-miss is the case a syntax-directed proposer would actually produce."""
    import qlport.regular as R

    # `ab.*` denotes ab-anything; proposing the bare literal `ab` drops every longer
    # word, which only a language-equivalence check notices.
    monkeypatch.setattr(R, "_pattern_globs", lambda pattern: ["ab"])
    got = R.regex_to_globs("ab.*")
    assert got.reason == "proposal-not-equivalent", got


def test_the_gate_passes_a_correct_proposal_through(monkeypatch):
    """The companion: the gate must not refuse everything, or the above is vacuous."""
    import qlport.regular as R

    monkeypatch.setattr(R, "_pattern_globs", lambda pattern: ["ab*"])
    got = R.regex_to_globs("ab.*")
    assert got.verified and got.exact and got.globs == ("ab*",)

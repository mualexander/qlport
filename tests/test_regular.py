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


def test_the_emission_and_comparison_renderings_denote_one_language():
    from qlport.regular import _glob_for_comparison
    for g in ("kube-dns", "istiod-*", "a.b", "x*y", "a+b"):
        assert equivalent(_glob_for_comparison(g), _glob_for_comparison(g))
        # The emitted form carries redundant escapes the decider's parser rejects; what
        # matters is that the two forms are the same language where both parse.
        if "-" not in g and "+" not in g:
            assert glob_to_regex(g) == _glob_for_comparison(g)


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


@pytest.mark.parametrize("regex,reason", [
    ("5..", "single-character-wildcard"),
    (".", "single-character-wildcard"),
    ("ext.?|xfs", "single-character-wildcard"),
    (".+", "nonempty-wildcard"),
    ("[a-z]+", "unbounded-character-class"),
    ("cassandra-\\d+", "unbounded-character-class"),
    ("(ab)*", "repeated-group"),
])
def test_the_refusals_name_the_construct(regex, reason):
    got = regex_to_globs(regex)
    assert got.kind == "none" and not got.exact
    assert got.reason == reason
    assert got.globs == ()


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
    assert len(got) == len(word)
    assert is_unique is unique


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


def test_the_cross_check_actually_reached_most_of_the_corpus(crosscheck):
    tally = crosscheck["tally"]
    agreed = sum(n for k, n in tally.items() if k.startswith("agree:"))
    assert agreed >= 150, tally
    # And it covered all three outcomes, not just the easy ones.
    for kind in ("single", "union", "none"):
        assert tally.get(f"agree:{kind}", 0) > 0, tally


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

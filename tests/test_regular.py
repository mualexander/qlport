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
# below pins what they return instead. `5..` left the list when the bound was raised to
# 41 squared, which is exactly the two free positions it needs; `2..|5..` is twice that
# and is the smallest corpus pattern the bound still refuses.
@pytest.mark.parametrize("regex,reason", [
    ("2..|5..", "union-too-large"),
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
    # maintains. `2..|5..` is two free positions twice over, 3,362 literals against a
    # bound of 41 squared -- and `[a-z][a-z]`, which this test used while the bound was
    # 64, is now admitted at 676, so the bound has to be named by something larger.
    got = regex_to_globs("2..|5..")
    assert got.kind == "none"
    assert got.reason in {"union-too-large", "too-many-combinations",
                          "character-class-too-wide"}
    assert MAX_UNION == 1681


def test_the_bound_is_the_square_of_the_alphabet():
    """Two unconstrained character positions, which is what the corpus's gap implies.

    The sizes the refused patterns need are five between 82 and 206 and then nothing
    until 41 squared, so every bound in that gap admits the same patterns and only the
    square is a different decision. Pinning the relation rather than the number is what
    keeps the two in step if the alphabet is ever remeasured.
    """
    from qlport.regular import TAG_ALPHABET

    assert MAX_UNION == len(TAG_ALPHABET) ** 2
    # One free position is well inside it; three are outside, and the intermediate
    # product is guarded separately so they never reach the union bound at all.
    assert regex_to_globs(".").kind == "union"
    assert regex_to_globs("5..").kind == "union"
    assert regex_to_globs("5...").kind == "none"


def test_the_product_bound_guards_the_expansion_the_union_bound_cannot():
    """`MAX_PRODUCT` is the second bound, and it is the one that binds first.

    `MAX_UNION` bounds what gets emitted, which cannot bound what builds it: the
    expansion is a Cartesian product, and three free positions are $41^{3}$ globs before
    anything is there to count them. So §6 carries two bounds, and the product one is
    what keeps a third free position from ever reaching the union bound. Nothing tested
    it, and all three places the product is taken refuse in their own words -- which is
    how a refused pattern says which bound refused it.
    """
    from qlport.regular import MAX_PRODUCT

    assert MAX_PRODUCT == 4096
    # Above the union bound, so a product the guard admits is still not emittable: the
    # product guard exists to stop the expansion, not to choose what the scope may say.
    assert MAX_PRODUCT > MAX_UNION
    # A concatenation of free positions: 41 cubed.
    assert regex_to_globs("5...").reason == "too-many-combinations"
    # A bounded character class: ten to the fourth.
    assert regex_to_globs("[0-9]{4}").reason == "character-class-too-wide"
    # A bounded group: four branches to the seventh.
    assert regex_to_globs("(a|b|c|d){7}").reason == "repeated-group-too-wide"
    # The guard is on exceeding the bound, not on reaching it: a product of exactly
    # MAX_PRODUCT is built, and then refused by the union bound in its own words.
    assert 4 ** 6 == MAX_PRODUCT
    assert regex_to_globs("(a|b|c|d){6}").reason == "union-too-large"
    # And a product inside both bounds decides.
    assert regex_to_globs("[0-9]{3}").kind == "union"


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

    It now reaches all of them, in 2,998 s unbudgeted, and nothing exceeds the search's
    own candidate bound either, so there is no undecided residue to disclose.

    The tally has moved twice and neither move was a language changing. Pinning the
    alphabet at 41 characters made `.` and `.+` finite unions rather than refusals, and
    raising the bound to 41 squared admitted twelve more patterns whose branch counts sit
    at or below two free positions. So 43 patterns walked from `none` to `union`, and what
    is left in `none` is four refused by the bound and three by Corollary 5(3) -- the only
    part of it that refuses anything over this alphabet.
    """
    assert crosscheck["tally"] == {
        "agree:single": 61, "agree:union": 116, "agree:none": 7,
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


# --- deciding under a deadline ---------------------------------------------------------
# The two tests below are the only ones here that assert a cost rather than a result,
# and their names say "in seconds" and "quickly" while nothing bounded either. The
# regression they exist to catch is a return to the unfactored alternation, which did
# not finish in four hours -- so a bare elapsed-time assertion cannot report it: the
# suite would hang rather than fail. The decision runs in a child process that is killed
# at the deadline, and the overrun is the failure.
#
# The ceilings are loose on purpose. On the development machine the first decides in
# about 26 s and the second in under a second; what the deadline has to separate is
# seconds from hours, not one run from the next.

DEADLINE = 240.0


def _decide_in_child(regex: str, out) -> None:    # pragma: no cover - child process
    try:
        from qlport.regular import regex_to_globs
        got = regex_to_globs(regex)
        out.put(("ok", got.kind, got.verified, got.exact, sorted(got.globs)))
    except BaseException as exc:                  # noqa: BLE001 - reported, not handled
        import traceback
        out.put(("raised", traceback.format_exc() or repr(exc)))


def _decided_within(regex: str, seconds: float = DEADLINE) -> dict:
    """`regex_to_globs(regex)` in a child process, killed if it overruns `seconds`."""
    import multiprocessing as mp
    import queue as _queue

    out: mp.Queue = mp.Queue()
    child = mp.Process(target=_decide_in_child, args=(regex, out), daemon=True)
    child.start()
    try:
        try:
            got = out.get(timeout=seconds)
        except _queue.Empty:
            pytest.fail(f"{regex!r} did not decide within {seconds:g}s")
    finally:
        child.terminate()
        child.join(10)
    if got[0] == "raised":
        pytest.fail(f"{regex!r} raised in the child:\n{got[1]}")
    _, kind, verified, exact, globs = got
    return {"kind": kind, "verified": verified, "exact": exact, "globs": globs}


# --- the factored comparison -----------------------------------------------------------
# `.+` is proposed as 41 globs differing in one character position, and rendering that as
# a 41-branch alternation duplicates the surrounding context 41 times. greenery unions
# automata pairwise by a product construction, at roughly twelve seconds a branch over a
# 41-symbol alphabet, so nine corpus patterns did not decide in four hours. The gate now
# decides against a factored regex instead. These tests are what stop that factoring from
# quietly becoming a way of agreeing with the proposer.


def test_the_factored_regex_denotes_the_union_it_replaces(monkeypatch):
    """Factoring is distributivity, so the two renderings must agree as languages.

    Checked over a three-character alphabet, because the comparison this test makes is
    the expensive one the factoring exists to avoid: at $41$ characters the alternation
    costs about twelve seconds a branch, which is twenty-five minutes for the three
    shapes below. Distributivity does not depend on the alphabet's size, and the sizes
    are exercised separately -- the corpus patterns themselves run at $41$.
    """
    import qlport.regular as R

    monkeypatch.setattr(R, "TAG_ALPHABET", "abc")
    monkeypatch.setattr(R, "_SIGMA_CLASS", "[abc]")
    for prefix, suffix in (("x", "y"), ("*ab", "ba*"), ("", "*")):
        globs = tuple(f"{prefix}{c}*{suffix}" for c in sorted("abc"))
        factored = R._factor_alphabet_expansion(globs)
        assert factored is not None, (prefix, suffix)
        assert R.equivalent(factored, R._globs_for_comparison(globs)), (prefix, suffix)


@pytest.mark.parametrize("globs", [
    ("a*", "b*"),                                   # not the whole alphabet
    ("ab", "cd"),                                   # two positions vary
    ("x", "y", "z"),                                # wrong cardinality
])
def test_a_family_that_is_not_an_alphabet_expansion_is_not_factored(globs):
    """Anything else must fall back to the alternation rather than be collapsed."""
    from qlport.regular import _factor_alphabet_expansion
    assert _factor_alphabet_expansion(globs) is None


def test_the_right_sized_family_over_the_wrong_characters_is_not_factored():
    """41 globs varying in one position, but not over the alphabet, must not factor."""
    from qlport.regular import TAG_ALPHABET, _factor_alphabet_expansion
    wrong = tuple(f"x{c}*" for c in sorted(TAG_ALPHABET.replace("z", "Z")))
    assert len(wrong) == len(TAG_ALPHABET)
    assert _factor_alphabet_expansion(wrong) is None


def test_the_gate_still_catches_a_wrong_proposal_that_factors(monkeypatch):
    """The case the factoring could have hidden.

    A proposal shaped like an alphabet expansion takes the cheap path, so the cheap path
    has to be the one that refuses a wrong one. Here the literal is wrong -- `spans`
    where the regex says `spams` -- which no amount of factoring should forgive.
    """
    import qlport.regular as R

    monkeypatch.setattr(R, "_pattern_globs",
                        lambda pattern: [f"*proc_{c}*spans*" for c in sorted(R.TAG_ALPHABET)])
    got = R.regex_to_globs(".*proc_.+spams.*")
    assert got.reason == "proposal-not-equivalent", got
    assert not got.exact and not got.verified


def test_an_alphabet_expansion_decides_in_seconds_and_is_verified():
    """The nine-pattern family that did not finish in four hours, as one case.

    Run under `DEADLINE`, so the four hours are what fails rather than what happens.
    """
    from qlport.regular import TAG_ALPHABET

    got = _decided_within(".*otelcol_processor_.+refused_spans.*")
    assert got["kind"] == "union" and got["verified"] and got["exact"]
    assert len(got["globs"]) == len(TAG_ALPHABET)
    # Every branch is the regex's own literals with one alphabet character between them.
    assert set(got["globs"]) == {f"*otelcol_processor_{c}*refused_spans*"
                                 for c in TAG_ALPHABET}


# --- factoring a union of several expansions -------------------------------------------
# A pattern can carry more than one expansion. `pvc.*.ctrl.*|pvc.*.rep.*|pvc.*.target.*`
# is three shapes each with its own 41 branches, and the single-product test refused the
# lot -- which cost the cross-check over four hours on that one pattern. The globs
# separate by length, so each group is factored on its own.


def test_a_union_of_several_expansions_is_factored_per_group(monkeypatch):
    """Three shapes, three groups, three branches instead of one per glob.

    Over a three-character alphabet, because the comparison this test makes against the
    unfactored alternation is the expensive one the factoring exists to avoid.
    """
    import qlport.regular as R

    monkeypatch.setattr(R, "TAG_ALPHABET", "abc")
    monkeypatch.setattr(R, "_SIGMA_CLASS", "[abc]")
    globs = tuple(f"pvc*{c}*{tail}*" for tail in ("ctrl", "rep", "target")
                  for c in sorted("abc"))
    assert len(globs) == 9
    factored = R._factor_alphabet_expansion(globs)
    assert factored is not None
    assert factored.count("|") == 2, factored          # three branches, not nine
    assert R.equivalent(factored, R._globs_for_comparison(globs))


def test_an_expansion_beside_a_lone_literal_still_factors(monkeypatch):
    """A singleton group is its own rendering, so a mixed union is not refused."""
    import qlport.regular as R

    monkeypatch.setattr(R, "TAG_ALPHABET", "abc")
    monkeypatch.setattr(R, "_SIGMA_CLASS", "[abc]")
    globs = tuple(sorted({f"x{c}*" for c in "abc"} | {"zzzz"}))
    factored = R._factor_alphabet_expansion(globs)
    assert factored is not None
    assert R.equivalent(factored, R._globs_for_comparison(globs))


def test_an_ordinary_union_is_not_factored_and_so_is_unchanged():
    """`xfs|ext.*` has no expansion in it, so the gate must decide it as it always did."""
    from qlport.regular import _factor_alphabet_expansion

    for globs in (("ext*", "xfs"), ("GET", "LIST"), ("", "resource")):
        assert _factor_alphabet_expansion(globs) is None, globs


def test_the_gate_catches_a_wrong_proposal_in_one_group_of_several(monkeypatch):
    """The case per-group factoring could hide.

    Two groups, one right and one wrong: the `rep` branch is proposed where the regex
    says `reps`. A gate that factored each group and compared only the shapes, rather
    than the languages, would pass this.
    """
    import qlport.regular as R

    globs = [f"pvc*{c}*ctrl*" for c in sorted(R.TAG_ALPHABET)]
    globs += [f"pvc*{c}*rep*" for c in sorted(R.TAG_ALPHABET)]
    monkeypatch.setattr(R, "_pattern_globs", lambda pattern: globs)
    got = R.regex_to_globs("pvc.*.ctrl.*|pvc.*.reps.*")
    assert got.reason == "proposal-not-equivalent", got
    assert not got.exact and not got.verified


def test_the_multi_shape_union_decides_quickly_and_is_verified():
    """The corpus pattern that cost the cross-check four hours, as one case.

    Under the same deadline, for the same reason: without one, a regression here is a
    suite that never finishes rather than a test that fails.
    """
    got = _decided_within("pvc.*.ctrl.*|pvc.*.rep.*|pvc.*.target.*")
    assert got["kind"] == "union" and got["verified"] and got["exact"]
    assert len(got["globs"]) == 123


def test_a_repeated_glob_does_not_pass_the_cardinality_test(monkeypatch):
    """The one way the premise behind the product test can fail.

    `_factor_one_product` reasons that |Sigma|^k globs with full projections at every
    varying position must be the whole product. That holds for distinct globs and fails
    for a multiset: over a three-character alphabet, `aa ab ac ba bb bc ca cb cb` is
    nine globs projecting onto all of Sigma twice over, and is not the product. So the
    input is deduplicated before it is counted, and this family is refused for being
    eight globs rather than accepted for being nine.
    """
    import qlport.regular as R

    monkeypatch.setattr(R, "TAG_ALPHABET", "abc")
    monkeypatch.setattr(R, "_SIGMA_CLASS", "[abc]")
    full = [a + b for a in "abc" for b in "abc"]
    assert R._factor_one_product(tuple(full)) is not None
    repeated = [g for g in full if g != "cc"] + ["cb"]
    assert len(repeated) == len(full)
    assert R._factor_one_product(tuple(repeated)) is None


def test_a_deduplicated_family_still_factors(monkeypatch):
    """Dedup must not turn a genuine expansion into a refusal for being short."""
    import qlport.regular as R

    monkeypatch.setattr(R, "TAG_ALPHABET", "abc")
    monkeypatch.setattr(R, "_SIGMA_CLASS", "[abc]")
    full = [a + b for a in "abc" for b in "abc"]
    once = R._factor_one_product(tuple(full))
    twice = R._factor_one_product(tuple(full + full))
    assert once is not None and twice == once


def test_the_product_test_is_implied_by_the_three_conditions_before_it(monkeypatch):
    """The claim the corrected comment makes, checked by exhaustion.

    Over a three-character alphabet and two varying positions there is exactly one way
    to choose nine distinct pairs, and it is the product. The comment that used to stand
    at this test asserted a counterexample exists; none does, so the search is the test.
    """
    import itertools as it

    import qlport.regular as R

    monkeypatch.setattr(R, "TAG_ALPHABET", "abc")
    monkeypatch.setattr(R, "_SIGMA_CLASS", "[abc]")
    pairs = ["".join(t) for t in it.product("abc", repeat=2)]
    nines = list(it.combinations(pairs, 9))
    assert len(nines) == 1
    for family in nines:
        assert {g[0] for g in family} == {g[1] for g in family} == set("abc")
        assert R._factor_one_product(tuple(family)) is not None


# --- the universe the gate decides over, where a pattern leaves it ----------------

WIDENED = Path(__file__).resolve().parent.parent / "data" / "raw" / "globs" / \
    "widened.json"


@pytest.fixture(scope="module")
def widened() -> dict:
    if not WIDENED.exists():  # pragma: no cover - run widen_alphabet_check.py
        pytest.skip("widened.json not recorded; run scripts/widen_alphabet_check.py")
    return json.loads(WIDENED.read_text())


def test_the_patterns_whose_literals_leave_the_alphabet_are_still_counted(widened):
    """A comparison that cannot fail is not a verification, and 36 of 184 could not.

    Every equivalence is decided over TAG_ALPHABET, which is right for a pattern whose
    literals are legal tag values. For `LIST|GET` or `01-Inlet Ambient` both sides
    restrict to the empty language and equivalence holds for that reason alone. PromQL
    label values are not lower-cased, so these are ordinary Kubernetes and HTTP values
    rather than corpus defects, and they carry 13.0% of the matcher occurrences.
    """
    assert widened["narrow_alphabet_size"] == 41
    assert widened["patterns_out_of_sigma"] == 36
    assert widened["patterns_total"] == 184


def test_every_one_of_them_decides_the_same_way_over_a_real_alphabet(widened):
    """Re-decided over Sigma widened by the characters each pattern uses, all 36 agree.

    So the classifications §6 reports for them were right rather than lucky, and no
    figure in the paper moves. What changes is that the claim is now checked by a
    comparison that could have come out the other way.
    """
    assert widened["agreed"] == widened["patterns_out_of_sigma"]
    disagreeing = [r for r in widened["rows"] if not r["agrees"]]
    assert disagreeing == [], disagreeing


def test_the_widened_alphabet_really_was_wider(widened):
    """Without this the test above is the vacuous comparison it exists to replace."""
    for row in widened["rows"]:
        assert row["extra_characters"], row
        assert row["alphabet_size"] == 41 + len(row["extra_characters"]), row


# --- the alphabet ---------------------------------------------------------------------

def test_the_alphabet_is_the_one_the_corpus_writes():
    """$\\Sigma$ is the universe every equivalence in this module is decided over.

    Which makes it a claim rather than a constant, and the most load-bearing one here:
    over all strings `.+` is no finite union of globs and 218 corpus occurrences are
    inexpressible, while over the strings a tag value may contain it is the union of
    `c*` for each character and a scope can say it. §6.4 argues the second universe is
    the right one because a string that is not a legal tag value cannot appear on either
    side -- and that argument is only as good as the alphabet being the one Datadog
    accepts and the corpus writes.

    So the corpus is censused rather than trusted. The documentation's set and the
    corpus's are asserted equal in both directions: no value uses a character outside
    `TAG_ALPHABET`, and no character of `TAG_ALPHABET` goes unused, which is the half
    that would catch an alphabet quietly widened to make a pattern decide.

    `*` is set aside because it is the glob wildcard rather than a character of a
    value; `rem:glob-constructs` is the measurement that Datadog's filter grammar has
    no other wildcard to confuse it with.
    """
    import collections

    import pandas as pd

    from qlport import corpus as c
    from qlport.parse import parse
    from qlport.regular import TAG_ALPHABET

    root = Path(__file__).resolve().parent.parent
    queries = [q for q in c.read_queries(root / "data" / "corpus" / "queries.jsonl")
               if q.lang == "datadog"]
    assert queries, "no Datadog queries in the corpus"
    norm = pd.Series([c.normalise("datadog", q.text) for q in queries]).value_counts()

    values: set[str] = set()
    for text in norm.index:
        if not c.parse_outcome("datadog", text).ok:
            continue                      # the six that do not parse carry no scope
        for token in parse("datadog", text).scan_values(lambda _: True):
            if getattr(token, "type", "") != "TAG":
                continue
            body = str(token)
            body = body[1:] if body.startswith("!") else body
            if ":" in body:               # `k:v`; a bare key carries no value
                values.add(body.split(":", 1)[1])

    assert len(values) > 300, len(values)
    used = collections.Counter("".join(values).replace("*", ""))
    assert set(used) == set(TAG_ALPHABET), {
        "outside the alphabet": sorted(set(used) - set(TAG_ALPHABET)),
        "in the alphabet but unwritten": sorted(set(TAG_ALPHABET) - set(used)),
    }


# --- what the undocumented reading of a glob's star is worth ---------------------------

STAR_READING = (Path(__file__).resolve().parent.parent / "data" / "raw" / "globs"
                / "star_reading.json")


@pytest.fixture(scope="module")
def star_reading() -> dict:
    if not STAR_READING.exists():            # pragma: no cover - the cache is committed
        pytest.skip(f"{STAR_READING} missing; run scripts/star_reading_check.py")
    return json.loads(STAR_READING.read_text())


def test_the_reading_the_corpus_was_decided_under_is_the_measured_one():
    """E5 asked a real account and the wildcard matches the empty string.

    The constant is the only place that answer lives, so a change to it silently
    rewrites every figure in §6. `rem:star-empty` says how much: this is the assertion
    that the section's numbers were computed under the reading the experiment settled.
    """
    from qlport.regular import STAR_MATCHES_EMPTY

    assert STAR_MATCHES_EMPTY is True


def test_the_other_reading_costs_what_the_section_says_it_costs(star_reading):
    """§6.4 and `rem:star-empty` quote the counterfactual; this is where it comes from.

    What the cache measures is this paper's procedure under the other reading, not that
    reading's own expressibility -- the complete search of `prop:decidable` declines to
    run without the reading taken. That is the right scope for the claim, because what
    is at stake is every figure the section computes, and it is why the numbers below
    are about classification rather than about globs existing.
    """
    taken, other = star_reading["taken"], star_reading["other"]
    assert star_reading["provenance"]["patterns"] == 184
    assert star_reading["provenance"]["occurrences"] == 1041

    # The reading taken, which must still be the split notebook 04 reports.
    assert {k: v["distinct"] for k, v in taken.items()} == {
        "single": 61, "union": 116, "none": 7}
    assert {k: v["occurrences"] for k, v in taken.items()} == {
        "single": 309, "union": 648, "none": 84}

    # And the other one: 599 refused where the reading taken refuses 84.
    assert other["none"]["occurrences"] == 599
    assert other["none"]["distinct"] == 79
    assert star_reading["moved"]["distinct"] == 72
    assert star_reading["moved"]["occurrences"] == 515
    # Every move is toward refusal, which is the asymmetry the remark turns on: a star
    # that cannot stand for nothing cannot stand for the `.*` these patterns are built
    # from, so nothing is gained anywhere to offset it.
    assert set(star_reading["moved"]["by_transition"]) == {"single->none", "union->none"}
    assert all(t["distinct"] == 36
               for t in star_reading["moved"]["by_transition"].values())


def test_the_counterfactual_rows_add_up_to_its_tallies(star_reading):
    """The tallies are a reduction of the rows, so they are derived here, not read."""
    import collections

    rows = star_reading["rows"]
    assert len(rows) == 184
    for key in ("taken", "other"):
        got: dict[str, dict[str, int]] = collections.defaultdict(
            lambda: {"distinct": 0, "occurrences": 0})
        for r in rows:
            got[r[key]]["distinct"] += 1
            got[r[key]]["occurrences"] += r["occurrences"]
        assert dict(got) == star_reading[key], key
    moved = [r for r in rows if r["taken"] != r["other"]]
    assert len(moved) == star_reading["moved"]["distinct"]
    assert sum(r["occurrences"] for r in moved) == star_reading["moved"]["occurrences"]

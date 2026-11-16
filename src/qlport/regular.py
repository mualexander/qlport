"""The regular sub-problem (paper §6): regex matcher <-> Datadog glob.

The two directions are not symmetric, and the asymmetry is the point.

**Glob -> regex** is exact by construction, and is what the main direction needs
(`lower/datadog.py`). A Datadog tag glob is a sequence of literal strings separated by
`*`: the documentation supports prefix, suffix and substring wildcards and nothing
else (docs.datadoghq.com/metrics/advanced-filtering/). In particular there is **no single-character wildcard** --- no
`?` --- so `?` in a tag value is a literal, and `is_glob` in the lowering module treats
it as one.

**Regex -> glob** is the reverse direction's problem, and most regexes have no glob.
This module proposes a glob, or a finite union of globs, from the regex's structure and
then *verifies* the proposal by deciding equivalence of the two regular languages
(Rich Thm 9.5, via `greenery`). Nothing is returned unverified, so a returned glob is
exact; a refusal carries a reason naming the construct that caused it.

A finite union is the right target rather than a single glob because a Datadog scope can
disjoin over one tag key --- `k IN (g1, ..., gn)`, or `k:g1 OR ... OR k:gn` --- which is
exactly the construction §5's single-key disjunction proposition reads in the
other direction. So the
interesting classification is three-way: one glob, a union of globs, or neither.

What `*` means when a gap is empty is not documented: `web-*` is published as a prefix
filter without saying whether it matches `web-` itself. The two readings are mutually
exclusive for `.*` and `.+` --- a glob has one wildcard, which cannot both require and
not require a character --- so under either reading one of those two common regexes is
inexpressible. `STAR_MATCHES_EMPTY` records the reading taken, and
`docs/trial_account_checks.md` carries the check that would settle it.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

__all__ = [
    "STAR_MATCHES_EMPTY", "GlobConversion", "glob_to_regex", "globs_to_regex",
    "equivalent", "unanchor", "regex_to_globs", "regex_to_glob",
    "single_glob_by_search", "shortest_word", "shortest_words", "SearchTooLarge",
]

# Whether a Datadog `*` matches the empty string. Undocumented; see the module docstring.
STAR_MATCHES_EMPTY = True

# How many globs a union may contain, and how large an intermediate product may get,
# before the conversion gives up. A union is emitted as `IN (...)` in a scope, so a
# thousand-branch union is not a translation anyone wants even when it is exact.
MAX_UNION = 64
MAX_PRODUCT = 4096


def glob_to_regex(g: str) -> str:
    """A Datadog tag glob as an (implicitly fully anchored) regular expression.

    `*` is the only wildcard; every other character, `?` included, is a literal. `*`
    becomes `.*`, so the alphabet is the newline-free one `.` ranges over --- see
    `_dot_charclass`.
    """
    return "".join(".*" if c == "*" else re.escape(c) for c in g)


def globs_to_regex(globs) -> str:
    """A finite union of globs as one regular expression, for emission."""
    parts = [glob_to_regex(g) for g in globs]
    if not parts:
        return "[]"            # the empty language
    return "|".join(f"(?:{p})" for p in parts) if len(parts) > 1 else parts[0]


# Metacharacters that must carry a backslash for the *comparison* renderer below.
_META = frozenset(".^$*+?()[]{}|\\")


def _glob_for_comparison(g: str) -> str:
    """The same language as `glob_to_regex(g)`, in the dialect the decider parses.

    `glob_to_regex` escapes with `re.escape`, which emits redundant escapes such as
    `\\-`. RE2 accepts those --- the emitted matchers were verified against a real
    Prometheus, so that rendering is the one to keep --- but `greenery`'s parser rejects
    `\\-` outright, which previously surfaced as two dozen regexes reported
    ``undecided'' when the only problem was a backslash. This renderer escapes exactly
    the metacharacters and nothing else. The two renderings denote the same language and
    differ only in which redundant escapes they carry.
    """
    return "".join(".*" if c == "*" else ("\\" + c if c in _META else c) for c in g)


def _globs_for_comparison(globs) -> str:
    parts = [_glob_for_comparison(g) for g in globs]
    if not parts:
        return "[]"
    return "|".join(f"(?:{p})" for p in parts) if len(parts) > 1 else parts[0]


def equivalent(regex_a: str, regex_b: str) -> bool:
    """Decide L(a) = L(b) by DFA equivalence (Rich Thm 9.5)."""
    from greenery import parse  # optional dependency, extra "regular"
    return parse(regex_a).equivalent(parse(regex_b))


def unanchor(r: str) -> str | None:
    """Strip the redundant anchors a fully anchored matcher may still carry.

    PromQL anchors a label matcher's regex over the whole value, so a leading `^` and a
    trailing `$` say nothing. An anchor anywhere else does say something, and nothing in
    a glob can say it, so this returns None rather than quietly dropping it.
    """
    body = r[1:] if r.startswith("^") else r
    if body.endswith("$") and not body.endswith("\\$"):
        body = body[:-1]
    if re.search(r"(?<!\\)[\^$]", body):
        return None
    return body


def normalise_glob(g: str) -> str:
    """Collapse runs of `*`, which denote the same language as one."""
    return re.sub(r"\*+", "*", g)


@dataclass(frozen=True)
class GlobConversion:
    """The outcome of asking for a glob. `globs` is set only when `exact` holds."""

    regex: str
    globs: tuple[str, ...] = ()
    exact: bool = False
    reason: str = ""
    verified: bool = False

    @property
    def kind(self) -> str:
        """`single`, `union` or `none` --- the three-way classification of §6."""
        if not self.exact:
            return "none"
        return "single" if len(self.globs) == 1 else "union"


# ---------------------------------------------------------------------------
# the structural proposal
# ---------------------------------------------------------------------------

class _NoGlob(Exception):
    """Raised with a reason naming the construct that has no glob."""


def _dot_charclass():
    """The character class the regex `.` denotes, taken from the parser itself.

    Following the usual regex convention, greenery reads `.` as every character *except*
    the newline, so this is not the complement of the empty set. The alphabet of §6 is
    therefore newline-free, which costs nothing --- a tag value cannot contain one --- and
    keeps both sides of every equivalence check over the same alphabet, since
    `glob_to_regex` renders `*` as `.*`.
    """
    from greenery import parse
    (conc,) = parse(".").concs
    (mult,) = conc.mults
    return mult.multiplicand


def _is_full(cc) -> bool:
    """Is this character class the one `.` denotes?"""
    return cc == _dot_charclass()


def _mult_fragments(mult) -> list[str]:
    """The glob fragments one `Mult` (a multiplicand with a multiplier) can denote.

    Returns the alternatives for that position; the caller takes a product across the
    sequence. Every refusal names the construct, because §6 argues the refusals rather
    than only counting them.
    """
    from greenery.charclass import Charclass

    md, lo, hi = mult.multiplicand, mult.multiplier.min.v, mult.multiplier.max.v

    if isinstance(md, Charclass):
        full = _is_full(md)
        if full:
            if (lo, hi) == (0, None):                     # `.*`
                if not STAR_MATCHES_EMPTY:
                    raise _NoGlob("star-requires-a-character")
                return ["*"]
            if (lo, hi) == (1, None):                     # `.+`
                if STAR_MATCHES_EMPTY:
                    raise _NoGlob("nonempty-wildcard")
                return ["*"]
            # `.`, `.?`, `.{2,3}` -- a bounded run of arbitrary characters.
            raise _NoGlob("single-character-wildcard")

        chars = sorted(md.get_chars())
        if md.negated:
            raise _NoGlob("negated-character-class")
        if hi is None:
            # `[0-9]*`, `[a-z]+`: a glob's starred gap admits every string, so it
            # cannot be restricted to a proper subset of the alphabet.
            raise _NoGlob("unbounded-character-class")
        if any(c == "*" for c in chars):
            raise _NoGlob("literal-star")                 # no escape for `*` in a glob
        out = []
        for n in range(lo, hi + 1):
            if len(chars) ** max(n, 1) > MAX_PRODUCT:
                raise _NoGlob("character-class-too-wide")
            out += ["".join(t) for t in itertools.product(chars, repeat=n)]
        return out

    # A group. Translate it, then apply the multiplier.
    inner = _pattern_globs(md)
    if (lo, hi) == (1, 1):
        return inner
    if (lo, hi) == (0, 1):
        return ["", *inner]                               # `(...)?`
    if hi is None:
        # A starred group is expressible only when the group is itself `*`: then
        # `(.*)* = .*`. Any other repeated group admits structure a glob cannot state.
        if set(map(normalise_glob, inner)) == {"*"}:
            return ["*"]
        raise _NoGlob("repeated-group")
    out = []
    for n in range(lo, hi + 1):
        if len(inner) ** max(n, 1) > MAX_PRODUCT:
            raise _NoGlob("repeated-group-too-wide")
        out += ["".join(t) for t in itertools.product(inner, repeat=n)]
    return out


def _conc_globs(conc) -> list[str]:
    """A concatenation: the product of its positions' fragments."""
    per_position = [_mult_fragments(m) for m in conc.mults]
    size = 1
    for frs in per_position:
        size *= max(len(frs), 1)
        if size > MAX_PRODUCT:
            raise _NoGlob("too-many-combinations")
    return ["".join(t) for t in itertools.product(*per_position)] or [""]


def _pattern_globs(pattern) -> list[str]:
    """An alternation: the union over its branches. `concs` is a set, so results sort."""
    out: list[str] = []
    for conc in pattern.concs:
        out += _conc_globs(conc)
    out = sorted({normalise_glob(g) for g in out})
    if len(out) > MAX_UNION:
        raise _NoGlob("union-too-large")
    return out


def regex_to_globs(r: str) -> GlobConversion:
    """Propose a glob or finite union of globs for `r`, and verify it before returning.

    The proposal is syntax-directed; the verification is a decision procedure. So a
    returned conversion is exact, and a refusal is a statement about this procedure
    unless §6's lemmas say the named construct has no glob at all.
    """
    body = unanchor(r)
    if body is None:
        return GlobConversion(r, reason="interior-anchor")
    try:
        from greenery import parse
    except ImportError:  # pragma: no cover - declared in the "regular" extra
        return GlobConversion(r, reason="greenery-missing")
    try:
        pattern = parse(body)
    except Exception:
        return GlobConversion(r, reason="unparsed")

    try:
        globs = _pattern_globs(pattern)
    except _NoGlob as exc:
        return GlobConversion(r, reason=str(exc))

    if not globs:
        return GlobConversion(r, reason="empty-language")
    try:
        ok = equivalent(body, _globs_for_comparison(globs))
    except Exception:
        return GlobConversion(r, reason="equivalence-undecided")
    if not ok:
        # The proposal was wrong. That is a bug in this module rather than a fact about
        # the languages, so it is named as one instead of being reported as a gap.
        return GlobConversion(r, reason="proposal-not-equivalent")
    return GlobConversion(r, tuple(globs), exact=True, reason="", verified=True)


def regex_to_glob(r: str) -> str | None:
    """The single glob equivalent to `r`, or None. A union does not count."""
    got = regex_to_globs(r)
    return got.globs[0] if got.kind == "single" else None


# ---------------------------------------------------------------------------
# the decision procedure of §6
# ---------------------------------------------------------------------------

_REPRESENTATIVES = "abcxyz0189-_./:=+ "


def _one_char(cc) -> str:
    """Some character the class admits, preferring a readable one."""
    chars = frozenset(cc.get_chars())
    if not cc.negated:
        return sorted(chars)[0]
    for c in _REPRESENTATIVES:
        if c not in chars:
            return c
    for o in range(33, 0x11000):           # pragma: no cover - the pool always serves
        if chr(o) not in chars:
            return chr(o)
    raise ValueError("character class admits nothing")


def _class_size(cc, cap: int = 2) -> int:
    """How many characters the class admits, counted no higher than `cap`.

    A negated class stands for the whole alphabet minus a finite set, so it admits more
    than one character whenever the alphabet is larger than that set --- which it is, the
    alphabet being Unicode. Only ``more than one'' matters here, so this returns `cap`.
    """
    if cc.negated:
        return cap
    return min(len(frozenset(cc.get_chars())), cap)


def shortest_word(r: str) -> tuple[str | None, bool]:
    """The shortest word of L(r) and whether it is the only word of that length.

    Breadth-first over the DFA, which is what the question actually asks: the shortest
    accepted string is the shortest accepting path. greenery's own `strings()` enumerates
    candidates in order instead, and on a corpus regex whose shortest word is sixty
    characters long over a twenty-character alphabet it does not finish.

    Returns `(None, False)` for the empty language. Word counts are capped at two,
    because the caller needs uniqueness rather than a census.
    """
    from greenery import parse
    fsm = parse(r).to_fsm()
    # count[state] and witness[state] at the state's own minimal distance.
    count = {fsm.initial: 1}
    witness = {fsm.initial: ""}
    layer = [fsm.initial]
    seen = {fsm.initial}
    while layer:
        finals = [s for s in layer if s in fsm.finals]
        if finals:
            total = sum(count[s] for s in finals)
            return witness[finals[0]], total == 1
        nxt: dict[int, int] = {}
        for s in layer:
            for cc, dst in fsm.map.get(s, {}).items():
                if dst in seen:
                    continue
                nxt[dst] = min(nxt.get(dst, 0) + count[s] * _class_size(cc), 2)
                if dst not in witness:
                    witness[dst] = witness[s] + _one_char(cc)
        for dst, n in nxt.items():
            count[dst] = n
            seen.add(dst)
        layer = list(nxt)
    return None, False


def shortest_words(r: str, *, limit: int = 2) -> list[str]:
    """Kept for callers that only want a witness; see `shortest_word`."""
    w, _ = shortest_word(r)
    return [] if w is None else [w]


class SearchTooLarge(Exception):
    """The bounded search would need more candidates than the cap allows."""


def _insert_stars(w: str, gaps) -> str:
    pieces = []
    for i, ch in enumerate(w):
        if i in gaps:
            pieces.append("*")
        pieces.append(ch)
    if len(w) in gaps:
        pieces.append("*")
    return normalise_glob("".join(pieces))


def single_glob_by_search(r: str, *, max_candidates: int = 4096) -> str | None:
    r"""Decide single-glob expressibility by searching star insertions (§6, Prop. 5).

    If $L(r)=L(g)$ for $g=s_0*s_1*\cdots*s_n$ then, with `*` matching the empty string,
    the shortest word of $L(g)$ is $s_0s_1\cdots s_n$ and it is the *only* word of that
    length. So $L(r)$ has a unique shortest word $w$, the literals of $g$ concatenate to
    $w$, and $g$ is $w$ with stars inserted into some subset of its $|w|+1$ gaps --- at
    most $2^{|w|+1}$ candidates, each settled by equivalence. The question is therefore
    decidable, and this is the procedure.

    One soundness-preserving pruning keeps it usable on the corpus, where a shortest word
    of sixty characters would otherwise mean $2^{61}$ candidates. Adding stars only grows
    the language, so every star of a solution must on its own keep the candidate inside
    $L(r)$; a gap failing that single test cannot appear in any solution and is dropped.
    What remains is enumerated smallest-first. The pruning discards no solution, so the
    procedure stays complete, and `SearchTooLarge` is raised rather than a wrong answer
    returned if even the pruned space exceeds `max_candidates`.

    `regex_to_globs` is the fast structural route used in bulk. Keeping both is what lets
    the tests show they agree, and lets §6 check its refusals against a complete
    procedure rather than trusting the proposer.
    """
    from greenery import parse

    if not STAR_MATCHES_EMPTY:  # pragma: no cover - the proposition assumes this reading
        raise NotImplementedError("the search assumes * matches the empty string")
    body = unanchor(r)
    if body is None:
        return None
    w, unique = shortest_word(body)
    if w is None or not unique:
        return None                    # empty language, or no unique shortest word
    target = parse(body).to_fsm()
    if parse(_glob_for_comparison(w)).to_fsm().equivalent(target):
        return w                       # a literal glob, no stars needed

    usable = [i for i in range(len(w) + 1)
              if parse(_glob_for_comparison(_insert_stars(w, {i}))).to_fsm()
              .issubset(target)]
    if 1 << len(usable) > max_candidates:
        raise SearchTooLarge(f"{len(usable)} usable gaps in a {len(w)}-character word")

    for size in range(1, len(usable) + 1):
        for combo in itertools.combinations(usable, size):
            cand = _insert_stars(w, set(combo))
            if parse(_glob_for_comparison(cand)).to_fsm().equivalent(target):
                return cand
    return None

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
    "equivalent", "unanchor", "drop_redundant_escapes", "regex_to_globs",
    "regex_to_glob",
    "single_glob_by_search", "shortest_word", "shortest_words", "SearchTooLarge",
]

# Whether a Datadog `*` matches the empty string. Undocumented; see the module docstring.
STAR_MATCHES_EMPTY = True

# How many globs a union may contain, and how large an intermediate product may get,
# before the conversion gives up. A union is emitted as `IN (...)` in a scope, so a
# thousand-branch union is not a translation anyone wants even when it is exact.
MAX_UNION = 64
MAX_PRODUCT = 4096

# The alphabet a Datadog tag value is drawn from, which is the universe every
# equivalence in this module is decided over.
#
# §6 asked whether a PromQL matcher and a Datadog scope denote the same set, and left
# "the same set of what?" unstated for eleven passes. The two readings are not close.
# Over all strings, `.+` is no finite union of globs and 218 corpus occurrences are
# inexpressible; over the strings a Datadog tag value may contain, `.+` is the union of
# `c*` for the 41 characters below and those occurrences are a union the bound admits.
# The second is the right universe --- a value that is not a legal tag value cannot
# appear on either side --- and it is the one §6.4's own arithmetic already assumed
# when it spoke of "one branch per alphabet character".
#
# Datadog documents the set: lower-case letters, digits, underscores, minuses, colons,
# periods and slashes, with the whole value lower-cased on ingest. The corpus agrees
# exactly: a census of all 419 distinct tag values in the instantiated corpus, taken
# through this project's own Datadog parser, finds these same 41 characters and no
# others. E12 is the reason to trust the documentation here, having put the adjacent
# claim about glob constructs to the account and had it confirmed.
TAG_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789_:./-"
# greenery's parser wants the hyphen escaped and not final inside a class. Kept as the
# written form of the alphabet; the decider substitutes a Charclass rather than parsing
# this, because parsing it was never the slow part and intersecting with it was.
_SIGMA_CLASS = r"[a-z0-9_:.\-/]"


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


def _sigma_charclass():
    """`TAG_ALPHABET` as a greenery character class."""
    from greenery import Charclass
    return Charclass(tuple((c, c) for c in sorted(TAG_ALPHABET)))


def _restrict_to_sigma(obj):
    """The same pattern with every full-alphabet class replaced by $\\Sigma$.

    Recursive over greenery's term structure: a `Charclass` is swapped when it is the
    one `.` denotes, a `Mult` keeps its multiplier, and a `Conc` and a `Pattern` are
    rebuilt from their restricted parts.
    """
    from greenery import Charclass, Pattern
    from greenery.rxelems import Conc, Mult

    if isinstance(obj, Charclass):
        return _sigma_charclass() if obj == _dot_charclass() else obj
    if isinstance(obj, Mult):
        return Mult(_restrict_to_sigma(obj.multiplicand), obj.multiplier)
    if isinstance(obj, Conc):
        return Conc(*(_restrict_to_sigma(m) for m in obj.mults))
    if isinstance(obj, Pattern):
        return Pattern(*(_restrict_to_sigma(c) for c in obj.concs))
    raise TypeError(f"not a greenery term: {type(obj).__name__}")


def _fsm_over_sigma(regex: str):
    r"""The pattern's DFA with `.` reading as $\Sigma$ rather than as every code point.

    §6 decides its questions over the tag alphabet, and `equivalent` does this by
    substitution. The complete procedure of the decidability proposition did not: it
    built its automata with `parse(...).to_fsm()` directly, so the structural proposer
    was deciding over $41$ characters while the search it is checked against decided
    over every code point. The cross-check reported zero disagreements throughout, but
    they were agreements between answers to two different questions. What hid it is that
    the search bails at ``no unique shortest word'' under either reading, which is the
    outcome for every pattern containing a `.`; the two would have parted company on a
    pattern writing the tag alphabet out as a class, which this corpus happens not to
    contain. Both sides now read `.` the same way.

    It is also the faster route, for the reason given on `equivalent`: the automata stay
    over $41$ characters instead of over Unicode.
    """
    from greenery import parse
    return _restrict_to_sigma(parse(regex)).to_fsm()


def equivalent(regex_a: str, regex_b: str) -> bool:
    r"""Decide L(a) = L(b) by DFA equivalence (Rich Thm 9.5), over `TAG_ALPHABET`.

    The question §6 asks is whether two patterns select the same series, and a string
    that is not a legal tag value selects nothing on either side, so the decision is
    taken over $\Sigma$ rather than over every string. Both sides are rewritten to put
    $\Sigma$ where `.` stood, which makes the glob side's `*` range over $\Sigma$ too
    and so needs no separate rendering.

    Restricting can only identify more pairs, never fewer, so no equivalence this
    module reported before is withdrawn by it.

    Substitution rather than intersection, and the difference is not cosmetic. An
    earlier version of this function restricted by taking $L \cap \Sigma^{*}$, which is
    a DFA product against the class `.` denotes --- every code point --- and the product
    grows with the pattern. On `.*refused_spans.*` it cost $4.1$ seconds; on
    `.*otelcol_processor_.+refused_spans.*` it did not finish in $85$. Two runs of §6's
    apparatus spent fourteen hours on a single pattern because of it and wrote nothing.
    Substituting first costs $2$ milliseconds and leaves the automata over $41$
    characters throughout: the same two decisions take $0.33$ seconds and $24$ seconds.
    """
    from greenery import parse  # optional dependency, extra "regular"
    a = _restrict_to_sigma(parse(regex_a))
    b = _restrict_to_sigma(parse(regex_b))
    return a.equivalent(b)


# Characters that genuinely need a backslash. Anything else escaped is redundant.
_NEEDS_ESCAPE = frozenset(".^$*+?()[]{}|\\")


def drop_redundant_escapes(r: str) -> str:
    """Remove backslashes that escape characters which never needed one.

    `re.escape` escapes freely --- it emits `\\-` for a hyphen --- and a regex that has
    been through it, or through a hand that was being careful, carries escapes the
    decision procedure's parser rejects outright. The escape means nothing to the
    language either way, so dropping it changes no denotation and lets a regex be
    decided rather than reported as unparseable. Alphanumerics keep their backslash,
    since `\\d` and `\\1` are not redundant.
    """
    out, i = [], 0
    while i < len(r):
        if r[i] == "\\" and i + 1 < len(r):
            nxt = r[i + 1]
            out.append(("\\" + nxt) if (nxt.isalnum() or nxt in _NEEDS_ESCAPE) else nxt)
            i += 2
        else:
            out.append(r[i])
            i += 1
    return "".join(out)


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

    greenery 4.2.3 reads `.` as *every* character, newline included: `parse(".")` gives
    `~Charclass(())`, the complement of the empty set, and `parse(".").matches chr(10)`.
    This docstring claimed the opposite -- the usual regex convention, which greenery
    does not follow -- until the eleventh pass. Nothing measured depends on it: a tag
    value contains no newline either way, and both sides of every equivalence check use
    this same class, since `glob_to_regex` renders `*` as `.*`. What did depend on it
    was §6's sentence about the size of the alphabet, which is unbounded rather than
    merely large.
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
                if not STAR_MATCHES_EMPTY:
                    return ["*"]
                # A star admits the empty string (E5), so `.+` is not `*`. Over
                # `TAG_ALPHABET` it is the union of `c*`: one character, then a star
                # for a remainder that is arbitrary anyway. This was refused as
                # `nonempty-wildcard` until the alphabet was pinned.
                return [c + "*" for c in TAG_ALPHABET]
            # `.`, `.?`, `.{2,3}`: a bounded run over the alphabet, which is the same
            # case as an explicit class and falls through to it. Whether the resulting
            # union is worth emitting is `MAX_UNION`'s business, not this function's:
            # `5..` yields 1,681 branches and is refused for its size, not its kind.
            chars = sorted(TAG_ALPHABET)
        else:
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
    body = drop_redundant_escapes(body)
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
        # Anything the decision raises is recorded as undecided rather than as a gap,
        # because a decision this module could not take is not a fact about the two
        # languages. A caller imposing a time limit must raise something deriving from
        # `BaseException`, or this very line will file the timeout as a refusal: a
        # three-second cap in `scripts/probe_proposer_cost.py` did exactly that and
        # produced fourteen refusals that an uncapped run decides as exact unions.
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
    r"""How many characters the class admits, counted no higher than `cap`.

    A negated class stands for the whole alphabet minus a finite set, so it admits more
    than one character whenever the alphabet is larger than that set --- which it is for
    every negated class the corpus contains. Only ``more than one'' matters here, so this
    returns `cap`.

    `_fsm_over_sigma` substitutes $\Sigma$ for the class `.` denotes before this is
    reached, and $\Sigma$ is not negated, so `.` now counts $41$ rather than `cap`.
    Both exceed one, which is all this is asked, so no uniqueness verdict moves.
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
    fsm = _fsm_over_sigma(r)
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
    r"""Decide single-glob expressibility by searching star insertions
    (§6, the decidability proposition).

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
    procedure rather than trusting the proposer --- over the same alphabet as the
    proposer, which `_fsm_over_sigma` records was not true before the twelfth pass.
    """
    if not STAR_MATCHES_EMPTY:  # pragma: no cover - the proposition assumes this reading
        raise NotImplementedError("the search assumes * matches the empty string")
    body = unanchor(r)
    if body is None:
        return None
    w, unique = shortest_word(body)
    if w is None or not unique:
        return None                    # empty language, or no unique shortest word
    target = _fsm_over_sigma(body)
    if _fsm_over_sigma(_glob_for_comparison(w)).equivalent(target):
        return w                       # a literal glob, no stars needed

    usable = [i for i in range(len(w) + 1)
              if _fsm_over_sigma(_glob_for_comparison(_insert_stars(w, {i})))
              .issubset(target)]
    if 1 << len(usable) > max_candidates:
        raise SearchTooLarge(f"{len(usable)} usable gaps in a {len(w)}-character word")

    for size in range(1, len(usable) + 1):
        for combo in itertools.combinations(usable, size):
            cand = _insert_stars(w, set(combo))
            if _fsm_over_sigma(_glob_for_comparison(cand)).equivalent(target):
                return cand
    return None

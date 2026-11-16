"""Datadog AST -> IR (main direction).

Ordered by the corpus feature frequency of `paper/tables/implementation_order.tex`:
the kernel is scope terms, the space aggregators, `by {...}`, `.as_count()`,
`.as_rate()` and `.rollup(...)`, which together complete ~87% of the corpus. Anything
outside it returns a `Bottom` carrying one `Gap`, so the refusals are a catalogue
rather than a crash.

Two things the front end has to decide that the query text does not say:

* **The metric's type.** A dashboard's JSON never states whether a metric was
  submitted as COUNT or GAUGE. `.as_count()` and `.as_rate()` only apply to COUNT, so
  their presence is taken as evidence; otherwise the kind stays "unknown" and the
  back end refuses the counter rules rather than guessing.
* **The rollup interval.** Datadog derives it from the dashboard's time range when
  `.rollup(agg, interval)` does not give one, so the `Rollup` carries `interval=None`
  and the back end is the one that has to be told what to assume.

Every query gets a `Rollup`, including one with no method at all: Datadog always
aggregates in time, and its documented default aggregator is `avg`. Emitting the bare
instant selector instead would quietly claim that Datadog shows raw samples.

The `Rollup` goes *inside* the `Agg`. Datadog evaluates "first by time, then by space":
the rollup reduces each series on its own, and the space aggregator then combines those
reduced values. Nesting it the other way round would describe a different query, and
would raise a commutation question that the real order never asks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lark import Token, Tree

from .. import ir
from ..diagnostics import Bottom, Gap
from ..regular import glob_to_regex

# Datadog names are dotted; Prometheus metric names are [a-zA-Z_:][a-zA-Z0-9_:]* and
# label names forbid the colon as well. The conventional mapping (and the one the
# Datadog agent's own OpenMetrics check uses) is to replace every illegal character by
# an underscore. It is not injective -- `a.b` and `a_b` collide -- so a corpus with
# both would need a disambiguating rename; none of the three sources here has one.
_ILLEGAL_METRIC = re.compile(r"[^a-zA-Z0-9_:]")
_ILLEGAL_LABEL = re.compile(r"[^a-zA-Z0-9_]")

SPACE_AGGS = {"sum", "avg", "min", "max", "count"}
TIME_AGGS = {"sum", "avg", "min", "max", "count"}
DEFAULT_TIME_AGG = "avg"   # Datadog's documented default rollup aggregator


class _Refused(Exception):
    def __init__(self, bottom: Bottom):
        self.bottom = bottom


def _refuse(gap: Gap, detail: str = "") -> None:
    raise _Refused(Bottom(gap, detail))


def metric_name(name: str) -> str:
    out = _ILLEGAL_METRIC.sub("_", name)
    return out if re.match(r"[a-zA-Z_:]", out) else "_" + out


def label_name(name: str) -> str:
    out = _ILLEGAL_LABEL.sub("_", name)
    return out if re.match(r"[a-zA-Z_]", out) else "_" + out


def is_glob(value: str) -> bool:
    r"""Does this tag value carry a wildcard?

    `*` only. Datadog documents prefix, suffix and substring wildcards and names no
    other construct, so there is no single-character wildcard and a `?` in a tag value
    is an ordinary character (paper §6, Definition 4). Treating `?` as a wildcard, as an
    earlier version did, would translate the literal value `web-?` into the matcher
    `web\-.`, which selects `web-x` and not `web-?` --- wrong in both directions at
    once. No query in the corpus carries a `?` in a scope, so nothing measured moves.
    """
    return "*" in value


# ---------------------------------------------------------------------------
# scope -> label matchers
# ---------------------------------------------------------------------------
def _tag_matchers(term: str, *, negated: bool) -> tuple[ir.Matcher, ...]:
    """One scope term as zero or more label matchers.

    `*` is match-all and contributes nothing. `key` and `key:*` both mean the tag is
    present, which in PromQL is `key=~".+"` -- not `.*`, which also matches series
    that have no such label at all.
    """
    if term == "*":
        if negated:
            # `{*,!*}` excludes every series; nothing in PromQL says "match nothing".
            _refuse(Gap.UNSUPPORTED_FUNCTION, "negated match-all")
        return ()
    key, sep, value = term.partition(":")
    if is_glob(key):
        _refuse(Gap.LABEL_NAME_GLOB, term)
    lab = label_name(key)
    if not sep or value == "*":
        return (ir.Matcher(lab, "!~" if negated else "=~", ".+"),)
    if is_glob(value):
        return (ir.Matcher(lab, "!~" if negated else "=~", glob_to_regex(value)),)
    return (ir.Matcher(lab, "!=" if negated else "=", value),)


# A PromQL selector is a conjunction of matchers, so a disjunction of scope terms has
# no selector-level form in general. It has one in the case the corpus almost always
# uses: when every branch constrains the *same* tag key, the union of values is a single
# regex matcher, `k=~"v1|v2"`. That rests on Prometheus anchoring the whole regex rather
# than each side of the alternation -- under `^v1|v2$` the matcher would also select a
# value with v1 as a prefix. It does; `scripts/verify_matchers.py` reads the answer off
# a real server and `tests/test_harness.py` checks the recorded responses.
#
# Branches across different keys are still refused. They remain expressible in PromQL,
# as a set union with the rollup distributed over the branches, but not as the
# syntax-directed translation of a scope, which is what this module is.
_PRESENT = ".+"   # the tag is present, whatever its value


def _value_regex(value: str) -> str:
    return glob_to_regex(value) if is_glob(value) else re.escape(value)


def _or_branches(node) -> list:
    """Flatten a left-nested `scope_or` chain into its branches."""
    if isinstance(node, Tree) and str(node.data) == "scope_or":
        return _or_branches(node.children[0]) + _or_branches(node.children[1])
    return [node]


def _branch_value(node) -> tuple[str, str] | None:
    """One disjunct as (label, value regex), or None when it is match-all.

    A branch is not always a bare term. Datadog's `or` binds looser than `,`, so
    `a:x or a:y,*,*` parses as `a:x OR (a:y AND * AND *)` and the right branch is a
    conjunction -- one whose match-all terms contribute nothing, leaving a single
    constraint. So the branch is lowered first and accepted if what comes back is one
    positive matcher. A branch yielding two matchers really does constrain two things
    and cannot join a union of values; a negated one cannot either.
    """
    matchers = _scope(node)
    if not matchers:
        return None
    if len(matchers) > 1:
        _refuse(Gap.SCOPE_DISJUNCTION, "disjunction of compound terms")
    m = matchers[0]
    if m.op == "=":
        return m.label, re.escape(m.value)
    if m.op == "=~":
        return m.label, m.value
    _refuse(Gap.SCOPE_DISJUNCTION, "negated branch in a disjunction")
    raise AssertionError("unreachable")


def _one_label_disjunction(pairs: list[tuple[str, str] | None], *, negated: bool,
                           detail: str) -> tuple[ir.Matcher, ...]:
    """`[(k, "a"), (k, "b")]` -> `k=~"a|b"`, or bottom if the keys differ.

    A `None` branch is match-all, and a union containing everything is everything, so
    the whole disjunction contributes no matcher at all.
    """
    if any(p is None for p in pairs):
        if negated:
            # `NOT (a:1 OR *)` excludes every series; PromQL cannot say "match nothing".
            _refuse(Gap.UNSUPPORTED_FUNCTION, "negated match-all")
        return ()
    keys = {k for k, _ in pairs}
    if len(keys) > 1:
        _refuse(Gap.SCOPE_DISJUNCTION, detail)
    key = pairs[0][0]
    regexes = [r for _, r in pairs]
    op = "!~" if negated else "=~"
    if _PRESENT in regexes:
        # One branch asks only that the tag be present, which subsumes every value.
        return (ir.Matcher(key, op, _PRESENT),)
    return (ir.Matcher(key, op, "|".join(dict.fromkeys(regexes))),)


def _scope(node, *, negated: bool = False) -> tuple[ir.Matcher, ...]:
    if isinstance(node, Token):                      # a bare TAG inside IN (...)
        return _tag_matchers(str(node), negated=negated)
    data = str(node.data)
    if data == "tag":
        return _tag_matchers(str(node.children[0]), negated=negated)
    if data == "scope_and":
        if negated:                                  # NOT (a AND b) = NOT a OR NOT b
            _refuse(Gap.SCOPE_DISJUNCTION, "negated conjunction is a disjunction")
        return _scope(node.children[0]) + _scope(node.children[1])
    if data == "scope_not":
        return _scope(node.children[1], negated=not negated)
    if data == "scope_or":
        detail = "disjunction across tag keys"
        pairs = [_branch_value(b) for b in _or_branches(node)]
        return _one_label_disjunction(pairs, negated=negated, detail=detail)
    if data == "membership":
        # membership: TAG [NOT] IN "(" tag ("," tag)* ")" -- one key, several values.
        key = label_name(str(node.children[0]))
        inverted = node.children[1] is not None          # `k NOT IN (...)`
        values = [str(v.children[0]) for v in node.children[3:]]
        if any(":" in v for v in values):
            # `k IN (a:1)` would be a tag inside a tag; Datadog's IN takes bare values.
            _refuse(Gap.SCOPE_DISJUNCTION, "IN (...) over a compound term")
        pairs = [None if v == "*" else (key, _value_regex(v)) for v in values]
        return _one_label_disjunction(pairs, negated=negated != inverted,
                                      detail="IN (...) across tag keys")
    _refuse(Gap.UNSUPPORTED_FUNCTION, f"scope node {data}")
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# methods -> time aggregation
# ---------------------------------------------------------------------------
@dataclass
class _Methods:
    time_agg: str | None = None
    interval: float | None = None
    as_rate: bool = False


def _arg_text(node) -> str:
    if isinstance(node, Token):
        return str(node)
    if str(node.data) == "string":
        return str(node.children[0])[1:-1]
    if str(node.data) == "signed_number":
        sign, num = node.children
        return f"{'-' if sign is not None and str(sign) == '-' else ''}{num}"
    return str(node.children[0])


def _methods(nodes: list[Tree]) -> _Methods:
    """Fold the method chain. A later `.rollup` interval wins over an earlier one,
    which is what the query editor does: the rollup names the bucket width and
    `.as_count()` / `.as_rate()` only choose the time aggregator."""
    m = _Methods()
    for node in nodes:
        name = str(node.children[0])
        args = [a for a in node.children[1:] if a is not None]
        if name == "as_count":
            m.time_agg = "sum"
        elif name == "as_rate":
            m.time_agg, m.as_rate = "sum", True
        elif name == "rollup":
            if not args:
                _refuse(Gap.UNSUPPORTED_FUNCTION, ".rollup() with no aggregator")
            agg = _arg_text(args[0])
            if agg not in TIME_AGGS:
                _refuse(Gap.UNSUPPORTED_FUNCTION, f".rollup({agg})")
            m.time_agg = agg
            if len(args) > 1:
                try:
                    m.interval = float(_arg_text(args[1]))
                except ValueError:
                    _refuse(Gap.UNSUPPORTED_FUNCTION, f".rollup(, {_arg_text(args[1])})")
        elif name == "fill":
            # fill(null) is the default and changes nothing; every other mode
            # invents samples, which no PromQL expression does.
            if not args or _arg_text(args[0]) != "null":
                _refuse(Gap.FILL_INTERPOLATION,
                        f".fill({_arg_text(args[0]) if args else ''})")
        else:
            _refuse(Gap.UNSUPPORTED_FUNCTION, f".{name}()")
    return m


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------
def _query(node: Tree) -> ir.Expr:
    agg_tok, metric, scope, grouping, *methods = node.children
    space_agg = str(agg_tok.children[0]) if agg_tok is not None else "avg"
    if space_agg not in SPACE_AGGS:
        _refuse(Gap.UNSUPPORTED_FUNCTION, f"{space_agg}:")   # pNN: percentiles
    m = _methods(methods)

    # Only a COUNT metric can be rolled up by summing, and `.as_count()` /
    # `.as_rate()` only apply to COUNT, so either is evidence of the type the
    # dashboard JSON never states.
    kind = "counter" if m.as_rate or m.time_agg == "sum" else "unknown"
    e: ir.Expr = ir.Select(metric_name(str(metric)), _scope(scope.children[0]), kind=kind)
    keys = tuple(label_name(str(k)) for k in grouping.children[1:]) if grouping else ()

    # Time first, per series; then space. `.as_rate()`'s division by the interval sits
    # on the inner node for the same reason -- it is part of the time aggregation. The
    # placement is only observable for `count:`, where dividing after the space
    # aggregation would scale a count of series by the interval.
    e = ir.Rollup(m.time_agg or DEFAULT_TIME_AGG, m.interval, e, per_second=m.as_rate)
    return ir.Agg(space_agg, keys, "by", e)


def _expr(node) -> ir.Expr:
    if isinstance(node, Token):
        _refuse(Gap.UNSUPPORTED_FUNCTION, f"bare token {node.type}")
    data = str(node.data)
    if data == "query":
        return _query(node)
    if data == "number":
        return ir.Number(float(node.children[0]))
    if data == "binop":
        lhs, op, rhs = node.children
        return ir.BinOp(str(op), _expr(lhs), _expr(rhs))
    if data == "neg":
        sign = -1.0 if str(node.children[0]) == "-" else 1.0
        return ir.Scale(sign, _expr(node.children[1]))
    if data == "wrap":
        _refuse(Gap.UNSUPPORTED_FUNCTION, f"{node.children[0]}(...)")
    if data == "query_list":
        _refuse(Gap.UNSUPPORTED_FUNCTION, "several queries in one request")
    _refuse(Gap.UNSUPPORTED_FUNCTION, f"node {data}")
    raise AssertionError("unreachable")


def lower(tree: Tree) -> ir.Expr | Bottom:
    """Datadog parse tree -> IR, or a Bottom naming the one gap that stopped it."""
    try:
        return _expr(tree)
    except _Refused as r:
        return r.bottom


def lower_all(tree: Tree) -> list[ir.Expr | Bottom]:
    """Every query in the request: a widget may hold a comma-separated list."""
    nodes = tree.children if isinstance(tree, Tree) and str(tree.data) == "query_list" else [tree]
    return [lower(n) for n in nodes]

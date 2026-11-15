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
    return "*" in value or "?" in value


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
        _refuse(Gap.SCOPE_DISJUNCTION, "OR between scope terms")
    if data == "membership":
        _refuse(Gap.SCOPE_DISJUNCTION, "IN (...) is a disjunction over tag values")
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
    e = ir.Agg(space_agg, keys, "by", e)

    return ir.Rollup(m.time_agg or DEFAULT_TIME_AGG, m.interval, e, per_second=m.as_rate)


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

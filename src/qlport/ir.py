"""Typed series algebra (proposal §3, Step 2).

Each node has a denotation ⟦·⟧ : Database × Time → Result defined in semantics/.
Front ends lower their ASTs into these nodes; back ends emit from them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

MatchOp = Literal["=", "!=", "=~", "!~", "glob"]


@dataclass(frozen=True)
class Matcher:
    label: str
    op: MatchOp
    value: str


@dataclass(frozen=True)
class Select:
    """Instant selection of series by metric name and label matchers."""
    metric: str
    matchers: tuple[Matcher, ...] = ()
    kind: Literal["counter", "gauge", "unknown"] = "unknown"


@dataclass(frozen=True)
class Window:
    """Range selection: the samples of `expr` in (t - width, t]."""
    expr: Expr
    width: float  # seconds


@dataclass(frozen=True)
class RangeFn:
    """A function of a window, e.g. rate, increase, increase_anchored, avg_over_time."""
    fn: str
    arg: Window


@dataclass(frozen=True)
class Agg:
    """Space aggregation across series, keeping the labels in `keys` (by) or dropping them (without)."""
    op: Literal["sum", "avg", "min", "max", "count"]
    keys: tuple[str, ...]
    mode: Literal["by", "without"]
    expr: Expr


@dataclass(frozen=True)
class BinOp:
    op: str
    lhs: Expr
    rhs: Expr
    matching: tuple[str, ...] = field(default=())  # on(...) labels; empty = all labels


@dataclass(frozen=True)
class Rollup:
    """Time aggregation into epoch-aligned buckets of width `interval` (Datadog's rollup).

    `interval=None` means Datadog's automatic interval, which it derives from the
    dashboard's time range: the query does not state it, so the IR does not invent it
    and the back end has to be told what to assume.

    `per_second` is Datadog's `.as_rate()`: the bucket total divided by the interval.
    It rides on this node rather than a separate Scale because the divisor is the
    interval, which is not known until the back end resolves it.
    """
    op: Literal["sum", "avg", "min", "max", "count"]
    interval: float | None
    expr: Expr
    per_second: bool = False


@dataclass(frozen=True)
class Scale:
    """Multiply by a constant; used for as_rate()/as_count() normalisation."""
    factor: float
    expr: Expr


@dataclass(frozen=True)
class Number:
    value: float


Expr = Union[Select, Window, RangeFn, Agg, BinOp, Rollup, Scale, Number]


# ---------------------------------------------------------------------------
# shape: the IR with its names abstracted away
# ---------------------------------------------------------------------------

def shape(e: Expr) -> tuple:
    """A hashable canonical form of `e` with metric names, tag keys and tag values
    replaced by positional placeholders.

    Two queries that differ only in which metric they name, or which tag value they
    filter on, have the same shape. That is the right granularity for a differential
    test: what such a test exercises is the operator structure --- which rollup, which
    aggregation, which matcher shapes --- and testing `cpu{env:prod}` after
    `mem{env:staging}` re-runs an experiment already done.

    Regex matcher values collapse to a single placeholder rather than being numbered,
    because the regular expression's own content is tested in §6 and not here.

    Placeholders are allocated in traversal order and are per-call, so the form is
    canonical for one expression and comparable across expressions.
    """
    names: dict[str, str] = {}
    keys: dict[str, str] = {}
    values: dict[str, str] = {}

    def label(d: dict[str, str], k: str, prefix: str) -> str:
        return d.setdefault(k, f"{prefix}{len(d)}")

    def go(x: Expr) -> tuple:
        if isinstance(x, Select):
            matchers = tuple(sorted(
                (label(keys, m.label, "k"), m.op,
                 "*" if m.op in ("=~", "!~") else label(values, m.value, "v"))
                for m in x.matchers))
            return ("Select", label(names, x.metric, "m"), matchers, x.kind)
        if isinstance(x, Window):
            return ("Window", x.width, go(x.expr))
        if isinstance(x, RangeFn):
            return ("RangeFn", x.fn, go(x.arg))
        if isinstance(x, Rollup):
            # Whether the interval was *stated* is structural; its value is not.
            return ("Rollup", x.op, x.interval is not None, x.per_second, go(x.expr))
        if isinstance(x, Agg):
            return ("Agg", x.op, tuple(label(keys, k, "k") for k in x.keys),
                    x.mode, go(x.expr))
        if isinstance(x, BinOp):
            return ("BinOp", x.op, go(x.lhs), go(x.rhs),
                    tuple(label(keys, k, "k") for k in x.matching))
        if isinstance(x, Scale):
            return ("Scale", x.factor, go(x.expr))
        if isinstance(x, Number):
            return ("Number",)
        raise TypeError(f"unknown IR node {type(x).__name__}")

    return go(e)


def shape_slots(e: Expr) -> tuple[int, int]:
    """How many distinct metric names and tag keys a shape needs to be instantiated.

    The series budget of a differential test is this, times however many label values
    each grouping key needs to make an aggregation non-trivial.
    """
    names: set[str] = set()
    keys: set[str] = set()

    def go(x: Expr) -> None:
        if isinstance(x, Select):
            names.add(x.metric)
            keys.update(m.label for m in x.matchers)
        if isinstance(x, Agg):
            keys.update(x.keys)
        if isinstance(x, BinOp):
            keys.update(x.matching)
            go(x.lhs); go(x.rhs)
            return
        inner = getattr(x, "expr", None) or getattr(x, "arg", None)
        if inner is not None and not isinstance(inner, (str, float, int, tuple)):
            go(inner)

    go(e)
    return len(names), len(keys)

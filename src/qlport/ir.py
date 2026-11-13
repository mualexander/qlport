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
    """Time aggregation into epoch-aligned buckets of width `interval` (Datadog's rollup)."""
    op: Literal["sum", "avg", "min", "max", "count"]
    interval: float
    expr: Expr


@dataclass(frozen=True)
class Scale:
    """Multiply by a constant; used for as_rate()/as_count() normalisation."""
    factor: float
    expr: Expr


@dataclass(frozen=True)
class Number:
    value: float


Expr = Union[Select, Window, RangeFn, Agg, BinOp, Rollup, Scale, Number]

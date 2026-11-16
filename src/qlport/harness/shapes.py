"""Instantiate a corpus query: synthesise series that its filters actually select.

The corpus-scale differential test runs one representative query per shape
(paper §7). For a representative to mean anything, the two systems have to hold data
that the query selects -- a query whose scope asks for `result:miss` returns nothing
unless a series carries that tag, and a comparison of two empty results proves nothing.

This module reads the requirement off the lowered IR and produces series specifications
that satisfy it: the metric name, the label sets, and how many series the grouping needs
to be non-trivial. It deliberately does not invent values for constructs it cannot
satisfy -- a negated matcher, say, is satisfied by *not* carrying the label, which is a
different kind of instruction -- and reports those rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import ir

__all__ = ["SeriesSpec", "Instantiation", "instantiate", "value_for_glob"]

FILL = "x"          # stands in for a glob's wildcard
GROUP_VALUES = 3    # label values per grouping key, so an aggregation has work to do


def value_for_glob(glob: str) -> str:
    """A concrete tag value that a Datadog glob selects.

    `*` matches any string including the empty one (measured; see
    `docs/trial_account_checks.md` E5), so replacing each wildcard with a filler yields
    a value inside the glob's language. A bare `*` would give the empty string, which is
    not a usable tag value, so it yields the filler itself.
    """
    out = glob.replace("*", FILL)
    return out or FILL


def _value_matching(regex: str) -> str | None:
    """A concrete tag value the regex selects, or None if we cannot name one.

    Most of these regexes were globs before `lower/datadog.py` converted them, so §6's
    decision procedure converts them back: if the language is a union of globs, any
    branch gives a value, and `value_for_glob` fills the wildcards. That reuses the
    machinery of paper 6 rather than guessing at the pattern, and it
    inherits its honesty -- a regex with no glob gets no value invented for it.
    """
    from ..regular import regex_to_globs

    got = regex_to_globs(regex)
    if got.exact and got.globs:
        return value_for_glob(sorted(got.globs)[0])
    return None


@dataclass(frozen=True)
class SeriesSpec:
    """One series to create in both systems."""

    metric: str
    labels: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.metric, tuple(sorted(self.labels.items())))


@dataclass(frozen=True)
class Instantiation:
    """What a representative query needs, or why it cannot be met."""

    series: tuple[SeriesSpec, ...] = ()
    group_keys: tuple[str, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.reason


def _selects(node: ir.Select) -> tuple[dict[str, str], str]:
    """The labels a series must carry to be selected, or a reason it cannot be built."""
    labels: dict[str, str] = {}
    for m in node.matchers:
        if m.op == "=":
            labels[m.label] = m.value
        elif m.op == "=~":
            if m.value == ".+":                 # the tag merely has to be present
                labels[m.label] = FILL
            else:
                value = _value_matching(m.value)
                if value is None:
                    return {}, f"cannot satisfy regex matcher {m.label}=~{m.value!r}"
                labels[m.label] = value
        elif m.op in ("!=", "!~"):
            # Satisfied by absence, which needs no label at all -- but only if nothing
            # else in the query requires that label to exist.
            continue
        else:
            return {}, f"unsupported matcher op {m.op!r}"
    return labels, ""


def instantiate(e: ir.Expr, *, group_values: int = GROUP_VALUES) -> Instantiation:
    """Series specifications that make `e` return something from both systems."""
    selects: list[ir.Select] = []
    groups: list[str] = []

    def walk(x: ir.Expr) -> str:
        if isinstance(x, ir.Select):
            selects.append(x)
            return ""
        if isinstance(x, ir.Agg):
            groups.extend(x.keys)
            return walk(x.expr)
        if isinstance(x, ir.BinOp):
            return walk(x.lhs) or walk(x.rhs)
        if isinstance(x, (ir.Rollup, ir.Scale)):
            return walk(x.expr)
        if isinstance(x, ir.Window):
            return walk(x.expr)
        if isinstance(x, ir.RangeFn):
            return walk(x.arg)
        if isinstance(x, ir.Number):
            return ""
        return f"unsupported node {type(x).__name__}"

    bad = walk(e)
    if bad:
        return Instantiation(reason=bad)
    if not selects:
        return Instantiation(reason="no selector to instantiate")

    group_keys = tuple(dict.fromkeys(groups))
    specs: list[SeriesSpec] = []
    seen: set[tuple] = set()
    for sel in selects:
        base, why = _selects(sel)
        if why:
            return Instantiation(reason=why)
        # Vary the grouping keys so the aggregation combines more than one series. A
        # grouping key the scope already pins to a value keeps that value: the query
        # asked for it, and overriding it would make the query select nothing.
        free = [k for k in group_keys if k not in base]
        n = group_values if free else 1
        for i in range(n):
            labels = dict(base)
            for k in free:
                labels[k] = f"g{i}"
            spec = SeriesSpec(sel.metric, labels)
            if spec.key() not in seen:
                seen.add(spec.key())
                specs.append(spec)
    return Instantiation(tuple(specs), group_keys)

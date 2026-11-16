"""IR -> PromQL text (main direction).

The counter rules come from notebook 02, which measured them against a reference
evaluator of Datadog's rollup on four workloads:

  Rollup(sum, Δ, counter)        ->  x - x offset Δs            exact without a reset
                                 or  increase(x[Δs] anchored)   exact with resets too
  Scale(1/Δ, Rollup(sum, Δ, .))  ->  (x - x offset Δs) / Δ      i.e. Datadog `.as_rate()`
                                 or  rate(x[Δs])                ε-sound only

`.as_count()` is the undivided form and outnumbers `.as_rate()` in the corpus by about
6.6 to 1, so it is the rule that matters most.

Datadog evaluates a query "first by time, then by space", so a rollup reduces each
series alone and the space aggregator wraps it: `max:x{*}.rollup(avg, 60)` is
`max(avg_over_time(x[60s]))`. PromQL's range functions have the same shape, which is
why no subquery is needed and why the two orders never have to commute.

Nothing here is merely ε-sound. The exact cumulative difference is always available in
standard PromQL, so `rate(x[Δs])` -- the translation everyone reaches for first -- is
never the best this emitter can do; notebook 02 measures how far off it is. What the
emitter does instead is state the preconditions its exactness rests on, as caveats:

* `ROLLUP_ALIGNMENT` -- Datadog's buckets are epoch-aligned, while a PromQL window is
  aligned to the evaluation step. The forms above agree with Datadog exactly at
  epoch-aligned evaluation times, which is what notebook 02 measures; off that grid the
  two systems are summing different sample sets.
* `ROLLUP_DEFAULT` -- when the interval was assumed rather than read from
  `.rollup(agg, interval)`, the result is only as good as the assumption.
* `METRIC_TYPE_UNKNOWN` -- an `avg`/`min`/`max`/`count` rollup reduces Datadog's stored
  points, which are samples for a GAUGE and per-interval deltas for a COUNT. The
  emitted `*_over_time` is exact for the first reading and wrong for the second, and
  the dashboard never says which. It is reported, not refused, because a rule *can* be
  chosen here; for a `sum` rollup none can, so that case refuses with the same code.
* `COUNTER_RESET` -- the cumulative difference is exact on every bucket without a
  reset (notebook 02) and undercounts across one. `anchored=True` removes that caveat
  for `ANCHORED_VERSION`: the modifier is reset-exact, verified against a real
  Prometheus by `scripts/verify_anchored.py`, but it only exists from 3.7.0 on. It is
  off by default so the output runs anywhere; turn it on when the target is 3.7+.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import ir
from ..diagnostics import Bottom, Gap

_RANGE_FNS = {"avg": "avg_over_time", "min": "min_over_time",
              "max": "max_over_time", "count": "count_over_time"}


@dataclass(frozen=True)
class Emission:
    """A translation plus the preconditions its correctness rests on."""
    promql: str
    caveats: tuple[Gap, ...] = ()

    def __str__(self) -> str:
        return self.promql


class _Refused(Exception):
    def __init__(self, bottom: Bottom):
        self.bottom = bottom


def _refuse(gap: Gap, detail: str = "") -> None:
    raise _Refused(Bottom(gap, detail))


def duration(seconds: float) -> str:
    return f"{int(seconds)}s" if float(seconds).is_integer() else f"{seconds:g}s"


def _number(v: float) -> str:
    return f"{int(v)}" if float(v).is_integer() else f"{v:g}"


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _selector(e: ir.Select) -> str:
    if not e.matchers:
        return e.metric
    inner = ",".join(f"{m.label}{m.op}{_quote(m.value)}" for m in e.matchers)
    return f"{e.metric}{{{inner}}}"


def _agg(op: str, keys: tuple[str, ...], mode: str, inner: str) -> str:
    if not keys:
        return f"{op}({inner})"
    return f"{op} {mode} ({', '.join(keys)}) ({inner})"


def _atom(text: str) -> str:
    """Parenthesise unless the text is already a single bracketed term."""
    if not re.search(r"[-+*/]| (?:and|or|unless) ", text):
        return text
    depth = 0
    for i, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and i < len(text) - 1:
            return f"({text})"
    return text if text.startswith("(") else f"({text})"


def _kind_of(e: ir.Expr) -> str:
    """The metric type the IR carries for the series this node reduces."""
    if isinstance(e, ir.Select):
        return e.kind
    if isinstance(e, ir.Agg):
        return _kind_of(e.expr)
    return "unknown"


def _is_counter(e: ir.Expr) -> bool:
    return _kind_of(e) == "counter"


class _Emitter:
    def __init__(self, *, anchored: bool, default_rollup: float | None):
        self.anchored = anchored
        self.default_rollup = default_rollup
        self.caveats: list[Gap] = []

    def note(self, gap: Gap) -> None:
        if gap not in self.caveats:
            self.caveats.append(gap)

    # -- counter difference over a bucket -----------------------------------
    def _bucket_total(self, inner: ir.Expr, width: float) -> str:
        """Datadog's sum of deltas over (t-Δ, t] as a cumulative difference."""
        if self.anchored:
            # Proposal 0052. Introduced in Prometheus 3.7.0 behind
            # --enable-feature=promql-extended-range-selectors, and always-on since the
            # modifiers went stable in September 2026, so the flag is now a no-op.
            # Reset-exact, and checked against a real server in notebook 02.
            self.note(Gap.ANCHORED_VERSION)
            return f"increase({self.expr(inner)}[{duration(width)}] anchored)"
        self.note(Gap.COUNTER_RESET)
        base = self.expr(inner)
        return f"{base} - {base} offset {duration(width)}"

    def _interval(self, e: ir.Rollup) -> float:
        if e.interval is not None:
            return e.interval
        if self.default_rollup is None:
            _refuse(Gap.ROLLUP_DEFAULT,
                    "the query gives no .rollup(agg, interval) and no default was set")
        self.note(Gap.ROLLUP_DEFAULT)
        return float(self.default_rollup)

    def _rollup(self, e: ir.Rollup) -> str:
        """Datadog's time aggregation: one series in, one series out.

        It reduces each series on its own and any space aggregation wraps the result,
        which is what PromQL's range functions already do -- so there is no rollup to
        push anywhere and no commutation question to answer.
        """
        width = self._interval(e)
        self.note(Gap.ROLLUP_ALIGNMENT)
        if e.op == "sum":
            if not _is_counter(e.expr):
                # Not the information barrier of paper §3: the data is there, we just
                # cannot tell from a dashboard whether this metric was submitted as
                # COUNT, and the rule to use depends on that.
                _refuse(Gap.METRIC_TYPE_UNKNOWN,
                        "summing over time needs a counter, and the type is unknown")
            body = self._bucket_total(e.expr, width)
        else:
            # `avg`, `min`, `max` and `count` reduce the points Datadog *stores*. For a
            # GAUGE those are the samples Prometheus holds, and the correspondence is
            # exact. For a COUNT they are per-interval deltas, and no `*_over_time` of
            # the cumulative series computes a function of them -- averaging deltas is
            # not averaging running totals. The dashboard never states which, so the
            # emission is right for a gauge and says so rather than implying both.
            if _kind_of(e.expr) != "gauge":
                self.note(Gap.METRIC_TYPE_UNKNOWN)
            body = f"{_RANGE_FNS[e.op]}({self.expr(e.expr)}[{duration(width)}])"
        return f"{_atom(body)} / {_number(width)}" if e.per_second else body

    # -- dispatch ------------------------------------------------------------
    def expr(self, e: ir.Expr) -> str:
        if isinstance(e, ir.Number):
            return _number(e.value)
        if isinstance(e, ir.Select):
            return _selector(e)
        if isinstance(e, ir.Agg):
            return _agg(e.op, e.keys, e.mode, self.expr(e.expr))
        if isinstance(e, ir.Rollup):
            return self._rollup(e)
        if isinstance(e, ir.Scale):
            if e.factor == 1.0:
                return self.expr(e.expr)
            if e.factor == -1.0:                      # a leading unary sign
                return f"-{_atom(self.expr(e.expr))}"
            return f"{_atom(self.expr(e.expr))} * {_number(e.factor)}"
        if isinstance(e, ir.BinOp):
            return f"{_atom(self.expr(e.lhs))} {e.op} {_atom(self.expr(e.rhs))}"
        if isinstance(e, ir.Window):
            _refuse(Gap.UNSUPPORTED_FUNCTION, "bare window")
        if isinstance(e, ir.RangeFn):
            _refuse(Gap.UNSUPPORTED_FUNCTION, f"{e.fn}()")
        _refuse(Gap.UNSUPPORTED_FUNCTION, type(e).__name__)
        raise AssertionError("unreachable")


def emit_detail(e: ir.Expr, *, anchored: bool = False,
                default_rollup: float | None = None) -> Emission | Bottom:
    """Emit PromQL, with the preconditions the result depends on.

    `anchored` selects the experimental reset-aware form; it is off by default because
    the plain cumulative difference uses only standard PromQL and is the form notebook
    02 verified. `default_rollup` is the bucket width to assume when the query does not
    state one; without it, such a query is refused rather than guessed at.
    """
    em = _Emitter(anchored=anchored, default_rollup=default_rollup)
    try:
        text = em.expr(e)
    except _Refused as r:
        return r.bottom
    return Emission(text, tuple(em.caveats))


def emit(e: ir.Expr, *, anchored: bool = False,
         default_rollup: float | None = None) -> str | Bottom:
    got = emit_detail(e, anchored=anchored, default_rollup=default_rollup)
    return got if isinstance(got, Bottom) else got.promql

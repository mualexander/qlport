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
  Prometheus by `scripts/verify_anchored.py`, but it needs 3.7.0 or newer *started with*
  --enable-feature=promql-extended-range-selectors, which is still required at 3.15.0.
  It is off by default so the output runs anywhere; turn it on only when you control how
  the target server is launched.
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


def _blank_strings(text: str) -> str:
    """The text with the contents of quoted literals replaced by dots.

    Used only to decide parenthesisation. A label value is opaque: `result="hit-wait"`
    contains a hyphen that is not an operator, and reading it as one wraps the whole
    expression in parentheses that mean nothing. Harmless to a server and not harmless
    to a reader comparing two translations.
    """
    return re.sub(r'"(?:[^"\\]|\\.)*"', lambda m: '"' + "." * (len(m.group(0)) - 2) + '"',
                  text)


def _atom(text: str) -> str:
    """Parenthesise unless the text is already a single bracketed term."""
    if not re.search(r"[-+*/]| (?:and|or|unless) ", _blank_strings(text)):
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


_SCALAR = object()


def _match_keys(e: ir.Expr):
    """The label set PromQL gives this subexpression, for vector matching.

    An empty frozenset means one unlabelled series --- what a space aggregation with no
    `by` returns. `_SCALAR` is a number, which matches anything. `None` means two
    operands of a nested operation already disagreed, so nothing can be said.

    Datadog and PromQL part company here. Datadog broadcasts: dividing a per-group
    series by an ungrouped one divides every group by that one value. PromQL matches on
    labels, and two vectors with different label sets have no matching pairs at all, so
    the same expression returns the empty vector rather than a ratio --- silently, since
    an empty result is not an error. The corpus-scale differential test of §7 is what
    surfaced it: two shapes where Datadog returned data and PromQL returned nothing.
    """
    if isinstance(e, ir.Number):
        return _SCALAR
    if isinstance(e, ir.Agg):
        return frozenset(e.keys) if e.mode == "by" else None
    if isinstance(e, ir.BinOp):
        lhs, rhs = _match_keys(e.lhs), _match_keys(e.rhs)
        if lhs is _SCALAR:
            return rhs
        if rhs is _SCALAR:
            return lhs
        return lhs if lhs == rhs else None
    inner = getattr(e, "expr", None)
    if inner is None:
        inner = getattr(e, "arg", None)
    if inner is not None and not isinstance(inner, (str, float, int, tuple)):
        return _match_keys(inner)
    return frozenset()


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
            # --enable-feature=promql-extended-range-selectors, and still gated by it on
            # 3.15.0: a server without the flag answers "anchored modifier is
            # experimental and not enabled". So ANCHORED_VERSION is a precondition on how
            # the target server was *started*, not merely on its version -- which is why
            # this form stays off by default. Reset-exact, checked against a real server.
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

    def _rollup(self, e: ir.Rollup, *, scale: bool = True) -> str:
        """Datadog's time aggregation: one series in, one series out.

        It reduces each series on its own and any space aggregation wraps the result,
        which is what PromQL's range functions already do -- so the reduction itself
        needs no pushing anywhere. `.as_rate()`'s division is a separate matter: it
        scales the query's result rather than each series, and that does raise a
        commutation question, answered in the `Agg` branch of `expr`.

        `scale=False` emits the reduction without `.as_rate()`'s division, for the one
        caller that has to place that division outside an enclosing aggregation; see
        the `Agg` branch of `expr`.
        """
        width = self._interval(e)
        self.note(Gap.ROLLUP_ALIGNMENT)
        if e.op == "sum":
            if _is_counter(e.expr):
                # The kind is an inference, never a fact: E6 submitted a GAUGE and
                # found `.as_count()` and `.as_rate()` accepted and silently ignored
                # on it, so a query carrying either says what its author believed the
                # metric to be and no reading of the query settles it. The rule below
                # is right for a COUNT and wrong for a GAUGE, which is exactly what
                # this caveat means: the translation depends on a type the dashboard
                # never states.
                self.note(Gap.METRIC_TYPE_UNKNOWN)
            else:
                # Not the information barrier of paper §3: the data is there, we just
                # cannot tell from a dashboard whether this metric was submitted as
                # COUNT, and the rule to use depends on that.
                _refuse(Gap.METRIC_TYPE_UNKNOWN,
                        "summing over time needs a counter, and the type is unknown")
            body = self._bucket_total(e.expr, width)
        elif _kind_of(e.expr) == "counter" and e.op in ("avg", "min", "max"):
            # The metric is *known* to be a counter, so Datadog is reducing its stored
            # deltas and `avg_over_time` of the cumulative series is not a function of
            # them at all. Emitting it here was wrong rather than merely unqualified,
            # and the corpus-scale differential test of §7 is what caught it.
            #
            # The mean of a bucket's deltas is their total over their number, and both
            # are expressible: the total is the same cumulative difference the `sum`
            # branch uses, and the number of deltas in a window is the number of
            # samples in it, away from the start of the series.
            if e.op != "avg":
                # min and max would need the individual deltas, which PromQL can only
                # reach through a subquery over the sample grid -- and the grid spacing
                # is not in the query.
                _refuse(Gap.UNSUPPORTED_FUNCTION,
                        f".rollup({e.op}) over a counter needs the individual deltas")
            # Counter-ness is inferred, not stated; see the `sum` branch above.
            self.note(Gap.METRIC_TYPE_UNKNOWN)
            total = self._bucket_total(e.expr, width)
            n = f"count_over_time({self.expr(e.expr)}[{duration(width)}])"
            body = f"{_atom(total)} / {n}"
        else:
            # `avg`, `min`, `max` and `count` reduce the points Datadog *stores*. For a
            # GAUGE those are the samples Prometheus holds, and the correspondence is
            # exact. For a COUNT they are per-interval deltas, and no `*_over_time` of
            # the cumulative series computes a function of them. Where the type is
            # unknown the emission is right for a gauge and says so; where it is known
            # to be a counter the branch above applies instead.
            if _kind_of(e.expr) != "gauge":
                self.note(Gap.METRIC_TYPE_UNKNOWN)
            body = f"{_RANGE_FNS[e.op]}({self.expr(e.expr)}[{duration(width)}])"
        # Every form that reads a counter needs a sample older than the bucket, and for
        # two different reasons. The cumulative difference has nothing to subtract
        # without one, so it returns no result where Datadog returns the deltas it does
        # hold. The mean- and count-of-deltas forms divide by, or return,
        # `count_over_time`, which counts *samples* where Datadog counts *deltas* --- and
        # a series beginning inside the bucket has one fewer delta than samples, since
        # phi drops the first. The anchored form is the one exception: it anchors to the
        # first sample it can find and returns Datadog's figure for a new series too,
        # which is a second advantage of it beyond reset handling and is measured rather
        # than assumed.
        if _is_counter(e.expr) and not (e.op == "sum" and self.anchored):
            self.note(Gap.PHI_SERIES_START)
        return f"{_atom(body)} / {_number(width)}" if (e.per_second and scale) else body

    def _binop(self, e: ir.BinOp) -> str:
        """A binary operation, with the vector matching PromQL needs and Datadog does not.

        Where both sides carry the same label set the operation is written plainly and
        PromQL's default matching is already right. Where one side is a single
        unlabelled series --- a space aggregation with no `by` --- Datadog broadcasts it
        across the other side's groups, and `on() group_left()` (or `group_right()`)
        is the PromQL that does the same. Where both sides are grouped on *different*
        non-empty sets there is no faithful reading: the operation would have to pick a
        set to match on, and the query does not say which. No corpus query does this.
        """
        lhs, rhs = self.expr(e.lhs), self.expr(e.rhs)
        lk, rk = _match_keys(e.lhs), _match_keys(e.rhs)
        op = e.op
        if lk is not _SCALAR and rk is not _SCALAR and lk != rk:
            if lk is None or rk is None or (lk and rk):
                _refuse(Gap.UNSUPPORTED_FUNCTION,
                        f"operands of {e.op} are grouped on different label sets")
            # The empty side is the single series; the other is the "many" side.
            op = f"{e.op} on() {'group_left()' if rk == frozenset() else 'group_right()'}"
        return f"{_atom(lhs)} {op} {_atom(rhs)}"

    # -- dispatch ------------------------------------------------------------
    def expr(self, e: ir.Expr) -> str:
        if isinstance(e, ir.Number):
            return _number(e.value)
        if isinstance(e, ir.Select):
            return _selector(e)
        if isinstance(e, ir.Agg):
            inner = e.expr
            if e.op == "count" and isinstance(inner, ir.Rollup) and inner.per_second:
                # `.as_rate()` divides the query's *result*, not each series. E6
                # measures `count:m{*} by {g}.as_rate()` as 1/R where the same query
                # without the modifier is 1, so the division sits outside the space
                # aggregation. Dividing by a positive constant commutes with sum, avg,
                # min and max, so emitting it inside is harmless for those and is kept
                # there for readability; `count` is blind to its operand's values and
                # does not commute with it. Emitting it inside was wrong rather than
                # merely unqualified, and the corpus-scale differential test of §7 is
                # what caught it.
                width = self._interval(inner)
                body = _agg(e.op, e.keys, e.mode, self._rollup(inner, scale=False))
                return f"{_atom(body)} / {_number(width)}"
            return _agg(e.op, e.keys, e.mode, self.expr(inner))
        if isinstance(e, ir.Rollup):
            return self._rollup(e)
        if isinstance(e, ir.Scale):
            if e.factor == 1.0:
                return self.expr(e.expr)
            if e.factor == -1.0:                      # a leading unary sign
                return f"-{_atom(self.expr(e.expr))}"
            return f"{_atom(self.expr(e.expr))} * {_number(e.factor)}"
        if isinstance(e, ir.BinOp):
            return self._binop(e)
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

"""IR -> PromQL text (main direction).

Planned rules (backed by notebook 02):
  Scale(1/Δ, Rollup(sum, Δ, Select counter))  ->  increase(x[Δs] anchored) / Δ   (exact, needs feature flag)
                                              or  (x - x offset Δs) / Δ          (exact when no resets)
                                              or  rate(x[Δs])                    (ε-sound only: RATE_EXTRAPOLATION)
"""
from .. import ir
from ..diagnostics import Bottom


def emit(e: ir.Expr, *, allow_epsilon: bool = False, anchored: bool = True) -> str | Bottom:
    raise NotImplementedError

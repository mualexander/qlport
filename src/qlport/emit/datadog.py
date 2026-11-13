"""IR -> Datadog query text (reverse direction). TODO."""
from .. import ir
from ..diagnostics import Bottom


def emit(e: ir.Expr, *, allow_epsilon: bool = False) -> str | Bottom:
    raise NotImplementedError

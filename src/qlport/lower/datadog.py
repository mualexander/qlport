"""Datadog AST -> IR (main direction). TODO: implement, ordered by corpus feature frequency."""
from lark import Transformer, Tree

from .. import ir
from ..diagnostics import Bottom


class _Lower(Transformer):
    # One method per production in grammars/datadog.lark, e.g.
    #   query -> Rollup/Scale(Agg(Select(...)))
    pass


def lower(tree: Tree) -> ir.Expr | Bottom:
    raise NotImplementedError

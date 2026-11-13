"""PromQL AST -> IR (reverse direction). TODO."""
from lark import Tree

from .. import ir
from ..diagnostics import Bottom


def lower(tree: Tree) -> ir.Expr | Bottom:
    raise NotImplementedError

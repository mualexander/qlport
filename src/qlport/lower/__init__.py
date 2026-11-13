"""Front ends: AST -> IR. One module per source language (lower/datadog.py, lower/promql.py).

Each module exposes `lower(tree) -> ir.Expr | diagnostics.Bottom` and is written as a
syntax-directed translation: one transformer method per grammar production (attribute
grammar, Knuth 1968).
"""

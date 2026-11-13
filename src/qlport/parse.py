"""LALR(1) parsers built from the grammars in grammars/.

Construction itself is the RQ1 check: Lark raises GrammarError on any shift/reduce or
reduce/reduce conflict, so `get_parser(lang)` succeeding means the grammar is LALR(1).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from lark import Lark, Tree

LANGS = ("promql", "datadog")


@lru_cache(maxsize=None)
def get_parser(lang: str) -> Lark:
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}")
    grammar = files("qlport.grammars").joinpath(f"{lang}.lark").read_text()
    return Lark(grammar, parser="lalr", lexer="contextual", strict=False,
                propagate_positions=True, maybe_placeholders=True)


def parse(lang: str, query: str) -> Tree:
    return get_parser(lang).parse(query)

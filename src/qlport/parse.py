"""LALR(1) parsers built from the grammars in grammars/.

Construction itself is the RQ1 check: Lark raises GrammarError on any shift/reduce or
reduce/reduce conflict, so `get_parser(lang)` succeeding means the grammar is LALR(1).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from lark import Lark, Tree

LANGS = ("promql", "datadog")

# v1 is the starter grammar inferred from vendor documentation alone; v2 adds the
# productions the corpus showed were missing. Both are kept so the gap table of
# paper §4 stays reproducible: it is a v1-vs-corpus measurement, and the same
# examples are then asserted to parse under v2.
VERSIONS = ("v1", "v2")
_FILES = {"v1": "{lang}_v1.lark", "v2": "{lang}.lark"}


@lru_cache(maxsize=None)
def get_parser(lang: str, version: str = "v2") -> Lark:
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}")
    if version not in VERSIONS:
        raise ValueError(f"unknown grammar version {version!r}")
    name = _FILES[version].format(lang=lang)
    grammar = files("qlport.grammars").joinpath(name).read_text()
    return Lark(grammar, parser="lalr", lexer="contextual", strict=False,
                propagate_positions=True, maybe_placeholders=True)


def parse(lang: str, query: str, version: str = "v2") -> Tree:
    return get_parser(lang, version).parse(query)

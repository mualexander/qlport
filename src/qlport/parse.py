"""LALR(1) parsers built from the grammars in grammars/.

Lark reports shift/reduce and reduce/reduce conflicts only under `strict=True`;
with the default `strict=False` it resolves them silently by preferring a shift. So
`get_parser()` succeeding is NOT by itself evidence that a grammar is LALR(1), and
`lalr_conflicts()` below is what actually asks the question. The parsers used for
everything else keep `strict=False` deliberately, because PromQL has one genuine
conflict that has to be resolved the way the reference implementation resolves it
(see `tests/test_parse.py`).
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
def get_parser(lang: str, version: str = "v2", strict: bool = False) -> Lark:
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}")
    if version not in VERSIONS:
        raise ValueError(f"unknown grammar version {version!r}")
    name = _FILES[version].format(lang=lang)
    grammar = files("qlport.grammars").joinpath(name).read_text()
    return Lark(grammar, parser="lalr", lexer="contextual", strict=strict,
                propagate_positions=True, maybe_placeholders=True)


def lalr_conflicts(lang: str, version: str = "v2") -> str | None:
    """`None` if the LALR(1) table is conflict-free, else the conflict Lark reports.

    This is the RQ1 question asked properly. Strict mode also needs `interegular` to
    check for terminal collisions, so a missing dependency surfaces as a LexError
    rather than a silent pass -- it is returned like any other answer rather than
    swallowed, because "we could not check" must not read as "there is no conflict".
    """
    from lark.exceptions import GrammarError, LexError

    try:
        get_parser(lang, version, strict=True)
        return None
    except (GrammarError, LexError) as e:
        return str(e).splitlines()[0]


def parse(lang: str, query: str, version: str = "v2") -> Tree:
    return get_parser(lang, version).parse(query)

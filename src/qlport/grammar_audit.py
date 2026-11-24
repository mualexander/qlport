"""The grammar-inference decisions of section 5, and what each costs the corpus.

Notebook 01 writes `paper/tables/grammar_decisions.tex` from this; the suite re-derives
the table's "queries lost" column from it. The list lived in the notebook alone, which
meant the only quantitative justification for the four decisions -- 474 distinct corpus
queries for one of them -- sat in a generated table no test read, editable to anything.
Putting it here is what lets both sides share one definition instead of the suite
keeping a second copy of the mutations it is checking.

Each entry perturbs ONE terminal back to the form it would naively take. The witness is
a minimal query written for the table, not a corpus query; the corpus evidence is the
count, which is the number of distinct corpus queries the real grammar parses and the
naive grammar does not.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# (language, decision, why, mutation, witness)
DECISIONS: list[tuple[str, str, str, object, str]] = [
    ("promql", r"keyword terminals carry \textbackslash{}b",
     "otherwise a keyword is a prefix of an identifier",
     lambda t: t.replace(
         r"/(?i:sum|avg|min|max|count_values|count|stddev|stdvar|topk|bottomk|quantile|group)\b/",
         r"/(?i:sum|avg|min|max|count_values|count|stddev|stdvar|topk|bottomk|quantile|group)/"),
     "max_over_time(x[5m])"),
    ("promql", "FUNC needs a following paren",
     "otherwise every lowercase identifier is a function name",
     lambda t: t.replace(r"FUNC.2: /[a-z_][a-z0-9_]*(?=\s*\()/",
                         r"FUNC.2: /[a-z_][a-z0-9_]*/"),
     'up{job="x"}'),
    ("datadog", "aggregator looks ahead for the colon",
     "consuming it would shadow a metric whose name starts with an aggregator",
     lambda t: t.replace(r"/(sum|avg|min|max|count|p\d{1,3})(?=\s*:)/",
                         r"/(sum|avg|min|max|count|p\d{1,3})/"),
     "sum.of.things{*}"),
    ("datadog", "no separate STAR terminal",
     "Lark breaks lexer ties on priority before length, so a high-priority STAR splits a"
     " wildcard tag key",
     lambda t: t.replace("tag: TAG", "tag: TAG | STAR").replace(
         "%ignore", 'STAR.5: "*"\n%ignore'),
     "avg:x{*-zone}"),
]


def corpus_texts(lang: str) -> list[str]:
    """Distinct normalised corpus queries of `lang` that the real grammar parses."""
    from . import corpus as c

    texts = {c.normalise(lang, q.text)
             for q in c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl")
             if q.lang == lang and (lang != "promql" or c.is_promql(q.text))}
    return sorted(t for t in texts if c.parse_outcome(lang, t).ok)


def grammar_text(lang: str, version: str = "v2") -> str:
    """A shipped grammar's source. Notebook 01 defined this and nothing else could."""
    from importlib.resources import files

    name = f"{lang}.lark" if version == "v2" else f"{lang}_{version}.lark"
    return files("qlport.grammars").joinpath(name).read_text()


def queries_lost(lang: str, mutate, texts: list[str] | None = None) -> int:
    """How many of those the naive grammar fails to parse."""
    from lark import Lark

    from .parse import get_parser

    naive_text = mutate(grammar_text(lang, "v2"))
    assert naive_text != grammar_text(lang, "v2"), f"{lang}: the mutation changed nothing"
    naive = Lark(naive_text, parser="lalr", lexer="contextual",
                 propagate_positions=True, maybe_placeholders=True)
    get_parser(lang)  # the real grammar has to load for the comparison to mean anything
    lost = 0
    for text in (corpus_texts(lang) if texts is None else texts):
        try:
            naive.parse(text)
        except Exception:
            lost += 1
    return lost

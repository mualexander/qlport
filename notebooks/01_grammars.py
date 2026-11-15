# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Grammars and LALR(1) analysis (paper §4, RQ1)
#
# Two things §4 claims about the grammars, both measured here rather than asserted.
#
# **1. Are they LALR(1)?** Lark reports shift/reduce and reduce/reduce conflicts, and
# terminal collisions, only under `strict=True`; with the default it resolves them
# silently by preferring a shift. So building a parser proves nothing about ambiguity,
# and `parse.lalr_conflicts` asks the question properly.
#
# **2. Which design decisions are load-bearing?** Each terminal in the grammars that
# carries a word boundary, a lookahead or a priority is there because something in the
# corpus failed without it. Rather than describe those in prose, each one is perturbed
# back to the naive form and the witness query is re-parsed: the table records what
# breaks. This is the same move as the repair table of notebook 00, run in reverse.
#
# Outputs: `paper/tables/grammar_stats.tex`, `paper/tables/grammar_decisions.tex`.
# The corpus parse rates live in notebook 00 (`parse_coverage.tex`), not here.

# %%
from importlib.resources import files
from pathlib import Path

import pandas as pd
from lark import Lark

from qlport.parse import LANGS, VERSIONS, get_parser, lalr_conflicts

ROOT = Path.cwd() if (Path.cwd() / "paper").exists() else Path.cwd().parent
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}"}


def tex(s: object) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def code(s: object, width: int | None = None) -> str:
    s = " ".join(str(s).split())
    if width and len(s) > width:
        s = s[: width - 1] + "…"
    return r"\texttt{" + tex(s).replace("…", r"\dots{}") + "}"


def tabular(spec: str, header: list[str], rows: list[list[str]], path: Path) -> None:
    lines = [rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def grammar_text(lang: str, version: str) -> str:
    name = f"{lang}.lark" if version == "v2" else f"{lang}_{version}.lark"
    return files("qlport.grammars").joinpath(name).read_text()


# %% [markdown]
# ## Table size and the LALR(1) question
# Terminals, rules and states are read off the constructed table. The last column is
# the answer to RQ1 for that grammar.

# %%
rows, stats = [], []
for lang in LANGS:
    for version in VERSIONS:
        p = get_parser(lang, version)
        states = len(p.parser.parser._parse_table.states)
        conflict = lalr_conflicts(lang, version)
        stats.append(dict(lang=lang, version=version, terminals=len(p.terminals),
                          rules=len(p.rules), states=states, conflict=conflict))
        rows.append([tex(lang), version, f"{len(p.terminals)}", f"{len(p.rules)}",
                     f"{states}",
                     "none" if conflict is None else code("Shift/Reduce on LPAR")])
stats = pd.DataFrame(stats)
tabular("llrrrl",
        ["Language", "Grammar", "Terminals", "Rules", "LALR states", "Conflicts"],
        rows, TABLES / "grammar_stats.tex")
print(stats.to_string(index=False))

# %% [markdown]
# ### The one PromQL conflict
# `group_left` takes an *optional* parenthesised label list, so after `group_left` a `(`
# may begin either that list or a parenthesised right operand, and one token of lookahead
# cannot tell which. Removing the optionality removes the conflict, which identifies it:

# %%
base = grammar_text("promql", "v2")
variant = base.replace('group: (GROUP_LEFT | GROUP_RIGHT) ["(" [label_list] ")"]',
                       'group: (GROUP_LEFT | GROUP_RIGHT) "(" [label_list] ")"')
assert variant != base
try:
    Lark(variant, parser="lalr", lexer="contextual", strict=True,
         propagate_positions=True, maybe_placeholders=True)
    print("label list made mandatory -> conflict-free, so the optionality is the cause")
except Exception as e:
    print("still conflicting:", e)

# %% [markdown]
# Lark resolves it by shifting — taking the label list — and so does the reference
# implementation. Prometheus 3.15.0 answers
# `1:25: parse error: unexpected end of input` for the ambiguous form, and our parser
# reaches the same verdict by the same route. `tests/test_parse.py` pins all four cases.

# %%
for q in ("a * on(x) group_left(y) b", "a * on(x) group_left b",
          "a * on(x) group_left() b", "a * on(x) group_left (y) (b)",
          "a * on(x) group_left (b)"):
    try:
        get_parser("promql").parse(q)
        print(f"  accepted  {q}")
    except Exception as e:
        print(f"  rejected  {q}   ({type(e).__name__})")

# %% [markdown]
# ## Which lexical decisions are load-bearing
# Each row perturbs one terminal back to the form it would naively take and re-parses a
# query from the corpus. A row that still parsed would mean the decision is dead weight.

# %%
DECISIONS = [
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

rows = []
for lang, decision, why, mutate, witness in DECISIONS:
    naive = mutate(grammar_text(lang, "v2"))
    assert naive != grammar_text(lang, "v2"), decision
    try:
        Lark(naive, parser="lalr", lexer="contextual",
             propagate_positions=True, maybe_placeholders=True).parse(witness)
        outcome = "parses (decision is dead weight!)"
    except Exception as e:
        outcome = f"{type(e).__name__}"
    get_parser(lang).parse(witness)          # the real grammar must accept it
    rows.append([tex(lang), decision, code(witness, 26), tex(outcome)])
    print(f"{lang:8s} {decision:42s} {outcome}  ({why})")
tabular("llll", ["Language", "Decision", "Witness", "Naive form"],
        rows, TABLES / "grammar_decisions.tex")

# %% [markdown]
# ## Findings (to carry into §4)
# * The Datadog grammar is conflict-free LALR(1) with no terminal collisions, in both
#   the documentation-derived v1 and the corpus-informed v2. Since no reference grammar
#   is published, that is a result rather than a reproduction.
# * PromQL is **not** conflict-free, and the one conflict is the language's rather than
#   ours: an optional parenthesised label list after `group_left` cannot be told from a
#   parenthesised operand with one token of lookahead. Both our parser and Prometheus
#   3.15.0 resolve it by taking the label list, and both therefore reject
#   `a * on(x) group_left (b)`. RQ1's answer is "LALR(1) modulo one documented
#   resolution", not "conflict-free".
# * Every lexical decision in the two grammars is load-bearing: perturbing any one of
#   them back to its naive form breaks a query the corpus actually contains.

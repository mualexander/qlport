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
# # 03 — Translation and the gap catalogue (paper §5 and §7, RQ2 and RQ3)
#
# Runs Datadog → IR → PromQL over every Datadog query in the corpus and sorts the
# outcomes into three classes:
#
# * **translated** — exact, *relative to* the preconditions it reports (the caveats
#   table below); nothing in this direction is merely ε-sound, because the exact
#   cumulative difference is always available and `rate()` is strictly worse — notebook
#   02 measures by how much;
# * **⊥** — refused, carrying one `Gap` code from `diagnostics.py`;
# * **unparsed** — outside the §4 grammar, kept in the denominator so κ is honest.
#
# Coverage $\kappa$ (RQ3) is the share of query *occurrences* translated, so a widget
# that appears in forty dashboards counts forty times: that is what a migration faces.
#
# **Two assumptions are stated here rather than hidden.** Datadog derives the rollup
# interval from a documented time-range x widget-type table, and the query text records
# neither, so queries without an explicit `.rollup(agg, interval)` are translated
# against `DEFAULT_ROLLUP` and carry the `ROLLUP_DEFAULT` caveat. And Datadog's buckets
# are epoch-aligned (documented) while a PromQL window follows the evaluation step, so
# every rollup carries `ROLLUP_ALIGNMENT`: the forms of notebook 02 are exact on the
# aligned grid and not off it.
#
# Outputs: `paper/tables/translation_coverage.tex`, `paper/tables/gap_catalogue.tex`,
# `paper/tables/translation_examples.tex`, `paper/tables/gap_by_feature.tex`,
# `paper/figures/coverage_by_gap.pdf`, `data/processed/translations.csv`.

# %%
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qlport import corpus as c
from qlport.diagnostics import Bottom
from qlport.emit.promql import emit_detail
from qlport.lower.datadog import lower_all
from qlport.parse import get_parser, parse

ROOT = Path.cwd() if (Path.cwd() / "paper").exists() else Path.cwd().parent
CORPUS = ROOT / "data" / "corpus"
PROCESSED = ROOT / "data" / "processed"
TABLES, FIGURES = ROOT / "paper" / "tables", ROOT / "paper" / "figures"
for p in (TABLES, FIGURES, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_ROLLUP = 60.0   # seconds; Datadog picks it from the dashboard's time range
EXAMPLE_CHARS = 46
# The worked-examples table carries two code columns, so each gets less room.
PAIR_CHARS = 38

# %% [markdown]
# ## LaTeX helpers
# Bare `tabular` environments; the section files wrap them in a float.

# %%
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


def pct(x: float) -> str:
    return f"{x:.1%}".replace("%", r"\%")


def tabular(spec: str, header: list[str], rows: list[list[str]], path: Path) -> None:
    lines = [rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


# %% [markdown]
# ## Translate the corpus
# One row per distinct instantiated query, with the number of occurrences it stands for.
# A query the grammar cannot parse is kept and marked, so κ is reported against the
# whole corpus rather than against the part that happened to parse.

# %%
queries = [q for q in c.read_queries(CORPUS / "queries.jsonl") if q.lang == "datadog"]
occurrences = pd.Series([c.normalise("datadog", q.text) for q in queries]).value_counts()
print(f"{len(queries):,} Datadog queries, {len(occurrences):,} distinct after instantiation")


def classify(text: str) -> dict:
    """One corpus row -> one outcome.

    A widget request may hold several comma-separated queries and plot a series for
    each, so the PromQL side is a list of expressions; the row counts as translated
    only if every one of them is.
    """
    row = dict(norm=text, promql="", parts=(), outcome="", gap="", detail="", caveats="")
    if not c.parse_outcome("datadog", text).ok:
        return {**row, "outcome": "unparsed"}
    parts, caveats = [], []
    for e in lower_all(parse("datadog", text)):
        got = e if isinstance(e, Bottom) else emit_detail(
            e, default_rollup=DEFAULT_ROLLUP)
        if isinstance(got, Bottom):
            return {**row, "outcome": "bottom", "gap": got.gap.name, "detail": got.detail}
        parts.append(got.promql)
        caveats += [g.name for g in got.caveats]
    return {**row, "promql": " ; ".join(parts), "parts": tuple(parts),
            "outcome": "translated", "caveats": "|".join(sorted(set(caveats)))}


tr = pd.DataFrame([classify(t) for t in occurrences.index])
tr["n"] = tr.norm.map(occurrences).astype(int)
tr.drop(columns="parts").to_csv(PROCESSED / "translations.csv", index=False)
print(f"wrote {(PROCESSED / 'translations.csv').relative_to(ROOT)}")
tr.groupby("outcome")["n"].agg(["size", "sum"])

# %% [markdown]
# ### Everything we emit is PromQL
# The §4 grammar is the check: if a translation does not parse, the emitter invented
# syntax. This is an assertion rather than a table because the only acceptable count
# is zero.

# %%
promql_parser = get_parser("promql")


def parses_as_promql(text: str) -> bool:
    try:
        promql_parser.parse(text)
        return True
    except Exception:
        return False


bad = [q for r in tr.itertuples() for q in r.parts if not parses_as_promql(q)]
assert not bad, bad[:3]
print(f"all {sum(len(r) for r in tr.parts):,} emitted expressions parse as PromQL")

# %% [markdown]
# ## Coverage κ (RQ3)
# By occurrence and by distinct query. The gap between the two columns is itself a
# result: the queries a translator fails on are not the ones dashboards repeat.

# %%
total_n, total_d = tr.n.sum(), len(tr)
order = ["translated", "bottom", "unparsed"]
label = {"translated": r"translated ($\kappa$)", "bottom": r"$\bot$ (gap catalogue)",
         "unparsed": "unparsed by the §4 grammar"}
rows = [[label[o], f"{int(tr[tr.outcome == o].n.sum()):,}",
         pct(tr[tr.outcome == o].n.sum() / total_n), f"{len(tr[tr.outcome == o]):,}",
         pct(len(tr[tr.outcome == o]) / total_d)] for o in order]
tabular("lrrrr", ["Outcome", "Occurrences", "Share", "Distinct", "Share"],
        rows, TABLES / "translation_coverage.tex")
translated = tr[tr.outcome == "translated"]
print(f"kappa = {translated.n.sum() / total_n:.1%} of occurrences, "
      f"{len(translated) / total_d:.1%} of distinct queries")

# %% [markdown]
# ## The gap catalogue (§5)
# One row per `Gap`, with what it cost and the shortest query that hits it. These are
# the semantic gaps the paper claims, each one reached by a real dashboard query.

# %%
gaps = []
for gap, sub in tr[tr.outcome == "bottom"].groupby("gap"):
    example = sub.loc[sub.norm.str.len().idxmin()]
    gaps.append(dict(gap=gap, occurrences=int(sub.n.sum()), distinct=len(sub),
                     share=sub.n.sum() / total_n, detail=example.detail,
                     example=example.norm))
gaps = pd.DataFrame(gaps).sort_values("occurrences", ascending=False)
tabular("lrrll", ["Gap", "Occurrences", "Distinct", "Share", "Shortest example"],
        [[code(r.gap), f"{r.occurrences:,}", f"{r.distinct:,}", pct(r.share),
          code(r.example, EXAMPLE_CHARS)] for _, r in gaps.iterrows()],
        TABLES / "gap_catalogue.tex")
print(gaps[["gap", "occurrences", "distinct"]].to_string(index=False))

# %% [markdown]
# ### Is a refusal a language gap or unfinished work?
# Having gap codes is only worth something if they separate those two, so the split is
# measured rather than asserted in the paper.
#
# `EXPRESSIVENESS` lists the refusals that rest on something PromQL cannot say at all.
# Everything else is a construct this translator has not implemented. The two totals are
# the ceiling on κ and the slack below it, and §7 quotes both.
#
# Disjunctive scopes used to be the largest entry here and are now mostly translated: a
# disjunction whose branches all constrain one tag key is a single regex matcher,
# `k=~"v1|v2"`, which is sound because Prometheus anchors the whole regex
# (`scripts/verify_matchers.py` asked a real server). What is left refused is disjunction
# across *different* keys, which no selector expresses.

# %%
# Refusals that rest on something PromQL cannot express, by (gap, detail).
EXPRESSIVENESS = {
    ("UNSUPPORTED_FUNCTION", ".weighted()"),          # time-weighted space average
    ("FILL_INTERPOLATION", None),                     # interpolation inside the query
    ("LABEL_NAME_GLOB", None),                        # label names are not queryable
    ("SCOPE_DISJUNCTION", None),                      # only cross-key cases survive now
}


def _is_gap(r) -> bool:
    if (r.gap, None) in EXPRESSIVENESS or (r.gap, r.detail) in EXPRESSIVENESS:
        return True
    # A percentile aggregator reads a sketch the metric carries, not a set of samples.
    return r.gap == "UNSUPPORTED_FUNCTION" and bool(re.fullmatch(r"p\d{1,3}:", str(r.detail)))


bot = tr[tr.outcome == "bottom"].assign(gap_kind=lambda d: [
    "language gap" if _is_gap(r) else "not implemented" for r in d.itertuples()])
kinds = bot.groupby("gap_kind")["n"].agg(occurrences="sum", distinct="size")
kinds["share"] = kinds.occurrences / total_n
print(kinds.to_string())
ceiling = (translated.n.sum() + kinds.occurrences["not implemented"]) / total_n
print(f"\nceiling for this direction: {ceiling:.1%}")

print("\nwhat is left of SCOPE_DISJUNCTION:")
print(bot[bot.gap == "SCOPE_DISJUNCTION"].groupby("detail")["n"]
      .agg(occurrences="sum", distinct="size").to_string())

# How far the single-label rule got: the features that carry a disjunctive scope.
for f in ("SCOPE_BOOL", "SCOPE_IN", "SCOPE_PAREN"):
    sub = tr[[f in c.features("datadog", t_) for t_ in tr.norm]]
    share = sub[sub.outcome == "translated"].n.sum() / sub.n.sum()
    print(f"{f:12s} {int(sub.n.sum()):5d} occurrences, {share:.1%} translated")

# §7 quotes these, so re-running the notebook has to reproduce them.
assert int(kinds.occurrences["language gap"]) == 175, kinds
assert int(kinds.occurrences["not implemented"]) == 207, kinds


# %% [markdown]
# ### Caveats on the translations we do claim
# An exact translation is exact *relative to* its preconditions. Every rollup depends
# on bucket alignment, and a rollup whose interval the query never stated also depends
# on the interval we assumed — the two trial-account checks of §7.

# %%
cav = (tr[tr.caveats != ""].assign(c=lambda d: d.caveats.str.split("|"))
       .explode("c").groupby("c")["n"].agg(occurrences="sum", distinct="size")
       .sort_values("occurrences", ascending=False).reset_index())
tabular("lrrr", ["Caveat", "Occurrences", "Distinct", "Share of translated"],
        [[code(r.c), f"{int(r.occurrences):,}", f"{int(r.distinct):,}",
          pct(r.occurrences / translated.n.sum())] for _, r in cav.iterrows()],
        TABLES / "translation_caveats.tex")
print(cav.to_string(index=False))

# %% [markdown]
# ## Which features cost the coverage
# Each refused query is attributed to the Datadog features it uses, so §5 can say which
# constructs the gaps sit behind rather than only naming the gaps.

# A feature at 0% is not always a feature we cannot translate: it can be one that only
# ever turns up next to something we cannot. `SCOPE_BARE` is the clear case —
# `avg:x{host}` translates to `x{host=~".+"}` on its own, but in this corpus a bare tag
# key appears only inside a boolean scope or beside a `*-zone` key glob. Read the rows
# as "queries using this feature", not "this feature".

# %%
feat = tr.assign(f=[sorted(c.features("datadog", t)) for t in tr.norm]).explode("f")
by_feature = (feat.pivot_table(index="f", columns="outcome", values="n",
                               aggfunc="sum", fill_value=0)
              .reindex(columns=order, fill_value=0))
by_feature["total"] = by_feature.sum(axis=1)
by_feature["share"] = by_feature.translated / by_feature.total
worst = by_feature[by_feature.total >= 50].sort_values("share").head(12)
tabular("llrr", ["Feature", "Construct", "Occurrences", "Translated"],
        [[code(f), tex(c.FEATURE_DOC.get(f, "")), f"{int(r.total):,}", pct(r.share)]
         for f, r in worst.iterrows()],
        TABLES / "gap_by_feature.tex")
print(worst[["total", "share"]].to_string())

# %% [markdown]
# ## Worked examples for §5
# The rules of the paper, taken from the corpus rather than invented.

# %%
wanted = ["M_AS_COUNT", "M_AS_RATE", "M_ROLLUP_INTERVAL", "GROUP_BY", "SCOPE_NEG",
          "SCOPE_GLOB", "SCOPE_IN", "ARITH"]
shown, rows = set(), []
for f in wanted:
    sub = feat[(feat.f == f) & (feat.outcome == "translated")]
    sub = sub[~sub.norm.isin(shown)]
    if sub.empty:
        continue
    r = sub.loc[sub.norm.str.len().idxmin()]
    shown.add(r.norm)
    rows.append([code(f), code(r.norm, PAIR_CHARS), code(r.promql, PAIR_CHARS)])
tabular("lll", ["Feature", "Datadog", "PromQL"], rows,
        TABLES / "translation_examples.tex")
for r in rows:
    print(r[0])

# %%
fig, ax = plt.subplots(figsize=(5.2, 2.9))
shares = tr.groupby("outcome")["n"].sum().reindex(order).fillna(0) / total_n
left = 0.0
for o, colour in zip(order, ["#2b6cb0", "#e2a03f", "#cbd5e0"]):
    if shares[o] <= 0:
        continue
    ax.barh(["corpus"], [shares[o]], left=left, color=colour,
            label=f"{label[o].replace(r'($\kappa$)', '(κ)').replace(r'$\bot$', '⊥')}"
                  f" ({shares[o]:.0%})")
    left += shares[o]
ax.set_xlim(0, 1)
ax.set_xlabel("share of Datadog query occurrences in the corpus")
ax.set_yticks([])
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
fig.tight_layout()
fig.savefig(FIGURES / "coverage_by_gap.pdf")
print("wrote paper/figures/coverage_by_gap.pdf")

# %% [markdown]
# ## Findings (to carry into §5 and §7)
# * κ is reported two ways on purpose. By occurrence it answers "how much of a real
#   dashboard estate ports", which is the migration question; by distinct query it
#   answers "how much of the language ports", which is the language question. Here they
#   come out at essentially the same value, which is itself worth saying: the queries a
#   translator refuses are no more and no less repeated across dashboards than the ones
#   it handles, so there is no cheap win hiding in the popular queries.
# * `.as_count()` and `.as_rate()` reduce to the *same* cumulative difference,
#   `x - x offset Δ`, divided by Δ only in the rate case. The count form is the one the
#   corpus overwhelmingly uses, and notebook 02 shows both are exact without a reset.
# * Every rollup translation rests on two facts about Datadog that no experiment here
#   settles: the reset policy ρ and whether buckets are right-closed. (Epoch alignment
#   and the interval table *are* documented; what the query text omits is the dashboard
#   time range that selects the interval.) They are reported as caveats, not folded into
#   the exact count — `docs/trial_account_checks.md` specifies the checks that close
#   them, and paper §7.9 reports them as open.
# * The largest single refusal is structural, not semantic: Datadog's scope is a
#   boolean query over tags, and a PromQL selector is a conjunction of matchers. `OR`
#   and `IN (...)` need either a regex over one label or a set union of selectors, and
#   neither is a drop-in for a scope sitting under an aggregation.
# * Datadog evaluates a query "first by time, then by space": a rollup reduces each
#   series alone and the space aggregator wraps the result. PromQL's range functions
#   have exactly that shape, so `max:x{*}.rollup(avg, 60)` is `max(avg_over_time(x[60s]))`
#   and the two layers never have to commute. An earlier version of this translator had
#   the order the other way round and refused 348 occurrences on the strength of a
#   commutation argument that the real semantics never raises; the documented order is
#   worth stating in §5 precisely because getting it backwards costs four points of κ
#   while leaving every emitted translation looking plausible.

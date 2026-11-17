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
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qlport import corpus as c
from qlport import ir
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

# §7 and §8 quote these, so re-running the notebook has to reproduce them.
assert int(kinds.occurrences["language gap"]) == 175, kinds
assert int(kinds.occurrences["not implemented"]) == 208, kinds

# §8 names the five constructs the language-gap share is made of, with counts, so the
# breakdown is asserted and not just its total.
gap_rows = bot[bot.gap_kind == "language gap"]
weighted = int(gap_rows[gap_rows.detail.astype(str).str.contains("weighted")].n.sum())
pctile = int(gap_rows[[bool(re.fullmatch(r"p\d{1,3}:", str(x))) for x in gap_rows.detail]].n.sum())
by_gap = gap_rows.groupby("gap")["n"].sum()
breakdown = {".weighted()": weighted, "percentiles": pctile,
             "FILL_INTERPOLATION": int(by_gap.get("FILL_INTERPOLATION", 0)),
             "LABEL_NAME_GLOB": int(by_gap.get("LABEL_NAME_GLOB", 0)),
             "SCOPE_DISJUNCTION": int(by_gap.get("SCOPE_DISJUNCTION", 0))}
print("language-gap breakdown:", breakdown)
assert breakdown == {".weighted()": 70, "percentiles": 34, "FILL_INTERPOLATION": 36,
                     "LABEL_NAME_GLOB": 31, "SCOPE_DISJUNCTION": 4}, breakdown
assert sum(breakdown.values()) == 175, breakdown


# %% [markdown]
# ### How far the soundness theorems reach
# §5 states two: one over the kernel $\mathcal{K}$ --- a space aggregation of a rollup
# of a selector --- and one over its arithmetic closure $\mathcal{A}$. Both are claims
# about IR *shape*, so their reach is computed from the lowered IR and not from the
# feature-completion figure of the staircase, which answers a different question.

# %%
def _is_kernel(e):
    return (isinstance(e, ir.Agg) and e.mode == "by"
            and isinstance(e.expr, ir.Rollup)
            and isinstance(e.expr.expr, ir.Select))


def _in_closure(e):
    if _is_kernel(e) or isinstance(e, ir.Number):
        return True
    if isinstance(e, ir.Scale):
        return _in_closure(e.expr)
    if isinstance(e, ir.BinOp):
        return _in_closure(e.lhs) and _in_closure(e.rhs)
    return False


kernel_occ = closure_occ = 0
for _text, _n in occurrences.items():
    try:
        _tree = parse("datadog", _text, "v2")
    except Exception:                                            # noqa: BLE001
        continue
    _lo = lower_all(_tree)
    if not _lo or any(isinstance(x, Bottom) for x in _lo):
        continue
    if any(isinstance(emit_detail(x, default_rollup=DEFAULT_ROLLUP), Bottom)
           for x in _lo):
        continue
    if all(_is_kernel(x) for x in _lo):
        kernel_occ += _n
    if all(_in_closure(x) for x in _lo):
        closure_occ += _n

_tot = int(total_n)
_tr = int(translated.n.sum())
print(f"kernel K : {kernel_occ:,} = {kernel_occ / _tot:.1%} of corpus, "
      f"{kernel_occ / _tr:.1%} of translated")
print(f"closure A: {closure_occ:,} = {closure_occ / _tot:.1%} of corpus, "
      f"{closure_occ / _tr:.1%} of translated")

# The closure covers every translation the emitter produces: nothing it emits is outside
# the fragment §5 proves exact. That is what makes thm:arith worth stating separately
# from thm:kernel rather than as a footnote to it.
assert closure_occ == _tr, (closure_occ, _tr)
assert kernel_occ == 8294, kernel_occ
assert closure_occ == 8375, closure_occ

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
#   settles. One is now settled from another direction: the agent is open source, and its
#   source gives ρ = zero on the OpenMetrics path (and clamp, not drop, as the other
#   branch) — so the open one is whether buckets are right-closed. (Epoch alignment and
#   the interval table *are* documented; what the query text omits is the dashboard time
#   range that selects the interval.) Both are reported as caveats, not folded into the
#   exact count — `docs/trial_account_checks.md` specifies the checks that remain, and
#   paper §7.9 separates what was read from what is still unmeasured.
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


# %% [markdown]
# ## How big is the corpus-scale differential test?
#
# §7 quotes these. The test that would turn κ from "willing to translate" into
# "translates correctly" needs each *shape* exercised once, not each occurrence: two
# queries differing only in which metric they name are one experiment. `ir.shape`
# abstracts metric names, tag keys and tag values away and keeps the operator structure,
# including whether the rollup interval was stated — which is what `ROLLUP_DEFAULT` is
# about and so cannot be abstracted.

# %%
import collections                            # noqa: E402
from collections import Counter               # noqa: E402
from qlport.ir import shape, shape_slots       # noqa: E402

# The population is the one `scripts/shape_differential.py` actually instantiates, so
# that what is asserted here is what §7 reports: a request that is a comma-separated
# list of queries is excluded, because the runner needs one expression per
# representative, and so is anything the emitter refuses. An earlier version of this
# cell shaped every sub-query of every lowered request and counted 331 classes, which
# is a defensible number of a different population and was quoted by nothing.
shapes, slots = Counter(), {}
for row in tr[tr.outcome == "translated"].itertuples():
    try:
        lowered = lower_all(parse("datadog", row.norm, "v2"))
    except Exception:
        continue
    if len(lowered) != 1 or isinstance(lowered[0], Bottom):
        continue
    e = lowered[0]
    if isinstance(emit_detail(e, default_rollup=DEFAULT_ROLLUP), Bottom):
        continue
    s = shape(e)
    shapes[s] += int(row.n)
    slots[s] = max(slots.get(s, (0, 0)), shape_slots(e))

total_shaped = sum(shapes.values())
print(f"{len(shapes):,} distinct shapes covering {total_shaped:,} occurrences")

cum, coverage = 0, []
for i, (s, n) in enumerate(shapes.most_common(), 1):
    cum += n
    coverage.append((i, cum / total_shaped))
milestones = {}
for target in (0.5, 0.8, 0.9, 0.95, 0.99):
    milestones[target] = next(i for i, frac in coverage if frac >= target - 1e-9)
    print(f"  {target:5.0%} of occurrences needs {milestones[target]:,} shapes")

LABEL_VALUES = 3     # per grouping key, to make an aggregation non-trivial
for k in (50, len(shapes)):
    series = sum(slots[s][0] for s, _ in shapes.most_common(k)) * LABEL_VALUES
    print(f"  top {k:3} shapes -> ~{series:,} synthetic series")

# §7 and §9 quote these, so a re-run has to reproduce them.
assert len(shapes) == 322, len(shapes)
assert total_shaped == 8270, total_shaped
assert milestones[0.5] == 6 and milestones[0.8] == 23, milestones
assert milestones[0.9] == 49 and milestones[0.95] == 103, milestones
print("shape-reduction assertions hold")


# %% [markdown]
# ### Is one representative per shape enough?
#
# The reduction above is only sound if every query in a shape class translates the same
# way — otherwise testing one member says nothing about the others. That is checkable
# rather than arguable: emit every translated query, mask the identifiers and numbers,
# and count how many distinct emitted forms each class has. One apiece means a
# representative suffices.

# %%
def _masked(text: str) -> str:
    """Emitted PromQL with identifiers, literals and numbers masked.

    Selector labels are sorted first: a conjunction of matchers has no order, so two
    members emitting them differently are not translating differently.
    """
    keep = {"by", "without", "offset", "anchored", "and", "or", "unless",
            "sum", "avg", "min", "max", "count", "rate", "increase",
            "avg_over_time", "min_over_time", "max_over_time", "count_over_time"}
    text = re.sub(r"\{([^{}]*)\}",
                  lambda m: "{" + ",".join(sorted(m.group(1).split(","))) + "}", text)
    t = re.sub(r'"[^"]*"', '"V"', text)
    t = re.sub(r"\b[A-Za-z_:][A-Za-z0-9_:]*\b",
               lambda m: m.group(0) if m.group(0) in keep else "N", t)
    return re.sub(r"\d+", "#", t)


forms = collections.defaultdict(set)
for row in tr[tr.outcome == "translated"].itertuples():
    try:
        lowered = lower_all(parse("datadog", row.norm, "v2"))
    except Exception:
        continue
    for e in lowered:
        if isinstance(e, Bottom):
            continue
        em = emit_detail(e, default_rollup=DEFAULT_ROLLUP)
        if not isinstance(em, Bottom):
            forms[shape(e)].add(_masked(em.promql))

divergent = {s: f for s, f in forms.items() if len(f) > 1}
print(f"{len(forms):,} shape classes; {len(divergent)} emit more than one form")
assert not divergent, list(divergent.values())[:2]
print("one representative per shape covers its class")

# %% [markdown]
# ## Figures the prose quotes
#
# Every number §1, §5, §7, §8 and §9 state in running text rather than read off a table
# is recorded here, so `tests/test_paper_figures.py` can check the prose against the
# corpus instead of against an earlier draft. Three figures went stale this way before
# the check existed — $\kappa$ as $95.6\%$ after it moved to $95.5\%$, and the
# translated and emitted counts by one each — and none of them was visible in any table.

# %%
_q = {
    "corpus_occurrences": int(total_n),
    "corpus_distinct": int(total_d),
    "distinct_after_instantiation": int(total_d),
    "kappa_occurrences": int(translated.n.sum()),
    "kappa_distinct": int(len(translated)),
    "kappa_pct": round(translated.n.sum() / total_n * 100, 1),
    "emitted_expressions": int(sum(len(r) for r in tr.parts)),
    "multi_expression_queries": int(sum(1 for r in tr.parts if len(r) > 1)),
    "language_gap_occurrences": int(kinds.occurrences["language gap"]),
    "not_implemented_occurrences": int(kinds.occurrences["not implemented"]),
    "language_gap_pct": round(kinds.occurrences["language gap"] / total_n * 100, 1),
    "not_implemented_pct": round(kinds.occurrences["not implemented"] / total_n * 100, 1),
    "ceiling_pct": round(ceiling * 100, 1),
    "kernel_occurrences": int(kernel_occ),
    "kernel_pct_corpus": round(kernel_occ / total_n * 100, 1),
    "kernel_pct_translated": round(kernel_occ / translated.n.sum() * 100, 1),
    "closure_occurrences": int(closure_occ),
    "closure_pct_corpus": round(closure_occ / total_n * 100, 1),
    "closure_pct_translated": round(closure_occ / translated.n.sum() * 100, 1),
    "shapes": int(len(shapes)),
    "shaped_occurrences": int(total_shaped),
    "shapes_for_half": int(milestones[0.5]),
    "shapes_for_80pct": int(milestones[0.8]),
    "shapes_for_90pct": int(milestones[0.9]),
    "shapes_for_95pct": int(milestones[0.95]),
}
for _, _row in cav.iterrows():
    _q[f"caveat_{_row.c.lower()}_pct"] = round(_row.occurrences / translated.n.sum() * 100, 1)
    _q[f"caveat_{_row.c.lower()}_occurrences"] = int(_row.occurrences)

# Next to the generated tables rather than in data/processed, which is derived data and
# gitignored: this is a paper input, and the check that reads it has to work on a fresh
# clone before any notebook has run.
_QF = ROOT / "paper" / "quoted_figures.json"
_existing = json.loads(_QF.read_text()) if _QF.exists() else {}
_existing.update({f"nb03.{k}": v for k, v in _q.items()})
_QF.write_text(json.dumps(_existing, indent=1, sort_keys=True) + "\n")
print(f"wrote {_QF.relative_to(ROOT)} ({len(_q)} figures from notebook 03)")
for _k in sorted(_q):
    print(f"  {_k:38s} {_q[_k]}")

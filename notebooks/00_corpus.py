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
# # 00 — Corpus and feature frequencies (paper §4 parse rate, §7 corpus; implementation order)
#
# **Plan**
# 1. Load the queries extracted by `scripts/fetch_corpus.py` from
#    kube-prometheus (compiled kubernetes-mixin and friends), the 200 most-downloaded
#    Grafana community dashboards with a Prometheus datasource, and
#    DataDog/integrations-core `*/assets/dashboards/*.json`.
# 2. Measure what fraction of real queries the grammars of §4 accept, and name the
#    terminal that blocks each failure.
# 3. Count syntactic features per language, then compute a greedy implementation order:
#    the sequence of features that covers the most queries *completely* per step.
#
# Outputs: `paper/tables/corpus_sources.tex`, `paper/tables/parse_coverage.tex`,
# `paper/tables/grammar_gaps.tex`,
# `paper/tables/operator_frequency.tex`, `paper/tables/implementation_order.tex`,
# `paper/figures/feature_frequency.pdf`, `paper/figures/cumulative_coverage.pdf`,
# `data/processed/features.csv`, `data/processed/parse_outcomes.csv`.

# %%
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qlport import corpus as c

ROOT = Path.cwd() if (Path.cwd() / "paper").exists() else Path.cwd().parent
CORPUS = ROOT / "data" / "corpus"
PROCESSED = ROOT / "data" / "processed"
TABLES, FIGURES = ROOT / "paper" / "tables", ROOT / "paper" / "figures"
for p in (TABLES, FIGURES, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

TOP_N = 20          # rows per language in the paper's feature-frequency figure
EXAMPLE_CHARS = 26  # excerpt width in the gap table

queries = c.read_queries(CORPUS / "queries.jsonl")
df = pd.DataFrame([q.__dict__ for q in queries])

# Some community dashboards listed under the Prometheus datasource have Loki panels.
# LogQL is a different language and is no evidence about PromQL, so it leaves here.
logql = (df.lang == "promql") & ~df.text.map(c.is_promql)
print(f"{len(df):,} queries; dropping {int(logql.sum())} LogQL panels "
      f"from {df.loc[logql, 'origin'].nunique()} dashboards")
df = df[~logql].reset_index(drop=True)
df.groupby(["lang", "kind"]).size()


# %% [markdown]
# ## LaTeX helpers
# Tables are emitted as bare `tabular` environments; the section files wrap them in a
# float, as `rollup_translation.tex` already does.

# %%
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}"}


def tex(s: object) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def code(s: object, width: int | None = None) -> str:
    s = str(s)
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


# %% [markdown]
# ## The corpus
# Two PromQL populations are kept apart on purpose: `kube-prometheus` is curated by the
# upstream projects, `grafana` is what practitioners actually publish. Datadog has no
# community-dashboard registry, so `integrations-core` — one vendor-authored dashboard
# per integration — is the only comparable public population.

# %%
SOURCE_LABEL = {
    "kube-prometheus": ("PromQL", "compiled mixins (curated)"),
    "grafana": ("PromQL", "community dashboards, top 200 by downloads"),
    "integrations-core": ("Datadog", "vendor integration dashboards"),
}
meta = {}
for name in SOURCE_LABEL:
    f = CORPUS / "sources" / name / ".fetch.json"
    meta[name] = json.loads(f.read_text()) if f.exists() else {}

rows = []
for name, (lang, what) in SOURCE_LABEL.items():
    sub = df[df.source == name]
    pin = meta.get(name, {}).get("commit", "")
    pin = pin[:10] if pin else meta.get(name, {}).get("fetched", "")
    rows.append([tex(name), lang, what, f"{sub.origin.nunique():,}",
                 f"{(sub.lang != 'datadog_formula').sum():,}", code(pin)])
tabular("llllrl", ["Source", "Language", "Population", "Documents", "Queries", "Pinned at"],
        rows, TABLES / "corpus_sources.tex")
pd.DataFrame(rows, columns=["source", "lang", "population", "files", "queries", "pin"])


# %% [markdown]
# ## Grammar coverage (RQ1)
# Dashboard queries are templates: `$pod`, `$__rate_interval`, `$host` stand for values a
# user picks. `corpus.normalise` instantiates them — a PromQL range variable becomes a
# duration, a Datadog scope variable becomes `*`, the instance a dashboard shows by
# default — and leaves everything else alone. The raw column is therefore a statement
# about dashboards, the instantiated column a statement about the grammars.
#
# Two grammar versions are measured. v1 (`grammars/*_v1.lark`) was inferred from vendor
# documentation alone, before the corpus existed. A v1 failure is explained by the
# smallest set of *repairs* that makes it parse (`corpus.REPAIRS`), each repair being one
# concrete production or terminal; v2 (`grammars/*.lark`) is v1 with every one of those
# productions added, so the gap table is verified by construction in both directions —
# the repair shows the construct was missing, and v2 parses it unrepaired. The residual
# is what v2 still cannot parse, reported rather than absorbed.

# %%
PARSED = ("datadog", "promql")   # the formula layer has its own grammar, not written yet

rows_out = []
for lang in PARSED:
    for text in sorted(set(df.loc[df.lang == lang, "text"])):
        norm = c.normalise(lang, text)
        v1, v2 = c.parse_outcome(lang, norm, "v1"), c.parse_outcome(lang, norm, "v2")
        repairs = c.repairs_needed(lang, norm, "v1")
        rows_out.append(dict(lang=lang, text=text, norm=norm,
                             raw_ok=c.parse_outcome(lang, text, "v2").ok,
                             v1_ok=v1.ok, ok=v2.ok,
                             blocker=v2.blocker, excerpt=v2.excerpt,
                             repairs="|".join(repairs) if repairs else
                             ("" if repairs == () else "UNEXPLAINED")))
outcomes = pd.DataFrame(rows_out)
outcomes = outcomes.join(df[df.lang.isin(PARSED)].groupby(["lang", "text"]).size().rename("n"),
                         on=["lang", "text"])
outcomes.to_csv(PROCESSED / "parse_outcomes.csv", index=False)
print(f"wrote {(PROCESSED / 'parse_outcomes.csv').relative_to(ROOT)}")

rows = []
for lang in PARSED:
    o = outcomes[outcomes.lang == lang]
    rows.append([
        tex(lang), f"{int(o.n.sum()):,}", f"{len(o):,}",
        f"{o.raw_ok.mean():.1%}".replace("%", r"\%"),
        f"{o.v1_ok.mean():.1%}".replace("%", r"\%"),
        f"{o.ok.mean():.1%}".replace("%", r"\%"),
        f"{int((~o.ok).sum()):,}",
    ])
tabular("lrrrrrr",
        ["Language", "Queries", "Distinct", "Raw", "v1, instantiated",
         "v2, instantiated", "Residual"],
        rows, TABLES / "parse_coverage.tex")
print(outcomes.groupby("lang")[["raw_ok", "v1_ok", "ok"]].mean())

# Every v1 failure the repair set cannot explain must be exactly what v2 still fails on;
# otherwise a repair is claiming a gap it does not close.
assert set(outcomes.loc[outcomes.repairs == "UNEXPLAINED", "norm"]) == \
       set(outcomes.loc[~outcomes.ok, "norm"]), "a repair claims a gap v2 does not close"


# %% [markdown]
# ### The gaps v1 had, as grammar changes
# One row per construct v1 was missing, with the number of distinct queries that need it
# and the shortest query that does. Each row is a production now in v2, and
# `tests/test_parse.py` parses the same construct there unrepaired.

# %%
gaps = []
for lang in PARSED:
    o = outcomes[(outcomes.lang == lang) & ~outcomes.repairs.isin(["", "UNEXPLAINED"])]
    assert not o.empty
    for repair in c.REPAIRS[lang]:
        name = repair[0]
        hit = o[o.repairs.str.split("|").apply(lambda rs: name in rs)]
        if len(hit):
            example = hit.loc[hit.norm.str.len().idxmin(), "norm"]
            gaps.append(dict(lang=lang, repair=name, doc=c.REPAIR_DOC[name],
                             queries=len(hit), example=example.replace("\n", " ")))
gaps = pd.DataFrame(gaps).sort_values(["lang", "queries"], ascending=[True, False])
tabular("lllrl", ["Language", "Missing construct", "Repair", "Queries", "Shortest example"],
        [[tex(r.lang), tex(r.doc), code(r.repair), f"{r.queries:,}",
          code(r.example, EXAMPLE_CHARS)] for _, r in gaps.iterrows()],
        TABLES / "grammar_gaps.tex")
print(gaps[["lang", "repair", "queries"]].to_string(index=False))

# %%
residual = outcomes[~outcomes.ok]
print(f"{len(residual)} distinct queries grammar v2 still cannot parse:")
for lang in PARSED:
    r = residual[residual.lang == lang]
    print(f"\n{lang}: {len(r)}")
    for _, row in r.head(5).iterrows():
        print(f"  [{row.blocker}] {row.norm[:100]}")


# %% [markdown]
# ## Feature frequencies
# Features are read off the raw text, so template variables and boolean scopes count as
# the features they are. Frequencies are over query occurrences, not distinct strings: a
# construct used by a thousand widgets costs a thousand dashboard panels if it is left
# out.

# %%
df["features"] = [c.features(l, t) for l, t in zip(df.lang, df.text)]
long = pd.DataFrame([(i, r.lang, r.source, r.kind, f)
                     for i, r in df.iterrows() for f in r.features],
                    columns=["query", "lang", "source", "kind", "feature"])
long.to_csv(PROCESSED / "features.csv", index=False)
print(f"wrote {(PROCESSED / 'features.csv').relative_to(ROOT)}: {len(long):,} rows")

freq = (long.groupby(["lang", "feature"]).size().rename("n").reset_index()
        .merge(df.groupby("lang").size().rename("total").reset_index(), on="lang"))
freq["share"] = freq.n / freq.total
freq = freq.sort_values(["lang", "n"], ascending=[True, False])
freq.head(15)


# %% [markdown]
# ### Table and figure

# %%

fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))
for ax, lang, title in zip(axes, ("datadog", "promql"),
                           ("Datadog metrics queries", "PromQL expressions")):
    top = freq[freq.lang == lang].head(TOP_N).iloc[::-1]
    ax.barh(range(len(top)), top.share, color="0.35")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.feature, fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of queries")
    ax.set_title(f"{title}\n(n = {int(top.total.iloc[0]):,})", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "feature_frequency.pdf")
print("wrote paper/figures/feature_frequency.pdf")


# %% [markdown]
# ## Functions and methods
# The Datadog side is dominated by `.as_count()`, not `.as_rate()`: notebook 02 solved
# the rollup translation for the rate form, and the count form is the one the corpus
# actually asks for.

# %%
fn = {}
for lang in ("datadog", "promql"):
    counter: Counter = Counter()
    for t in df.loc[df.lang == lang, "text"]:
        counter += c.function_names(lang, t)
    agg: Counter = Counter()
    for t in df.loc[df.lang == lang, "text"]:
        agg += c.aggregation_ops(lang, t)
    fn[lang] = (counter, agg)
    print(f"\n{lang} aggregators: {dict(agg.most_common(8))}")
    print(f"{lang} functions:   {dict(counter.most_common(12))}")

rows = []
for lang in ("datadog", "promql"):
    counter, agg = fn[lang]
    for kind, items in (("aggregator", agg.most_common(6)), ("function", counter.most_common(10))):
        for name, n in items:
            rows.append([tex(lang), kind, code(name), f"{n:,}"])
tabular("lllr", ["Language", "Kind", "Name", "Occurrences"],
        rows, TABLES / "operator_frequency.tex")


# %% [markdown]
# ## Implementation order
# Marginal frequency is the wrong order to build in: a query is only translatable when
# *every* feature it uses is supported. So pick greedily — at each step add the feature
# that completes the most queries — and report cumulative complete coverage. For a
# translator that supports exactly a feature set $S$, coverage$(S)$ is an upper bound on
# κ, since a query using anything outside $S$ cannot even be attempted; greedy gives an
# attainable $S$ at each size, not an optimal one, so the curve is a lower bound on the
# best ceiling a fragment of that size could have. The order is the build order for
# `lower/datadog.py` and `emit/promql.py`.

# %%
def greedy_order(feature_sets: list[frozenset[str]]) -> pd.DataFrame:
    """Greedily choose features, maximising queries whose feature set is fully covered."""
    total = len(feature_sets)
    missing = [set(f) for f in feature_sets]
    chosen: list[str] = []
    out = []
    covered = sum(not m for m in missing)
    out.append(dict(step=0, feature="(none)", gained=covered, pending=0,
                    cumulative=covered, share=covered / total))
    while True:
        completes: Counter = Counter()
        residual: Counter = Counter()
        for m in missing:
            if len(m) == 1:
                completes[next(iter(m))] += 1
            for f in m:
                residual[f] += 1
        if not residual:
            break
        pick = (completes.most_common(1)[0][0] if completes
                else residual.most_common(1)[0][0])
        gained = completes.get(pick, 0)
        for m in missing:
            m.discard(pick)
        chosen.append(pick)
        covered = sum(not m for m in missing)
        out.append(dict(step=len(chosen), feature=pick, gained=gained,
                        pending=residual[pick], cumulative=covered, share=covered / total))
    return pd.DataFrame(out)


order = {lang: greedy_order(list(df.loc[df.lang == lang, "features"]))
         for lang in ("datadog", "promql")}
for lang, o in order.items():
    hit = o[o.share >= 0.9]
    print(f"{lang}: {len(o) - 1} features; 90% complete coverage after "
          f"{int(hit.step.iloc[0]) if len(hit) else '>' + str(len(o) - 1)} features")
order["datadog"].head(14)

# %% [markdown]
# These values are recorded rather than read off the truncated table above, because the
# table stops at twenty rows. What they are *not* is a bound on $\kappa$: greedy gives an
# attainable set at each size and not an optimal one, so the curve is a lower bound on
# the best coverage a fragment of that size could reach, and a translator with $n$
# features may sit above it. This corpus shows that happening --- the translator's own
# 25-feature set covers 97.3% where the greedy value at 25 is 96.2% --- so the bound
# §7 states against $\kappa$ is computed in notebook 03 from the feature set the
# translator actually uses. An earlier draft of this cell claimed the opposite direction
# and §7 drew the wrong conclusion from it.

# %%
_dd = order["datadog"]
_ceil = {int(r.step): round(r.share * 100, 1) for _, r in _dd.iterrows()}
_QF = ROOT / "paper" / "quoted_figures.json"
_existing = json.loads(_QF.read_text()) if _QF.exists() else {}
_existing.update({f"nb00.ceiling_{k}_features_pct": v for k, v in _ceil.items()})
_existing["nb00.datadog_features"] = len(_dd) - 1

# %% [markdown]
# ### Does a PromQL query name its own window?
# §8 argues the design artefact is that a Datadog query is not self-contained: it states
# neither its rollup interval nor its metric type. That is a *comparative* claim, and the
# comparison was never run. A PromQL range selector is a slot in the grammar, but a
# dashboard is free to fill it with a template variable, in which case the window comes
# from the dashboard's time range exactly as Datadog's does -- by a different mechanism
# and with the same consequence for a translator.

# %%
_RANGE = re.compile(r"\[([^\]]+)\]")
# A regex character class inside a quoted matcher is not a range selector:
# `device=~"^/dev/[sv]d[a-z][1-9]$"` was counting as three, which inflated the
# denominator of §8's comparison by twelve queries. String literals are blanked first.
_LIT = re.compile(r'"(?:[^"\\]|\\.)*"' + r"|'(?:[^'\\]|\\.)*'" + r"|`[^`]*`")
_pq = df[df.lang == "promql"]
_ranged = _var = 0
for _text in _pq.text:
    _bare = _LIT.sub(lambda m: '""', _text)
    _spans = _RANGE.findall(_bare)
    if not _spans:
        continue
    _ranged += 1
    _var += any("$" in s for s in _spans)
_existing["nb00.promql_with_range"] = int(_ranged)
_existing["nb00.promql_window_from_variable"] = int(_var)
_existing["nb00.promql_window_from_variable_pct"] = round(_var / _ranged * 100, 1)
print(f"promql occurrences with a range selector: {_ranged:,}")
print(f"  window supplied by a template variable: {_var:,} "
      f"({_var / _ranged * 100:.1f}%)")

# %% [markdown]
# ### Which ranking functions does `top()` use, and in which direction?
# §7 settles whether `top(...)` is expressible by checking the candidate expression for
# every ranking function the corpus asks for, so which those are is a load-bearing fact
# and was asserted rather than counted. Counted here: four functions, not the five the
# probe covers, and two *directions* -- which matters more than the count, because
# `topk` is the descending one and an ascending `top()` needs `bottomk`.

# %%
_TOPCALL = re.compile(r"\btop\s*\(")
_RANKARGS = re.compile(r",\s*\d+\s*,\s*'([a-z0-9]+)'\s*,\s*'(asc|desc)'")
_rank, _dir = Counter(), Counter()
_top_occ = _asc_occ = 0
for _text in df.loc[df.lang == "datadog", "text"]:
    if not _TOPCALL.search(_text):
        continue
    _top_occ += 1
    _args = _RANKARGS.findall(_text)
    for _by, _d in _args:
        _rank[_by] += 1
    _dirs = {_d for _, _d in _args}
    for _d in _dirs:
        _dir[_d] += 1
    _asc_occ += "asc" in _dirs
# A query may carry two `top()` calls, so the per-function counts sum above _top_occ.
_existing["nb00.top_occurrences"] = int(_top_occ)
_existing["nb00.top_rank_functions"] = len(_rank)
_existing["nb00.top_asc_occurrences"] = int(_asc_occ)
_existing["nb00.top_desc_occurrences"] = int(_dir["desc"])
for _by, _n in _rank.items():
    _existing[f"nb00.top_rank_{_by}_calls"] = int(_n)
print(f"top() occurrences: {_top_occ:,}; ranking functions: "
      + ", ".join(f"{k} {v}" for k, v in _rank.most_common()))
print(f"  ascending: {_asc_occ}  descending: {_dir['desc']}")

# %% [markdown]
# ### The formula layer §7 holds aside
# A formula sits over a request's queries, so it belongs to the metrics language only
# when they do -- the extractor applied that filter to the queries and not to the
# formulas, and the shares §7 quotes were computed over a population that was 48%
# log, event and security-signal formulas, which are 90% bare references.

# %%
_fml = df[df.lang == "datadog_formula"]
_ffeat = Counter(f for t in _fml.text for f in c.features("datadog_formula", t))
_existing["nb00.formulas"] = len(_fml)
_existing["nb00.formulas_distinct"] = int(_fml.text.nunique())
for _k, _n in _ffeat.items():
    _existing[f"nb00.formula_{_k.removeprefix('FORMULA_').lower()}_pct"] = round(
        _n / len(_fml) * 100, 1)
# The population the filter removes, which §7 reports to say what the old shares were
# diluted with. Recomputed from the source trees rather than from the corpus, since the
# corpus no longer contains them.
_nonmetric = c.non_metric_formulas(CORPUS / "sources" / "integrations-core")
_existing["nb00.non_metric_formulas"] = len(_nonmetric)
_existing["nb00.non_metric_formula_bare_ref_pct"] = round(
    sum("FORMULA_BARE_REF" in c.features("datadog_formula", t) for t in _nonmetric)
    / len(_nonmetric) * 100, 1) if _nonmetric else None
print(f"{len(_fml):,} metric formulas ({_fml.text.nunique():,} distinct); "
      f"{len(_nonmetric):,} non-metric formulas excluded, "
      f"{_existing['nb00.non_metric_formula_bare_ref_pct']}% bare references")
for _k, _n in _ffeat.most_common():
    print(f"   {_k:20} {_n:6,} {_n / len(_fml) * 100:5.1f}%")

# %% [markdown]
# ### What each half of instantiation is worth
# §7 describes instantiation as two substitutions -- a PromQL range variable becomes a
# duration, a Datadog scope variable becomes `*` -- and used to say everything else was
# left alone. `normalise` also rewrites a variable outside a string literal into an
# identifier, and on the PromQL side that third step is worth three points of parse rate,
# so the description mattered to how the table is read.

# %%
_pqd = sorted({t for t in df.loc[df.lang == "promql", "text"]})
def _rate(f):
    return round(sum(1 for t in _pqd if c.parse_outcome("promql", f(t)).ok)
                 / len(_pqd) * 100, 1)
_existing["nb00.promql_parse_range_vars_only_pct"] = _rate(
    lambda t: c._outside_strings(
        t, lambda s: c._PROM_RANGE_TMPL.sub(c._prom_range_sub, s)))
print("promql distinct parse rate, range variables substituted only: "
      f"{_existing['nb00.promql_parse_range_vars_only_pct']}%")

# %% [markdown]
# ### How many queries carry a template variable?
# §4 and §7 compare "$87\%$ of Datadog and $82\%$ of PromQL occurrences contain at least
# one variable". The PromQL half was the any-variable share and the Datadog half was the
# share with a variable *inside the scope braces* --- two definitions in one comparison.
# Both are counted here, under the definition the sentence states.

# %%
for _lang in ("datadog", "promql"):
    _sub = df[df.lang == _lang]
    _anyvar = int(sum("$" in t for t in _sub.text))
    _existing[f"nb00.{_lang}_with_variable"] = _anyvar
    _existing[f"nb00.{_lang}_with_variable_pct"] = round(_anyvar / len(_sub) * 100, 1)
    print(f"{_lang}: {_anyvar:,} of {len(_sub):,} occurrences carry a variable "
          f"({_anyvar / len(_sub) * 100:.1f}%)")

# %%
_QF.write_text(json.dumps(_existing, indent=1, sort_keys=True) + "\n")
print("greedy ceilings near kappa:")
for _k in sorted(_ceil):
    if _ceil[_k] >= 94.0:
        print(f"  {_k:>2} features -> {_ceil[_k]}%")

# %%
dd = order["datadog"]
rows = [[f"{int(r.step)}", code(r.feature), tex(c.FEATURE_DOC.get(r.feature, "")),
         f"{int(r.pending):,}", f"{int(r.gained):,}", f"{r.share:.1%}".replace("%", r"\%")]
        for _, r in dd[dd.step > 0].head(TOP_N).iterrows()]
tabular("rllrrr",
        ["Step", "Feature", "Construct", "Pending uses", "Queries completed", "Cumulative"],
        rows, TABLES / "implementation_order.tex")

fig, ax = plt.subplots(figsize=(4.5, 2.8))
for lang, style in (("datadog", "-"), ("promql", "--")):
    o = order[lang]
    ax.step(o.step, o.share, style, where="post", color="0.2" if lang == "datadog" else "0.55",
            label=f"{lang} (n = {int((df.lang == lang).sum()):,})")
ax.axhline(0.9, color="0.8", lw=0.8, zorder=0)
ax.set_xlabel("features implemented (greedy order)")
ax.set_ylabel("queries fully covered")
ax.set_ylim(0, 1)
ax.legend(fontsize=7, frameon=False, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "cumulative_coverage.pdf")
print("wrote paper/figures/cumulative_coverage.pdf")


# %% [markdown]
# ## The formula layer
# A Datadog widget is a list of named queries plus a formula over them. If almost every
# formula is a bare reference, the metrics query language carries the semantics and
# scoping the paper to it costs little; that is what the corpus says.

# %%
fl = df[df.lang == "datadog_formula"]
fshare = (pd.Series([f for fs in fl.features for f in fs]).value_counts() / len(fl))
print(f"{len(fl):,} formulas")
print((fshare * 100).round(1).to_string())


# %% [markdown]
# ## Findings (to carry into §4, §5 and §7)
# * The grammars of §4 were first inferred from documentation; the corpus says how far
#   that got. `parse_coverage.tex` gives the raw, v1 and v2 rates, and `grammar_gaps.tex`
#   the exact productions v1 was missing, each verified twice: by the repair that makes
#   its queries parse under v1, and by v2 parsing the same construct unrepaired.
# * What is left is not a grammar gap. The PromQL residual is queries that only become
#   queries once a dashboard variable is substituted into the middle of an identifier
#   (`otelcol_..${divider:raw}..`), and the Datadog residual is one undocumented
#   construct, the conditional aggregator `count(v: v>=0):`, shipped in Datadog's own
#   dashboards. Inventing a production for an undocumented construct would be worse than
#   reporting it.
# * Template variables are not an edge case: they appear in the large majority of
#   dashboard queries in both languages, so a translator must take a query *schema*, not a
#   query.
# * Datadog's boolean scopes (`AND`/`OR`/`NOT`, `IN (...)`, parentheses) are a second
#   language inside the scope braces, and the regex↔glob work of §6 has to meet it.
# * `.as_count()` outnumbers `.as_rate()` several times over, so §5 needs the count
#   translation worked out as carefully as the rate one.
# * Coverage is a staircase, not a curve: the first few features complete almost nothing
#   on their own because nearly every real query combines several, so a fragment is only
#   worth anything once the whole kernel (scope variables, `sum:`/`avg:`, grouping,
#   `.as_count()`, plain and `{*}` scopes) is in. `implementation_order.tex` is that
#   order; its cumulative column bounds κ for a translator limited to those features.

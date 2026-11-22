#!/usr/bin/env python
r"""The frozen grammar and the frozen translator, on dashboards nobody here has read.

`scripts/heldout_split.py` refits by cross-validation the two things that are functions
of the corpus. The grammar is not one of them --- it is handwritten, so no script can
re-derive what it would have been had it seen half the corpus --- and it is the channel
section 7 names first: grammar v2 is v1 plus every production the corpus showed was
missing. The only honest test of that fit is a corpus the grammar was not built against.

`scripts/heldout_fetch.py` provides three:

  grafana           Prometheus dashboards past download rank 200, which the corpus
                    stopped at. Never listed, downloaded or read here.
  integrations-core The same repository at a later commit, restricted to query strings
                    the corpus does not contain.
  integrations-extras A different repository: the community- and partner-maintained
                    sibling, 209 dashboards by other authors in the same file layout.
                    This is the one Datadog holdout with the size to estimate a rate.

This script reports, on each of them, exactly the columns Table~ref{tab:parse-coverage}
reports in sample --- raw, v1, v2, residual --- plus the feature-vocabulary ceiling and,
on the Datadog side, kappa from the frozen translator.

It is a measurement and not a development loop. If a number here is worse than the
in-sample one, that is the finding; repairing the grammar until it improves would
destroy the only held-out set this paper has.

Usage: .venv/bin/python scripts/heldout_eval.py [--write-registry]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport import corpus as c                                       # noqa: E402
from qlport.diagnostics import Bottom                                # noqa: E402
from qlport.emit.promql import emit_detail                           # noqa: E402
from qlport.lower.datadog import lower_all                           # noqa: E402
from qlport.parse import parse                                       # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "queries.jsonl"
HELDOUT = ROOT / "data" / "heldout"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "paper" / "tables"
REGISTRY = ROOT / "paper" / "quoted_figures.json"

DEFAULT_ROLLUP = 60.0    # as notebook 03 sets it; the translator is called unchanged


RESIDUAL_KINDS = (
    # Checked in order, so each residual string lands in exactly one bucket.
    ("logql", lambda t: "__error__" in t or "|=" in t or "unwrap" in t),
    ("metricsql", lambda t: t.lstrip().startswith("with (")),
    ("old_template", lambda t: "[[" in t),
)


def residual_kinds(strings: list[str]) -> dict[str, int]:
    """What the strings v2 cannot parse actually are.

    Reported rather than removed. A held-out population is dirtier than a curated one
    --- community dashboards filed under the Prometheus datasource carry Loki panels and
    VictoriaMetrics templates --- and the temptation is to filter those out until the
    rate improves. Doing it after seeing the rate is how a held-out set stops being one,
    so the number in the table is the number as measured and this is the decomposition.
    """
    out: dict[str, int] = {k: 0 for k, _ in RESIDUAL_KINDS}
    out["promql"] = 0
    for t in strings:
        low = t.lower()
        for kind, test in RESIDUAL_KINDS:
            if test(low):
                out[kind] += 1
                break
        else:
            out["promql"] += 1
    return out


def _promql_only(queries: list) -> list:
    """Notebook 00's filter, applied here for the same reason it is applied there.

    Community dashboards listed under the Prometheus datasource carry Loki panels.
    LogQL is a different language and is no evidence about a PromQL grammar, so the
    notebook drops those panels before it measures anything --- and a held-out set of
    500 community dashboards carries far more of them than the corpus's 200 did. Leaving
    them in would have scored the grammar against a language it does not claim to parse
    and called the result a holdout.
    """
    return [q for q in queries
            if q.lang != "promql" or c.is_promql(q.text)]


def tabular(colspec: str, head: list[str], rows: list[list[str]], path: Path) -> None:
    lines = [rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", r"\toprule",
             " & ".join(head) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def pct(x: float) -> str:
    return f"{x:.1%}".replace("%", r"\%")


# ---------------------------------------------------------------------------
# the frozen translator, called exactly as notebook 03 calls it
# ---------------------------------------------------------------------------
def classify(text: str) -> dict:
    row = dict(norm=text, outcome="", gap="")
    if not c.parse_outcome("datadog", text).ok:
        return {**row, "outcome": "unparsed"}
    for e in lower_all(parse("datadog", text)):
        got = e if isinstance(e, Bottom) else emit_detail(
            e, default_rollup=DEFAULT_ROLLUP)
        if isinstance(got, Bottom):
            return {**row, "outcome": "bottom", "gap": got.gap.name}
    return {**row, "outcome": "translated"}


# ---------------------------------------------------------------------------
# one side of the held-out set
# ---------------------------------------------------------------------------
def ceiling(texts: list[str], occ: Counter, feats: dict[str, set],
            vocabulary: set[str]) -> dict:
    """What share of occurrences use only features the vocabulary already has.

    `novel_features` counts occurrences per feature and a string with two novel
    features lands in two buckets, so those counts do not sum to anything. The number
    that means something --- and the only one the prose may total --- is
    `novel_occurrences`, which counts each string once however many features it adds.
    """
    novel: Counter = Counter()
    inside_occ = inside_d = 0
    novel_occ = 0
    for t in texts:
        extra = feats[t] - vocabulary
        if extra:
            novel_occ += occ[t]
            for f in extra:
                novel[f] += occ[t]
        else:
            inside_occ += occ[t]
            inside_d += 1
    total = sum(occ.values()) or 1
    # `occurrences` is here so that the distance between two of these ceilings can be
    # taken from the counts. Differencing the two rounded shares gave 2.6 where the
    # counts give 2.6101 --- the same to a tenth today and by arithmetic that cannot
    # promise to stay so, which is the defect this script spent a commit removing from
    # the shrinkages and left in place one function away.
    return dict(size=len(vocabulary),
                occ_pct=round(inside_occ / total * 100, 1),
                occurrences=total, inside_occurrences=inside_occ,
                distinct_pct=round(inside_d / (len(texts) or 1) * 100, 1),
                novel_occurrences=novel_occ,
                novel_features=dict(novel.most_common()))


def measure(queries: list, lang: str, vocabulary: set[str],
            corpus_forms: set[str], translator: set[str] | None = None) -> dict:
    """Parse rates over distinct queries, and the vocabulary ceiling over occurrences.

    Disjointness is enforced by dashboard, which is the unit the fit was made over: the
    grammar was extended until it parsed the corpus, and a dashboard the corpus does not
    contain contributed nothing to that. But a popular query string recurs across
    dashboards, so some held-out strings are ones the corpus also holds, and those are
    not held out in any useful sense. They are measured separately and reported
    separately, because the rate that belongs in the paper is the one on the strings
    whose instantiated form appears nowhere in the corpus.
    """
    texts = sorted({q.text for q in queries})
    occ = Counter(q.text for q in queries)
    forms = {t: c.normalise(lang, t) for t in texts}
    novel_texts = [t for t in texts if forms[t] not in corpus_forms]
    raw = v1 = v2 = 0
    nov_v1 = nov_v2 = 0
    residual = []
    blockers: Counter = Counter()
    for t in texts:
        norm = forms[t]
        raw += c.parse_outcome(lang, t, "v2").ok
        got_v1 = c.parse_outcome(lang, norm, "v1").ok
        out = c.parse_outcome(lang, norm, "v2")
        v1 += got_v1
        v2 += out.ok
        if forms[t] not in corpus_forms:
            nov_v1 += got_v1
            nov_v2 += out.ok
        if not out.ok:
            residual.append(norm)
            blockers[out.blocker] += 1
    n = len(texts) or 1
    nn = len(novel_texts) or 1

    feats = {t: c.features(lang, t) for t in texts}
    corpus_ceiling = ceiling(texts, occ, feats, vocabulary)
    out = dict(
        dashboards=len({q.origin for q in queries}),
        queries=len(queries), distinct=len(texts),
        # The counts, so a margin can be taken before the rounding rather than after it.
        # round(99.9 - 94.4, 1) is 5.5 and the gain is 5.6: subtracting two figures that
        # have each already lost four decimal places is how a tenth of a point appears
        # out of nowhere, and every gain this script reports is derived from these.
        raw_n=raw, v1_n=v1, v2_n=v2,
        raw_pct=round(raw / n * 100, 1), v1_pct=round(v1 / n * 100, 1),
        v2_pct=round(v2 / n * 100, 1), residual=len(residual),
        shared_distinct=len(texts) - len(novel_texts),
        novel_distinct=len(novel_texts),
        novel_v1_n=nov_v1, novel_v2_n=nov_v2,
        novel_v1_pct=round(nov_v1 / nn * 100, 1),
        novel_v2_pct=round(nov_v2 / nn * 100, 1),
        blockers=dict(blockers.most_common()),
        residual_kinds=residual_kinds(residual),
        residual_examples=[r[:120] for r in residual[:5]],
        # Named for the vocabulary it is measured against. The corpus shows 28 Datadog
        # features; the translator demonstrably handles 25 of them, and that smaller set
        # is the one that bounds kappa. A ceiling quoted without saying which set it is
        # against is two different numbers wearing one name.
        corpus_vocabulary=corpus_ceiling,
        ceiling_occ_pct=corpus_ceiling["occ_pct"],
        ceiling_distinct_pct=corpus_ceiling["distinct_pct"],
        novel_features=corpus_ceiling["novel_features"],
    )
    if translator is not None:
        out["translator_vocabulary"] = ceiling(texts, occ, feats, translator)
    return out


def _cluster_se(per: dict[str, list[int]]) -> tuple[float, float, float]:
    """How far a rate moves when the dashboards behind it change.

    Occurrences are not drawn one at a time. They arrive a dashboard at a time, and
    inside a dashboard they are near-copies of each other --- one integration's panels
    repeat one query shape over a dozen metrics --- so the binomial standard error of a
    rate over occurrences is the standard error of a sample far larger than the one we
    have. The arithmetic here is the linearised standard error of the ratio estimator
    with the dashboard as the resampling unit, plus the design effect it implies: the
    factor by which treating occurrences as independent would understate the spread.

    What it is not is a sampling error, and this docstring claimed it was for a commit:
    "the design that was actually used" was wrong, because no design was used. Both
    populations here are censuses of their repositories, so nothing was sampled and
    there is no sampling error to estimate. What the figure measures is how far the
    rate moves when the repository's dashboards change, which is the question the
    paper is actually asking of it, and the answer is that a tenth of a point is
    inside that movement several times over.
    """
    n = sum(v[0] for v in per.values())
    t = sum(v[1] for v in per.values())
    m = len(per)
    if not n or m < 2:
        return 0.0, 0.0, 0.0
    p = t / n
    resid = [v[1] - p * v[0] for v in per.values()]
    se = math.sqrt(sum(x * x for x in resid) * m / (m - 1)) / n * 100
    srs = math.sqrt(p * (1 - p) / n) * 100
    return se, srs, (se / srs) ** 2 if srs else 0.0


def gain_spread(queries: list, lang: str, corpus_forms: set[str]) -> dict:
    """v2's margin over v1, and the spread a dashboard-level jackknife puts on it.

    The paper's interest is not in this margin but in how much of it survives out of
    sample, which is a difference of two of these. A difference of two noisy numbers is
    noisier than either, and the fit decompositions this subsection used to report ---
    a third of the PromQL climb, a fifth of the Datadog one --- were differences
    quoted to a tenth of a point without anyone asking what a tenth of a point was
    worth. Dropping one dashboard and recomputing answers that, and the answer is that
    neither difference is larger than its own spread.

    Strings are the unit measured and dashboards are the unit resampled, so dropping a
    dashboard drops only the strings no other dashboard carries.
    """
    bydash: dict[str, set[str]] = {}
    for q in queries:
        bydash.setdefault(q.origin, set()).add(q.text)
    texts = sorted({q.text for q in queries})
    forms = {t: c.normalise(lang, t) for t in texts}
    keep = [t for t in texts if forms[t] not in corpus_forms]
    kept = set(keep)
    ok1 = {t: c.parse_outcome(lang, forms[t], "v1").ok for t in keep}
    ok2 = {t: c.parse_outcome(lang, forms[t], "v2").ok for t in keep}
    holders: Counter = Counter()
    for ts in bydash.values():
        for t in ts & kept:
            holders[t] += 1

    def rate(drop: set[str]) -> float:
        sel = [t for t in keep if t not in drop]
        return (sum(ok2[t] for t in sel) - sum(ok1[t] for t in sel)) / (len(sel) or 1) \
            * 100

    full = rate(set())
    reps = [rate({t for t in ts & kept if holders[t] == 1}) for ts in bydash.values()]
    n = len(reps)
    mean = sum(reps) / n if n else 0.0
    var = (n - 1) / n * sum((r - mean) ** 2 for r in reps) if n > 1 else 0.0
    # Rounded for the paper, exact for anything that subtracts two of these. The
    # shrink below is a difference of two margins about a point apart; taking it from
    # the rounded pair moved it a tenth of a point and the guard that checked the
    # subtraction checked the rounded one, so it agreed.
    # Same concentration question as the kappa jackknife: how many dashboards the
    # spread actually rests on. The replicates are the per-dashboard deviations.
    dev = [r - mean for r in reps]
    s2 = sum(d * d for d in dev)
    s4 = sum(d ** 4 for d in dev)
    return dict(distinct=len(keep), dashboards=n,
                effective_clusters=round(s2 * s2 / s4, 1) if s4 else None,
                gain=full, gain_se=math.sqrt(var),
                gain_points=round(full, 1), gain_se_points=round(math.sqrt(var), 1))


def feature_set_seen(queries: list, corpus_queries: list) -> dict:
    """Occurrences whose whole feature set is one the corpus already exhibits.

    String-level disjointness is what this holdout enforces and it is not structural
    novelty: a query can be a string no corpus dashboard contains and still be the
    same shape over a different metric. This measures how much of that there is, and
    it is the caveat the subsection ends on rather than a result.
    """
    seen = {frozenset(c.features("datadog", q.text)) for q in corpus_queries}
    n = sum(1 for q in queries if frozenset(c.features("datadog", q.text)) in seen)
    return dict(occurrences=len(queries), seen=n,
                pct=round(n / (len(queries) or 1) * 100, 1))


_BOOTSTRAP_DRAWS = 20_000
# Five seeds, not one. A single seed's third decimal is not a measurement: across
# eight seeds at this many draws the held-out standard deviation ran from 1.077 to
# 1.091, and the one originally shipped was the highest of them --- which is to say
# the one closest to the linearised 1.095, quoted in a commit message as evidence that
# the two agree. The Monte-Carlo error on a bootstrap standard deviation is about
# sd/sqrt(2B), some 0.005 here, so the spread is expected and only the mean is worth
# storing. The spread is stored beside it so that nothing has to take that on trust.
_BOOTSTRAP_SEEDS = (20261122, 1, 2, 7, 12345)


def _cluster_bootstrap_once(per: list[list[int]], seed: int) -> list[float]:
    """One seed's replicate rates: `len(per)` dashboards drawn with replacement."""
    import random

    m = len(per)
    rng = random.Random(seed)
    reps = []
    for _ in range(_BOOTSTRAP_DRAWS):
        pick = [per[rng.randrange(m)] for _ in range(m)]
        n = sum(v[0] for v in pick)
        if n:
            reps.append(sum(v[1] for v in pick) / n * 100)
    return reps


def _cluster_bootstrap(per: list[list[int]]) -> dict:
    """How the rate behaves when whole dashboards are resampled.

    The non-parametric companion to `_cluster_se`, which asks nothing of the
    residuals' shape. Where it agrees with the linearised error, the linearisation is
    doing no work; where it does not, the linearised figure is the one to distrust.

    It answers two different questions and they came out differently, which is why
    both are recorded. The MAGNITUDE agrees: the mean standard deviation over the
    seeds is within a per cent of the linearised error. The SHAPE does not: the
    replicate distribution is left-skewed, and its 95 per cent interval is displaced
    from the normal one by a third of a standard error on the held-out side against a
    sixth in sample --- the degradation an effective cluster count of two predicts. A
    comment in this file asserted the shape was fine on the strength of the magnitude
    agreeing, which does not follow, after an earlier one had asserted the opposite on
    no evidence at all.
    """
    if len(per) < 2:
        return {}
    sds, out = [], {}
    for seed in _BOOTSTRAP_SEEDS:
        reps = _cluster_bootstrap_once(per, seed)
        mean = sum(reps) / len(reps)
        sds.append(math.sqrt(sum((r - mean) ** 2 for r in reps) / (len(reps) - 1)))
        if seed == _BOOTSTRAP_SEEDS[0]:
            reps.sort()
            sd = sds[0]
            # The temporal holdout's three dashboards all translate everything, so
            # every replicate is the same rate and there is no shape to describe.
            if sd:
                out["skew"] = sum((r - mean) ** 3 for r in reps) / len(reps) / sd ** 3
            out["pct_2_5"] = reps[int(0.025 * len(reps))]
            out["pct_97_5"] = reps[int(0.975 * len(reps))]
    out["se_exact"] = sum(sds) / len(sds)
    out["se_spread"] = max(sds) - min(sds)
    return out


def jackknife(queries: list, translated: set[str]) -> dict:
    """kappa with each dashboard dropped in turn.

    Occurrences are not independent: they cluster inside dashboards, and one
    integration's dashboard can carry a double-figure share of a population's refusals.
    A rate quoted to a tenth of a point should be set against the spread this produces,
    not against its own standard error computed as though the occurrences were drawn
    one at a time.
    """
    per: dict[str, list[int]] = {}
    for q in queries:
        d = per.setdefault(q.origin, [0, 0])
        d[0] += 1
        d[1] += c.normalise("datadog", q.text) in translated
    tot_n = sum(v[0] for v in per.values())
    tot_t = sum(v[1] for v in per.values())
    spread = sorted((tot_t - v[1]) / (tot_n - v[0]) * 100
                    for v in per.values() if tot_n - v[0])
    worst = max(per.items(), key=lambda kv: kv[1][0] - kv[1][1])
    se, srs, deff = _cluster_se(per)
    # How many clusters the spread is really resting on. A cluster-robust error is
    # asymptotic in the number of clusters, and these residuals are concentrated:
    # (sum r^2)^2 / sum r^4 is the usual effective count. Recorded here so the prose
    # has to take it into account.
    resid = [v[1] - (tot_t / tot_n) * v[0] for v in per.values()] if tot_n else []
    s2 = sum(r * r for r in resid)
    s4 = sum(r ** 4 for r in resid)
    # Every scalar here is derivable from this list, which is why it is stored: a
    # cached number nothing can recompute is a number on the author's word.
    clusters = sorted([v[0], v[1]] for v in per.values())
    # And whether the linearised error can be read as a normal one at that effective
    # count, which is a question about the arithmetic and not an opinion about it. Two
    # comments here have now answered it wrongly in opposite directions: the first said
    # no on no evidence, the second said yes on evidence about the width. The answer is
    # that the width agrees and the tails do not. Drawn from `clusters` in its sorted
    # order, so the draw sequence is a function of what the cache holds rather than of
    # dictionary insertion order.
    boot = _cluster_bootstrap(clusters)
    return dict(dashboards=len(per), clusters=clusters,
                cluster_se_pct=round(se, 1), srs_se_pct=round(srs, 1),
                cluster_se_exact=se, srs_se_exact=srs,
                bootstrap_se_exact=boot.get("se_exact"),
                bootstrap_se_pct=round(boot["se_exact"], 1) if boot else None,
                bootstrap_se_spread=boot.get("se_spread"),
                bootstrap_skew=boot.get("skew"),
                bootstrap_pct_2_5=boot.get("pct_2_5"),
                bootstrap_pct_97_5=boot.get("pct_97_5"),
                bootstrap_draws=_BOOTSTRAP_DRAWS, bootstrap_seeds=list(_BOOTSTRAP_SEEDS),
                effective_clusters=round(s2 * s2 / s4, 1) if s4 else None,
                top_cluster_share_pct=round(max((r * r for r in resid), default=0.0)
                                            / s2 * 100, 1) if s2 else None,
                design_effect=round(deff, 1),
                dashboards_with_a_refusal=sum(1 for v in per.values() if v[0] != v[1]),
                worst_dashboard=worst[0],
                worst_dashboard_refusals=worst[1][0] - worst[1][1],
                low_pct=round(spread[0], 1) if spread else None,
                high_pct=round(spread[-1], 1) if spread else None)


def template_share(queries: list) -> dict:
    r"""Occurrences carrying a template variable, under notebook 00's definition.

    \S\ref{sec:threats} says the template share is the figure most likely to be inflated
    by using vendor and community dashboards, and before this holdout there was nothing
    to compare it against. There is now, so it is measured rather than conceded --- by
    the notebook's own test, a literal ``$'' in the shipped string, so that the
    in-sample figure here has to come back as the notebook's.
    """
    n = sum("$" in q.text for q in queries)
    return dict(occurrences=len(queries), with_variable=n,
                pct=round(n / (len(queries) or 1) * 100, 1))


def kappa(queries: list) -> dict:
    occ = Counter(c.normalise("datadog", q.text) for q in queries)
    rows = [{**classify(t), "n": occ[t]} for t in occ]
    total = sum(occ.values()) or 1
    by = Counter()
    byd = Counter()
    for r in rows:
        by[r["outcome"]] += r["n"]
        byd[r["outcome"]] += 1
    gaps = Counter(r["gap"] for r in rows if r["gap"])
    gaps_occ: Counter = Counter()
    for r in rows:
        if r["gap"]:
            gaps_occ[r["gap"]] += r["n"]
    return dict(
        occurrences=total, distinct=len(rows),
        translated_occ=by["translated"], translated_distinct=byd["translated"],
        kappa_occ_pct=round(by["translated"] / total * 100, 1),
        kappa_distinct_pct=round(byd["translated"] / (len(rows) or 1) * 100, 1),
        by_outcome_occ=dict(by), by_outcome_distinct=dict(byd),
        # Both units. The prose breaks the refusals down in a sentence that opens with
        # an occurrence count, so the distinct counts alone cannot be quoted there: they
        # sum to the number of refused *strings*, which is a different number.
        gaps=dict(gaps.most_common()), gaps_occ=dict(gaps_occ.most_common()),
        jackknife=jackknife(queries, {r["norm"] for r in rows
                                      if r["outcome"] == "translated"}),
        translated_norms=sorted(r["norm"] for r in rows
                                if r["outcome"] == "translated"),
    )


def translator_feature_set(queries: list, translated: list[str]) -> set[str]:
    r"""$S^\star$: the features of the queries the translator actually translates.

    Notebook 03 builds it this way and uses it to bound kappa, because a query using a
    feature the translator never handles is refused by construction. It is a subset of
    the features the corpus exhibits --- 25 of 28 --- so a ceiling measured against the
    corpus vocabulary is a different and looser quantity, and the two must not be
    quoted under one name.
    """
    by_norm: dict[str, set[str]] = {}
    for q in queries:
        norm = c.normalise("datadog", q.text)
        by_norm.setdefault(norm, set()).update(c.features("datadog", q.text))
    return set().union(*(by_norm[n] for n in translated if n in by_norm))

def _corpus_census() -> dict:
    """The corpus side's dashboard census, from `data/corpus/SOURCES.md`.

    That file is written by `fetch_corpus.py` at the commit the corpus was extracted
    from, and is the only record of it: `data/corpus/sources/` is gitignored. Parsed
    out of its own section rather than by taking the last matching line, which would
    silently become kube-prometheus's if the sections were reordered.
    """
    text = (ROOT / "data" / "corpus" / "SOURCES.md").read_text()
    section = text.split("## integrations-core", 1)[1].split("\n## ", 1)[0]

    def field(name: str) -> int:
        return int(re.search(rf"- {name}: `(\d+)`", section).group(1))

    return {"ho.dd_corpus_dashboards_read": field("dashboards"),
            "ho.dd_corpus_dashboards_at_commit": field("files"),
            "ho.dd_corpus_unreadable": field("unreadable")}


def figures(res: dict) -> dict:
    """Every `ho.*` key the paper may quote, derived from one run's cached output.

    This is a function and not a stretch of `main()` so that the test suite can call
    it on `data/processed/heldout_eval.json` and compare the result against
    `paper/quoted_figures.json` key by key. Without that comparison the registry was
    a file a value could be hand-edited in with every guard above it still green: the
    phrases pin the prose to the registry, and nothing pinned the registry to the run.
    The pass that added the comparison put a count on how many values that was. It had
    not counted them, and the count was wrong by more than a factor of two, so what is
    written here instead is the property, which the test asserts and which does not
    need a number: every key the registry holds under `ho.` is one this function
    returns, and every value it holds for one is the value this function computes.
    """
    manifest = res["manifest"]
    gi = res["grafana_in_sample"]
    di = res["datadog_in_sample"]

    def gain(m: dict) -> float:
        """v2 over v1 in points, taken from the counts.

        Not `round(v2_pct - v1_pct, 1)`: both of those have already been rounded, and
        on the extras population that subtraction gives 5.5 where the counts give 5.6.
        """
        return round((m["v2_n"] - m["v1_n"]) / (m["distinct"] or 1) * 100, 1)

    fig: dict = {}
    if "grafana" in res:
        g = res["grafana"]
        fig.update({
            "ho.gf_dashboards": g["dashboards"], "ho.gf_queries": g["queries"],
            "ho.gf_distinct": g["distinct"], "ho.gf_raw_pct": g["raw_pct"],
            "ho.gf_v1_pct": g["v1_pct"], "ho.gf_v2_pct": g["v2_pct"],
            "ho.gf_residual": g["residual"],
            "ho.gf_ceiling_pct": g["ceiling_occ_pct"],
            "ho.gf_novel_features": len(g["novel_features"]),
            "ho.gf_shared_distinct": g["shared_distinct"],
            "ho.gf_residual_metric": g["blockers"].get("METRIC", 0),
            "ho.gf_residual_matchop": g["blockers"].get("MATCHOP", 0),
            "ho.gf_residual_other": g["residual"] - g["blockers"].get("METRIC", 0)
                                    - g["blockers"].get("MATCHOP", 0),
            "ho.gf_residual_logql": g["residual_kinds"]["logql"],
            "ho.gf_residual_metricsql": g["residual_kinds"]["metricsql"],
            "ho.gf_residual_old_template": g["residual_kinds"]["old_template"],
            "ho.gf_residual_promql": g["residual_kinds"]["promql"],
            "ho.gf_catalogue": manifest["grafana"]["available"],
            "ho.gf_downloaded": manifest["grafana"]["downloaded"],
            "ho.gf_novel_distinct": g["novel_distinct"],
            "ho.gf_novel_v1_pct": g["novel_v1_pct"],
            "ho.gf_novel_v2_pct": g["novel_v2_pct"],
        })
    if "integrations-core" in res:
        d = res["integrations-core"]
        fig.update({
            "ho.dd_dashboards": d["dashboards"], "ho.dd_queries": d["queries"],
            "ho.dd_distinct": d["distinct"], "ho.dd_v1_pct": d["v1_pct"],
            "ho.dd_v2_pct": d["v2_pct"], "ho.dd_residual": d["residual"],
            "ho.dd_ceiling_pct": d["ceiling_occ_pct"],
            "ho.dd_novel_features": len(d["novel_features"]),
        })
    # The corpus side's census, read out of the corpus's own provenance note.
    #
    # It came off `manifest["integrations-core"]` for a commit, which is the entry
    # `heldout_fetch.py` writes for that repository at a LATER commit --- the temporal
    # holdout, which this subsection insists is a separate population. The two commits
    # ship the same 534 dashboard files with the same 9 unreadable, so the figures
    # agreed digit for digit and nothing could see it. They agree because nothing was
    # added or removed between them, not because they are the same tree: four of the
    # 534 files differ in content. One upstream commit adding a dashboard would have
    # had the paper reporting the holdout's census as the corpus's, with the
    # registry-to-run pin confirming it, because the run is where the wrong number
    # came from.
    fig.update(_corpus_census())
    if "integrations-extras" in res:
        e = res["integrations-extras"]
        xm = manifest["integrations-extras"]
        fig.update({
            # Read, and shipped, as two keys. They are equal on this repository, and
            # the prose said "209 of 209" with one key on both sides of the "of", so
            # the sentence asserting that everything was read could not have failed.
            # On the corpus side above they are not equal.
            "ho.ex_dashboards_read": xm["dashboards"],
            "ho.ex_dashboards_at_commit": xm["files"],
            "ho.ex_unreadable": xm["unreadable"],
            "ho.ex_dashboards": e["dashboards"], "ho.ex_queries": e["queries"],
            "ho.ex_distinct": e["distinct"], "ho.ex_raw_pct": e["raw_pct"],
            "ho.ex_v1_pct": e["v1_pct"], "ho.ex_v2_pct": e["v2_pct"],
            "ho.ex_residual": e["residual"],
            "ho.ex_ceiling_pct": e["ceiling_occ_pct"],
            "ho.ex_novel_features": len(e["novel_features"]),
            "ho.ex_shared_distinct": e["shared_distinct"],
            "ho.ex_novel_distinct": e["novel_distinct"],
            "ho.ex_novel_v1_pct": e["novel_v1_pct"],
            "ho.ex_novel_v2_pct": e["novel_v2_pct"],
            "ho.ex_queries_at_commit": xm["queries_at_commit"],
            # Normalised forms, not shipped strings: the manifest counts
            # `corpus.normalise(lang, text)`, and the shipped-string count for the
            # same population is a different and larger number that sits one row away
            # in tab:heldout-coverage. The key says which so the prose has to.
            "ho.ex_forms_at_commit": xm["distinct_at_commit"],
            "ho.ex_gain_points": gain(e),
            "ho.ex_sstar_size": e["translator_vocabulary"]["size"],
            "ho.ex_sstar_pct": e["translator_vocabulary"]["occ_pct"],
            "ho.ex_sstar_novel_features":
                len(e["translator_vocabulary"]["novel_features"]),
            "ho.ex_sstar_outside_occurrences":
                e["translator_vocabulary"]["novel_occurrences"],
            "ho.ex_ceiling_outside_occurrences":
                e["corpus_vocabulary"]["novel_occurrences"],
            "ho.ex_vocabulary_size": e["corpus_vocabulary"]["size"],
            # The distance between the two ceilings: quoting the corpus vocabulary's
            # 99.9% as though it were the translator's would overstate the reachable
            # ceiling by exactly this much. Taken from the occurrence counts, because
            # the two shares are already rounded and the difference of two rounded
            # figures is not the rounded difference.
            "ho.ex_ceiling_overstatement_points": round(
                (e["corpus_vocabulary"]["inside_occurrences"]
                 - e["translator_vocabulary"]["inside_occurrences"])
                / e["corpus_vocabulary"]["occurrences"] * 100, 1),
            # Dropped before anything was measured, and counted so the prose can say
            # how much of the sibling repository was not disjoint.
            "ho.ex_shared_forms": xm["shared_with_corpus"],
            # Set aside rather than scored. No grammar exists for the formula layer in
            # either version, so there is no parse rate to report; the corpus's own
            # formulas are held aside the same way, which is why there is no formula
            # row in Table~\ref{tab:parse-coverage} either.
            "ho.ex_formula_queries": xm["queries_formula_held_out"],
            "ho.ex_formula_distinct": xm["distinct_formula_held_out"],
        })
    for key, prefix in (("gain_gf_in_sample", "ho.gf_in_sample"),
                        ("gain_gf_held", "ho.gf_novel"),
                        ("gain_dd_in_sample", "ho.dd_in_sample"),
                        ("gain_ex_held", "ho.ex")):
        if key in res:
            fig[f"{prefix}_gain_se_points"] = res[key]["gain_se_points"]
            fig[f"{prefix}_gain_effective_dashboards"] = \
                res[key]["effective_clusters"]
    for name, prefix in (("shrink_gf", "ho.gf"), ("shrink_dd", "ho.ex")):
        if name in res:
            fig[f"{prefix}_shrink_points"] = res[name]["points"]
            fig[f"{prefix}_shrink_se_points"] = res[name]["se_points"]
    if "feature_set_seen" in res:
        fig["ho.ex_feature_set_seen_pct"] = res["feature_set_seen"]["pct"]
    # The formula layer as it ships, before anything is held out. The prose used to
    # quote the post-filter remnant as though it were the layer, which understated it
    # ninefold and left the dropped majority --- forms the corpus already holds ---
    # unmentioned.
    if "integrations-extras" in res:
        xm = manifest["integrations-extras"]
        fig.update({
            "ho.ex_formula_queries_at_commit": xm["queries_formula_at_commit"],
            "ho.ex_formula_forms_at_commit": xm["distinct_formula_at_commit"],
            "ho.ex_formula_shared_forms": xm["shared_formula_with_corpus"],
            "ho.ex_formula_shared_queries": (xm["queries_formula_at_commit"]
                                             - xm["queries_formula_held_out"]),
        })
    fig.update({"ho.dd_in_sample_sstar_pct":
                di["translator_vocabulary"]["occ_pct"],
                "ho.dd_in_sample_sstar_size": di["translator_vocabulary"]["size"],
                "ho.dd_in_sample_distinct": di["distinct"],
                "ho.dd_in_sample_queries": di["queries"],
                "ho.dd_in_sample_dashboards": di["dashboards"],
                "ho.dd_in_sample_v1_pct": di["v1_pct"],
                "ho.dd_in_sample_v2_pct": di["v2_pct"],
                "ho.dd_in_sample_residual": di["residual"],
                "ho.dd_in_sample_gain_points": gain(di)})
    fig.update({"ho.gf_in_sample_dashboards": gi["dashboards"],
                "ho.gf_in_sample_distinct": gi["distinct"],
                "ho.gf_in_sample_queries": gi["queries"],
                "ho.gf_in_sample_raw_pct": gi["raw_pct"],
                "ho.gf_in_sample_residual": gi["residual"],
                "ho.gf_in_sample_v1_pct": gi["v1_pct"],
                "ho.gf_in_sample_v2_pct": gi["v2_pct"],
                # What the corpus-driven productions are worth, in sample and out of it.
                # The pair is the point, so both are recorded rather than subtracted in
                # the prose.
                "ho.gf_in_sample_gain_points": gain(gi)})
    if "grafana" in res:
        g = res["grafana"]
        fig["ho.gf_novel_gain_points"] = round(
            (g["novel_v2_n"] - g["novel_v1_n"]) / (g["novel_distinct"] or 1) * 100, 1)
        fig["ho.gf_listed_to_rank"] = manifest["grafana"]["listed_to_rank"]
    if "template_extras" in res:
        fig["ho.ex_template_pct"] = res["template_extras"]["pct"]
        fig["ho.dd_in_sample_template_pct"] = res["template_in_sample"]["pct"]
    if "kappa" in res:
        fig["ho.dd_kappa_pct"] = res["kappa"]["kappa_occ_pct"]
        fig["ho.dd_kappa_occurrences"] = res["kappa"]["occurrences"]
        fig["ho.dd_kappa_distinct"] = res["kappa"]["distinct"]
    if "kappa_in_sample" in res:
        ki = res["kappa_in_sample"]
        fig["ho.kappa_in_sample_pct"] = ki["kappa_occ_pct"]
        fig["ho.kappa_in_sample_se_points"] = ki["jackknife"]["cluster_se_pct"]
        fig["ho.kappa_in_sample_effective_dashboards"] = (
            ki["jackknife"]["effective_clusters"])
        fig["ho.kappa_in_sample_jackknife_low_pct"] = ki["jackknife"]["low_pct"]
        fig["ho.kappa_in_sample_jackknife_high_pct"] = ki["jackknife"]["high_pct"]
        fig["ho.kappa_in_sample_distinct_pct"] = ki["kappa_distinct_pct"]
        fig["ho.kappa_in_sample_occurrences"] = ki["occurrences"]
    if "kappa_extras" in res:
        ke = res["kappa_extras"]
        fig["ho.ex_kappa_pct"] = ke["kappa_occ_pct"]
        fig["ho.ex_kappa_distinct_pct"] = ke["kappa_distinct_pct"]
        fig["ho.ex_kappa_occurrences"] = ke["occurrences"]
        fig["ho.ex_kappa_distinct"] = ke["distinct"]
        fig["ho.ex_translated"] = ke["by_outcome_occ"].get("translated", 0)
        fig["ho.ex_bottom"] = ke["by_outcome_occ"].get("bottom", 0)
        fig["ho.ex_unparsed"] = ke["by_outcome_occ"].get("unparsed", 0)
        fig["ho.ex_refused_distinct"] = ke["by_outcome_distinct"].get("bottom", 0)
        fig["ho.ex_jackknife_low_pct"] = ke["jackknife"]["low_pct"]
        fig["ho.ex_jackknife_high_pct"] = ke["jackknife"]["high_pct"]
        fig["ho.ex_kappa_se_points"] = ke["jackknife"]["cluster_se_pct"]
        fig["ho.ex_kappa_srs_se_points"] = ke["jackknife"]["srs_se_pct"]
        fig["ho.ex_kappa_design_effect"] = ke["jackknife"]["design_effect"]
        # What the dashboard-level spread rests on. 139 dashboards, but the
        # dispersion is concentrated in effectively two of them and one carries most
        # of it, so the spread is a statement about how much this figure moves when
        # the repository's dashboards change --- not an interval anything was
        # sampled to produce.
        fig["ho.ex_kappa_effective_dashboards"] = (
            ke["jackknife"]["effective_clusters"])
        fig["ho.ex_kappa_top_dashboard_pct"] = (
            ke["jackknife"]["top_cluster_share_pct"])
        fig["ho.ex_dashboards_with_a_refusal"] = (
            ke["jackknife"]["dashboards_with_a_refusal"])
        fig["ho.ex_worst_dashboard_refusals"] = (
            ke["jackknife"]["worst_dashboard_refusals"])
        for name, n in ke["gaps"].items():
            fig[f"ho.ex_gap_{name.lower()}"] = n
        for name, n in ke["gaps_occ"].items():
            fig[f"ho.ex_gap_occ_{name.lower()}"] = n
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-registry", action="store_true")
    args = ap.parse_args()

    path = HELDOUT / "queries.jsonl"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing --- run "
                         "`.venv/bin/python scripts/heldout_fetch.py --all`")
    held = _promql_only(c.read_queries(path))
    train = _promql_only(c.read_queries(CORPUS))
    manifest = json.loads((HELDOUT / "MANIFEST.json").read_text())

    # The vocabulary the translator was built against: every feature the corpus shows.
    vocab = {lang: set().union(*(c.features(lang, q.text)
                                 for q in train if q.lang == lang))
             for lang in ("datadog", "promql")}
    for lang, v in vocab.items():
        print(f"corpus {lang} vocabulary: {len(v)} features")
    forms = {lang: {c.normalise(lang, q.text) for q in train if q.lang == lang}
             for lang in ("datadog", "promql")}

    # Disjointness is the whole premise, so it is asserted rather than assumed.
    # Every held-out Datadog source, found from the file rather than named here: an
    # earlier version of this check listed `integrations-core` and `grafana` explicitly
    # and so asserted nothing about `integrations-extras`, which is the population the
    # held-out kappa rests on, while printing that none of them were in the corpus.
    seen_dd = {c.normalise("datadog", q.text) for q in train if q.lang == "datadog"}
    seen_ids = {q.origin.split()[0] for q in train if q.source == "grafana"}
    dd_sources = sorted({q.source for q in held if q.lang == "datadog"})
    for src in dd_sources:
        leaked = [q for q in held if q.source == src and q.lang == "datadog"
                  and c.normalise("datadog", q.text) in seen_dd]
        assert not leaked, f"{len(leaked)} held-out {src} queries are in the corpus"
    leaked_gf = [q for q in held if q.source == "grafana"
                 and q.origin.split()[0] in seen_ids]
    assert not leaked_gf, f"{len(leaked_gf)} held-out dashboards are in the corpus"
    print(f"held out: {len(held):,} queries; the Datadog sources checked for leakage "
          f"are {', '.join(dd_sources)}, and none of them leaked")

    res: dict = {"manifest": manifest}

    # The matched in-sample comparator. Table~\ref{tab:parse-coverage}'s promql row
    # mixes Grafana with kube-prometheus, whose queries are compiled from jsonnet and
    # are nothing like a hand-built dashboard's, while the held-out set is Grafana
    # alone. Comparing against the mixed row would confound source with holdout, so the
    # Grafana part of the corpus is measured here on its own.
    gf_train = [q for q in train if q.source == "grafana" and q.lang == "promql"]
    res["grafana_in_sample"] = measure(gf_train, "promql", vocab["promql"], set())
    gi = res["grafana_in_sample"]
    print(f"\nGrafana in sample: {gi['queries']:,} queries, {gi['distinct']:,} distinct, "
          f"raw {gi['raw_pct']}%  v1 {gi['v1_pct']}%  v2 {gi['v2_pct']}%  "
          f"residual {gi['residual']}")

    # The matched in-sample Datadog comparator, for the same reason. This one should
    # reproduce Table~\ref{tab:parse-coverage}'s datadog row outright, because that row
    # is integrations-core alone; if it does not, this script is not the paper's
    # pipeline and nothing below it means anything.
    dd_train = [q for q in train if q.source == "integrations-core"
                and q.lang == "datadog"]

    # kappa, in sample and on each held-out Datadog population. The in-sample run is a
    # check and not a result: it calls notebook 03's classify() over notebook 03's
    # input, so it must return notebook 03's kappa. It also runs first, because the
    # queries it translates are what define $S^\star$.
    for key, qs, what in (
            ("kappa_in_sample", dd_train, "the corpus"),
            ("kappa_extras",
             [q for q in held if q.source == "integrations-extras"
              and q.lang == "datadog"], "integrations-extras"),
            ("kappa", [q for q in held if q.source == "integrations-core"
                       and q.lang == "datadog"], "the later commit")):
        if not qs:
            continue
        res[key] = k = kappa(qs)
        print(f"\nfrozen translator on {what}: kappa = {k['kappa_occ_pct']}% "
              f"of {k['occurrences']:,} occurrences, {k['kappa_distinct_pct']}% of "
              f"{k['distinct']:,} distinct")
        print(f"  outcomes {k['by_outcome_occ']}")
        print(f"  gaps by string {k['gaps']}")
        print(f"  gaps by occurrence {k['gaps_occ']}")

    for key, qs in (("template_in_sample", dd_train),
                    ("template_extras",
                     [q for q in held if q.source == "integrations-extras"
                      and q.lang == "datadog"])):
        if qs:
            res[key] = t = template_share(qs)
            print(f"{key}: {t['with_variable']:,}/{t['occurrences']:,} "
                  f"= {t['pct']}% carry a template variable")

    sstar = translator_feature_set(
        dd_train, res["kappa_in_sample"].pop("translated_norms"))
    for key in ("kappa_extras", "kappa"):
        res.get(key, {}).pop("translated_norms", None)
    print(f"\ntranslator feature set: {len(sstar)} of {len(vocab['datadog'])} "
          f"(absent: {sorted(vocab['datadog'] - sstar)})")

    res["datadog_in_sample"] = measure(dd_train, "datadog", vocab["datadog"], set(),
                                       translator=sstar)
    di = res["datadog_in_sample"]
    print(f"Datadog in sample: {di['queries']:,} queries, {di['distinct']:,} distinct, "
          f"raw {di['raw_pct']}%  v1 {di['v1_pct']}%  v2 {di['v2_pct']}%  "
          f"residual {di['residual']}")

    # How much of v2's margin over v1 is corpus-specific, and what that difference is
    # worth. The pair of gains is the measurement; the difference between them is what
    # the prose wants to call the fit, and it is reported here beside the spread a
    # dashboard-level jackknife puts on it so that the prose cannot quote the one
    # without the other. On both languages the difference is smaller than its spread.
    for key, qs, lang, base in (
            ("gain_gf_in_sample", gf_train, "promql", set()),
            ("gain_gf_held", [q for q in held if q.source == "grafana"
                              and q.lang == "promql"], "promql", forms["promql"]),
            ("gain_dd_in_sample", dd_train, "datadog", set()),
            ("gain_ex_held", [q for q in held
                              if q.source == "integrations-extras"
                              and q.lang == "datadog"], "datadog", forms["datadog"])):
        if qs:
            res[key] = g = gain_spread(qs, lang, base)
            print(f"{key}: {g['gain_points']} +/- {g['gain_se_points']} points over "
                  f"{g['distinct']:,} strings in {g['dashboards']} dashboards")
    for name, a, b in (("shrink_gf", "gain_gf_in_sample", "gain_gf_held"),
                       ("shrink_dd", "gain_dd_in_sample", "gain_ex_held")):
        if a in res and b in res:
            d = res[a]["gain"] - res[b]["gain"]
            se = math.hypot(res[a]["gain_se"], res[b]["gain_se"])
            res[name] = dict(points=round(d, 1), se_points=round(se, 1))
            print(f"{name}: the margin shrinks {d:.1f} points, "
                  f"spread {se:.1f} --- {'' if abs(d) > 2 * se else 'not '}"
                  f"distinguishable from no shrink at all")

    ex_qs = [q for q in held if q.source == "integrations-extras"
             and q.lang == "datadog"]
    if ex_qs:
        res["feature_set_seen"] = f = feature_set_seen(ex_qs, dd_train)
        print(f"feature sets the corpus already exhibits: {f['seen']:,}/"
              f"{f['occurrences']:,} = {f['pct']}% of held-out occurrences")

    rows = [[rf"Grafana, the corpus's ${gi['dashboards']}$",
             f"{gi['queries']:,}", f"{gi['distinct']:,}", pct(gi["raw_pct"] / 100),
             pct(gi["v1_pct"] / 100), pct(gi["v2_pct"] / 100), f"{gi['residual']:,}"]]
    for source, lang, label in (
            ("grafana", "promql", r"Grafana, @N it never saw"),
            ("@dd-in-sample", "datadog",
             r"\texttt{integrations-core}, the corpus's $@N$"),
            ("integrations-extras", "datadog",
             r"\texttt{integrations-extras}, @N others"),
            # Last, and not indented. An indented row in this table restricts the row
            # above it; this one is the complement of the corpus's own rows --- the
            # strings a later commit added, which are by construction the ones the
            # corpus does not contain --- so indenting it under them said the opposite
            # of what it measures.
            ("integrations-core", "datadog",
             r"\texttt{integrations-core}, @N at a later commit")):
        if source == "@dd-in-sample":
            m = di
            rows.append([label.replace("@N", f"{m['dashboards']}"),
                         f"{m['queries']:,}", f"{m['distinct']:,}",
                         pct(m["raw_pct"] / 100), pct(m["v1_pct"] / 100),
                         pct(m["v2_pct"] / 100), f"{m['residual']:,}"])
            continue
        qs = [q for q in held if q.source == source and q.lang == lang]
        if not qs:
            print(f"\n{label}: nothing held out")
            continue
        m = measure(qs, lang, vocab[lang], forms[lang],
                    translator=sstar if lang == "datadog" else None)
        res[source] = m
        print(f"\n{label.replace('@N', str(m['dashboards']))}: """
              f"{m['dashboards']} dashboards, {m['queries']:,} queries, "
              f"{m['distinct']:,} distinct")
        print(f"  raw {m['raw_pct']}%  v1 {m['v1_pct']}%  v2 {m['v2_pct']}%  "
              f"residual {m['residual']} {m['blockers']}")
        print(f"  of those, {m['shared_distinct']:,} strings the corpus also holds; "
              f"on the {m['novel_distinct']:,} it does not: "
              f"v1 {m['novel_v1_pct']}%  v2 {m['novel_v2_pct']}%")
        print(f"  residual by kind: {m['residual_kinds']}")
        for which, cl in (("corpus", m["corpus_vocabulary"]),
                          ("translator", m.get("translator_vocabulary"))):
            if cl:
                print(f"  {which} vocabulary ({cl['size']} features): "
                      f"{cl['occ_pct']}% of occurrences, {cl['novel_occurrences']} "
                      f"outside it, novel {list(cl['novel_features'])}")
        for r in m["residual_examples"]:
            print(f"    [residual] {r}")
        rows.append([label.replace("@N", f"${m['dashboards']}$"),
                     f"{m['queries']:,}", f"{m['distinct']:,}",
                     pct(m["raw_pct"] / 100), pct(m["v1_pct"] / 100),
                     pct(m["v2_pct"] / 100), f"{m['residual']:,}"])
        if m["shared_distinct"]:
            rows.append([r"\quad its strings the corpus lacks", "---",
                         f"{m['novel_distinct']:,}", "---",
                         pct(m["novel_v1_pct"] / 100), pct(m["novel_v2_pct"] / 100),
                         "---"])

    if rows:
        tabular("lrrrrrr",
                ["Population", "Queries", "Distinct", "Raw", "v1", "v2",
                 "Residual"], rows,
                TABLES / "heldout_coverage.tex")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "heldout_eval.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")

    fig = figures(res)
    print("\nregistry:")
    for k_, v_ in sorted(fig.items()):
        print(f"  {k_:<28} {v_}")
    if args.write_registry:
        reg = json.loads(REGISTRY.read_text())
        reg.update(fig)
        REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
        print(f"\nmerged {len(fig)} keys into {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
r"""How much of the coverage ceiling is a fit to this corpus? A cross-validated answer.

Section 7 concedes that three things were fitted to `data/corpus/queries.jsonl`: grammar
v2's productions, the translator's feature vocabulary, and the greedy implementation
order. The grammar is handwritten and cannot be refitted by a script, so it is tested
out of sample instead, on dashboards nobody here has read (`scripts/heldout_eval.py`).
The other two *are* functions of the corpus, so they can be refitted on part of it and
scored on the rest, which is what this script does.

The fit it models is the one section 7 names: support exactly the features the
dashboards you read contain. Split the dashboards, take the vocabulary of one half, and
ask what share of the other half's queries use only features in it. Coverage on the
half you read is 100% by construction, so the shortfall on the half you did not is the
optimism of the ceiling --- the amount by which "what this corpus asks for, once someone
has read it" exceeds what a fresh corpus would ask for.

Splitting is by dashboard, never by query: queries from one dashboard share a house
style, and a split that let them straddle would leak that style across the boundary and
flatter the estimate.

Nothing here touches `src/qlport/`. The point of the exercise is lost if it does.

Usage: .venv/bin/python scripts/heldout_split.py [--write-registry] [--reps N]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")                                                # noqa: E402
import matplotlib.pyplot as plt                                      # noqa: E402

from qlport import corpus as c                                       # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "queries.jsonl"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "paper" / "figures"
REGISTRY = ROOT / "paper" / "quoted_figures.json"

LANGS = ("datadog", "promql")
SEED = 20261121
# The staircase cross-validation runs over all five, for the reason `staircase_cv`
# gives: at one seed its printed figure was a seed artefact. `SEED` still drives
# `vocabulary_cv`, whose figures are means over far less variable quantities and are
# printed with their own standard deviation beside them.
# The first is the one this figure was shipped at (`SEED + 1`, as the single-seed
# version spelled it), so the change here is the pooling and nothing else. The rest are
# `heldout_eval.py`'s bootstrap seeds, because two seeded scripts drawing from two
# private lists is one list too many.
CV_SEEDS = (20261122, 1, 2, 7, 12345)
FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------
# the corpus, as feature sets grouped by dashboard
# ---------------------------------------------------------------------------
def load() -> dict[str, dict[str, list[frozenset[str]]]]:
    """lang -> dashboard -> one feature set per query occurrence in it."""
    out: dict[str, dict[str, list[frozenset[str]]]] = {
        lang: defaultdict(list) for lang in LANGS}
    for q in c.read_queries(CORPUS):
        if q.lang not in LANGS:
            continue
        # Notebook 00 drops Loki panels before it counts anything, because LogQL is no
        # evidence about PromQL; the same queries have to leave here or the PromQL
        # vocabulary would include features of another language.
        if q.lang == "promql" and not c.is_promql(q.text):
            continue
        out[q.lang][f"{q.source}:{q.origin}"].append(c.features(q.lang, q.text))
    return {lang: dict(d) for lang, d in out.items()}


def greedy(sets: list[frozenset[str]]) -> list[str]:
    """The greedy maximum-coverage feature order of notebook 00, as a plain list.

    Mirrors `greedy_order` there: at each step take the feature that completes the most
    queries outright, falling back to the most frequent residual feature when nothing
    completes anything. `test_paper_tables.py` checks this reproduces notebook 00's
    recorded staircase on the whole corpus, so the two cannot drift apart.
    """
    missing = [set(f) for f in sets]
    chosen: list[str] = []
    while True:
        completes: Counter = Counter()
        residual: Counter = Counter()
        for m in missing:
            if len(m) == 1:
                completes[next(iter(m))] += 1
            for f in m:
                residual[f] += 1
        if not residual:
            return chosen
        pick = (completes.most_common(1)[0][0] if completes
                else residual.most_common(1)[0][0])
        for m in missing:
            m.discard(pick)
        chosen.append(pick)


def covered(sets: list[frozenset[str]], support: set[str]) -> int:
    return sum(1 for f in sets if f <= support)


def staircase(sets: list[frozenset[str]], order: list[str]) -> list[float]:
    """Share of `sets` fully covered after each prefix of `order`, as percentages."""
    out, support = [], set()
    total = len(sets) or 1
    out.append(covered(sets, support) / total * 100)
    for f in order:
        support.add(f)
        out.append(covered(sets, support) / total * 100)
    return out


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------
def vocabulary_cv(by_dash: dict[str, list[frozenset[str]]], reps: int) -> list[dict]:
    """For each train fraction: refit the vocabulary on a sample of dashboards, score
    the rest. Coverage on the training half is 100% by construction, so the only number
    worth reporting is the one on the half the vocabulary never saw."""
    names = sorted(by_dash)
    rng = random.Random(SEED)
    rows = []
    for frac in FRACTIONS:
        k = max(1, round(len(names) * frac))
        occ, novel, vocab = [], [], []
        for _ in range(reps):
            train_names = set(rng.sample(names, k))
            train = [f for n in train_names for f in by_dash[n]]
            test = [f for n in names if n not in train_names for f in by_dash[n]]
            if not test:
                continue
            support = set().union(*train) if train else set()
            occ.append(covered(test, support) / len(test) * 100)
            novel.append(len(set().union(*test) - support))
            vocab.append(len(support))
        rows.append(dict(
            fraction=frac, dashboards=k, reps=len(occ),
            vocab_mean=round(st.mean(vocab), 2),
            occ_mean=round(st.mean(occ), 2), occ_sd=round(st.stdev(occ), 2),
            occ_p05=round(sorted(occ)[int(0.05 * len(occ))], 2),
            occ_p95=round(sorted(occ)[int(0.95 * len(occ)) - 1], 2),
            novel_mean=round(st.mean(novel), 2),
            novel_max=max(novel)))
    return rows


def staircase_cv(by_dash: dict[str, list[frozenset[str]]], reps: int,
                 frac: float = 0.5, at: int = 16) -> dict:
    """Refit the greedy order on half the dashboards; score both halves at every size.

    The train curve is section 7's staircase recomputed on a sample. The test curve is
    what the same build order would have delivered on dashboards it was not chosen
    from, and the vertical distance between them at a given size is the part of the
    staircase that is a property of this corpus rather than of the language.

    OVER FIVE SEEDS, AND THE EXACT MEANS KEPT. This ran at one seed and the paper
    printed the distance at `at` features as "a gap of one point". At another seed it
    is 1.2: the Monte-Carlo error on the mean of 200 replicates is 0.18, which is
    nearly twice the last digit printed, and the replicate-level standard deviation is
    2.5, so a third of individual half-splits show no gap at all. Pooling five seeds
    and storing both spreads is what lets the paper say which of those it means. The
    same seed artefact was corrected in `heldout_eval.py` for the bootstrap in the
    pass before this one, and this script -- the other seeded one -- was not touched.

    The gap is also taken from the exact means rather than from the two curves after
    rounding. That made no difference at this run's values and it is the shape of the
    error that put a tenth of a point into three printed figures.
    """
    names = sorted(by_dash)
    k = max(1, round(len(names) * frac))
    width = len(set().union(*(f for n in names for f in by_dash[n])))
    train_runs, test_runs, by_seed = [], [], []
    for seed in CV_SEEDS:
        rng = random.Random(seed)
        tr_runs, te_runs = [], []
        for _ in range(reps):
            train_names = set(rng.sample(names, k))
            train = [f for n in train_names for f in by_dash[n]]
            test = [f for n in names if n not in train_names for f in by_dash[n]]
            if not test:
                continue
            order = greedy(train)
            tr = staircase(train, order)
            te = staircase(test, order)
            pad = width + 1 - len(tr)
            tr_runs.append(tr + [tr[-1]] * pad)
            te_runs.append(te + [te[-1]] * pad)
        if at <= width:
            by_seed.append(st.mean(r[at] for r in tr_runs)
                           - st.mean(r[at] for r in te_runs))
        train_runs += tr_runs
        test_runs += te_runs
    mean = lambda runs, i: st.mean(r[i] for r in runs)  # noqa: E731
    gaps = [t[at] - s[at] for t, s in zip(train_runs, test_runs)] if at <= width else []
    out = dict(
        fraction=frac, reps=len(train_runs), seeds=list(CV_SEEDS), width=width,
        train=[round(mean(train_runs, i), 2) for i in range(width + 1)],
        test=[round(mean(test_runs, i), 2) for i in range(width + 1)])
    if gaps:
        sd = st.stdev(gaps)
        out.update(
            at=at,
            train_at=mean(train_runs, at), test_at=mean(test_runs, at),
            gap_at=mean(train_runs, at) - mean(test_runs, at),
            # The uncertainty ON THE FIGURE PRINTED, which is the Monte-Carlo error of
            # the pooled mean and nothing else. The range of the per-seed means is
            # stored beside it as a tripwire rather than as that uncertainty: five
            # seeds chosen after the fact agreed to 0.11 while eight spanned 0.39, so
            # a range over a handful of seeds bounds the next seed by luck. That is
            # the same mistake as last pass's bootstrap spread, which bounded the mean
            # it was quoted beside by accident.
            gap_at_mc_se=sd / len(gaps) ** 0.5,
            gap_at_seed_range=max(by_seed) - min(by_seed),
            gap_at_per_seed=[round(g, 4) for g in by_seed],
            gap_at_replicate_sd=sd,
            gap_at_share_not_positive=sum(1 for g in gaps if g <= 0) / len(gaps))
    return out


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def figure(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    style = {"datadog": dict(color="#1f77b4", marker="o"),
             "promql": dict(color="#d62728", marker="s")}

    ax = axes[0]
    for lang in LANGS:
        rows = res[lang]["vocabulary"]
        xs = [r["fraction"] * 100 for r in rows]
        ys = [r["occ_mean"] for r in rows]
        ax.plot(xs, ys, label=lang, markersize=3.5, lw=1.2, **style[lang])
        ax.fill_between(xs, [r["occ_p05"] for r in rows],
                        [r["occ_p95"] for r in rows],
                        color=style[lang]["color"], alpha=0.15, lw=0)
    ax.set_xlabel("dashboards read (% of corpus)")
    ax.set_ylabel("coverage of the rest (%)")
    ax.set_title("vocabulary refitted on a sample", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for lang in LANGS:
        s = res[lang]["staircase"]
        xs = range(len(s["train"]))
        ax.plot(xs, s["train"], lw=1.2, color=style[lang]["color"],
                label=f"{lang}, read")
        ax.plot(xs, s["test"], lw=1.2, ls="--", color=style[lang]["color"],
                label=f"{lang}, unread")
    ax.set_xlabel("features implemented (greedy order on the read half)")
    ax.set_ylabel("queries covered (%)")
    ax.set_title("the staircase, scored both ways", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="lower right")

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "heldout_vocabulary.pdf")
    print(f"wrote {(FIGURES / 'heldout_vocabulary.pdf').relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--write-registry", action="store_true")
    args = ap.parse_args()

    data = load()
    res: dict = {}
    for lang in LANGS:
        by_dash = data[lang]
        n_q = sum(len(v) for v in by_dash.values())
        print(f"{lang}: {len(by_dash)} dashboards, {n_q:,} query occurrences")
        res[lang] = dict(dashboards=len(by_dash), occurrences=n_q,
                         vocabulary=vocabulary_cv(by_dash, args.reps),
                         staircase=staircase_cv(by_dash, args.reps))
        print(f"  {'read':>6} {'vocab':>6} {'unread coverage':>17} "
              f"{'novel features':>15}")
        for r in res[lang]["vocabulary"]:
            print(f"  {r['fraction']*100:5.0f}% {r['vocab_mean']:6.1f} "
                  f"{r['occ_mean']:9.1f}% +-{r['occ_sd']:<5.1f} "
                  f"{r['novel_mean']:10.1f} "
                  f"(max {r['novel_max']})")

    # The full-corpus greedy order, as a check that this script's greedy reproduces
    # notebook 00's recorded staircase rather than some near neighbour of it.
    full = {lang: staircase([f for v in data[lang].values() for f in v],
                            greedy([f for v in data[lang].values() for f in v]))
            for lang in LANGS}
    res["full_staircase"] = {lang: [round(x, 1) for x in s] for lang, s in full.items()}

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "heldout_cv.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    figure(res)

    half = {lang: next(r for r in res[lang]["vocabulary"] if r["fraction"] == 0.5)
            for lang in LANGS}
    fig = {}
    for lang, r in half.items():
        p = "dd" if lang == "datadog" else "pq"
        rows = res[lang]["vocabulary"]
        tenth = next(x for x in rows if x["fraction"] == 0.1)
        # The first sample size at which no replicate found a feature the sample missed.
        done = next((x["fraction"] for x in rows if x["novel_max"] == 0), None)
        fig[f"cv.{p}_half_coverage_pct"] = round(r["occ_mean"], 1)
        fig[f"cv.{p}_half_optimism_pct"] = round(100 - r["occ_mean"], 1)
        fig[f"cv.{p}_half_vocab"] = round(r["vocab_mean"], 1)
        fig[f"cv.{p}_tenth_coverage_pct"] = round(tenth["occ_mean"], 1)
        fig[f"cv.{p}_tenth_coverage_sd"] = round(tenth["occ_sd"], 1)
        fig[f"cv.{p}_tenth_vocab"] = round(tenth["vocab_mean"], 1)
        fig[f"cv.{p}_tenth_dashboards"] = tenth["dashboards"]
        fig[f"cv.{p}_tenth_novel_features"] = round(tenth["novel_mean"], 1)
        if done is not None:
            fig[f"cv.{p}_complete_at_pct"] = round(done * 100)
        fig[f"cv.{p}_dashboards"] = res[lang]["dashboards"]
        fig[f"cv.{p}_vocabulary"] = int(res[lang]["staircase"]["width"])
    fig["cv.reps"] = args.reps
    s = res["datadog"]["staircase"]
    # From the exact means, not from the two curves after rounding, and with both
    # spreads stored: across seeds, and across the replicates themselves.
    fig["cv.dd_stair_gap_16_pct"] = round(s["gap_at"], 1)
    fig["cv.dd_stair_train_16_pct"] = round(s["train_at"], 1)
    fig["cv.dd_stair_test_16_pct"] = round(s["test_at"], 1)
    fig["cv.dd_stair_gap_16_mc_se"] = round(s["gap_at_mc_se"], 2)
    fig["cv.dd_stair_gap_16_seed_range"] = round(s["gap_at_seed_range"], 2)
    fig["cv.dd_stair_gap_16_replicate_sd"] = round(s["gap_at_replicate_sd"], 1)
    fig["cv.dd_stair_gap_16_share_not_positive_pct"] = round(
        s["gap_at_share_not_positive"] * 100)
    print("\nregistry:")
    for k, v in sorted(fig.items()):
        print(f"  {k:<34} {v}")
    if args.write_registry:
        reg = json.loads(REGISTRY.read_text())
        reg.update(fig)
        REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
        print(f"\nmerged {len(fig)} keys into {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

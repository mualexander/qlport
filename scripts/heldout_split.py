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
# The first is the one the staircase figure was shipped at (`SEED + 1`, as the
# single-seed version spelled it). The rest are `heldout_eval.py`'s bootstrap seeds,
# because two seeded scripts drawing from two private lists is one list too many.
#
# Both cross-validations pool over all five. What the replicate count bought, measured:
# the staircase gap read 1.19 at 200 replicates per seed, 1.04 at 1,000 and 1.06 at
# 5,000, with a Monte-Carlo error of 0.02 at the last of those. Which is to say 200 per
# seed was never enough to print a figure from, and that one decimal place is not a
# resolution this procedure has at any budget worth running -- so the registry carries
# the gap to two places and the paper quotes it to none.
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
        # Ties broken by name, not by insertion order. `most_common` returns the
        # first-inserted of an equal-count group, and these Counters are fed by
        # iterating `m`, a set of strings, so the insertion order followed Python's
        # per-process hash salt. Two runs of this function on the same seed therefore
        # chose different features at a tie, which is about one tie per half-split:
        # the cached cross-validation figures could not be reproduced by the code that
        # wrote them, and the guard over them pinned the registry to the cache rather
        # than to a derivation, so it could not notice.
        #
        # THE WHOLE-CORPUS ORDER IS NOT TIE-FREE, and the first version of this comment
        # said it was. There is exactly one tie in it, at the fifth pick, between two
        # features that each complete one query; resolving it the other way delays the
        # step in section 7's staircase by a feature and lowers every ceiling past it,
        # which is what three drafts printed. So this is not a change confined to the
        # cross-validation: twelve of the registry's `nb00.ceiling_*` keys moved with
        # it. `test_the_one_tie_in_the_whole_corpus_order_is_broken_the_defensible_way`
        # pins both the count of ties and which feature this rule takes at it.
        source = completes if completes else residual
        pick = min(sorted(source.items()), key=lambda kv: (-kv[1], kv[0]))[0]
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
    rows = []
    for frac in FRACTIONS:
        k = max(1, round(len(names) * frac))
        occ, novel, vocab = [], [], []
        # Over the same five seeds the staircase uses. These statistics are means and
        # so are not sensitive to which seed produced them -- but `novel_max` is a
        # MAXIMUM, and `cv.*_complete_at_pct` is read off where it first reaches zero,
        # which makes it the least stable figure in this paragraph and the one most
        # sensitive to how many replicates were drawn. The comment that used to sit at
        # `CV_SEEDS` justified leaving this at one seed on the grounds that its figures
        # are "printed with their own standard deviation beside them" -- true of the
        # coverage means, false of the one figure that needed it.
        for seed in CV_SEEDS:
            rng = random.Random(seed)
            for _ in range(reps):
                train_names = set(rng.sample(names, k))
                train = [f for n in sorted(train_names) for f in by_dash[n]]
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

    IT WAS NOT REPRODUCIBLE AT ALL UNTIL THIS PASS. `greedy` broke ties between
    features completing the same number of queries by taking the first out of a
    `Counter`, whose insertion order came from iterating a `set` of dashboard names.
    CPython salts string hashing per process, so the same seed gave a different build
    order in a different interpreter: three runs of the shipped seed gave three
    answers at `at` features and none of them matched the committed cache. Ties now
    break on the feature name. A test in the suite runs this path in two subprocesses
    at different values of PYTHONHASHSEED and requires one answer; it fails on the
    old tie-break.

    That is also what the previous draft's seed spread was measuring. It stored a
    0.23-point range over five seeds AS A TRIPWIRE AGAINST SEED ARTEFACTS, and most
    of the range was the hash salt rather than the seeds: broken deterministically
    the five per-seed means span 0.08. A tripwire that fires on the wrong thing reads
    as a clean bill of health for the thing it was pointed at.

    OVER FIVE SEEDS AT 5,000 REPLICATES EACH, AND THE EXACT MEANS KEPT. At 200
    replicates -- what this ran at for three drafts -- the Monte-Carlo error is 0.18
    against a printed last digit of 0.1, so the tenth of a point was never resolved
    whatever the tie-break did. Pooled, the gap is 1.06 with an error of 0.02, which
    does not settle a first decimal place: 1.0 and 1.1 are both inside it. So the
    registry carries the gap to two places with its error beside it, and the paper
    leans on the replicate standard deviation -- 2.6 points, a hundred and thirty
    times the error on the mean -- because that is the figure that describes one
    reader's split rather than the average of twenty-five thousand.

    The gap is also taken from the exact means rather than from the two curves after
    rounding, which is why `cv.dd_stair_gap_16_points` is not the difference of the
    two printed percentages and must not be checked as though it were.
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
            train = [f for n in sorted(train_names) for f in by_dash[n]]
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
            # stored beside it as a tripwire rather than as that uncertainty: a range
            # over a handful of seeds bounds the next seed by luck, which is the same
            # mistake as the bootstrap spread corrected two passes ago. It earned its
            # keep by failing -- it read 0.23 while the tie-break was salt-dependent
            # and 0.08 once it was not -- but what it bounds is the seeds, and the
            # suite now checks it against what the replicate spread predicts for it
            # rather than treating it as an error bar on anything.
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


def figures(res: dict) -> dict:
    """Every `cv.*` registry key this run produces, from the cached run alone.

    Lifted out of `main` so the suite can call it on the committed cache and
    compare key by key, which is what `heldout_eval.figures` already allowed for
    the `ho.*` keys. Nothing did that here, and it showed: renaming
    `cv.dd_stair_gap_16_pct` to `..._points` left the old key in the registry,
    because `--write-registry` merges rather than replaces. A dead key is a figure
    the paper can quote and no run stands behind, and it also licenses any prose
    numeral of the same magnitude through the figure ledger.
    """
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
    # The POOLED count, not `args.reps`: `vocabulary_cv` draws `reps` replicates
    # per seed over five seeds, so the band in the figure is five times the
    # argument. The caption quotes this key and said 200 of a band over 1,000.
    fig["cv.reps"] = res["datadog"]["vocabulary"][0]["reps"]
    s = res["datadog"]["staircase"]
    # From the exact means, not from the two curves after rounding, and with both
    # spreads stored: across seeds, and across the replicates themselves.
    # To two places, and not to one. The one-decimal figure was the whole trouble:
    # 1.2 at 200 replicates per seed, 1.0 at 1,000, 1.06 at 5,000, and a Monte-Carlo
    # interval that straddles the rounding boundary at every budget. So the registry
    # carries the mean and its error and the paper quotes neither to a tenth.
    fig["cv.dd_stair_gap_16_points"] = round(s["gap_at"], 2)
    fig["cv.dd_stair_gap_16_mc_se"] = round(s["gap_at_mc_se"], 2)
    fig["cv.dd_stair_train_16_pct"] = round(s["train_at"], 1)
    fig["cv.dd_stair_test_16_pct"] = round(s["test_at"], 1)
    fig["cv.dd_stair_gap_16_seed_range"] = round(s["gap_at_seed_range"], 2)
    fig["cv.dd_stair_gap_16_replicate_sd"] = round(s["gap_at_replicate_sd"], 1)
    fig["cv.dd_stair_gap_16_share_not_positive_pct"] = round(
        s["gap_at_share_not_positive"] * 100)
    fig["cv.stair_reps"] = s["reps"]
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    # The staircase cross-validation runs far more replicates than the vocabulary one,
    # because the quantity it measures is a difference of two noisy curves and the
    # vocabulary figures are not. At 200 per seed the gap read 1.2, at 1,000 it read
    # 1.0 and at 5,000 it reads 1.06 with a Monte-Carlo error of 0.02 -- which is to
    # say that 200 per seed was not enough to put a figure in a paper, and that one
    # decimal place is not a resolution this procedure has at any budget we would run.
    ap.add_argument("--stair-reps", type=int, default=5000)
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
                         staircase=staircase_cv(by_dash, args.stair_reps))
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

    fig = figures(res)
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

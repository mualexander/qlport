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
# # 02 — Ingestion mapping and exact rollup translation (paper §3)
#
# **Scenario.** A company sends counters to Datadog and wants to send them to Prometheus
# instead. Going forward, the same source counter reaches both systems:
# Prometheus stores the cumulative value; Datadog stores per-interval deltas, $\phi(D)$.
#
# **Questions.**
# 1. Which PromQL expression reproduces Datadog's `.as_rate()` rollup exactly?
# 2. How far off is the "obvious" translation, `rate()`?
# 3. How do counter resets affect each candidate?
#
# Outputs: `paper/figures/rate_error_vs_window.pdf`,
# `paper/tables/model_verification.tex`, `paper/tables/rollup_translation_live.tex`.

# %%
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qlport.harness.workload import bursty, poisson, scrape, steady
from qlport.metrics import reldiff
from qlport.semantics import datadog_eval as dd
from qlport.semantics import promql_eval as pq
from qlport.semantics.ingest import phi_series

ROOT = Path.cwd() if (Path.cwd() / "paper").exists() else Path.cwd().parent
TABLES, FIGURES = ROOT / "paper" / "tables", ROOT / "paper" / "figures"
HOUR = 3600

# %% [markdown]
# ## Workloads
# Six hours each, 15 s scrapes with ±2 s jitter. "Bursty" adds a 10× spike for two minutes
# every 30 minutes (deploys, incidents): exactly the windows in which alerts fire.

# %%
restarts = tuple(np.arange(1800, 6 * HOUR, HOUR))
WORKLOADS = {
    "steady":            dict(rate_fn=steady(10), jitter=0),
    "poisson":           dict(rate_fn=poisson(10), jitter=2),
    "bursty":            dict(rate_fn=bursty(10, 1800, 120, 10), jitter=2),
    "bursty + restarts": dict(rate_fn=bursty(10, 1800, 120, 10), jitter=2, restarts=restarts),
}

CANDIDATES = {
    "rate(x[Δ])":                 lambda s, t, D: pq.rate(s.ts, s.vs, t, D),
    "(x - x offset Δ) / Δ":       lambda s, t, D: pq.offset_diff_rate(s.ts, s.vs, t, D),
    "increase(x[Δ] anchored) / Δ": lambda s, t, D: pq.increase_anchored(s.ts, s.vs, t, D) / D,
}


def compare(series, interval, reset="zero"):
    """Each candidate against Datadog's bucket, bucket by bucket.

    Datadog's bucket reported at `b` covers `[b, b + interval)` — measured, not assumed
    (`scripts/trial_checks.py e2`). A PromQL range selector covers `(u - interval, u]`,
    so the candidate is evaluated *inside* the bucket rather than on its right edge: at
    `b + interval` the window would take a sample sitting exactly on the boundary that
    Datadog assigns to the next bucket, which with a 15 s scrape and a 60 s rollup is
    every bucket.
    """
    p = phi_series(series, reset)
    rows = []
    for b in dd.bucket_starts(600, series.ts[-1] - interval, interval):
        truth = dd.as_rate(p.ts, p.vs, b, interval)
        u = dd.eval_instant(b, interval)
        for name, f in CANDIDATES.items():
            rows.append(dict(t=b, candidate=name, err=reldiff(f(series, u, interval), truth)))
    return pd.DataFrame(rows)


# %%
results = []
for wname, kw in WORKLOADS.items():
    s = scrape(start_value=5000, interval=15, duration=6 * HOUR, seed=681, **kw)
    for interval in (60, 300):
        df = compare(s, interval)
        df["workload"], df["interval"] = wname, interval
        results.append(df)
results = pd.concat(results, ignore_index=True).dropna()

summary = (results.groupby(["workload", "interval", "candidate"], sort=False)["err"]
           .agg(exact=lambda e: (e < 1e-12).mean(), median="median", max="max")
           .reset_index())
summary

# %% [markdown]
# ## The modelled summary
#
# This is the version §3 *used* to cite, computed from our own PromQL evaluator. The
# live table below replaces it, read off a real server, and the paper shows only that
# one. What this cell is still for is the comparison between the two: §3 claims they
# agree on every summary statistic, so the modelled numbers have to exist to be
# compared against. They are no longer written to `paper/tables/`, because a generated
# file the paper does not `\input` is a cell computing for nobody.

# %%
TABLES.mkdir(parents=True, exist_ok=True)
print(summary.to_string(index=False))

# %% [markdown]
# ## How the `rate()` error shrinks with the window
# `rate()` is the only candidate whose error depends on the data; plot it against Δ/scrape.

# %%
widths = [30, 60, 120, 300, 600, 900]
fig, ax = plt.subplots(figsize=(4.5, 2.8))
for wname in ("poisson", "bursty"):
    s = scrape(start_value=5000, interval=15, duration=6 * HOUR, seed=681, **WORKLOADS[wname])
    p = phi_series(s)
    med, p95 = [], []
    for D in widths:
        e = np.array([reldiff(pq.rate(s.ts, s.vs, dd.eval_instant(b, D), D),
                              dd.as_rate(p.ts, p.vs, b, D))
                      for b in dd.bucket_starts(600, s.ts[-1] - D, D)])
        e = e[~np.isnan(e)]
        med.append(np.median(e)); p95.append(np.percentile(e, 95))
    ax.plot(widths, med, marker="o", label=f"{wname} (median)")
    ax.plot(widths, p95, marker="o", linestyle="--", label=f"{wname} (p95)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"rollup interval $\Delta$ (s), scrape = 15 s")
ax.set_ylabel(r"$d$ vs. Datadog")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / "rate_error_vs_window.pdf")

# %% [markdown]
# ## Against a real Prometheus
# Everything above reads PromQL through `semantics/promql_eval.py`, which is a *model*.
# `increase(x[Δ] anchored)` was written from the text of Prometheus proposal 0052, so a
# claim about it was a claim about our own code.
#
# `scripts/verify_anchored.py` closes that: it renders each workload as OpenMetrics,
# backfills it into a TSDB with `promtool tsdb create-blocks-from openmetrics`, serves it
# from a pinned Prometheus with no scrape targets, and reads each candidate back with one
# range query evaluated on the same epoch-aligned grid used above. Responses are cached in
# `data/raw/anchored/`, which is all this notebook reads.
#
# Two separate questions follow, and they are worth keeping apart: whether our model of
# PromQL is right (model vs. server), and whether the translation is right (server vs.
# Datadog). Only the second is a claim about the paper's thesis; the first is a claim
# about our instruments.

# %%
import glob

from qlport.harness.prometheus import read_raw, series_from_response
from qlport.semantics.model import Series

MODELS = {
    "rate(x[Δ])": lambda s, t, D: pq.rate(s.ts, s.vs, t, D),
    "(x - x offset Δ) / Δ": lambda s, t, D: pq.offset_diff_rate(s.ts, s.vs, t, D),
    "increase(x[Δ] anchored) / Δ": lambda s, t, D: pq.increase_anchored(s.ts, s.vs, t, D) / D,
}

# Selected by WHAT THE RUN HOLDS, not by the glob. `verify_anchored.py --no-anchor`
# writes into this directory too, and its candidate set is the unanchored pair rather
# than the three models above, so a bare `*.json` picked it up and the loop below died
# on `KeyError: 'rate(x[D])'` -- before every write in this notebook, which took
# `make notebooks` and therefore `make all` down with it and left three committed
# tables unreproducible. The crash was in the repository for as long as the fifth run
# was. A run that carries all three keys belongs here whatever it is called; one that
# does not is somebody else's experiment.
_WANT = {k.replace("Δ", "D") for k in MODELS}
LIVE = [p for p in sorted(glob.glob(str(ROOT / "data" / "raw" / "anchored" / "*.json")))
        if _WANT <= set(read_raw(p).get("candidates", {}))]
_SKIPPED = len(sorted(glob.glob(str(ROOT / "data" / "raw" / "anchored" / "*.json")))) - len(LIVE)
print(f"{len(LIVE)} cached runs, {_SKIPPED} skipped for carrying other candidates"
      if LIVE else "no cache: run `python scripts/verify_anchored.py` (needs Docker)")

# %%
rows = []
for path in LIVE:
    rec = read_raw(path)
    t0, D = rec["t0"], rec["interval"]
    s = Series({}, np.array(rec["series"]["ts"]), np.array(rec["series"]["vs"]),
               kind="counter")
    p = phi_series(s)
    for name, model in MODELS.items():
        key = name.replace("Δ", "D")
        eps = rec["grid"]["eps"]
        for t_abs, server in series_from_response(rec["candidates"][key]["response"]).items():
            # The server was evaluated one epsilon inside a bucket's right edge, so the
            # Datadog bucket it answers for starts an interval earlier. Snap to the
            # millisecond Prometheus actually used; the float it reports back carries
            # dirt that `rate`'s extrapolation magnifies into a spurious 1e-9.
            t = round((t_abs - t0) * 1000) / 1000
            b = round(t - D + eps, 3)
            rows.append(dict(workload=rec["workload"], interval=int(D), candidate=name,
                             version=rec["prometheus_version"],
                             model_err=reldiff(server, model(s, t, D)),
                             truth_err=reldiff(server, dd.as_rate(p.ts, p.vs, b, D))))
live = pd.DataFrame(rows)
live.groupby("candidate")[["model_err", "truth_err"]].max()

# %% [markdown]
# ### Is our model of PromQL right? (RQ-hygiene, not a result)
# One row per candidate, over every bucket of every workload. `NaN mismatch` counts the
# buckets where the server returned a value and the model declined to, or the reverse:
# an empty cell is as much a semantic claim as a number, so it is counted too.

# %%
ver = []
for name, sub in live.groupby("candidate", sort=False):
    finite = sub.model_err.dropna()
    ver.append(f"\\texttt{{{name.replace('Δ', '$\\Delta$')}}} & {len(sub):,} & "
               f"{finite.max():.2e} & {int(sub.model_err.isna().sum())} \\\\")
lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"PromQL candidate & Buckets & Max $d$ vs. server & NaN mismatch \\",
         r"\midrule", *ver, r"\bottomrule", r"\end{tabular}"]
(TABLES / "model_verification.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
version = live.version.iloc[0] if len(live) else "n/a"
print(f"\nverified against Prometheus {version}")

# %% [markdown]
# ### Is the translation right? (the result)
# The same comparison as the table above, but with the PromQL side read from the server
# instead of computed by us. This is the version §3 should cite.

# %%
live_summary = (live.groupby(["workload", "interval", "candidate"], sort=False)["truth_err"]
                .agg(exact=lambda e: (e < 1e-12).mean(), median="median", max="max")
                .reset_index())
lines = [r"\begin{tabular}{@{}lrlrrr@{}}", r"\toprule",
         r"Workload & $\Delta$ (s) & PromQL candidate & Exact & Median $d$ & Max $d$ \\",
         r"\midrule"]
for _, r in live_summary.iterrows():
    exact = f"{r.exact:.1%}".replace("%", r"\%")
    cand = r.candidate.replace("Δ", r"$\Delta$")
    lines.append(f"{r.workload} & {int(r.interval)} & \\texttt{{{cand}}} & {exact} & "
                 f"{r['median']:.3f} & {r['max']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "rollup_translation_live.tex").write_text("\n".join(lines) + "\n")
print(live_summary.to_string(index=False))

# §3 claims the modelled and measured versions of this table agree on every summary
# statistic. That was a sentence rather than a check, and it was false: `compare()` reads
# 600 s as the first bucket *start* while `verify_anchored.bucket_grid` read it as the
# first evaluation *instant*, so the live grid began one bucket earlier -- for Δ=300 a
# bucket wholly inside the declared warm-up -- and six of the twenty-four rows differed in
# a printed cell. The grid is fixed; the claim is now asserted here.
_fmt = lambda r: (f"{r.exact:.1%}", f"{r['median']:.3f}", f"{r['max']:.3f}")
_model = {(r.workload, int(r.interval), r.candidate): _fmt(r)
          for _, r in summary.iterrows()}
_live = {(r.workload, int(r.interval), r.candidate): _fmt(r)
         for _, r in live_summary.iterrows()}
assert set(_model) == set(_live), (set(_model) ^ set(_live))
_differ = {k: (_model[k], _live[k]) for k in _model if _model[k] != _live[k]}
assert not _differ, _differ
print(f"\nmodelled and measured agree on all {len(_live)} rows, cell for cell")


# %% [markdown]
# ## Findings (to carry into §3 and §5)
# * `(x - x offset Δ) / Δ` is **exact** on every bucket without a reset, whatever the traffic shape.
# * `increase(x[Δ] anchored) / Δ` is exact **with** resets too, and this is now checked
#   against a real Prometheus rather than against our model of it: over every bucket of
#   every workload the server and `promql_eval` agree to float64 precision, with no bucket
#   where one produced a value and the other did not. The reset-aware form is therefore
#   safe to cite, and `emit/promql.py` can offer it as more than a guess.
# * `anchored` is **still** experimental. It arrived in Prometheus 3.7.0 behind
#   `--enable-feature=promql-extended-range-selectors` and is still gated by it at
#   3.15.0: a server without the flag answers "anchored modifier is experimental and not
#   enabled". Every run here passes the flag, which is why the claim went unchecked for
#   so long — `ANCHORED_VERSION` is a precondition on how the target server was started,
#   not just on its version.
# * `rate()` is ε-sound only, and its error depends on traffic shape and Δ/scrape. The
#   live table confirms it is never exact outside the perfectly steady workload.
# * ρ is settled by reading the agent, not by this notebook: `zero` on the OpenMetrics
#   path, `clamp` with the flag clear, and `drop` is not a reset policy at all. Note what
#   the restart rows still do *not* say — they compare the two PromQL forms against `φ`
#   under that `ρ`, so they show the anchored form tracks that φ, not that Datadog's
#   backend agrees.
# * The dual-write replay above now closes that for the counter rules on real traffic:
#   both exact forms are exact against Datadog itself, bucket by bucket. What it does not
#   reach is the reset case — a request counter does not restart in half an hour — so the
#   restart rows remain the evidence there.


# %% [markdown]
# ## The dual-write replay: against Datadog, not against a model of it
#
# Everything above compares a PromQL candidate against `datadog_eval`, which is our
# reading of Datadog's rollup. `scripts/replay_dualwrite.py` closes that gap. It takes
# the busiest half hour of a real 1995 web-server log, counts requests into a 15 s scrape
# grid to get a counter, and writes the *same* observations to both systems — the
# cumulative series to Prometheus, its image under φ to Datadog — then reads every
# candidate back from each and compares bucket by bucket.
#
# It runs twice. A real scrape is jittered, so its samples almost never land on a rollup
# boundary, and off the boundary `[b, b+Δ)` and `(b, b+Δ]` read the same samples: the
# closure convention is invisible. The aligned run snaps the scrape grid onto the rollup
# grid on purpose, which is the only case where the convention can be seen at all.

# %%
REPLAY = ROOT / "data" / "raw" / "replay"
replay_runs = {name: json.loads((REPLAY / f"dualwrite_{name}.json").read_text())
               for name in ("unaligned", "aligned")
               if (REPLAY / f"dualwrite_{name}.json").exists()}

# Exactness is re-derived from the recorded readings rather than read out of the run's
# own `summary`, because the two were not computed to the same standard: the script
# scored bit-identity while every other exactness column in this paper -- the
# model-versus-server check above, notebook 05's differential -- uses a 1e-12 relative
# bound. One bucket reading 236.0 against 235.99999999999997 counted as a disagreement
# and moved a published cell by 1.2 points. `replay_dualwrite.py` now records both.
EXACT_TOL = 1e-12

if replay_runs:
    rows = []
    for name, run in replay_runs.items():
        per_key: dict[tuple[str, str], list[float]] = {}
        for r in run["rows"]:
            if r.get("datadog") is None or r.get("promql") is None:
                continue
            key = (str(r["instant"]).strip(), str(r["candidate"]).strip())
            per_key.setdefault(key, []).append(reldiff(r["promql"], r["datadog"]))
        for (instant, candidate), ds in sorted(per_key.items()):
            rows.append(dict(grid=name, instant=instant, candidate=candidate,
                             buckets=len(ds),
                             exact=sum(1 for d in ds if d <= EXACT_TOL) / len(ds),
                             bit_identical=sum(1 for d in ds if d == 0) / len(ds),
                             median_d=float(np.median(ds)), max_d=max(ds)))
    replay = pd.DataFrame(rows)
    print(replay.to_string(index=False))

    order = ["x - x offset D", "increase(x[D] anchored)", "rate(x[D]) * D"]
    pretty = {"x - x offset D": r"\texttt{x - x offset $\Delta$}",
              "increase(x[D] anchored)": r"\texttt{increase(x[$\Delta$] anchored)}",
              "rate(x[D]) * D": r"\texttt{rate(x[$\Delta$])} $\times\,\Delta$"}
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Scrape grid & PromQL candidate & Evaluated inside & On the edge & Max $d$ \\",
             r"\midrule"]
    for grid in ("unaligned", "aligned"):
        sub = replay[replay.grid == grid]
        if sub.empty:
            continue
        for cand in order:
            ins = sub[(sub.candidate == cand) & (sub.instant == "inside")]
            edge = sub[(sub.candidate == cand) & (sub.instant == "on the edge")]
            if ins.empty:
                continue
            label = grid if cand == order[0] else ""
            lines.append(
                f"{label} & {pretty[cand]} & {ins.exact.iloc[0]:.1%} & "
                f"{edge.exact.iloc[0]:.1%} & {ins.max_d.iloc[0]:.3g} \\\\".replace("%", r"\%"))
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "replay_dualwrite.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {(TABLES / 'replay_dualwrite.tex').relative_to(ROOT)}")

    # §3.1 quotes these, so a re-run has to reproduce them.
    agg = replay.set_index(["grid", "instant", "candidate"])
    for grid in replay.grid.unique():
        for cand in ("x - x offset D", "increase(x[D] anchored)"):
            assert agg.loc[(grid, "inside", cand), "exact"] == 1.0, (grid, cand)
    assert agg.loc[("unaligned", "on the edge", "x - x offset D"), "exact"] == 1.0
    assert agg.loc[("aligned", "on the edge", "x - x offset D"), "exact"] < 0.5
    print("replay assertions hold")

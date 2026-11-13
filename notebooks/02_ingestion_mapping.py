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
# Outputs: `paper/tables/rollup_translation.tex`, `paper/figures/rate_error_vs_window.pdf`.

# %%
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
    p = phi_series(series, reset)
    rows = []
    for t in dd.bucket_ends(600, series.ts[-1], interval):
        truth = dd.as_rate(p.ts, p.vs, t, interval)
        for name, f in CANDIDATES.items():
            rows.append(dict(t=t, candidate=name, err=reldiff(f(series, t, interval), truth)))
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
# ## Table for the paper

# %%
def fmt_row(r):
    exact = f"{r.exact:.0%}".replace("%", r"\%")
    return f"{r.workload} & {int(r.interval)} & \\texttt{{{r.candidate}}} & {exact} & {r["median"]:.3f} & {r['max']:.3f} \\\\"

lines = [r"\begin{tabular}{@{}lrlrrr@{}}", r"\toprule",
         r"Workload & $\Delta$ (s) & PromQL candidate & Exact & Median $d$ & Max $d$ \\", r"\midrule"]
lines += [fmt_row(r).replace("Δ", r"$\Delta$") for _, r in summary.iterrows()]
lines += [r"\bottomrule", r"\end{tabular}"]
TABLES.mkdir(parents=True, exist_ok=True)
(TABLES / "rollup_translation.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:8]))

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
        e = np.array([reldiff(pq.rate(s.ts, s.vs, t, D), dd.as_rate(p.ts, p.vs, t, D))
                      for t in dd.bucket_ends(600, s.ts[-1], D)])
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
# ## Findings (to carry into §3 and §5)
# * `(x - x offset Δ) / Δ` is **exact** on every bucket without a reset, whatever the traffic shape.
# * The anchored form is exact with resets too, **under our model** of `anchored`.
#   Verify against real Prometheus (`--enable-feature=promql-extended-range-selectors`).
# * `rate()` is ε-sound only, and its error depends on traffic shape and Δ/scrape.
# * Open: which reset policy Datadog actually uses (trial-account test).

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
# # 00 — Corpus and feature frequencies (paper §7 figure; implementation order)
#
# **Plan**
# 1. Fetch kubernetes-mixin + Grafana community dashboards (PromQL) and DataDog/integrations-core assets/dashboards (Datadog) via scripts/fetch_corpus.py into data/corpus/.
# 2. Extract every query string; parse; count feature occurrences per language.
# 3. Write paper/figures/feature_frequency.pdf and data/processed/features.parquet.

# %%
# TODO

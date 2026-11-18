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
# # 05 — Corpus-scale differential testing (paper §7, RQ3)
#
# κ (notebook 03) measures what the translator is *willing* to translate. This notebook
# reports what it translates *correctly*, by reading back a run of
# `scripts/shape_differential.py`: every distinct operator shape in the corpus
# instantiated with synthetic series, written to a real Datadog account and to a local
# Prometheus, and the two evaluated bucket by bucket on the same data.
#
# The reduction that makes this affordable is the shape abstraction of `ir.shape()`:
# two queries differing only in the metric they name are one experiment, not two. Its
# soundness is checked rather than assumed — notebook 03 verifies that every query in a
# class emits one PromQL form, so a representative stands for its class.
#
# Live calls happen in the script, not here: this reads `data/raw/shapes/differential.json`
# only.
#
# Output: `paper/tables/differential.tex`.

# %%
import json
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "paper").exists() else Path.cwd().parent
RAW = ROOT / "data" / "raw" / "shapes" / "differential.json"
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def pct(x: float) -> str:
    return f"{x:.1%}".replace("%", r"\%")


def tabular(spec: str, header: list[str], rows: list[list[str]], path: Path) -> None:
    lines = [rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


# %%
run = json.loads(RAW.read_text())
rows = run["rows"]
prov = run["provenance"]
print("provenance:", prov)
print("status counts:", run["status_counts"])

# %% [markdown]
# ## What was compared, and what was not
#
# A shape is *compared* when both systems returned data for at least one bucket of the
# instantiated series. The rest are reported rather than quietly dropped: a shape whose
# lookback exceeds the run's window cannot be read in either system, and one whose
# Datadog groups never matched the PromQL groups is an unmatched comparison and not an
# agreement. Each is weighted by the corpus occurrences the shape stands for.

# %%
LABELS = {
    "no-data": "Not compared: one side returned no data",
    "unmatched": "Not compared: group keys did not line up",
    "window-too-short": "Not compared: lookback exceeds the run window",
}
by_status: dict[str, dict[str, int]] = {}
for r in rows:
    acc = by_status.setdefault(r["status"], {"shapes": 0, "occ": 0})
    acc["shapes"] += 1
    acc["occ"] += int(r["occurrences"])

total_shapes = sum(a["shapes"] for a in by_status.values())
total_occ = sum(a["occ"] for a in by_status.values())
print("shapes", total_shapes, "occurrences", total_occ)

# %% [markdown]
# ## Agreement among the shapes that were compared
#
# A bucket agrees when the two readings differ by no more than the run's tolerance.
# That tolerance is `1e-12` relative, except for shapes whose PromQL contains a `+` or
# `-` between aggregations, where subtracting two nearly equal floats leaves a relative
# difference of order `1e-11` that is catastrophic cancellation and not a translation
# error; those use `1e-9`.

# %%
cmp_rows = [r for r in rows if r["status"] == "compared"]
agree = [r for r in cmp_rows if r["exact"] == 1.0]
mismatch = [r for r in cmp_rows if r["exact"] < 1.0]
buckets = sum(int(r["buckets"]) for r in cmp_rows)
occ_cmp = sum(int(r["occurrences"]) for r in cmp_rows)
occ_ok = sum(int(r["occurrences"]) for r in agree)

assert run["shapes_compared"] == len(cmp_rows)
assert run["shapes_exact"] == len(agree)
assert run["occurrences_compared"] == occ_cmp
assert run["occurrences_exact"] == occ_ok

print(f"{len(agree)}/{len(cmp_rows)} shapes agree on every bucket")
print(f"{occ_ok:,}/{occ_cmp:,} occurrences; {buckets:,} bucket comparisons")

not_cmp = [s_ for s_ in ("no-data", "unmatched", "window-too-short") if s_ in by_status]
tabular("lrr", ["Outcome", "Shapes", "Occurrences"],
        [["Compared against both systems", f"{len(cmp_rows):,}", f"{occ_cmp:,}"],
         [r"\quad agreeing on every bucket", f"{len(agree):,}", f"{occ_ok:,}"],
         [r"\quad disagreeing somewhere", f"{len(mismatch):,}", f"{occ_cmp - occ_ok:,}"]]
        + [[LABELS[s_], f"{by_status[s_]['shapes']:,}", f"{by_status[s_]['occ']:,}"]
           for s_ in not_cmp]
        + [[r"\midrule Agreement among those compared",
            pct(len(agree) / len(cmp_rows)), pct(occ_ok / occ_cmp)]],
        TABLES / "differential.tex")

# %% [markdown]
# ## The disagreements
#
# Printed in full rather than summarised, and not written to a table: a differential
# test earns its place by what it finds, and the four defects this one found are
# described in §7 as prose because each needed explaining rather than tabulating. If a
# future run disagrees, this cell is where it shows up.

# %%
for r in sorted(mismatch, key=lambda r: -r["occurrences"]):
    print(f"\n{int(r['occurrences'])} occ, agree {r['exact']:.0%}, "
          f"median d {r['median_d']:.3g}, max d {r['max_d']:.3g}")
    print("  dd:", r["datadog"])
    print("  pq:", r["promql"])
if not mismatch:
    print("no shape disagreed on any bucket")

# %% [markdown]
# ## Figures §7 quotes
# The table holds the shape and occurrence counts; these are the run's scale and its
# agreement bound, which §7 states in prose.

# %%
ROOT = Path.cwd() if (Path.cwd() / "paper").exists() else Path.cwd().parent
_QF = ROOT / "paper" / "quoted_figures.json"
_existing = json.loads(_QF.read_text()) if _QF.exists() else {}
_existing.update({
    "nb05.bucket_comparisons": int(sum(r["buckets"] for r in cmp_rows)),
    "nb05.synthetic_series": int(prov["series"]),
    "nb05.shapes": int(prov["shapes"]),
    "nb05.bit_identical_shapes": int(sum(1 for r in cmp_rows
                                         if r.get("bit_identical") in (1.0, True))),
})
_QF.write_text(json.dumps(_existing, indent=1, sort_keys=True) + "\n")
print(f"wrote {_QF.relative_to(ROOT)}")
for _k in sorted(k for k in _existing if k.startswith("nb05.")):
    print(f"  {_k:32s} {_existing[_k]}")

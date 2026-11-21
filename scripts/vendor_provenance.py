#!/usr/bin/env python
r"""A provenance table for every measurement taken against the vendor.

\apref{app:experiments} said the experiments ran "against a trial account" and nothing
more: no dates, no site, no record of which run each figure came from. The caches under
`data/raw/datadog/` carry all three, so the table is generated from them rather than
written by hand -- eleven vendor measurements against an unversioned SaaS deserve to say
when they were taken, because the thing measured can change without notice and a reader
cannot otherwise tell whether two figures came from the same behaviour.

`type_declared.json` is deliberately absent: it censuses the vendor's published
integration repository rather than asking the API, which \apref{app:experiments} already
separates. `as_rate_order.json` carries a run id and no timestamp, so it is listed with
its run rather than dropped.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "datadog"
OUT = ROOT / "paper" / "tables" / "vendor_provenance.tex"

# What each cache asked, in the paper's own terms. Keyed by file so a new cache with no
# entry fails loudly rather than being summarised as "an experiment".
ASKED = {
    "e1.json": ("E1", "reset policy: a counter restarting from zero"),
    "e1_fast.json": ("E1", "the same, on the rig that replaced reading the agent"),
    "e2.json": ("E2", "bucket boundaries: two probes half a bucket apart"),
    "e3.json": ("E3", "whether a submitted metric answers \\texttt{p}$NN$"),
    "e4.json": ("E4", "automatic rollup interval at five query ranges"),
    "e5.json": ("E5", "flush behaviour of a partially filled interval"),
    "e7.json": ("E7", "\\texttt{.as\\_rate()} placement inside the aggregation"),
    "e8.json": ("E8", "gauge handling under \\texttt{.as\\_count()}"),
    "e9.json": ("E9", "type inference against a declared gauge"),
    "e12.json": ("E12", "glob constructs the filter grammar accepts"),
    "e13.json": ("E13", "scope forms over an empty-valued tag"),
    "e14.json": ("E14", "\\texttt{top()} under a narrowed window"),
    "e15.json": ("E15", "boolean scope evaluation"),
    "e16.json": ("E16", "negated match-all scopes"),
    "e17_scope_length.json": ("E17", "longest scope the account accepts"),
    "e18.json": ("E18", "automatic rollup interval at all ten declared spans"),
    "as_rate_order.json": ("E7", "the same question on a second series"),
    "scope_empty_tag.json": ("E13", "three series: \\texttt{k:v}, no \\texttt{k}, \\texttt{k:}"),
    "top.json": ("E14", "\\texttt{top()} ranking at full window"),
}
SKIP = {"type_declared.json"}        # the repository census, not an API call


def _provenance(doc: dict) -> tuple[str | None, str | None, str | None]:
    p = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
    at = p.get("collected_at") or doc.get("collected_at")
    return at, p.get("site") or doc.get("site"), p.get("run") or doc.get("run")


def main() -> int:
    rows = []
    for path in sorted(RAW.glob("*.json")):
        if path.name in SKIP:
            continue
        if path.name not in ASKED:
            raise SystemExit(f"{path.name}: no entry in ASKED -- add one")
        exp, asked = ASKED[path.name]
        at, site, run = _provenance(json.loads(path.read_text()))
        when = datetime.fromisoformat(at).strftime("%d %b %Y") if at else f"run {run}"
        rows.append((exp, when, site or "---", asked, path.name))

    rows.sort(key=lambda r: (int(r[0][1:]), r[4]))
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
             r"Experiment & Taken & Site & What was submitted \\", r"\midrule"]
    for exp, when, site, asked, _ in rows:
        site = site.replace("_", r"\_")
        lines.append(f"{exp} & {when} & \\texttt{{{site}}} & {asked} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} measurements")
    for r in rows:
        print(f"   {r[0]:5} {r[1]:12} {r[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

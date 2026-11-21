#!/usr/bin/env python
r"""Every emitted expression, from both translators, through Prometheus's own parser.

The paper has always judged "is this PromQL?" with the grammar of section 4. That
grammar is this paper's instrument: it predates both translators' output and is tuned to
neither, which is what makes it fair, but it is not the thing a migration actually faces.
Table~\ref{tab:baseline} said so in its own caption -- neither figure had been put to a
Prometheus parser at corpus scale -- and that sentence was a standing invitation to check.
This script accepts it.

The check is a parse, not an evaluation: `promtool check rules` runs
`promql.ParseExpr` over every expression and reports each failure with a position. No
server, no data, no network. Prometheus 3.15.0 is the same build every other reading in
this paper came off, so a disagreement here is a disagreement with that build and not
with a version of PromQL nobody runs.

Two things make the comparison the same comparison section 7.4 already reports:

  * Our expressions are recomputed from the cached corpus rather than read out of
    `translations.csv`, because that file stores a widget's several expressions joined
    with `" ; "` for display and the joined string is not PromQL. The parse unit is one
    expression, which is what Prometheus is handed.
  * OpenObserve's are produced by the same cached bundle, over the same held-out set
    (multi-query sources dropped, as `openobserve_baseline.py` drops them), so the
    denominators match the `oo.*` figures key for key.

A failure here is a defect: section 5's emitter is supposed to produce PromQL, and
nothing in the corpus reaches Prometheus through a template variable or a placeholder,
so there is no class of expression that may legitimately be rejected.

Usage: .venv/bin/python scripts/promtool_parse.py [--write-registry]
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "data" / "corpus"
TRANSLATIONS = ROOT / "data" / "processed" / "translations.csv"
OUT = ROOT / "data" / "processed" / "promtool.json"
REGISTRY = ROOT / "paper" / "quoted_figures.json"
IMAGE = "prom/prometheus:v3.15.0"
DEFAULT_ROLLUP = 60.0   # the same default notebook 03 translates against

# promtool reports a failure as `rule N, "<name>"`, and the name is how an expression is
# recovered. Recording-rule names must be metric names, so the index carries the identity.
FAILED = re.compile(
    r'rule \d+, "([a-z]+_\d+)": could not parse expression: (.*)$', re.M)


def _baseline():
    """`openobserve_baseline` as a module: its cached bundle and hold-out rule, reused."""
    path = ROOT / "scripts" / "openobserve_baseline.py"
    spec = importlib.util.spec_from_file_location("oo_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ours() -> list[dict]:
    """One row per distinct instantiated Datadog query, with its emitted expressions.

    Mirrors notebook 03's `classify`. The occurrence total is asserted against
    `translations.csv`, so a drift in either is a failure here and not a re-weighting.
    """
    from qlport import corpus as c
    from qlport.diagnostics import Bottom
    from qlport.emit.promql import emit_detail
    from qlport.lower.datadog import lower_all
    from qlport.parse import parse

    import pandas as pd

    queries = [q for q in c.read_queries(CORPUS / "queries.jsonl") if q.lang == "datadog"]
    occ = pd.Series([c.normalise("datadog", q.text) for q in queries]).value_counts()

    rows = []
    for text in occ.index:
        n = int(occ[text])
        if not c.parse_outcome("datadog", text).ok:
            rows.append({"norm": text, "n": n, "outcome": "unparsed", "parts": []})
            continue
        parts, bottom = [], False
        for e in lower_all(parse("datadog", text)):
            got = e if isinstance(e, Bottom) else emit_detail(
                e, default_rollup=DEFAULT_ROLLUP)
            if isinstance(got, Bottom):
                bottom = True
                break
            parts.append(got.promql)
        rows.append({"norm": text, "n": n,
                     "outcome": "bottom" if bottom else "translated",
                     "parts": [] if bottom else parts})

    have = {r["norm"]: r for r in rows}
    import csv as _csv
    for r in _csv.DictReader(TRANSLATIONS.open()):
        mine = have.get(r["norm"])
        assert mine is not None, f"query absent from the recomputation: {r['norm']!r}"
        assert mine["outcome"] == r["outcome"], (r["norm"], mine["outcome"], r["outcome"])
        assert mine["n"] == int(r["n"]), (r["norm"], mine["n"], r["n"])
    assert len(have) == sum(1 for _ in _csv.DictReader(TRANSLATIONS.open()))
    return rows


def theirs(mod, our_rows: list[dict]) -> dict[str, dict]:
    """OpenObserve's emission per query, over the same held-out set."""
    single = [r for r in our_rows if not mod.toplevel_comma(r["norm"])]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "run.cjs").write_text(mod.SHIM % mod.extract_module())
        (tmp / "q.json").write_text(json.dumps([r["norm"] for r in single]))
        done = subprocess.run(
            ["node", str(tmp / "run.cjs"), str(tmp / "q.json"), str(tmp / "r.json")],
            capture_output=True, text=True)
        if done.returncode:
            raise SystemExit("node failed:\n" + done.stdout + done.stderr)
        return {r["norm"]: r for r in json.loads((tmp / "r.json").read_text())}


def check(exprs: list[tuple[str, str]]) -> dict[str, str]:
    """`{name: error}` for every expression Prometheus's parser rejects.

    One rule file, one container, every failure reported. `json.dumps` quotes the
    expression: JSON scalars are valid YAML, so nothing in a label value can end the
    scalar early.
    """
    body = "".join(
        f"  - record: {name}\n    expr: {json.dumps(expr)}\n" for name, expr in exprs)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "rules.yaml").write_text("groups:\n- name: parse\n  rules:\n" + body)
        done = subprocess.run(
            ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
             "-v", f"{tmp}:/w", "--entrypoint", "/bin/promtool", IMAGE,
             "check", "rules", "/w/rules.yaml"],
            capture_output=True, text=True)
    out = done.stdout + done.stderr
    bad = dict(FAILED.findall(out))
    if done.returncode and not bad:
        raise SystemExit("promtool failed without naming a rule:\n" + out)
    return bad


def main() -> int:
    mod = _baseline()
    rows = ours()
    oo = theirs(mod, rows)

    # Name every distinct expression once; occurrences are re-attached afterwards.
    ours_id, theirs_id, exprs = {}, {}, []
    for r in rows:
        for p in r["parts"]:
            if p not in ours_id:
                ours_id[p] = f"ours_{len(ours_id)}"
                exprs.append((ours_id[p], p))
    for norm, o in oo.items():
        got = o["oo"]
        if got is None or got.startswith("# TODO"):
            continue
        if got not in theirs_id:
            theirs_id[got] = f"theirs_{len(theirs_id)}"
            exprs.append((theirs_id[got], got))

    print(f"{len(exprs):,} distinct expressions "
          f"({len(ours_id):,} ours, {len(theirs_id):,} theirs) -> promtool")
    bad = check(exprs)
    print(f"promtool rejects {len(bad):,}")

    single = {r["norm"] for r in rows if not mod.toplevel_comma(r["norm"])}
    by_n = {r["norm"]: r["n"] for r in rows}

    ours_fail_occ = sum(r["n"] for r in rows
                        if any(ours_id[p] in bad for p in r["parts"]))
    ours_fail_exprs = sorted(p for p in ours_id if ours_id[p] in bad)
    ours_occ = sum(r["n"] for r in rows if r["outcome"] == "translated")

    th_emits = {k: v["oo"] for k, v in oo.items()
                if v["oo"] is not None and not v["oo"].startswith("# TODO")}
    theirs_fail_occ = sum(by_n[k] for k, e in th_emits.items() if theirs_id[e] in bad)
    theirs_occ = sum(by_n[k] for k in th_emits)

    # The point of the exercise. Section 4's grammar is this paper's instrument; the
    # question is whether it answers "is this PromQL?" the way Prometheus does. Totals
    # agreeing would not settle it, so the two verdicts are compared expression by
    # expression, in both directions, over every expression either translator emitted.
    from qlport.parse import parse as _parse

    def by_grammar(q: str) -> bool:
        try:
            _parse("promql", q)
            return True
        except Exception:
            return False

    we_reject = {e for e in list(ours_id) + list(theirs_id) if not by_grammar(e)}
    prom_rejects = {e for e in list(ours_id) + list(theirs_id)
                    if (ours_id.get(e) or theirs_id.get(e)) in bad}
    only_us = sorted(we_reject - prom_rejects)
    only_prom = sorted(prom_rejects - we_reject)
    print(f"grammar rejects {len(we_reject):,}, Prometheus {len(prom_rejects):,}, "
          f"disagreements {len(only_us) + len(only_prom)}")
    assert not only_us, ("section 4's grammar rejects PromQL that Prometheus accepts",
                         only_us[:3])
    assert not only_prom, ("section 4's grammar accepts what Prometheus rejects",
                           only_prom[:3])

    fig = {
        "pt.exprs": len(exprs),
        "pt.ours_exprs": len(ours_id),
        "pt.ours_occ": ours_occ,
        "pt.ours_rejected_exprs": len(ours_fail_exprs),
        "pt.ours_rejected_occ": ours_fail_occ,
        "pt.theirs_exprs": len(theirs_id),
        "pt.theirs_occ": theirs_occ,
        "pt.theirs_rejected_exprs": sum(1 for e in theirs_id if theirs_id[e] in bad),
        "pt.theirs_rejected_occ": theirs_fail_occ,
        "pt.disagreements": len(only_us) + len(only_prom),
    }
    reasons = Counter(
        re.sub(r"^\d+:\d+: parse error: ", "", v).strip() for v in bad.values())

    OUT.write_text(json.dumps({
        "provenance": {"image": IMAGE, "command": "promtool check rules",
                       "held_out_multi_query": len(rows) - len(single)},
        "figures": fig,
        "reasons": dict(reasons.most_common()),
        "ours_rejected": ours_fail_exprs[:50],
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")

    for k, v in fig.items():
        print(f"  {k:<28} {v:,}")
    if reasons:
        print("\nreasons:")
        for r, k in reasons.most_common(10):
            print(f"  {k:>5}  {r}")

    if "--write-registry" in sys.argv:
        reg = json.loads(REGISTRY.read_text())
        reg.update(fig)
        REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
        print(f"\nmerged {len(fig)} keys into {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

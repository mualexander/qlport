#!/usr/bin/env python3
"""How often is the metric type the translator infers the one the vendor declares?

Paper §5's Remark rem:gauge establishes that a Datadog dashboard does not record
whether a metric was submitted as COUNT or GAUGE, so the front end infers the kind from
the query text -- `.as_count()`, `.as_rate()` or `.rollup(sum, D)` are read as evidence
of a counter, anything else leaves the kind unknown and the emitter translates it as a
gauge. Every translation therefore carries METRIC_TYPE_UNKNOWN, which §7 reports at
100% and §8 calls the largest qualification on kappa.

`fetch_metric_types.py` makes that assumption checkable for the metrics this corpus
names: the repository the dashboards come from also declares each metric's type in
`metadata.csv`. This script joins the two and counts, by query occurrence, how often
the inference is right, wrong, or unknowable.

The join uses the paper's own `metric_name()` mangling, since the IR stores the mangled
PromQL name. Where two declared names mangle together (Remark rem:mangling finds three
such pairs in this corpus) the lookup is recorded as ambiguous rather than guessed.

What a disagreement means, and does not: `metadata.csv` is what the integration says it
submits. A dashboard can query a metric the integration does not ship, and an operator
can submit the same name as something else. So a disagreement is evidence that the
inference is wrong, not proof -- which is the same standard §5 applies to the modifier
evidence it replaces.

  python scripts/type_declared.py

Writes data/raw/datadog/type_declared.json.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
from pathlib import Path

from qlport import corpus as c
from qlport import ir
from qlport.diagnostics import Bottom
from qlport.emit.promql import emit_detail
from qlport.lower.datadog import lower_all, metric_name
from qlport.parse import parse

ROOT = Path(__file__).resolve().parent.parent
DECLARED = (ROOT / "data" / "corpus" / "sources" / "integrations-core-metadata"
            / "declared.json")
OUT = ROOT / "data" / "raw" / "datadog" / "type_declared.json"
ROLLUP = 60.0

# Datadog stores COUNT as per-interval deltas and GAUGE as the samples themselves, which
# is what phi models. RATE is a third submission type phi does not cover.
COUNTERISH = {"count"}
GAUGEISH = {"gauge"}


def selects(e: ir.Expr) -> list[ir.Select]:
    found: list[ir.Select] = []

    def go(x):
        if isinstance(x, ir.Select):
            found.append(x)
        for field in ("expr", "lhs", "rhs", "arg"):
            kid = getattr(x, field, None)
            if isinstance(kid, ir.Expr):
                go(kid)
    go(e)
    return found


def verdict(inferred: str, declared: str | None) -> str:
    """One of: agree, wrong-counter, wrong-gauge, rate, ambiguous, undeclared."""
    if declared is None:
        return "undeclared"
    if declared in ("conflict", "ambiguous"):
        return "ambiguous"
    if declared in ("rate",):
        return "rate"
    # The emitter treats an unknown kind as a gauge, so that is the effective inference.
    effective = "counter" if inferred == "counter" else "gauge"
    if effective == "counter":
        return "agree" if declared in COUNTERISH else "wrong-counter"
    return "agree" if declared in GAUGEISH else "wrong-gauge"


def main() -> int:
    if not DECLARED.exists():
        raise SystemExit(f"{DECLARED} missing; run scripts/fetch_metric_types.py")
    fetched = json.loads(DECLARED.read_text())
    declared_raw: dict[str, str] = fetched["declared"]

    # mangled PromQL name -> declared type, with collisions marked rather than resolved
    by_mangled: dict[str, str] = {}
    for name, kind in declared_raw.items():
        key = metric_name(name)
        if key in by_mangled and by_mangled[key] != kind:
            by_mangled[key] = "ambiguous"
        else:
            by_mangled.setdefault(key, kind)

    occ = collections.Counter(
        c.normalise("datadog", q.text)
        for q in c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl")
        if q.lang == "datadog")

    consulted: set[str] = set()
    by_verdict: collections.Counter = collections.Counter()
    distinct: collections.Counter = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    translated = 0

    for text, n in sorted(occ.items()):
        try:
            trees = lower_all(parse("datadog", text, "v2"))
        except Exception:
            continue
        kinds: list[str] = []
        for e in trees:
            if isinstance(e, Bottom):
                continue
            if isinstance(emit_detail(e, default_rollup=ROLLUP), Bottom):
                continue
            for s in selects(e):
                consulted.add(s.metric)
                kinds.append(verdict(s.kind, by_mangled.get(s.metric)))
        if not kinds:
            continue
        translated += n
        # A query is as wrong as its worst selector: one mistranslated operand is enough.
        for rank in ("wrong-gauge", "wrong-counter", "ambiguous", "rate",
                     "undeclared", "agree"):
            if rank in kinds:
                by_verdict[rank] += n
                distinct[rank] += 1
                if len(examples[rank]) < 5:
                    examples[rank].append(text)
                break

    known = sum(by_verdict[k] for k in ("agree", "wrong-gauge", "wrong-counter"))
    wrong = by_verdict["wrong-gauge"] + by_verdict["wrong-counter"]
    used = {name: kind for name, kind in declared_raw.items()
            if metric_name(name) in consulted}
    OUT.write_text(json.dumps({
        "declared_for_corpus_metrics": dict(sorted(used.items())),
        "declared_population": fetched["type_counts"],
        "occurrences": dict(by_verdict),
        "distinct": dict(distinct),
        "examples": {k: v for k, v in sorted(examples.items())},
        "translated_occurrences": translated,
        "decidable_occurrences": known,
        "wrong_occurrences": wrong,
        "wrong_share_of_decidable_pct": round(wrong / known * 100, 1) if known else None,
        "decidable_share_of_translated_pct": round(known / translated * 100, 1),
        "provenance": {
            "declared_from": str(DECLARED.relative_to(ROOT)),
            "declarations_in_population": fetched["provenance"]["metrics_declared"],
            "declarations_recorded_here": len(used),
            "commit": json.loads(DECLARED.read_text())["provenance"]["commit"],
            "default_rollup_s": ROLLUP,
            "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }, indent=1, sort_keys=True) + "\n")

    print(f"translated occurrences            {translated:,}")
    for k, v in by_verdict.most_common():
        print(f"  {k:14} {v:>6,}  ({distinct[k]:,} distinct)")
    print(f"\ndecidable (declared gauge/count)  {known:,} "
          f"({known / translated * 100:.1f}% of translated)")
    if known:
        print(f"  inference wrong                 {wrong:,} ({wrong / known * 100:.1f}%)")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

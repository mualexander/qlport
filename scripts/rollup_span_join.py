#!/usr/bin/env python
r"""E18's lookup joined to the corpus: how often is the emitter's 60 s the right one?

`ROLLUP_DEFAULT` is carried by 97.0 per cent of translations, and §7.3 says the 60 s the
emitter writes is "systematic rather than arbitrary" because Datadog publishes the
interval as a lookup on the query's time range. E18 measured that lookup at all ten
spans the corpus's own widgets declare, and 60 s is the four-hour row and nothing else.
RQ4 asks how often a translator supplying the obvious default is wrong; the paper answers
it for the metric type (6.5 per cent, 491 occurrences) and had never answered it for the
interval, although both halves of the answer were already on disk.

What can be measured and what cannot. A Datadog widget may pin its own span in
`definition.time.live_span`; 460 widgets across the corpus do, all of them leaves, so
there is no group inheritance to model. A widget that pins nothing takes the span from
the dashboard's time selector, which is a viewer's choice at view time and is not in the
JSON at all -- so for those occurrences the default is *unverified*, which is what §7.3
said, and not *wrong*, which is what an extrapolation would say. Both are reported: the
measured subset stands on its own, and the extrapolation is labelled as one.

The walk mirrors `corpus.datadog_dashboard_queries` and asserts it recovers the same
occurrence count, so a drift in either is a failure here rather than a silent re-weighting.

Usage: .venv/bin/python scripts/rollup_span_join.py [--write-registry]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport import corpus as c  # noqa: E402

SOURCES = ROOT / "data" / "corpus" / "sources" / "integrations-core"
E18 = ROOT / "data" / "raw" / "datadog" / "e18.json"
TRANSLATIONS = ROOT / "data" / "processed" / "translations.csv"
OUT = ROOT / "data" / "processed" / "rollup_spans.json"
REGISTRY = ROOT / "paper" / "quoted_figures.json"

EMITTED = 60.0          # the interval the emitter writes into every range selector


def _span_of(defn: dict) -> str | None:
    time = defn.get("time")
    if not isinstance(time, dict):
        return None
    span = time.get("live_span")
    return str(span) if span else None


def _definitions():
    """Every widget definition in the corpus's Datadog dashboards, in traversal order.

    Both walks below need it: one counts the query occurrences under each declared
    span, the other counts the widgets that declare one at all. They have to see the
    same definitions or the gap between them is an artefact of two traversals.
    """
    for path in sorted(SOURCES.rglob("assets/dashboards/*.json")):
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        yield from c._dd_definitions(doc)


def widgets_with_spans() -> Counter[str]:
    """Every widget that pins its own span, counted by the span it pins.

    The occurrence join can only reach a span that some metric query sits under, and
    the corpus declares two values no metric query sits under at all. Counting widgets
    as well as occurrences is what makes that gap visible: without it the join's eight
    spans read as the corpus's ten. One of the two it misses is interior and widens
    nothing; the other is the widest span declared anywhere, where the emitted constant
    is furthest from the interval Datadog reports and where E18 observed no spacing to
    check that interval against.
    """
    out: Counter[str] = Counter()
    for defn in _definitions():
        span = _span_of(defn)
        if span:
            out[span] += 1
    return out


def occurrences_with_spans() -> list[tuple[str, str | None]]:
    """Every Datadog metric query occurrence, paired with its widget's declared span.

    One record per occurrence, in the extractor's own traversal order, so the count is
    comparable to the corpus's without matching on titles -- which are not unique
    within a dashboard.
    """
    out: list[tuple[str, str | None]] = []
    for defn in _definitions():
        span = _span_of(defn)
        for req in c._dd_requests(defn):
            q = req.get("q")
            if isinstance(q, str) and q.strip():
                out.append((q.strip(), span))
            subs = list(req.get("queries") or [])
            single = req.get("query")
            if isinstance(single, dict):
                subs.append(single)
            for sub in subs:
                if not isinstance(sub, dict) or sub.get("data_source") != "metrics":
                    continue
                text = sub.get("query")
                if isinstance(text, str) and text.strip():
                    out.append((text.strip(), span))
    return out


def main() -> int:
    lookup = json.loads(E18.read_text())["interpretation"]["reported_by_range"]

    import csv
    rollup_default: set[str] = set()
    translated: set[str] = set()
    for row in csv.DictReader(TRANSLATIONS.open()):
        if row["outcome"] != "translated":
            continue
        translated.add(row["norm"])
        if "ROLLUP_DEFAULT" in (row["caveats"] or "").split("|"):
            rollup_default.add(row["norm"])

    occ = occurrences_with_spans()
    corpus = [q for q in c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl")
              if q.lang == "datadog" and q.kind == "widget"]
    assert len(occ) == len(corpus), (
        f"walk recovered {len(occ)} occurrences, corpus has {len(corpus)}")

    by_span: Counter[str] = Counter()
    declared_total = undeclared_total = 0
    for text, span in occ:
        norm = c.normalise("datadog", text)
        if norm not in rollup_default:
            continue
        if span is None:
            undeclared_total += 1
        else:
            by_span[span] += 1
            declared_total += 1

    right = sum(n for s, n in by_span.items() if lookup.get(s) == EMITTED)
    wrong = declared_total - right
    unknown_span = sum(n for s, n in by_span.items() if s not in lookup)

    # The factor by which the emitted 60 s misses, at each span that declares one.
    factors = {s: round(EMITTED / lookup[s], 3) if lookup.get(s) else None
               for s in sorted(by_span)}
    modal_span, modal_n = by_span.most_common(1)[0]

    # What the occurrence join cannot see. A span reaches the join only if some metric
    # query sits under a widget that pins it, and two of the ten values the corpus
    # declares are pinned by widgets holding no metric query occurrence at all -- so
    # they are absent from `by_span` by having nothing to join rather than by failing
    # the caveat test, and the range the paragraph in 7.6 quotes is eight values wide
    # and not ten. Only one of the two is outside that range: thirty minutes sits
    # between spans the join reaches and widens nothing, while at a month Datadog
    # reports an interval the emitted 60 s is a 240th of, a larger factor than anything
    # the join's range contains and the one row of E18's lookup with no observed
    # spacing behind it.
    #
    # Ordered by the interval rather than by the span, which `rem:rollup-spans` records
    # as monotone in the range, so the two orders agree on the endpoint this needs; the
    # ties among the short spans are immaterial because nothing below reads the middle.
    widgets = widgets_with_spans()
    by_interval = sorted(widgets, key=lambda s: lookup[s])
    unjoined = [s for s in by_interval if s not in by_span]
    widest = by_interval[-1]

    fig = {
        "rs.rollup_default_occurrences": len(
            [1 for t, _ in occ if c.normalise("datadog", t) in rollup_default]),
        "rs.span_declared_occurrences": declared_total,
        "rs.span_undeclared_occurrences": undeclared_total,
        "rs.span_declared_right": right,
        "rs.span_declared_wrong": wrong,
        "rs.span_declared_right_pct": round(100 * right / declared_total, 1),
        "rs.span_declared_wrong_pct": round(100 * wrong / declared_total, 1),
        "rs.span_declared_pct_of_rollup_default": round(
            100 * declared_total / (declared_total + undeclared_total), 1),
        "rs.modal_declared_span_occurrences": modal_n,
        "rs.modal_declared_span_factor": factors[modal_span],
        "rs.span_unknown_to_e18": unknown_span,
        # The emitted 60 s against the interval Datadog reports at the widest span the
        # corpus declares, as a divisor: 60 s is a 240th of it. Quoted in 7.6 because
        # it is the largest error the lookup implies and it sits outside the range that
        # paragraph used to give as "the declared spans".
        "rs.widest_declared_span_divisor": round(lookup[widest] / EMITTED),
        # What the measured share would mean applied to every ROLLUP_DEFAULT
        # occurrence. Reported as an extrapolation and not a measurement, and the
        # selection runs one way: an author who pins a span is doing something
        # deliberate, most plausibly because the dashboard default was wrong for that
        # panel, so the declaring widgets are if anything enriched for disagreement
        # and this is an upper estimate.
        "rs.extrapolated_wrong": round(
            (declared_total + undeclared_total) * wrong / declared_total),
    }

    OUT.write_text(json.dumps({
        "figures": fig,
        "by_span": {s: {"occurrences": n, "interval_seconds": lookup.get(s),
                        "factor": factors[s]} for s, n in by_span.most_common()},
        # The widget-level census, which the occurrence join is a subset of. Kept
        # outside `figures` because the paragraph spells these in words; the test that
        # reads them reads them from here.
        "by_widget_span": {s: widgets[s] for s in by_interval},
        "declared_span_values": len(widgets),
        "joined_span_values": len(by_span),
        "unjoined_spans": unjoined,
        "unjoined_span_widgets": sum(widgets[s] for s in unjoined),
        "unjoined_span_factors": {s: round(EMITTED / lookup[s], 3) for s in unjoined},
        "widest_declared_span": widest,
        "modal_declared_span": modal_span,
        "lookup": lookup,
        "note": "spans are widget-declared live_span values; occurrences whose widget "
                "declares none take the dashboard's time selector, which is not in the "
                "JSON",
    }, indent=1, sort_keys=True) + "\n")

    print(f"ROLLUP_DEFAULT occurrences: {fig['rs.rollup_default_occurrences']:,}")
    print(f"  widget declares a span: {declared_total:,} "
          f"({fig['rs.span_declared_pct_of_rollup_default']}%)")
    print(f"  60 s is right for:      {right:,} ({fig['rs.span_declared_right_pct']}%)")
    print(f"  60 s is wrong for:      {wrong:,} ({fig['rs.span_declared_wrong_pct']}%)")
    print(f"  modal declared span:    {modal_span} "
          f"({modal_n:,} occurrences, off by {factors[modal_span]}x)")
    print(f"  span unknown to E18:    {unknown_span:,}")
    print(f"  declared span values:   {len(widgets)} "
          f"({len(by_span)} reached by the join; "
          f"{', '.join(unjoined)} pinned by "
          f"{sum(widgets[s] for s in unjoined)} widgets with no occurrence)")
    print(f"  widest declared span:   {widest}, interval {lookup[widest]}s, "
          f"emitted 60 s is a {round(lookup[widest] / EMITTED)}th of it")
    for s, n in by_span.most_common():
        print(f"     {s:>5}  {n:6,}  interval {lookup.get(s)}  factor {factors[s]}")

    if "--write-registry" in sys.argv:
        reg = json.loads(REGISTRY.read_text())
        reg.update(fig)
        REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
        print(f"\nmerged {len(fig)} keys into {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

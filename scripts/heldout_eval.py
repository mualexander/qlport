#!/usr/bin/env python
r"""The frozen grammar and the frozen translator, on dashboards nobody here has read.

`scripts/heldout_split.py` refits by cross-validation the two things that are functions
of the corpus. The grammar is not one of them --- it is handwritten, so no script can
re-derive what it would have been had it seen half the corpus --- and it is the channel
section 7 names first: grammar v2 is v1 plus every production the corpus showed was
missing. The only honest test of that fit is a corpus the grammar was not built against.

`scripts/heldout_fetch.py` provides two:

  grafana           Prometheus dashboards past download rank 200, which the corpus
                    stopped at. Never listed, downloaded or read here.
  integrations-core The same repository at a later commit, restricted to query strings
                    the corpus does not contain.

This script reports, on each of them, exactly the columns Table~ref{tab:parse-coverage}
reports in sample --- raw, v1, v2, residual --- plus the feature-vocabulary ceiling and,
on the Datadog side, kappa from the frozen translator.

It is a measurement and not a development loop. If a number here is worse than the
in-sample one, that is the finding; repairing the grammar until it improves would
destroy the only held-out set this paper has.

Usage: .venv/bin/python scripts/heldout_eval.py [--write-registry]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlport import corpus as c                                       # noqa: E402
from qlport.diagnostics import Bottom                                # noqa: E402
from qlport.emit.promql import emit_detail                           # noqa: E402
from qlport.lower.datadog import lower_all                           # noqa: E402
from qlport.parse import parse                                       # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "queries.jsonl"
HELDOUT = ROOT / "data" / "heldout"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "paper" / "tables"
REGISTRY = ROOT / "paper" / "quoted_figures.json"

DEFAULT_ROLLUP = 60.0    # as notebook 03 sets it; the translator is called unchanged


RESIDUAL_KINDS = (
    # Checked in order, so each residual string lands in exactly one bucket.
    ("logql", lambda t: "__error__" in t or "|=" in t or "unwrap" in t),
    ("metricsql", lambda t: t.lstrip().startswith("with (")),
    ("old_template", lambda t: "[[" in t),
)


def residual_kinds(strings: list[str]) -> dict[str, int]:
    """What the strings v2 cannot parse actually are.

    Reported rather than removed. A held-out population is dirtier than a curated one
    --- community dashboards filed under the Prometheus datasource carry Loki panels and
    VictoriaMetrics templates --- and the temptation is to filter those out until the
    rate improves. Doing it after seeing the rate is how a held-out set stops being one,
    so the number in the table is the number as measured and this is the decomposition.
    """
    out: dict[str, int] = {k: 0 for k, _ in RESIDUAL_KINDS}
    out["promql"] = 0
    for t in strings:
        low = t.lower()
        for kind, test in RESIDUAL_KINDS:
            if test(low):
                out[kind] += 1
                break
        else:
            out["promql"] += 1
    return out


def _promql_only(queries: list) -> list:
    """Notebook 00's filter, applied here for the same reason it is applied there.

    Community dashboards listed under the Prometheus datasource carry Loki panels.
    LogQL is a different language and is no evidence about a PromQL grammar, so the
    notebook drops those panels before it measures anything --- and a held-out set of
    500 community dashboards carries far more of them than the corpus's 200 did. Leaving
    them in would have scored the grammar against a language it does not claim to parse
    and called the result a holdout.
    """
    return [q for q in queries
            if q.lang != "promql" or c.is_promql(q.text)]


def tabular(colspec: str, head: list[str], rows: list[list[str]], path: Path) -> None:
    lines = [rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", r"\toprule",
             " & ".join(head) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def pct(x: float) -> str:
    return f"{x:.1%}".replace("%", r"\%")


# ---------------------------------------------------------------------------
# the frozen translator, called exactly as notebook 03 calls it
# ---------------------------------------------------------------------------
def classify(text: str) -> dict:
    row = dict(norm=text, outcome="", gap="")
    if not c.parse_outcome("datadog", text).ok:
        return {**row, "outcome": "unparsed"}
    for e in lower_all(parse("datadog", text)):
        got = e if isinstance(e, Bottom) else emit_detail(
            e, default_rollup=DEFAULT_ROLLUP)
        if isinstance(got, Bottom):
            return {**row, "outcome": "bottom", "gap": got.gap.name}
    return {**row, "outcome": "translated"}


# ---------------------------------------------------------------------------
# one side of the held-out set
# ---------------------------------------------------------------------------
def measure(queries: list, lang: str, vocabulary: set[str],
            corpus_forms: set[str]) -> dict:
    """Parse rates over distinct queries, and the vocabulary ceiling over occurrences.

    Disjointness is enforced by dashboard, which is the unit the fit was made over: the
    grammar was extended until it parsed the corpus, and a dashboard the corpus does not
    contain contributed nothing to that. But a popular query string recurs across
    dashboards, so some held-out strings are ones the corpus also holds, and those are
    not held out in any useful sense. They are measured separately and reported
    separately, because the rate that belongs in the paper is the one on the strings
    whose instantiated form appears nowhere in the corpus.
    """
    texts = sorted({q.text for q in queries})
    occ = Counter(q.text for q in queries)
    forms = {t: c.normalise(lang, t) for t in texts}
    novel_texts = [t for t in texts if forms[t] not in corpus_forms]
    raw = v1 = v2 = 0
    nov_v1 = nov_v2 = 0
    residual = []
    blockers: Counter = Counter()
    for t in texts:
        norm = forms[t]
        raw += c.parse_outcome(lang, t, "v2").ok
        got_v1 = c.parse_outcome(lang, norm, "v1").ok
        out = c.parse_outcome(lang, norm, "v2")
        v1 += got_v1
        v2 += out.ok
        if forms[t] not in corpus_forms:
            nov_v1 += got_v1
            nov_v2 += out.ok
        if not out.ok:
            residual.append(norm)
            blockers[out.blocker] += 1
    n = len(texts) or 1
    nn = len(novel_texts) or 1

    feats = {t: c.features(lang, t) for t in texts}
    novel: Counter = Counter()
    inside_occ = inside_d = 0
    for t in texts:
        extra = feats[t] - vocabulary
        if extra:
            for f in extra:
                novel[f] += occ[t]
        else:
            inside_occ += occ[t]
            inside_d += 1
    total_occ = sum(occ.values()) or 1

    return dict(
        dashboards=len({q.origin for q in queries}),
        queries=len(queries), distinct=len(texts),
        raw_pct=round(raw / n * 100, 1), v1_pct=round(v1 / n * 100, 1),
        v2_pct=round(v2 / n * 100, 1), residual=len(residual),
        shared_distinct=len(texts) - len(novel_texts),
        novel_distinct=len(novel_texts),
        novel_v1_pct=round(nov_v1 / nn * 100, 1),
        novel_v2_pct=round(nov_v2 / nn * 100, 1),
        blockers=dict(blockers.most_common()),
        residual_kinds=residual_kinds(residual),
        residual_examples=[r[:120] for r in residual[:5]],
        ceiling_occ_pct=round(inside_occ / total_occ * 100, 1),
        ceiling_distinct_pct=round(inside_d / n * 100, 1),
        novel_features=dict(novel.most_common()),
    )


def kappa(queries: list) -> dict:
    occ = Counter(c.normalise("datadog", q.text) for q in queries)
    rows = [{**classify(t), "n": occ[t]} for t in occ]
    total = sum(occ.values()) or 1
    by = Counter()
    byd = Counter()
    for r in rows:
        by[r["outcome"]] += r["n"]
        byd[r["outcome"]] += 1
    gaps = Counter(r["gap"] for r in rows if r["gap"])
    return dict(
        occurrences=total, distinct=len(rows),
        kappa_occ_pct=round(by["translated"] / total * 100, 1),
        kappa_distinct_pct=round(byd["translated"] / (len(rows) or 1) * 100, 1),
        by_outcome_occ=dict(by), by_outcome_distinct=dict(byd),
        gaps=dict(gaps.most_common()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-registry", action="store_true")
    args = ap.parse_args()

    path = HELDOUT / "queries.jsonl"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing --- run "
                         "`.venv/bin/python scripts/heldout_fetch.py --all`")
    held = _promql_only(c.read_queries(path))
    train = _promql_only(c.read_queries(CORPUS))
    manifest = json.loads((HELDOUT / "MANIFEST.json").read_text())

    # The vocabulary the translator was built against: every feature the corpus shows.
    vocab = {lang: set().union(*(c.features(lang, q.text)
                                 for q in train if q.lang == lang))
             for lang in ("datadog", "promql")}
    for lang, v in vocab.items():
        print(f"corpus {lang} vocabulary: {len(v)} features")
    forms = {lang: {c.normalise(lang, q.text) for q in train if q.lang == lang}
             for lang in ("datadog", "promql")}

    # Disjointness is the whole premise, so it is asserted rather than assumed.
    seen_dd = {c.normalise("datadog", q.text)
               for q in train if q.source == "integrations-core"}
    seen_ids = {q.origin.split()[0] for q in train if q.source == "grafana"}
    leaked_dd = [q for q in held if q.source == "integrations-core"
                 and c.normalise("datadog", q.text) in seen_dd]
    leaked_gf = [q for q in held if q.source == "grafana"
                 and q.origin.split()[0] in seen_ids]
    assert not leaked_dd, f"{len(leaked_dd)} held-out Datadog queries are in the corpus"
    assert not leaked_gf, f"{len(leaked_gf)} held-out dashboards are in the corpus"
    print(f"held out: {len(held):,} queries, none of them in the corpus")

    res: dict = {"manifest": manifest}

    # The matched in-sample comparator. Table~\ref{tab:parse-coverage}'s promql row
    # mixes Grafana with kube-prometheus, whose queries are compiled from jsonnet and
    # are nothing like a hand-built dashboard's, while the held-out set is Grafana
    # alone. Comparing against the mixed row would confound source with holdout, so the
    # Grafana part of the corpus is measured here on its own.
    gf_train = [q for q in train if q.source == "grafana" and q.lang == "promql"]
    res["grafana_in_sample"] = measure(gf_train, "promql", vocab["promql"], set())
    gi = res["grafana_in_sample"]
    print(f"\nGrafana in sample: {gi['queries']:,} queries, {gi['distinct']:,} distinct, "
          f"raw {gi['raw_pct']}%  v1 {gi['v1_pct']}%  v2 {gi['v2_pct']}%  "
          f"residual {gi['residual']}")

    rows = [[r"Grafana, the corpus's $200$",
             f"{gi['queries']:,}", f"{gi['distinct']:,}", pct(gi["raw_pct"] / 100),
             pct(gi["v1_pct"] / 100), pct(gi["v2_pct"] / 100), f"{gi['residual']:,}"]]
    for source, lang, label in (
            ("grafana", "promql", r"Grafana, @N it never saw"),
            ("integrations-core", "datadog",
             r"\texttt{integrations-core}, later commit")):
        qs = [q for q in held if q.source == source and q.lang == lang]
        if not qs:
            print(f"\n{label}: nothing held out")
            continue
        m = measure(qs, lang, vocab[lang], forms[lang])
        res[source] = m
        print(f"\n{label.replace('@N', str(m['dashboards']))}: """
              f"{m['dashboards']} dashboards, {m['queries']:,} queries, "
              f"{m['distinct']:,} distinct")
        print(f"  raw {m['raw_pct']}%  v1 {m['v1_pct']}%  v2 {m['v2_pct']}%  "
              f"residual {m['residual']} {m['blockers']}")
        print(f"  of those, {m['shared_distinct']:,} strings the corpus also holds; "
              f"on the {m['novel_distinct']:,} it does not: "
              f"v1 {m['novel_v1_pct']}%  v2 {m['novel_v2_pct']}%")
        print(f"  residual by kind: {m['residual_kinds']}")
        print(f"  vocabulary ceiling {m['ceiling_occ_pct']}% of occurrences, "
              f"{len(m['novel_features'])} novel features {list(m['novel_features'])}")
        for r in m["residual_examples"]:
            print(f"    [residual] {r}")
        rows.append([label.replace("@N", f"${m['dashboards']}$"),
                     f"{m['queries']:,}", f"{m['distinct']:,}",
                     pct(m["raw_pct"] / 100), pct(m["v1_pct"] / 100),
                     pct(m["v2_pct"] / 100), f"{m['residual']:,}"])
        if m["shared_distinct"]:
            rows.append([r"\quad its strings the corpus lacks", "---",
                         f"{m['novel_distinct']:,}", "---",
                         pct(m["novel_v1_pct"] / 100), pct(m["novel_v2_pct"] / 100),
                         "---"])

    dd = [q for q in held if q.source == "integrations-core" and q.lang == "datadog"]
    if dd:
        res["kappa"] = kappa(dd)
        k = res["kappa"]
        print(f"\nfrozen translator on the later commit: kappa = {k['kappa_occ_pct']}% "
              f"of {k['occurrences']:,} occurrences, {k['kappa_distinct_pct']}% of "
              f"{k['distinct']:,} distinct")
        print(f"  outcomes {k['by_outcome_occ']}")
        print(f"  gaps {k['gaps']}")

    if rows:
        tabular("lrrrrrr",
                ["Population", "Queries", "Distinct", "Raw", "v1", "v2",
                 "Residual"], rows,
                TABLES / "heldout_coverage.tex")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "heldout_eval.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")

    fig: dict = {}
    if "grafana" in res:
        g = res["grafana"]
        fig.update({
            "ho.gf_dashboards": g["dashboards"], "ho.gf_queries": g["queries"],
            "ho.gf_distinct": g["distinct"], "ho.gf_raw_pct": g["raw_pct"],
            "ho.gf_v1_pct": g["v1_pct"], "ho.gf_v2_pct": g["v2_pct"],
            "ho.gf_residual": g["residual"],
            "ho.gf_ceiling_pct": g["ceiling_occ_pct"],
            "ho.gf_novel_features": len(g["novel_features"]),
            "ho.gf_shared_distinct": g["shared_distinct"],
            "ho.gf_residual_metric": g["blockers"].get("METRIC", 0),
            "ho.gf_residual_matchop": g["blockers"].get("MATCHOP", 0),
            "ho.gf_residual_other": g["residual"] - g["blockers"].get("METRIC", 0)
                                    - g["blockers"].get("MATCHOP", 0),
            "ho.gf_residual_logql": g["residual_kinds"]["logql"],
            "ho.gf_residual_metricsql": g["residual_kinds"]["metricsql"],
            "ho.gf_residual_old_template": g["residual_kinds"]["old_template"],
            "ho.gf_residual_promql": g["residual_kinds"]["promql"],
            "ho.gf_catalogue": manifest["grafana"]["available"],
            "ho.gf_downloaded": manifest["grafana"]["downloaded"],
            "ho.gf_novel_distinct": g["novel_distinct"],
            "ho.gf_novel_v1_pct": g["novel_v1_pct"],
            "ho.gf_novel_v2_pct": g["novel_v2_pct"],
        })
    if "integrations-core" in res:
        d = res["integrations-core"]
        fig.update({
            "ho.dd_dashboards": d["dashboards"], "ho.dd_queries": d["queries"],
            "ho.dd_distinct": d["distinct"], "ho.dd_v1_pct": d["v1_pct"],
            "ho.dd_v2_pct": d["v2_pct"], "ho.dd_residual": d["residual"],
            "ho.dd_ceiling_pct": d["ceiling_occ_pct"],
            "ho.dd_novel_features": len(d["novel_features"]),
        })
    fig.update({"ho.gf_in_sample_distinct": gi["distinct"],
                "ho.gf_in_sample_queries": gi["queries"],
                "ho.gf_in_sample_raw_pct": gi["raw_pct"],
                "ho.gf_in_sample_residual": gi["residual"],
                "ho.gf_in_sample_v1_pct": gi["v1_pct"],
                "ho.gf_in_sample_v2_pct": gi["v2_pct"],
                # What the corpus-driven productions are worth, in sample and out of it.
                # The pair is the point, so both are recorded rather than subtracted in
                # the prose.
                "ho.gf_in_sample_gain_points": round(gi["v2_pct"] - gi["v1_pct"], 1)})
    if "grafana" in res:
        g = res["grafana"]
        fig["ho.gf_novel_gain_points"] = round(g["novel_v2_pct"] - g["novel_v1_pct"], 1)
        fig["ho.gf_listed_to_rank"] = manifest["grafana"]["listed_to_rank"]
    if "kappa" in res:
        fig["ho.dd_kappa_pct"] = res["kappa"]["kappa_occ_pct"]
        fig["ho.dd_kappa_occurrences"] = res["kappa"]["occurrences"]
        fig["ho.dd_kappa_distinct"] = res["kappa"]["distinct"]
    print("\nregistry:")
    for k_, v_ in sorted(fig.items()):
        print(f"  {k_:<28} {v_}")
    if args.write_registry:
        reg = json.loads(REGISTRY.read_text())
        reg.update(fig)
        REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
        print(f"\nmerged {len(fig)} keys into {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

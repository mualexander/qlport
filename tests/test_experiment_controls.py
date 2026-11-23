"""Whether each experiment could have come out the other way.

Every other test about the experiments asks whether a quoted figure matches its cache.
None asked the prior question: could something other than the stated cause have produced
the observation? That is not idle. E14's first run read back points and reasoned from
them, and the reasoning was one observation short -- a scope term the parser *discards*
returns exactly what a term matching *everything* returns -- so the run was re-done with
three negation controls before its claim could stand.

So this file is the inventory of how each experiment discriminates, and the checks that
the discrimination actually happened. What it found: the designs are sound, including the
two where the weakness would have been easiest to have -- E2 probes on the grid *and*
mid-bucket because one probe cannot separate closure from labelling, and E6 submits at
10 s while rolling up over 30 s because a harness scraping at the rollup width cannot see
the order of `.as_rate()` at all. What it also found is that E4's cache had recorded
E18's second finding two weeks before anyone read it, which is the reason
`rem:rollup-spans` now says so.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from texsource import builds, flat

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# Every experiment cache, and how it rules the alternative explanation out. The kinds:
#   "predictions"  a table of what each candidate rule would give, with one match
#   "narrowing"    a `consistent_with` list the observation cuts down
#   "controls"     probes whose expected answers are checked alongside the result
#   "negative"     a case the claim says must come back empty or rejected
#   "separation"   a workload built so two candidate rules give different answers
# A cache with none of these is an experiment whose result cannot be argued with, which
# is the state E14 was in before it was re-run.
DISCRIMINATION: dict[str, tuple[str, ...]] = {
    "datadog/e1.json": ("predictions", "narrowing"),
    "datadog/e1_fast.json": ("predictions", "narrowing"),
    "datadog/e2.json": ("narrowing", "separation"),
    "datadog/e3.json": ("predictions",),
    "datadog/e4.json": ("controls",),
    "datadog/e5.json": ("controls", "negative"),
    "datadog/as_rate_order.json": ("separation", "controls"),
    "datadog/e7.json": ("controls",),
    "datadog/e8.json": ("negative", "separation"),
    "datadog/e9.json": ("separation",),
    "datadog/e12.json": ("controls", "negative"),
    "datadog/e13.json": ("negative",),
    "datadog/e14.json": ("controls", "negative"),
    # E15's discrimination is the COUNT series beside the RATE one: the same two
    # rollups asked of both, so a rescaling of the RATE metric would show as a
    # difference between them. Its `note` says which comparison to make.
    "datadog/e15.json": ("controls",),
    "datadog/e16.json": ("negative", "controls"),
    "datadog/e17_scope_length.json": ("controls", "negative"),
    "datadog/e18.json": ("controls", "separation"),
    "datadog/scope_empty_tag.json": ("predictions",),
    "datadog/top.json": ("predictions", "controls"),
    "gap_rows/e10_fill.json": ("negative", "separation"),
    "gap_rows/e11_label_name.json": ("controls", "negative"),
}


def test_every_experiment_cache_is_accounted_for_here():
    """A new experiment has to say how it discriminates, or this list stops meaning
    anything. `type_declared.json` is excluded: it joins a repository rather than
    asking a system, so there is no alternative cause to rule out."""
    on_disk = {f"{d}/{p.name}" for d in ("datadog", "gap_rows")
               for p in (RAW / d).glob("*.json")} - {"datadog/type_declared.json"}
    assert on_disk == set(DISCRIMINATION), {
        "cached but not accounted for": sorted(on_disk - set(DISCRIMINATION)),
        "accounted for but not cached": sorted(set(DISCRIMINATION) - on_disk),
    }


@pytest.mark.parametrize("cache", sorted(DISCRIMINATION))
def test_each_cache_carries_the_evidence_its_kind_implies(cache):
    d = json.loads((RAW / cache).read_text())
    kinds = DISCRIMINATION[cache]
    blob = json.dumps(d)
    if "narrowing" in kinds:
        assert "consistent_with" in blob, cache
    if "controls" in kinds:
        assert any(k in blob for k in ("control", "stored", "gauge", "e4_expected",
                                       "too_fine", "attempts")), cache
    if "predictions" in kinds:
        assert any(k in blob for k in ("predict", "model", "expected", "prediction")), \
            cache


# --- the three where the discrimination is the whole design ---------------------------

def test_the_reset_policy_narrows_to_one_only_where_it_can():
    """E1's three cases, and which of them does the work.

    A run with no reset is consistent with every policy; a reset to zero cannot
    separate `zero` from `clamp`, because both predict the same bucket; only a reset to
    a non-zero value separates all four. The paper's claim rests on the third case and
    the other two are what show the first two could not have carried it.
    """
    cases = json.loads((RAW / "datadog" / "e1.json").read_text())["cases"]
    assert set(cases["none"]["consistent_with"]) == {"zero", "clamp", "drop", "raw"}
    assert set(cases["to0"]["consistent_with"]) == {"zero", "clamp"}
    assert cases["to100"]["consistent_with"] == ["zero"]
    # And the narrowing is the policy table, not an assertion beside it.
    policies = cases["to100"]["policies"]
    assert [k for k, v in policies.items() if v["consistent"]] == ["zero"]


def test_the_bucket_convention_needed_two_probes_and_got_them():
    """E2: one probe cannot answer it, which is why there are two.

    A point exactly on the grid is reported at the same timestamp under ``right-closed,
    labelled by end'' as under ``left-closed, labelled by start''. The mid-bucket probe
    fixes the labelling and the on-grid probe then fixes the closure, so the pair cuts
    four readings to one. This is the experiment whose subject is the convention the
    corpus-scale differential was later found unable to see, so its own grid mattering
    is not a hypothetical.
    """
    d = json.loads((RAW / "datadog" / "e2.json").read_text())
    probes, interp = d["provenance"]["probes"], d["interpretation"]
    delta = d["provenance"]["delta"]
    # The two probes are half a bucket apart, which is what separates the readings.
    assert probes["midbucket"] - probes["ongrid"] == delta / 2
    assert interp["conclusive"] is True
    assert interp["consistent_with"] == [{"closure": "left", "label": "start"}]


def test_the_as_rate_order_workload_separates_the_two_orders():
    """E6: the submission grid is deliberately not the rollup width.

    `.as_rate().rollup(avg, R)` could divide each delta by its own interval and average
    the rates, or average the deltas and divide once by R. The two agree whenever the
    grid equals R, so a harness that scrapes at the rollup width cannot see the
    difference -- which is a fact about harnesses this project learned elsewhere, the
    hard way. The constants are the design.
    """
    import re

    source = (ROOT / "scripts" / "as_rate_order.py").read_text()
    step = float(re.search(r"^STEP = ([\d.]+)", source, re.M).group(1))
    rollup = float(re.search(r"^ROLLUP = ([\d.]+)", source, re.M).group(1))
    assert step != rollup, (step, rollup)
    prov = json.loads((RAW / "datadog" / "as_rate_order.json").read_text())
    assert prov["step"] == step and prov["rollup"] == rollup
    # And the gauge probe, which is the undocumented case the modifier says nothing
    # about, plus the `count:` aggregator, which is the only one that does not commute
    # with dividing by a constant and so the only one the order is visible through.
    assert "gauge_level" in prov
    assert any("count:" in q for q in json.dumps(prov["queries"]).split('"'))


# --- what a cache knew before the paper did -------------------------------------------

def test_e4_recorded_the_divergence_e18_is_credited_with_finding():
    """`rem:rollup-spans` says so, and this is why it can.

    E4 asked the automatic interval at five ranges and its cache stored both what the
    response reported and what the returned points were spaced at. They differ at two
    of the five, which is E18's second finding, recorded a fortnight before anybody read
    it. The remark now says that rather than claiming the observation as new, and this
    test is what keeps the sentence honest.
    """
    interp = json.loads((RAW / "datadog" / "e4.json").read_text())["interpretation"]
    reported, observed = interp["reported_by_range"], interp["observed_by_range"]
    diverging = {k for k in reported if reported[k] != observed[k]}
    assert diverging == {"1m", "5m"}, diverging
    assert reported["1m"] == 1 and observed["1m"] == 10.0
    for remark in builds(ROOT / "paper" / "sections" / "moved" / "ing_rollup_spans.tex"):
        assert "it was not new" in remark
        assert "two of its five ranges" in remark


def test_the_controls_of_the_two_experiments_that_were_re_run_all_hold():
    """E14 and E17 are the two whose first runs were not conclusive, so their controls
    are the evidence that the second runs were."""
    e14 = json.loads((RAW / "datadog" / "e14.json").read_text())
    assert e14["interpretation"]["conclusive"] is True
    assert e14["interpretation"]["controls_hold"] is True
    assert all(e14["controls"].values()), e14["controls"]
    e17 = json.loads((RAW / "datadog" / "e17_scope_length.json").read_text())
    assert all(c["holds"] for c in e17["controls"]), e17["controls"]
    assert {c["control"] for c in e17["controls"]} >= {
        "bare-scope-present", "bare-scope-absent"}


# --- what the extractor does not reach -------------------------------------------------
# The corpus is the layer under every figure in the paper, and the only check it had was
# that `fetch_corpus.py --extract-only` reproduces it. That proves the extractor is
# deterministic, not that it is complete. `rem:extraction-fixes` now states the edge of
# the population; these are the counts it states.

EXTRACTION_EDGE = {"annotations": 9, "plugin_wrapper": 5}


def _source_strings() -> dict[str, set[str]]:
    """Query-shaped strings in the source trees, by the path kind that holds them."""
    import json as _json

    out: dict[str, set[str]] = {k: set() for k in
                                ("annotations", "plugin_wrapper", "fill", "metadata")}
    for path in sorted((ROOT / "data" / "corpus" / "sources").rglob("*.json")):
        try:
            doc = _json.loads(path.read_text())
        except (ValueError, OSError):
            continue

        def walk(node, trail):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, trail + [str(key)])
            elif isinstance(node, list):
                for value in node:
                    walk(value, trail + ["[]"])
            elif isinstance(node, str) and node.strip():
                flat = ".".join(t for t in trail if t != "[]")
                if flat == "annotations.list.expr":
                    out["annotations"].add(node.strip())
                elif "CustomPanel.targets.expr" in flat:
                    out["plugin_wrapper"].add(node.strip())
                elif flat.endswith("requests.fill.q"):
                    out["fill"].add(node.strip())
                elif flat.endswith("requests.metadata.expression"):
                    out["metadata"].add(node.strip())

        walk(doc, [])
    return out


@pytest.fixture(scope="module")
def corpus_texts() -> set[str]:
    import json as _json

    path = ROOT / "data" / "corpus" / "queries.jsonl"
    return {_json.loads(line)["text"].strip() for line in path.read_text().splitlines()
            if line.strip()}


def test_the_two_kinds_outside_the_corpus_are_the_two_the_remark_names(corpus_texts):
    """And at the counts it gives, so the stated edge is the measured one."""
    found = _source_strings()
    for kind, expected in EXTRACTION_EDGE.items():
        missing = {q for q in found[kind] if q not in corpus_texts}
        assert len(missing) == expected, (kind, len(missing), sorted(missing)[:3])
    for remark in builds(ROOT / "paper" / "sections" / "moved"
                         / "eval_extraction_fixes.tex"):
        assert "$9$ Grafana" in remark and "$5$ more behind" in remark


def test_the_two_correct_exclusions_really_are_duplicates(corpus_texts):
    """A `fill` sub-request restates the widget's own query, and a request's metadata
    names each component of a comma-separated one to hang an alias on it.

    Taking either would count a query twice. The first is checked outright -- every one
    of those strings is already in the corpus. The second cannot be, because the corpus
    holds the comma-joined string rather than its components, so what is checked is
    that each component appears inside some corpus query.
    """
    found = _source_strings()
    assert found["fill"], "no fill sub-requests found; the census broke"
    assert all(q in corpus_texts for q in found["fill"]), sorted(
        q for q in found["fill"] if q not in corpus_texts)[:3]

    import re

    joined = "\n".join(corpus_texts)
    orphans = [q for q in found["metadata"]
               if q not in corpus_texts and q not in joined]
    # Two of the fifty-one are neither: they preserve an earlier version of their own
    # request's query, from before the author replaced a `{*}` scope with template
    # variables. The dashboard carries a stale copy of its own query, which is not a
    # query it runs -- so the test is that each names a metric the corpus does hold.
    assert len(orphans) <= 2, orphans[:4]
    for q in orphans:
        metric = re.match(r"[a-z0-9]+:([A-Za-z0-9_.]+)", q)
        assert metric, q
        assert metric.group(1) in joined, q


def test_the_extractor_reproduces_the_committed_corpus():
    """`--extract-only` over the cached sources must give back the committed file.

    Determinism rather than completeness, which is the distinction the remark draws,
    and the cheaper half of it: the counts above are what completeness costs to check.
    """
    import subprocess

    path = ROOT / "data" / "corpus" / "queries.jsonl"
    before = path.read_bytes()
    proc = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "fetch_corpus.py"), "--extract-only"],
        capture_output=True, cwd=ROOT, timeout=600)
    try:
        assert proc.returncode == 0, proc.stderr.decode()[-400:]
        assert path.read_bytes() == before, "re-extraction changed the corpus"
    finally:
        path.write_bytes(before)


def test_the_span_table_written_by_hand_is_the_table_e18_recorded():
    r"""Ten spans, two rows, typed into a section file and checked by nothing.

    `tables/` is generated and `test_only_four_hours_takes_the_emitted_interval` reads
    the cache, but this table is a `tabular` inside `moved/ing_rollup_spans.tex`, so it
    is neither. Changing the observed interval at `1h` from $20$ to $60$ and the
    sentence under it from "out by a factor of three" to "exactly right" answered RQ4
    backwards with the suite green: the emitter's assumed 60 seconds would have been
    right for the modal span, repudiating section 7.6 and
    `rs.modal_declared_span_factor`.

    Both rows are in E18's own cache, so both are read off it here. The reported row is
    what a response says its interval is; the observed row is the spacing of the
    samples it actually returned, and the gap between them is the finding.
    """
    interp = json.loads((RAW / "datadog" / "e18.json").read_text())["interpretation"]
    reported, observed = interp["reported_by_range"], interp["observed_by_range"]

    rows = {}
    for text in builds(ROOT / "paper" / "sections" / "moved" / "ing_rollup_spans.tex"):
        head, body = None, {}
        for line in text.split("\n"):
            if line.startswith("Range &"):
                head = line
            elif head is not None and line.strip().startswith(("Reported", "Observed")):
                body[line.split("&")[0].strip()] = line
        assert head and set(body) == {"Reported (s)", "Observed (s)"}, sorted(body)
        # The header's span names, and each row's cells, both wrap across two lines in
        # the source, so the table is taken from the flattened text a reader sees.
        flat_text = flat(text)
        spans = re.search(r"Range ((?:& \\texttt\{\w+\}\s*)+)\\\\", flat_text)
        assert spans, "the span header no longer parses"
        names = re.findall(r"\\texttt\{(\w+)\}", spans.group(1))
        assert len(names) == 10, names
        for label, key in (("Reported", reported), ("Observed", observed)):
            cells = re.search(rf"{label} \(s\) ((?:& [^\\]+)+)\\\\", flat_text)
            assert cells, f"the {label} row no longer parses"
            values = [c.strip() for c in cells.group(1).split("&") if c.strip()]
            assert len(values) == 10, (label, values)
            rows[label] = dict(zip(names, values))

    for span, printed in rows["Reported"].items():
        assert _as_seconds(printed) == reported[span], (
            f"the table reports {printed} at {span}; E18 recorded {reported[span]}")
    for span, printed in rows["Observed"].items():
        want = observed[span]
        if want is None:
            assert printed == "---", (
                f"E18 read no spacing at {span} and the table prints {printed}")
            continue
        assert _as_seconds(printed) == want, (
            f"the table observes {printed} at {span}; E18 measured {want}")
    # And the sentence the second row exists to support, which is the one an adversary
    # inverted: the modal declared span is out by the factor the cache computes.
    factor = json.loads(
        (ROOT / "data" / "processed" / "rollup_spans.json").read_text())["figures"][
            "rs.modal_declared_span_factor"]
    assert factor > 1, "the modal span is no longer out by a factor at all"
    for text in builds(ROOT / "paper" / "sections" / "moved" / "ing_rollup_spans.tex"):
        assert "out by a factor of three" in flat(text), (
            "the remark no longer states the divergence the second row shows")


def _as_seconds(cell: str) -> int:
    """One table cell as a number: `$3{,}600$` is 3600."""
    return int(cell.replace("$", "").replace("{,}", "").replace(",", "").strip())


def test_every_anchored_figure_is_the_one_the_cache_produced():
    r"""Six registry keys that were the only unchecked family left.

    `na.*` is the whole evidence for Proposition~\ref{prop:anchored}'s second branch,
    and `scripts/verify_anchored.py` wrote the keys straight to the registry with no
    cache-to-registry pin of the kind `ho.*` has. Registry and prose moved together
    and nothing objected: an adversary took `na.no_anchor_plain` from $31.7$ to $51.7$
    and rewrote the proof to match, which changes the extrapolation control reading
    that separates the proposition from one of its two plausible failures.

    Every one is re-derived here from `data/raw/anchored/no_anchor_60.json`, which is
    committed, so this needs no server and no live Prometheus.
    """
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "scripts"))
    import verify_anchored as va

    payload = json.loads(
        (RAW / "anchored" / "no_anchor_60.json").read_text())
    sums = {float(k): v for k, v in payload["model_bucket_sums"].items()}
    t0, interval = payload["t0"], payload["interval"]
    read = {label: {round(t - t0 - interval + va.EVAL_EPSILON): v
                    for t, v in va.series_from_response(c["response"]).items()}
            for label, c in payload["candidates"].items()}
    anchored, plain = read["increase(x[D] anchored)"], read["increase(x[D])"]
    buckets = sorted(sums)
    exact = [b for b in buckets if abs(anchored.get(b, 1e30) - sums[b]) < 1e-9]
    b0 = buckets[0]
    series = va.no_anchor_series()

    want = {
        "na.buckets": len(buckets),
        "na.buckets_exact": len(exact),
        "na.no_anchor_bucket_sum": round(sums[b0], 6),
        "na.no_anchor_anchored": round(anchored[b0], 6),
        "na.no_anchor_plain": round(plain[b0], 1),
        "na.first_value": round(float(series.vs[0]), 6),
    }
    reg = json.loads((ROOT / "paper" / "quoted_figures.json").read_text())
    wrong = [f"  {k}: registry {reg.get(k)!r}, cache gives {v!r}"
             for k, v in want.items() if reg.get(k) != v]
    assert not wrong, "\n".join(wrong)

    # And the two relations the proof turns on, so that a consistent edit of all six
    # still has to be a consistent edit of the physics.
    assert want["na.no_anchor_anchored"] == want["na.no_anchor_bucket_sum"], (
        "the anchored form no longer returns the unextrapolated sum, which is what "
        "the proposition needs")
    assert want["na.no_anchor_plain"] > want["na.no_anchor_bucket_sum"], (
        "plain increase no longer overshoots, so the extrapolation control reading "
        "the proof quotes is not a control for anything")
    assert want["na.first_value"] > 10 * want["na.no_anchor_bucket_sum"], (
        "the first sample's value no longer separates the reading from the failure "
        "where the form takes that sample for a delta")

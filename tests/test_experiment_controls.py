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
from collections import Counter
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


# --- what each cache says about itself, replayed from its own evidence ----------------
# Eleven caches -- eight Datadog probes, both gap-row runs and the matcher reading --
# held 1,468 numbers between them, and changing all 1,468 at once failed three tests,
# all three of them E14's and all three about the same two fields. So ten of the eleven
# were held by nothing whatever, and the eleventh by three assertions. This was measured
# rather than reasoned about; four of the caches were also perturbed one at a time
# against the whole suite, which is where the per-cache verdicts come from.
#
# The caches are not unread. The suite reads their verdicts, which are the booleans and
# set memberships the scripts wrote, and reads them closely. What nothing read was the
# arithmetic those verdicts are supposed to rest on: a cache could say `stored` where
# its own response said nothing of the kind, and the response could report a window
# that excludes the point it carries.
#
# The first attempt at that measurement was wrong and is worth recording. It ran, for
# each cache, only the tests a `sys.addaudithook` trace saw open that file, which is
# sound in principle: `data/processed` is not rebuilt inside a run, so nothing else can
# see a `data/raw` edit. It missed E14's three readers anyway, because these caches come
# through module-scoped fixtures: the file is opened once, the trace attributes the open
# to whichever test ran first, and the other tests on the same fixture look blind. It
# reported eleven of eleven green, which is one cache too many.
#
# What these tests do not do is re-measure. The trial account has lapsed and these runs
# cannot be repeated, so there is no second source for any of it; what there is, in each
# cache, is a chain -- the points the rig submitted, the vendor's acknowledgement, the
# response it sent, and the summary the script wrote from that response. Each test below
# walks one of those chains by importing the script's own derivation and running it over
# the cached response. A drift in either end now has to be a consistent drift in both,
# which is a re-run rather than an edit.
#
# E8 is the one where that is not quite the shape. It caches no response, only the two
# sides' answers, and a test already compares them -- which is why the uniform
# perturbation could not catch it: adding 1 to both sides of an equality keeps it. Its
# numbers are recounted here from the four series the experiment declares, so the
# arithmetic is held and not just the agreement.


def _script(name: str):
    """A script imported for its own derivation, the way `verify_anchored` already is."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    return __import__(name)


def _raw_values(raw: dict) -> list[float]:
    return [float(v) for s in (raw.get("series") or [])
            for _t, v in (s.get("pointlist") or []) if v is not None]


def _raw_seconds(raw: dict) -> set[int]:
    return {int(t / 1000) for s in (raw.get("series") or [])
            for t, v in (s.get("pointlist") or []) if v is not None}


# Which response in each cache is the control: the one asked with no scope under test,
# whose job is to show the points were stored at all. `scope_empty_tag` keeps its
# probes under another name and calls that one `all`.
CONTROL_RESPONSE = {
    "datadog/e5.json": ("responses", "control"),
    "datadog/e12.json": ("responses", "control"),
    "datadog/e13.json": ("responses", "control"),
    "datadog/e14.json": ("responses", "control"),
    "datadog/e16.json": ("responses", "control"),
    "datadog/scope_empty_tag.json": ("probes", "all"),
}


def test_every_cached_response_agrees_with_its_own_bookkeeping():
    """Each response's metadata against its own points, over every Datadog cache.

    The window asked for, the window the vendor echoes, the per-series start and end,
    and the per-series length are all numbers in these files, and they are the numbers
    that say which run a response belongs to. None of them was read anywhere: a
    response could report a length of two over one point, or a window that excludes
    the point it carries, which is the shape a copied-and-edited cache would have.

    Over every cache rather than the eleven, because the check costs nothing and the
    E4 cache is the standing argument for breadth here -- it had carried E18's second
    finding for two weeks before anybody read that far into it.
    """
    seen = 0
    for path in sorted((RAW / "datadog").glob("*.json")):
        d = json.loads(path.read_text())
        entries = d.get("responses") or d.get("probes") or {}
        for name, entry in (entries if isinstance(entries, dict) else {}).items():
            raw = entry.get("raw") if isinstance(entry, dict) else None
            if not isinstance(raw, dict):
                continue
            seen += 1
            where = f"{path.name}/{name}"
            assert entry.get("query"), (
                f"{where}: the cache does not say what was asked, so the echo below "
                f"would be compared against nothing")
            assert raw.get("query") == entry.get("query"), (
                f"{where}: the response is to {raw.get('query')!r}, not to the "
                f"{entry.get('query')!r} the cache says was asked")
            if "from" in entry and "from_date" in raw:
                assert raw["from_date"] == int(entry["from"]) * 1000 and \
                    raw["to_date"] == int(entry["to"]) * 1000, (
                    f"{where}: the echoed window is not the one asked for")
            for s in raw.get("series") or []:
                pts = s.get("pointlist") or []
                assert s.get("length") == len(pts), (
                    f"{where}: {s.get('scope')} reports length {s.get('length')} over "
                    f"{len(pts)} points")
                stamps = [t for t, _v in pts]
                if stamps and "from_date" in raw:
                    assert raw["from_date"] <= min(stamps) and \
                        max(stamps) <= raw["to_date"], (
                        f"{where}: {s.get('scope')} carries points outside the window")
                if stamps and "start" in s:
                    assert s["start"] <= min(stamps) and max(stamps) <= s["end"], (
                        f"{where}: {s.get('scope')} carries points outside its own "
                        f"start and end")
    assert seen >= 35, f"only {seen} responses were examined; the scan found nothing"


@pytest.mark.parametrize("cache", sorted(CONTROL_RESPONSE))
def test_every_submitted_point_is_one_the_control_read_back(cache):
    """The first link: what the rig put in is what the vendor gave back.

    Every one of these experiments reasons from a series being present, so a cache whose
    control response is not the submission it claims to be is reasoning about someone
    else's data. Values as multisets rather than per series, because the scope a series
    comes back under is the vendor's construction and E16 shows it is not the submitted
    tag list -- a tag key the query groups by and the series does not carry comes back
    as `N/A`.
    """
    d = json.loads((RAW / cache).read_text())
    block, key = CONTROL_RESPONSE[cache]
    raw = d[block][key]["raw"]
    series = [s for p in d["submitted"] for s in p["series"]]
    points = [pt for s in series for pt in s["points"]]

    assert len(series) == len(raw.get("series") or []), (
        f"{cache}: {len(series)} series submitted, "
        f"{len(raw.get('series') or [])} came back under {key!r}")
    assert sorted(round(p["value"], 6) for p in points) == \
        sorted(round(v, 6) for v in _raw_values(raw)), (
        f"{cache}: the control's values are not the submitted ones")
    assert {int(p["timestamp"]) for p in points} == _raw_seconds(raw), (
        f"{cache}: the control's timestamps are not the submitted ones")

    t_star = d["provenance"].get("t_star", d["provenance"].get("t0"))
    assert min(int(p["timestamp"]) for p in points) == int(t_star), (
        f"{cache}: the first submitted point is not at t_star")
    for ack in d.get("acks", []):
        assert not ack.get("errors"), f"{cache}: the vendor rejected a submission: {ack}"


# The summary field each glob experiment wrote, and the response it was written from.
# `trial_checks._series_by_scope` is the function that wrote them.
BY_SCOPE = {
    "datadog/e5.json": {"stored": "control", "selected": "globbed"},
    "datadog/e12.json": {"stored": "control"},
    "datadog/e13.json": {"stored": "control", "selected": "asked"},
    "datadog/e16.json": {"stored": "control"},
}


@pytest.mark.parametrize("cache", sorted(BY_SCOPE))
def test_every_glob_probe_recounts_its_selection_from_its_own_response(cache):
    """`stored` and `selected` are the scripts' readings of responses in the same file.

    Nothing compared the two. E5's reading is the whole of rem:star-empty -- that a
    glob's `*` matches the empty string -- and it is a comparison of which scopes
    appear in two of these maps, so a map that is not its response's own content is the
    only thing the remark rests on.
    """
    tc = _script("trial_checks")
    d = json.loads((RAW / cache).read_text())
    for field, response in BY_SCOPE[cache].items():
        want = tc._series_by_scope(d["responses"][response]["raw"])
        assert d[field] == pytest.approx(want), (
            f"{cache}: `{field}` is not a recount of the {response!r} response\n"
            f"  cached: {d[field]}\n  response gives: {want}")

    # And the negative half, where there is one: a probe the parser refused has no
    # selection, and the cache has to say so in both places or it is saying two things.
    rejected = set(d.get("rejected") or {})
    if rejected:
        assert not d.get("selected") and not d.get("selected_points"), (
            f"{cache}: {sorted(rejected)} were rejected, so nothing was selected, "
            f"but the cache records a selection")
        for name, message in (d.get("rejected") or {}).items():
            assert "unable to parse" in message or "Invalid" in message, (
                f"{cache}: {name} is recorded as rejected with {message!r}, which is "
                f"not a parser refusal")


def test_the_negated_match_all_is_read_off_the_echo_and_not_off_the_points():
    r"""E14's three numbers and five verdicts, replayed from the responses.

    rem:negated-all turns on a distinction points cannot make -- a term that matched
    everything and a term the parser discarded return the same series -- so the verdict
    is read off the expression each response says Datadog evaluated. That reading is
    `scope_term_survived`, and nothing checked that the cache's copy of it is what the
    function returns for the cached response. The terms below are checked against the
    query each probe actually asked, so this table cannot drift from the run either.
    """
    dd = _script("trial_checks")
    from qlport.harness.datadog import scope_term_survived

    e14 = json.loads((RAW / "datadog" / "e14.json").read_text())

    terms = {
        "match-all-and-negated": "!*",
        "negated-alone": "!*",
        "absent-value": "rig:zzz",
        "negated-present-value": "!rig:b",
        "negated-absent-value": "!rig:zzz",
    }
    assert set(terms) == set(e14["selected_points"]), (
        "the probes in this table are not the ones the cache holds")

    for name, term in terms.items():
        entry = e14["responses"][name]
        assert term in entry["query"], (
            f"{name}: the cache asked {entry['query']!r}, which does not carry {term!r}")
        assert e14["term_survived_the_parse"][name] is \
            scope_term_survived(term, entry["raw"]), (
            f"{name}: the recorded survival is not what the echo says")
        assert e14["selected_points"][name] == len(dd._defined(entry["raw"])), (
            f"{name}: the recorded point count is not the response's")

    assert e14["stored_points"] == len(dd._defined(e14["responses"]["control"]["raw"])), (
        "stored_points is not the control response's point count")
    controls = {"absent-value": False, "negated-present-value": False,
                "negated-absent-value": True}
    assert e14["controls"] == {n: (e14["selected_points"][n] > 0) == expect
                               for n, expect in controls.items()}, (
        "the controls block is not what the points say")
    # The contrast the remark needs, stated as the inequality rather than as three
    # booleans: something had to fail to select, or "selects the series" says nothing.
    assert e14["selected_points"]["negated-absent-value"] > 0
    assert e14["selected_points"]["absent-value"] == 0
    assert e14["selected_points"]["negated-present-value"] == 0


def test_the_rate_and_count_rollups_are_the_submitted_points_reduced():
    r"""rem:rate-identity's four cells, and the boundary effect it explains.

    The remark prints them in a hand-written table -- $7.0$ and $28.0$ twice -- and
    they are the whole evidence that $\phi_\rho$ is the identity on the $134$
    occurrences naming a vendor-declared `rate`. Nothing read them off the cache, and
    nothing read the cache off its responses.

    The $28$ is the interesting one, because it is the number a reader would take for a
    rescaling. Five points at interval $60$ fall four-and-one across a $300$-second
    rollup boundary, so the sum of the fuller bucket is four times the submitted value
    and the other bucket is one times it. That is asserted here as the split, not as
    the literal $28$: the cache's own note still says to compare the sum against five
    times the submitted value, which is what the arithmetic would be if the boundary
    fell elsewhere, and the remark is the thing that gets this right.
    """
    tc = _script("trial_checks")
    d = json.loads((RAW / "datadog" / "e15.json").read_text())
    want = {name: sorted(tc._defined(r["raw"]).values())
            for name, r in d["responses"].items()}
    assert set(d["values"]) == set(want), (
        f"`values` names {sorted(d['values'])}, the responses {sorted(want)}")
    for name, got in want.items():
        assert d["values"][name] == pytest.approx(got), (
            f"`values[{name}]` is not a recount of its response\n"
            f"  cached: {d['values'][name]}\n  response gives: {got}")

    value = d["interpretation"]["submitted_value"]
    series = [s for p in d["submitted"] for s in p["series"]]
    assert {len(s["points"]) for s in series} == {d["interpretation"]["points"]}, (
        "the submitted series do not carry the number of points the cache counts")
    assert {pt["value"] for s in series for pt in s["points"]} == {value}, (
        "the submitted points are not all the one value the interpretation names")
    assert {s["interval"] for s in series} == {d["provenance"]["interval"]}, (
        "the submitted series are not at the interval the provenance declares")
    points = series[0]["points"]

    # The rollup's buckets are absolute multiples of its width, not offsets from the
    # first point, which is what makes the split four-and-one rather than five-and-none.
    rollup = 300
    bucket = Counter(int(pt["timestamp"]) // rollup for pt in points)
    split = sorted(bucket.values(), reverse=True)
    assert len(split) == 2 and sum(split) == len(points), (
        f"the five points no longer straddle one rollup boundary: {dict(bucket)}")
    assert {b * rollup for b in bucket} == {
        int(t) for t in tc._defined(d["responses"]["rate-sum"]["raw"])}, (
        "the response's buckets are not the ones the submitted points fall in")
    for metric in ("rate", "count"):
        assert d["values"][f"{metric}-avg"] == pytest.approx([value, value]), (
            f"{metric}-avg is no longer the submitted value in both buckets")
        assert d["values"][f"{metric}-sum"] == pytest.approx(
            sorted(n * value for n in split)), (
            f"{metric}-sum is not the submitted value summed over {split} points")
    assert d["values"]["rate-avg"] == pytest.approx(d["values"]["count-avg"])
    assert d["values"]["rate-sum"] == pytest.approx(d["values"]["count-sum"])

    # And the table a reader sees. Two rows of two cells, in every build that carries
    # the remark.
    seen = 0
    for text in builds(ROOT / "paper" / "sections" / "moved" / "reg_trial_probes.tex"):
        for kind in ("avg", "sum"):
            row = re.search(rf"rollup\({kind}, 300\)\}}((?:\s*&\s*\$[\d.]+\$)+)",
                            flat(text))
            assert row, f"the {kind} row of rem:rate-identity's table is not there"
            cells = [float(c) for c in re.findall(r"\$([\d.]+)\$", row.group(1))]
            assert len(cells) == 2, f"the {kind} row prints {cells}, not two cells"
            assert cells == pytest.approx([max(d["values"][f"rate-{kind}"]),
                                           max(d["values"][f"count-{kind}"])]), (
                f"the {kind} row prints {cells}, not what the cache holds")
            seen += 1
    assert seen >= 2, "no build carries rem:rate-identity, so its table went unchecked"


def test_the_binop_pairings_are_the_arithmetic_of_the_four_declared_series():
    """E8's eight cases recounted from the four series the experiment declares.

    thm:arith's premise is that the emitted PromQL pairs series the way Datadog does,
    and the existing tests here compare the two sides to each other. That is the claim,
    but it leaves both sides free together: adding one to every number in this file
    keeps the equality and was the one perturbation of these eleven caches that stayed
    green for a reason other than nothing reading it. What pins them is the workload --
    `a` at 10 and 20, `b` at 2 and 5 -- which is the file's own provenance.
    """
    e8 = json.loads((RAW / "datadog" / "e8.json").read_text())
    series = e8["provenance"]["series"]

    def by(metric: str, keys: tuple[str, ...]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, value in series.items():
            m, tags = name.split(":", 1)
            if m != metric:
                continue
            labels = dict(t.split("=") for t in tags.split(","))
            group = ",".join(f"{k}={labels[k]}" for k in keys)
            out[group] = out.get(group, 0.0) + value
        return out

    a1, b1 = by("a", ("g",)), by("b", ("g",))
    a2, b2 = by("a", ("g", "h")), by("b", ("g", "h"))
    atot, btot = sum(a1.values()), sum(b1.values())
    shared = a1.keys() & b1.keys()
    want = {
        "both_grouped_add": {k: a1[k] + b1[k] for k in shared},
        "both_grouped_div": {k: a1[k] / b1[k] for k in shared},
        "lhs_ungrouped": {k: atot / v for k, v in b1.items()},
        "rhs_ungrouped": {k: v / btot for k, v in a1.items()},
        "rhs_ungrouped_default_matching": {k: v / btot for k, v in a1.items()},
        "subset_lhs_finer": {k: v / b1[k.split(",")[0]] for k, v in a2.items()
                             if k.split(",")[0] in b1},
        "subset_rhs_finer": {k: a1[k.split(",")[0]] / v for k, v in b2.items()
                             if k.split(",")[0] in a1},
    }
    assert set(want) | {"different_keys"} == set(e8["cases"]), (
        "the cases in this recount are not the ones the cache holds")

    for case, expect in want.items():
        got = e8["cases"][case]["datadog"]
        assert "__error__" not in got, f"{case}: {got}"
        assert {k: round(v, 6) for k, v in expect.items()} == \
            pytest.approx({k: v[0] for k, v in got.items() if v}, abs=1e-6), (
            f"{case}: Datadog's answer is not the arithmetic of the declared series")
        if e8["cases"][case]["promql"]:
            assert {k: round(v, 6) for k, v in expect.items()} == \
                pytest.approx(e8["cases"][case]["promql"], abs=1e-6), (
                f"{case}: the emitted PromQL's answer is not that arithmetic either")

    # The two numbers the `by` above cannot derive, which the queries themselves carry:
    # the range the emitted form averages over, and the two groups that make the join
    # an inner one. A workload where both sides carried the same groups would make
    # every case above pass and say nothing about pairing.
    interval = int(e8["provenance"]["interval"])
    assert all(f"[{interval}s]" in c["promql_query"] for c in e8["cases"].values()), (
        f"the emitted forms no longer average over the declared interval of {interval}s")
    assert a1.keys() != b1.keys() and shared, (
        f"the two operands now carry the same groups ({sorted(a1)}), so no case here "
        f"separates an inner join from an outer one")


def test_the_scope_probes_replay_the_prediction_table_they_were_run_against():
    """`scope_empty_tag`'s five probes, recounted with the script's own reader.

    This is the cache behind the claim that Datadog's three tag states do not map onto
    PromQL's two, and the thing it reports is `forms_whose_translation_selects_a_
    different_set` -- four of five. Every number under it was editable: the selections,
    the per-probe verdicts, the count of points, and the grid they sit on.
    """
    se = _script("scope_empty_tag")
    d = json.loads((RAW / "datadog" / "scope_empty_tag.json").read_text())
    p = d["provenance"]

    assert set(d["probes"]) == set(se.PREDICTIONS), (
        "the cached probes are not the prediction table the script runs")
    assert p["series"] == se.SERIES, "the cached workload is not the script's"
    assert (p["step"], p["n_points"]) == (se.STEP, se.N), (
        "the cached grid is not the script's")

    diverging = []
    for form, probe in d["probes"].items():
        predicted, matcher, promql = se.PREDICTIONS[form]
        got = se.selected(probe["raw"])
        assert probe["selected"] == got, (
            f"{form}: the cached selection is not what its own response holds\n"
            f"  cached: {probe['selected']}\n  response gives: {got}")
        scope = se.scope_of(form, p["run"])
        assert probe["query"].endswith(f"{{{scope}}} by {{probe}}"), (
            f"{form}: the cached query is not the one the script builds")
        assert probe["datadog_predicted"] == sorted(predicted), f"{form}: prediction"
        assert probe["promql_matcher"] == matcher, f"{form}: matcher"
        assert probe["promql_selects"] == sorted(promql), f"{form}: promql selection"
        assert probe["matches_prediction"] is (got == sorted(predicted)), f"{form}"
        same = matcher is None or got == sorted(promql)
        assert probe["same_series_set"] is same, f"{form}: same_series_set"
        if not same:
            diverging.append(form)
        # The points, and that they sit on the grid the provenance declares.
        for s in probe["raw"]["series"]:
            seconds = sorted(t / 1000 for t, _v in s["pointlist"])
            assert len(seconds) == se.N, (
                f"{form}: {len(seconds)} points, not the {se.N} declared")
            assert seconds == [p["t0"] + i * se.STEP for i in range(se.N)], (
                f"{form}: the points are not on the declared grid")

    assert d["forms_whose_translation_selects_a_different_set"] == sorted(diverging), (
        "the diverging forms are not the ones the probes diverge on")
    assert d["datadog_matches_prediction"] is all(
        pr["matches_prediction"] for pr in d["probes"].values())
    assert len(diverging) == 4, (
        f"{len(diverging)} of five forms diverge now, not four, which is the number "
        f"rem:empty-tag reports")


def test_the_gap_row_summaries_replay_the_scripts_own_condensation():
    """E10 and E11, recounted with `verify_gap_rows`' own reader.

    `points_past_lookback` is the measurement: whether an expression supplied a value
    where the database holds none, asked outside the staleness window so that anything
    present is the expression's doing. The summaries were editable, the responses were
    editable, and the exposition the server was loaded with was editable -- and that
    last one is what decides where the gap is.
    """
    vg = _script("verify_gap_rows")
    fill = json.loads((RAW / "gap_rows" / "e10_fill.json").read_text())
    names = json.loads((RAW / "gap_rows" / "e11_label_name.json").read_text())

    for cache, d in (("e10_fill", fill), ("e11_label_name", names)):
        assert d["meta"]["exposition"] == vg.exposition(), (
            f"{cache}: the recorded exposition is not the one the script writes")
        for field, value in (("t0", vg.T0), ("step_s", vg.STEP),
                             ("span_min", vg.SPAN), ("stop_min", vg.STOP)):
            assert d["meta"][field] == value, f"{cache}: meta.{field}"

    for name, candidate in fill["candidates"].items():
        assert candidate["summary"] == vg.series_present(candidate["response"]), (
            f"{name}: the cached summary is not its response condensed")
    for name, attempt in names["attempts"].items():
        result = (attempt["response"].get("data") or {}).get("result") or []
        assert attempt["series_returned"] == len(result), (
            f"{name}: series_returned is not the response's series count")

    # What the two caches are for. `svc=b` stops at STOP, so a bare selector has nothing
    # past the lookback and the fill candidates are judged on whether they do -- and the
    # one that holds a value does so at the value the exposition last gave that series.
    bare = fill["candidates"]["bare selector"]["summary"]
    b = next(v for k, v in bare.items() if '"svc": "b"' in k)
    assert b["points_past_lookback"] == 0, (
        "the bare selector now reaches past the lookback, so nothing here is a gap")
    held = fill["candidates"]["last_over_time, held"]["summary"]
    hb = next(v for k, v in held.items() if '"svc": "b"' in k)
    assert hb["points_past_lookback"] > 0 and hb["first_value"] == hb["last_value"], (
        "last_over_time no longer holds a constant past the gap")
    assert float(hb["first_value"]) == 100 + vg.STOP, (
        f"the held value is {hb['first_value']}, not the last one the exposition gave "
        f"svc=b at minute {vg.STOP}")
    assert all(a["series_returned"] == 0 for n, a in names["attempts"].items()
               if not n.startswith("control")), (
        "a pattern in label-name position now returns series, which is the claim")
    assert all(a["series_returned"] > 0 for n, a in names["attempts"].items()
               if n.startswith("control")), (
        "the controls return nothing either, so the refusal above is not about patterns")

    # rem:fill's table is the six candidates and one word each. A candidate counts as
    # reaching past the lookback if any label set carrying svc=b does -- which is how
    # the unioned one does it, under a label set the fill invented and not the stopped
    # series' own. Counted rather than matched row by row, because the table's names
    # for the candidates are prose and the cache's are expressions.
    reaching = {name: any(v["points_past_lookback"] > 0
                          for k, v in c["summary"].items() if '"svc": "b"' in k)
                for name, c in fill["candidates"].items()}
    seen = 0
    for text in builds(ROOT / "paper" / "sections" / "moved" / "eval_gap_rows.tex"):
        block = re.search(r"past the lookback.*?\\bottomrule", flat(text), re.S)
        assert block, "rem:fill's table is not there to check"
        verdicts = re.findall(r"& (absent|present)\b", block.group(0))
        assert len(verdicts) == len(reaching), (
            f"rem:fill's table prints {len(verdicts)} candidates, not the "
            f"{len(reaching)} the cache holds")
        assert sum(1 for v in verdicts if v == "present") == \
            sum(1 for ok in reaching.values() if ok), (
            f"the table calls {sum(1 for v in verdicts if v == 'present')} candidates "
            f"present; the cache has {sum(1 for ok in reaching.values() if ok)}")
        seen += 1
    assert seen >= 2, "no build carries rem:fill, so its table went unchecked"

    # And rem:label-name's table, which is the only place those five readings are
    # printed. Keyed on the verdict column rather than on the row names, because the
    # table calls the first attempt something shorter than the cache does; matching on
    # a name that is nearly the cache's would be the weaker check of the two anyway.
    refused = [a for a in names["attempts"].values() if a["status_code"] != 200]
    assert len(refused) == 2 and all(a["error"] for a in refused), (
        f"{len(refused)} of the five attempts were refused, not the two the table "
        f"prints as parse errors")
    seen = 0
    for text in builds(ROOT / "paper" / "sections" / "moved" / "eval_gap_rows.tex"):
        printed = re.findall(r"& (parse error|accepted, \$(\d+)\$ series) \\\\",
                             flat(text))
        assert len(printed) == len(names["attempts"]), (
            f"rem:label-name's table prints {len(printed)} readings, not the "
            f"{len(names['attempts'])} the cache holds")
        assert sum(1 for verdict, _n in printed if verdict == "parse error") == 2, (
            "the table no longer prints two parse errors")
        assert sorted(int(n) for _v, n in printed if n) == \
            sorted(a["series_returned"] for a in names["attempts"].values()
                   if a["status_code"] == 200), (
            "the series counts in the table are not the ones the cache holds")
        seen += 1
    assert seen >= 2, "no build carries rem:label-name, so its table went unchecked"


def test_the_matcher_readings_replay_the_scripts_own_selection():
    """`label_matchers`, recounted with `verify_matchers`' own reader.

    Six matchers read off a real Prometheus, and the corpus's own regexes submitted to
    it to see which RE2 accepts. The selections were editable and so was the series set
    the server was loaded with, which is what makes a selection mean anything.
    """
    vm = _script("verify_matchers")
    d = json.loads((RAW / "matchers" / "label_matchers.json").read_text())

    assert d["prometheus_version"] in d["image"], (
        "the recorded version is not the image's")
    assert d["series"] == vm.SERIES, "the cached workload is not the script's"
    assert d["collected_unix"] - d["t0"] == 3600, (
        "the backfill is no longer the hour before the run, which is the window the "
        "queries ask over")

    # The exposition the server was loaded with, as {labels: {second: value}}. Taken
    # from the script's own `exposition` rather than rebuilt here, so the values a
    # response reports are held to the ones the backfill wrote and not to a formula
    # restated in this file.
    served: dict[tuple, dict[int, str]] = {}
    for line in vm.exposition(d["t0"]).splitlines():
        if line.startswith("#"):
            continue
        selector, value, second = line.rsplit(" ", 2)
        labels = dict(
            kv.split("=", 1) for kv in
            selector[selector.index("{") + 1:selector.rindex("}")].split(","))
        key = tuple(sorted((k, v.strip('"')) for k, v in labels.items()))
        served.setdefault(key, {})[int(float(second))] = value

    for name, m in d["matchers"].items():
        assert m["selected"] == vm.selected(m["response"]), (
            f"{name}: the cached selection is not its own response\n"
            f"  cached: {m['selected']}\n  response gives: {vm.selected(m['response'])}")
        assert name in m["expr"], f"{name}: the cached expression does not carry it"
        for s in m["response"]["data"]["result"]:
            key = tuple(sorted((k, v) for k, v in s["metric"].items()
                               if k != "__name__"))
            assert key in served, f"{name}: {s['metric']} was never served"
            assert {int(t): v for t, v in s["values"]} == served[key], (
                f"{name}: the points for {s['metric']} are not the ones the "
                f"exposition wrote")

    # The reading itself. The sixth series carries no `k` at all, which is the case the
    # workload exists for: `.*` selects it and `.+` does not, because Prometheus treats
    # an absent label as the empty string. A workload of five labelled series would
    # make every assertion above pass and leave that difference unmeasured.
    def labelled(entries: list[dict]) -> set[tuple]:
        return {tuple(sorted(e.items())) for e in entries}

    star = labelled(d["matchers"]['k=~".*"']["selected"])
    plus = labelled(d["matchers"]['k=~".+"']["selected"])
    assert [dict(e) for e in star - plus] == [{"other": "none"}], (
        f"`.*` and `.+` now differ by {[dict(e) for e in star - plus]}, not by the "
        f"one series with no `k`")
    assert star == labelled(d["series"]), (
        "`.*` no longer selects every series the server was given")

    # And the corpus side, which is the claim that no corpus regex needs anything RE2
    # refuses. Recomputed from the committed corpus, so a regex entering the corpus
    # without a reading here fails rather than passing unnoticed.
    assert sorted(d["corpus_regexes"]) == vm.corpus_regexes(), (
        "the cached regexes are not the ones the translator emits over the corpus")
    refused = sorted(r for r, v in d["corpus_regexes"].items() if not v["accepted"])
    assert not refused, f"RE2 refused {refused}, which §6 says cannot happen"


# Which test above replays each cache's summary from its own evidence, and which caches
# nothing here replays -- their numbers are held by a dedicated test or by the figure
# census instead, which is a weaker thing to rest on and the reason the second list is a
# ratchet. A new experiment has to join one of them, because the test below counts.
REPLAYED: dict[str, tuple[str, ...]] = {
    "test_every_glob_probe_recounts_its_selection_from_its_own_response": (
        "datadog/e5.json", "datadog/e12.json", "datadog/e13.json", "datadog/e16.json"),
    "test_the_negated_match_all_is_read_off_the_echo_and_not_off_the_points": (
        "datadog/e14.json",),
    "test_the_rate_and_count_rollups_are_the_submitted_points_reduced": (
        "datadog/e15.json",),
    "test_the_binop_pairings_are_the_arithmetic_of_the_four_declared_series": (
        "datadog/e8.json",),
    "test_the_scope_probes_replay_the_prediction_table_they_were_run_against": (
        "datadog/scope_empty_tag.json",),
    "test_the_gap_row_summaries_replay_the_scripts_own_condensation": (
        "gap_rows/e10_fill.json", "gap_rows/e11_label_name.json"),
    "test_the_matcher_readings_replay_the_scripts_own_selection": (
        "matchers/label_matchers.json",),
}
UNREPLAYED: tuple[str, ...] = (
    "datadog/e1.json", "datadog/e1_fast.json", "datadog/e2.json", "datadog/e3.json",
    "datadog/e4.json", "datadog/e7.json", "datadog/e9.json",
    "datadog/as_rate_order.json",
    "datadog/e17_scope_length.json", "datadog/e18.json", "datadog/top.json",
)
UNREPLAYED_CEILING = 11


def test_every_experiment_cache_says_which_test_replays_it():
    """The ratchet, and that the names in it are real.

    It was 21 of 21 unreplayed before this section. The eleven left are the caches the
    same measurement found load-bearing already, so the next one to come off this list
    needs a derivation rather than an entry.
    """
    replayed = {c for caches in REPLAYED.values() for c in caches}
    both = replayed & set(UNREPLAYED)
    assert not both, f"listed as both replayed and not: {sorted(both)}"
    missing = set(DISCRIMINATION) - replayed - set(UNREPLAYED)
    assert not missing, (
        f"accounted for above but in neither list here: {sorted(missing)}")
    for test, caches in REPLAYED.items():
        assert test in globals(), f"{test} is not a test in this file"
        assert caches, f"{test} is listed as replaying nothing"
    for cache in replayed | set(UNREPLAYED):
        assert (RAW / cache).exists(), f"{cache} is listed here but not on disk"
    # A cache outside `DISCRIMINATION` is here only because this section replays it,
    # so one of those on the unreplayed list would be an entry saying nothing at all.
    stray = [c for c in UNREPLAYED if c not in DISCRIMINATION]
    assert not stray, f"not experiment caches and replayed by nothing: {stray}"

    assert len(UNREPLAYED) <= UNREPLAYED_CEILING, (
        f"{len(UNREPLAYED)} caches are replayed by nothing here, over a ceiling of "
        f"{UNREPLAYED_CEILING}:\n" + "\n".join(f"  {c}" for c in UNREPLAYED))
    assert len(UNREPLAYED) >= UNREPLAYED_CEILING - 1, (
        f"only {len(UNREPLAYED)} are unreplayed now; lower UNREPLAYED_CEILING to "
        f"{len(UNREPLAYED)} so the ratchet keeps its grip")

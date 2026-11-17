"""The Prometheus harness, and a regression test on the verification it produced.

The last test is the point of the file: `scripts/verify_anchored.py` established that
`semantics/promql_eval.py` models real PromQL, and that result is only worth anything
if it keeps holding. It reads the cached responses in `data/raw/` -- no Docker, no
network -- so a change to the evaluators that breaks the correspondence fails here
rather than quietly invalidating paper §3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qlport.harness.prometheus import (
    openmetrics, quantise, read_raw, series_from_response,
)
from qlport.metrics import reldiff
from qlport.semantics import datadog_eval as dd
from qlport.semantics import promql_eval as pq
from qlport.semantics.ingest import phi_series
from qlport.semantics.model import Series

ROOT = Path(__file__).resolve().parent.parent
ANCHORED = ROOT / "data" / "raw" / "anchored"

MODELS = {
    "rate(x[D])": lambda s, t, D: pq.rate(s.ts, s.vs, t, D),
    "(x - x offset D) / D": lambda s, t, D: pq.offset_diff_rate(s.ts, s.vs, t, D),
    "increase(x[D] anchored) / D": lambda s, t, D: pq.increase_anchored(s.ts, s.vs, t, D) / D,
}


def counter(ts, vs, **labels) -> Series:
    return Series(labels, np.array(ts, dtype=float), np.array(vs, dtype=float),
                  kind="counter")


# --- exposition ----------------------------------------------------------------------
def test_openmetrics_names_the_counter_and_suffixes_the_samples():
    out = openmetrics(counter([15.0, 30.0], [1.0, 2.0]), "qlport_requests", 1000.0)
    lines = out.splitlines()
    assert lines[0] == "# TYPE qlport_requests counter"
    assert lines[1].startswith("qlport_requests_total 1 1015.000")
    assert lines[-1] == "# EOF"


def test_openmetrics_renders_labels():
    out = openmetrics(counter([15.0], [1.0], job="api", env="prod"), "m", 0.0)
    assert 'm_total{env="prod",job="api"} 1 15.000' in out


def test_openmetrics_round_trips_the_float_exactly():
    """Fewer digits would hand the server a different series from the model's."""
    v = 5000.123456789012
    out = openmetrics(counter([15.0], [v]), "m", 0.0)
    assert float(out.splitlines()[1].split()[1]) == v


def test_openmetrics_refuses_a_non_counter():
    gauge = Series({}, np.array([1.0]), np.array([1.0]), kind="gauge")
    with pytest.raises(ValueError):
        openmetrics(gauge, "m", 0.0)


# --- quantisation --------------------------------------------------------------------
def test_quantise_rounds_to_whole_milliseconds():
    got = quantise(counter([1.00049, 2.0005], [1.0, 2.0]))
    assert list(got.ts) == [1.0, 2.001]
    assert got.kind == "counter"


def test_quantise_refuses_to_collide_two_samples():
    with pytest.raises(ValueError, match="collided"):
        quantise(counter([1.00001, 1.00002], [1.0, 2.0]))


# --- reading responses ---------------------------------------------------------------
def test_series_from_response_is_keyed_by_timestamp():
    raw = {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": {}, "values": [[100.0, "1.5"], [160.0, "2.5"]]}]}}
    assert series_from_response(raw) == {100.0: 1.5, 160.0: 2.5}


def test_series_from_response_handles_an_empty_result():
    raw = {"status": "success", "data": {"resultType": "matrix", "result": []}}
    assert series_from_response(raw) == {}


def test_series_from_response_raises_on_an_error():
    raw = {"status": "error", "errorType": "bad_data", "error": "parse error"}
    with pytest.raises(ValueError, match="parse error"):
        series_from_response(raw)


def test_series_from_response_refuses_more_than_one_series():
    raw = {"status": "success", "data": {"result": [
        {"metric": {"a": "1"}, "values": [[1.0, "1"]]},
        {"metric": {"a": "2"}, "values": [[1.0, "2"]]}]}}
    with pytest.raises(ValueError, match="one series"):
        series_from_response(raw)


# --- the verification, as a regression test -------------------------------------------
RUNS = sorted(ANCHORED.glob("*.json"))


def test_the_verification_cache_is_present():
    assert RUNS, "run scripts/verify_anchored.py (needs Docker)"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.stem)
def test_promql_models_agree_with_a_real_prometheus(path):
    """Every candidate, every bucket: our evaluator and the server, to float64.

    A NaN mismatch counts as a failure in either direction -- declining to produce a
    value is as much a semantic claim as producing one.
    """
    rec = read_raw(path)
    t0, width = rec["t0"], rec["interval"]
    s = counter(rec["series"]["ts"], rec["series"]["vs"])
    for label, model in MODELS.items():
        got = series_from_response(rec["candidates"][label]["response"])
        assert got, f"{label} returned nothing"
        for t_abs, server in got.items():
            # Prometheus evaluates on whole milliseconds and reports seconds as a float,
            # so the timestamp comes back with float dirt (599.9990000724792 for
            # 599999 ms). Handing that straight to the model evaluates it a fraction of
            # a microsecond away from where the server did, which `rate`'s extrapolation
            # turns into a 1e-9 relative difference -- a measurement artefact, not a
            # disagreement. Snap to the millisecond the server actually used.
            t = round((t_abs - t0) * 1000) / 1000
            ours = model(s, t, width)
            assert not np.isnan(ours), f"{label} at {t}: server has a value, model does not"
            assert reldiff(server, ours) < 1e-12, f"{label} at {t}: {server} vs {ours}"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.stem)
def test_the_anchored_form_reproduces_datadog_on_every_bucket(path):
    """The §3 claim, read off the server rather than the model.

    The cumulative difference is checked only where no reset falls in the window; the
    anchored form is checked everywhere, which is the whole reason it is interesting.
    """
    rec = read_raw(path)
    t0, width = rec["t0"], rec["interval"]
    eps = rec["grid"]["eps"]
    s = counter(rec["series"]["ts"], rec["series"]["vs"])
    p = phi_series(s)
    anchored = series_from_response(
        rec["candidates"]["increase(x[D] anchored) / D"]["response"])
    for t_abs, server in anchored.items():
        # The server was evaluated strictly inside the bucket, so the bucket it answers
        # for starts one interval earlier: Datadog's bucket is [b, b+width) and PromQL's
        # window is (u-width, u], and only an instant inside the bucket reads the same
        # samples when a scrape lands on the boundary (paper §3.1).
        b = round(t_abs - t0 - width + eps, 3)
        truth = dd.as_rate(p.ts, p.vs, b, width)
        assert reldiff(server, truth) < 1e-12, f"bucket {b}: {server} vs {truth}"


def test_the_runs_record_which_server_answered():
    """A measurement nobody can attribute is not reproducible."""
    for path in RUNS:
        rec = read_raw(path)
        assert rec["prometheus_version"].startswith("3.")
        assert rec["image_digest"].count("@sha256:") == 1
        # The *buckets* are epoch-aligned; the evaluation grid sits one epsilon inside
        # each bucket's right edge, which is what makes it read the same samples as
        # Datadog's left-closed bucket (paper §3.1).
        grid = rec["grid"]
        assert (grid["start"] + grid["eps"]) % rec["interval"] == 0, \
            "bucket grid is not epoch-aligned"
        assert 0 < grid["eps"] < 1, grid["eps"]


def test_the_cache_is_valid_json_and_names_every_candidate():
    for path in RUNS:
        rec = json.loads(path.read_text())
        assert set(rec["candidates"]) == set(MODELS)


# --- what a label matcher matches, as a real server answers it ------------------------
# `lower/datadog.py` translates a single-label Datadog disjunction into one regex
# matcher. That is sound only because Prometheus anchors the whole regex, which is a
# claim about Prometheus. `scripts/verify_matchers.py` asked a real 3.15.0 server; these
# tests hold the answers so a future change to the lowering cannot drift away from them.
MATCHERS = ROOT / "data" / "raw" / "matchers" / "label_matchers.json"


def _matcher_cache() -> dict:
    if not MATCHERS.exists():                       # pragma: no cover - needs Docker
        pytest.skip(f"{MATCHERS.relative_to(ROOT)} missing; run scripts/verify_matchers.py")
    return read_raw(MATCHERS)


def _values(rec: dict, matcher: str) -> list[str]:
    got = rec["matchers"][matcher]["selected"]
    return sorted(d.get("k", "(no k)") for d in got)


def test_alternation_in_a_matcher_is_fully_anchored():
    """`k=~"aws|gcp"` must not select `awsx`, or `k=~"v1|v2"` is the wrong translation.

    The series set includes `awsx` and `xgcp` precisely so that anchoring the whole
    regex, `^(?:aws|gcp)$`, is distinguishable from anchoring each alternative.
    """
    assert _values(_matcher_cache(), 'k=~"aws|gcp"') == ["aws", "gcp"]


def test_a_single_value_matcher_does_not_match_its_extensions():
    assert _values(_matcher_cache(), 'k=~"aws"') == ["aws"]


def test_a_negated_alternation_keeps_series_without_the_label():
    """`k!~"a|b"` includes series carrying no `k` at all, which is what Datadog's
    `NOT (k:a OR k:b)` means: the absent tag is not one of the excluded values."""
    assert _values(_matcher_cache(), 'k!~"aws|gcp"') == ["(no k)", "awsx", "azure", "xgcp"]


def test_present_is_dot_plus_and_not_dot_star():
    """The distinction `_tag_matchers` relies on for a bare tag key."""
    rec = _matcher_cache()
    assert "(no k)" not in _values(rec, 'k=~".+"')
    assert "(no k)" in _values(rec, 'k=~".*"')


def test_glob_alternation_matches_every_branch():
    assert _values(_matcher_cache(), 'k=~".*ws.*|.*cp.*"') == ["aws", "awsx", "gcp", "xgcp"]


def test_every_regex_the_translator_emits_is_one_re2_accepts():
    """Our PromQL grammar does not validate regex syntax, so nothing else checks this.

    Two halves. The recorded run must show every regex accepted; and the set the
    translator emits *today* must be inside the set that was verified, so that a change
    emitting a new regex shape fails here rather than shipping a query Prometheus
    rejects. Re-run `scripts/verify_matchers.py` when it does.
    """
    import re

    from qlport import corpus as c
    from qlport.diagnostics import Bottom
    from qlport.emit.promql import emit_detail
    from qlport.lower.datadog import lower_all
    from qlport.parse import parse

    verified = _matcher_cache()["corpus_regexes"]
    rejected = {r: v["error"] for r, v in verified.items() if not v["accepted"]}
    assert not rejected, rejected

    emitted, seen = set(), set()
    for q in c.read_queries(ROOT / "data" / "corpus" / "queries.jsonl"):
        if q.lang != "datadog":
            continue
        text = c.normalise("datadog", q.text)
        if text in seen or not c.parse_outcome("datadog", text).ok:
            continue
        seen.add(text)
        for e in lower_all(parse("datadog", text)):
            got = e if isinstance(e, Bottom) else emit_detail(e, default_rollup=60.0)
            if not isinstance(got, Bottom):
                emitted.update(re.findall(r'=~\s*"((?:[^"\\]|\\.)*)"', got.promql))
    assert emitted, "no regex matchers emitted at all; the extraction broke"
    assert emitted <= set(verified), sorted(emitted - set(verified))


def test_the_matcher_run_records_which_server_answered():
    rec = _matcher_cache()
    assert rec["prometheus_version"].startswith("3.")
    assert rec["image_digest"].startswith("prom/prometheus@sha256:")


# --------------------- top(): expressible, and what it costs (paper §7) -------

TOPK = Path(__file__).resolve().parent.parent / "data" / "raw" / "topk" / "ranking.json"


@pytest.fixture(scope="module")
def topk_run() -> dict:
    if not TOPK.exists():  # pragma: no cover - run scripts/verify_topk.py
        pytest.skip("ranking.json not recorded; run scripts/verify_topk.py (needs Docker)")
    return json.loads(TOPK.read_text())


def test_top_is_expressible_in_promql(topk_run):
    """The claim §7 classifies `top(...)` on, read off a real server.

    Datadog ranks by an aggregation over the whole range; PromQL's `topk` ranks per
    instant. They are different operators, so this could have been a language gap -- and
    139 occurrences moving to that column would invert §8's comparison. It is not: the
    `@`-pinned construction reproduces the ranking exactly.
    """
    assert topk_run["cases"], "no ranking cases recorded"
    for case in topk_run["cases"]:
        assert case["matches_model"], case
        assert case["constant"], case


def test_every_ranking_function_was_covered(topk_run):
    covered = {c["by"] for c in topk_run["cases"]}
    assert {"max", "mean", "min", "sum", "last"} <= covered, covered


def test_both_ranking_directions_were_covered(topk_run):
    """`topk` is the descending half of `top()`, and the corpus uses both halves.

    Five of the 139 occurrences rank ascending. While only `topk` was probed, the
    verified claim covered 134 of them and the paper said 139; `bottomk` closes that.
    """
    covered = {(c["by"], c["direction"]) for c in topk_run["cases"]}
    assert {d for _, d in covered} == {"desc", "asc"}, sorted(covered)
    for by in ("max", "mean", "min", "sum", "last"):
        assert (by, "asc") in covered and (by, "desc") in covered, by
    selectors = {c["direction"]: c["selector"] for c in topk_run["cases"]}
    assert selectors == {"desc": "topk", "asc": "bottomk"}, selectors


def test_the_workload_is_the_one_the_vendor_was_shown(topk_run):
    """Both halves of the comparison must have seen the same series, or it compares
    nothing. `harness/ranking.py` holds the one definition; this pins the cache to it.
    """
    from qlport.harness import ranking

    recorded = topk_run["provenance"]["series"]
    assert recorded == {k: ranking.values(k) for k in ranking.SERIES}
    assert (topk_run["provenance"]["n_points"], topk_run["provenance"]["step"]) \
        == (ranking.N, ranking.STEP)


def test_the_pinned_ranking_is_constant_while_the_naive_one_is_not(topk_run):
    """Why the `@` modifier is load-bearing rather than decorative."""
    assert all(c["constant"] for c in topk_run["cases"])
    # At least one naive case must drift, or the workload fails to separate the two
    # operators and the comparison above proves nothing.
    assert any(not c["constant"] for c in topk_run["naive"]), topk_run["naive"]


def test_the_expression_needs_a_range_and_an_anchor(topk_run):
    """The point §8 takes from this: the translation is not self-contained.

    Both the window width and the `@` anchor come from the dashboard's time range, which
    the Datadog query text does not carry -- the same omission as the rollup interval.
    """
    prov = topk_run["provenance"]
    assert prov["range_seconds"] and prov["anchor"]
    for case in topk_run["cases"]:
        assert f"[{prov['range_seconds']}s]" in case["expr"], case["expr"]
        assert f"@ {prov['anchor']}" in case["expr"], case["expr"]


def test_the_ranking_window_covers_the_first_sample(topk_run):
    """A range selector is left-open, so a width of exactly (n-1)*step drops it.

    That is where the spike separating `max` from `mean` lives, so getting this wrong
    silently makes the two rankings agree and the experiment prove nothing.
    """
    prov = topk_run["provenance"]
    assert prov["range_seconds"] > (prov["n_points"] - 1) * prov["step"]

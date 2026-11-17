"""The Datadog harness, offline: payload shape, response parsing, credential handling.

Everything here runs without an account. The network functions are not exercised --
`scripts/trial_checks.py` does that and caches what comes back -- but the parts that are
easy to get silently wrong are, because each of them would corrupt a measurement rather
than fail:

  * the metric-type enum (3 is GAUGE, and submitting a gauge makes E3 meaningless);
  * the seconds/milliseconds asymmetry between submission and `pointlist`;
  * dropping nulls, which are the measurement in E2 rather than noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlport.harness.datadog import (
    COUNT, GAUGE, RATE, UNSPECIFIED, Credentials, bucket_of, count_payload, credentials,
    interpret_closure, interpret_default_aggregator, pointlist, reported_interval,
)


def test_the_metric_type_enum_matches_the_intake():
    # datadog-api-client-python, v2/model/metric_intake_type.py. Asserted rather than
    # trusted: `docs/trial_account_checks.md` originally said type 3 was COUNT.
    assert (UNSPECIFIED, COUNT, RATE, GAUGE) == (0, 1, 2, 3)


def test_a_count_payload_submits_whole_second_timestamps():
    body = count_payload("qlport.e2", [(1_700_000_040, 1000.0)], tags=["rig:b"])
    (series,) = body["series"]
    assert series["metric"] == "qlport.e2"
    assert series["type"] == COUNT
    assert series["tags"] == ["rig:b"]
    assert series["points"] == [{"timestamp": 1_700_000_040, "value": 1000.0}]
    assert "interval" not in series


def test_a_fractional_timestamp_is_refused_rather_than_rounded():
    # Rounding here would move the point across exactly the edge E2 probes.
    with pytest.raises(ValueError, match="whole seconds"):
        count_payload("qlport.e2", [(1_700_000_040.5, 1.0)])


def test_an_empty_submission_is_refused():
    with pytest.raises(ValueError, match="no points"):
        count_payload("qlport.e2", [])


def test_an_interval_is_included_only_when_given():
    body = count_payload("qlport.e3", [(1_700_000_000, 1.0)], interval=60)
    assert body["series"][0]["interval"] == 60


def test_pointlist_converts_milliseconds_to_seconds():
    raw = {"series": [{"pointlist": [[1_700_000_040_000, 1000.0]]}]}
    assert pointlist(raw) == {1_700_000_040.0: 1000.0}


def test_pointlist_keeps_empty_buckets_as_none():
    # The null buckets are what show where the grid falls, which is half of E2's answer.
    raw = {"series": [{"pointlist": [
        [1_700_000_000_000, None], [1_700_000_060_000, 1000.0], [1_700_000_120_000, None],
    ]}]}
    got = pointlist(raw)
    assert got == {1_700_000_000.0: None, 1_700_000_060.0: 1000.0, 1_700_000_120.0: None}
    assert [t for t, v in got.items() if v is not None] == [1_700_000_060.0]


def test_an_empty_response_is_an_empty_pointlist_not_an_error():
    assert pointlist({"series": []}) == {}
    assert pointlist({}) == {}
    assert reported_interval({"series": []}) is None


def test_more_than_one_series_is_refused():
    raw = {"series": [{"pointlist": []}, {"pointlist": []}]}
    with pytest.raises(ValueError, match="expected one series"):
        pointlist(raw)


def test_reported_interval_is_read_from_the_response():
    assert reported_interval({"series": [{"interval": 20, "pointlist": []}]}) == 20
    assert reported_interval({"series": [{"pointlist": []}]}) is None


def test_the_site_decides_the_api_host():
    assert Credentials("a", "b", "datadoghq.com").base_url == "https://api.datadoghq.com"
    assert Credentials("a", "b", "datadoghq.eu").base_url == "https://api.datadoghq.eu"
    assert Credentials("a", "b", "datadoghq.eu/").base_url == "https://api.datadoghq.eu"


def test_both_keys_travel_as_headers():
    h = Credentials("secret-api", "secret-app", "datadoghq.com").headers()
    assert h["DD-API-KEY"] == "secret-api"
    assert h["DD-APPLICATION-KEY"] == "secret-app"


def test_credentials_never_appear_in_repr():
    r = repr(Credentials("secret-api", "secret-app", "datadoghq.com"))
    assert "secret-api" not in r and "secret-app" not in r
    assert "datadoghq.com" in r


@pytest.mark.parametrize("missing", ["DD_API_KEY", "DD_APP_KEY", "DD_SITE"])
def test_a_missing_variable_is_named_and_no_default_is_invented(missing):
    env = {"DD_API_KEY": "a", "DD_APP_KEY": "b", "DD_SITE": "datadoghq.com"}
    del env[missing]
    with pytest.raises(RuntimeError, match=missing):
        credentials(env)


def test_an_empty_variable_counts_as_missing():
    env = {"DD_API_KEY": "", "DD_APP_KEY": "b", "DD_SITE": "datadoghq.com"}
    with pytest.raises(RuntimeError, match="DD_API_KEY"):
        credentials(env)


def test_credentials_are_read_from_the_mapping_when_given():
    c = credentials({"DD_API_KEY": "a", "DD_APP_KEY": "b", "DD_SITE": "datadoghq.eu"})
    assert (c.api_key, c.app_key, c.site) == ("a", "b", "datadoghq.eu")


# --- E2's inference: which bucket, labelled by which edge ---------------------

T = 1_700_000_040.0   # a multiple of 60
D = 60.0


@pytest.mark.parametrize("closure,label,ongrid,mid", [
    ("right", "start", T - D, T),
    ("right", "end",   T,     T + D),
    ("left",  "start", T,     T),
    ("left",  "end",   T + D, T + D),
])
def test_bucket_of_places_both_probes(closure, label, ongrid, mid):
    assert bucket_of(T, D, closure=closure, label=label) == ongrid
    assert bucket_of(T + D / 2, D, closure=closure, label=label) == mid


def test_an_instant_on_the_grid_falls_below_it_when_right_closed():
    # (t-D, t] contains its upper edge, so a point at exactly T is in the bucket below.
    assert bucket_of(T, D, closure="right", label="start") == T - D
    assert bucket_of(T, D, closure="left", label="start") == T


@pytest.mark.parametrize("closure,label", [
    ("right", "start"), ("right", "end"), ("left", "start"), ("left", "end"),
])
def test_every_hypothesis_is_recovered_from_its_own_probes(closure, label):
    got = interpret_closure(
        T, D,
        bucket_of(T, D, closure=closure, label=label),
        bucket_of(T + D / 2, D, closure=closure, label=label),
    )
    assert got["conclusive"]
    assert got["consistent_with"] == [{"closure": closure, "label": label}]


def test_the_on_grid_probe_alone_cannot_separate_two_of_the_hypotheses():
    # This is why E2 submits a second point: "right-closed, labelled by end" and
    # "left-closed, labelled by start" report an on-grid point at the same timestamp.
    assert (bucket_of(T, D, closure="right", label="end")
            == bucket_of(T, D, closure="left", label="start"))
    # The mid-bucket probe is what tells them apart.
    assert (bucket_of(T + D / 2, D, closure="right", label="end")
            != bucket_of(T + D / 2, D, closure="left", label="start"))


def test_an_impossible_pair_of_probes_is_inconclusive_rather_than_guessed():
    got = interpret_closure(T, D, T - D, T + D)   # no hypothesis predicts this pair
    assert got["consistent_with"] == []
    assert not got["conclusive"]


def test_bucket_of_rejects_an_unknown_convention():
    with pytest.raises(ValueError, match="closure"):
        bucket_of(T, D, closure="middle", label="start")
    with pytest.raises(ValueError, match="label"):
        bucket_of(T, D, closure="right", label="centre")


# --- E3's inference: which time aggregator did a bare query apply? ------------

E3_POINTS = [(T + 31, 1.0), (T + 32, 2.0), (T + 33, 3.0), (T + 34, 10.0)]
E3_VALUES = [1.0, 2.0, 3.0, 10.0]


@pytest.mark.parametrize("value,expected", [
    (4.0, "avg"),
    (16.0, "sum (as if .as_count())"),
    (16.0 / 60.0, "sum/interval (as if .as_rate())"),
])
def test_the_bare_query_aggregator_is_named_from_the_value(value, expected):
    got = interpret_default_aggregator(60, E3_POINTS, E3_VALUES, {"x": value})
    assert got["conclusive"]
    assert got["behaved_as"] == [expected]


def test_no_reported_interval_is_inconclusive():
    got = interpret_default_aggregator(None, E3_POINTS, E3_VALUES, {"x": 16.0})
    assert not got["conclusive"] and "no interval" in got["why"]


def test_samples_straddling_two_buckets_are_inconclusive():
    # At a 2 s interval the four samples are spread over several buckets, so the value
    # is a partial sum and naming an aggregator from it would be a confident error.
    got = interpret_default_aggregator(2, E3_POINTS, E3_VALUES, {"x": 3.0})
    assert not got["conclusive"] and "share one" in got["why"]


def test_a_sample_on_a_bucket_edge_is_inconclusive():
    points = [(T, 1.0), (T + 1, 2.0)]
    got = interpret_default_aggregator(60, points, [1.0, 2.0], {"x": 3.0})
    assert not got["conclusive"] and "on_edge" in got["why"]


def test_more_than_one_non_zero_bucket_is_inconclusive():
    got = interpret_default_aggregator(60, E3_POINTS, E3_VALUES, {"a": 4.0, "b": 4.0})
    assert not got["conclusive"] and "one non-zero bucket" in got["why"]


# --------------------- the recorded E2 measurement (paper §3.1, bucket closure)


E2 = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "e2.json"


@pytest.fixture(scope="module")
def e2() -> dict:
    if not E2.exists():  # pragma: no cover - run scripts/trial_checks.py e2
        pytest.skip("e2.json not recorded; run scripts/trial_checks.py e2")
    return json.loads(E2.read_text())


def test_the_measured_convention_is_left_closed_labelled_by_start(e2):
    """The answer §3.1 is written against, read off a real Datadog account.

    A point submitted at exactly a bucket boundary is reported under the bucket
    *starting* there, not the one ending there. That rules out the right-closed
    reading the paper originally assumed.
    """
    got = e2["interpretation"]
    assert got["conclusive"], got
    assert got["consistent_with"] == [{"closure": "left", "label": "start"}], got


def test_the_two_probes_actually_discriminated(e2):
    # Both probes must have returned exactly one bucket, or the run proves nothing.
    for name in ("ongrid", "midbucket"):
        assert len(e2["reported_at"][name]) == 1, (name, e2["reported_at"])


def test_the_run_was_isolated_from_earlier_runs(e2):
    # Re-running without a per-run tag let a later run read an earlier run's points
    # back and report a bucket from the wrong experiment.
    prov = e2["provenance"]
    assert prov.get("run"), prov
    for r in e2["responses"].values():
        assert f"run:{prov['run']}" in r["query"], r["query"]


def test_the_interpretation_follows_from_the_recorded_buckets(e2):
    # Recompute from the cache rather than trusting the stored verdict.
    from qlport.harness.datadog import interpret_closure
    prov = e2["provenance"]
    again = interpret_closure(prov["t_star"], prov["delta"],
                              e2["reported_at"]["ongrid"][0],
                              e2["reported_at"]["midbucket"][0],
                              mid_submitted_at=prov["probes"]["midbucket"])
    assert again["consistent_with"] == e2["interpretation"]["consistent_with"]


# --------------------- E3 and E5, recorded (paper §5 and §6) ------------------


def _cached(name: str) -> dict:
    path = E2.parent / f"{name}.json"
    if not path.exists():  # pragma: no cover - run scripts/trial_checks.py <name>
        pytest.skip(f"{name}.json not recorded; run scripts/trial_checks.py {name}")
    return json.loads(path.read_text())


def test_a_bare_api_query_on_a_count_averages_the_deltas():
    """E3, which confirms rather than narrows METRIC_TYPE_UNKNOWN.

    The documentation says `avg` is the enforced default but that COUNT metrics get
    `.as_count()` appended *in the UI*. If the API did the same, the emitter's caveat
    would narrow to genuinely ambiguous gauges. It does not: four deltas of 1, 2, 3, 10
    come back as 4, their mean, not 16. So Datadog averages the stored deltas while
    `avg_over_time` averages the cumulative series, and the preconditions §5 reports on
    three quarters of the output are load-bearing.
    """
    got = _cached("e3")["interpretation"]
    assert got["conclusive"], got
    assert got["behaved_as"] == ["avg"], got
    assert got["value"] == pytest.approx(4.0)


def test_a_glob_star_matches_the_empty_string():
    """E5, which settles the reading §6's largest refusal class turns on.

    A glob has one wildcard, which either requires a character or does not, so `.*` and
    `.+` cannot both be expressible. `probe:web-*` selects the series whose tag value is
    exactly `web-`, so `*` matches the empty string: `.*` has a glob and `.+` has none.
    """
    got = _cached("e5")["interpretation"]
    assert got["conclusive"], got
    assert got["star_matches_empty"] is True, got
    assert {"web-", "web-1"} <= set(got["selected"]), got


def test_the_measured_reading_is_the_one_regular_py_takes():
    # §6's table was computed under this constant; if the measurement and the constant
    # ever disagree, the table is describing the wrong language.
    from qlport.regular import STAR_MATCHES_EMPTY
    assert _cached("e5")["interpretation"]["star_matches_empty"] is STAR_MATCHES_EMPTY


def test_e5_distinguished_the_two_probes_before_concluding():
    # A glob that selects nothing looks the same as a point that never arrived; the
    # control query is what separates them.
    e5 = _cached("e5")
    assert len(e5["stored"]) == 2, e5["stored"]


def test_the_automatic_rollup_interval_is_a_function_of_the_query_range():
    """E4. The interval a bare query gets, measured rather than assumed.

    The documentation publishes a time-frame table for *widgets* without promising the
    API agrees; it does, at least at one hour, where both say 20 s.
    """
    got = _cached("e4")["interpretation"]["reported_by_range"]
    assert got["1h"] == 20, got
    assert got["4h"] == 60, got
    assert got["1d"] == 300, got
    # Coarser range, coarser bucket -- monotone, which is what makes a single assumed
    # default wrong in a knowable direction rather than arbitrarily.
    order = ["1m", "5m", "1h", "4h", "1d"]
    widths = [got[k] for k in order]
    assert widths == sorted(widths), got


def test_the_assumed_default_corresponds_to_a_four_hour_view():
    """Why `ROLLUP_DEFAULT` is a real cost and not a formality.

    The translator assumes 60 s when the query states no interval. That is the right
    answer for a four-hour dashboard and wrong by a factor of three for the one-hour
    view, which is the range the documented table is written around.
    """
    got = _cached("e4")["interpretation"]["reported_by_range"]
    assert got["4h"] == 60
    assert got["1h"] != 60


def test_a_finer_interval_than_enforced_is_refused_not_widened():
    """The second half of E4, and the better of the two outcomes for a translator.

    Silently widening would make a translated query quietly disagree with its source;
    an error is visible. The limit is on points per query, not on the interval itself.
    """
    tf = _cached("e4")["too_fine_rollup"]
    assert tf["accepted"] is False, tf
    assert "limit" in tf["error"], tf["error"]


# ------------- the recorded E6 measurement (as_rate's placement and its evidence)


E6 = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "as_rate_order.json"


@pytest.fixture(scope="module")
def e6() -> dict:
    if not E6.exists():  # pragma: no cover - run scripts/as_rate_order.py
        pytest.skip("as_rate_order.json not recorded; run scripts/as_rate_order.py")
    return json.loads(E6.read_text())


def _approx(x: float):
    """The recorder rounds to six decimals, so comparisons match that precision."""
    return pytest.approx(x, abs=1e-6)


def _only(e6: dict, key: str) -> float:
    """The single value every series of a recorded query settled on.

    Each probe submits a constant, so a query that reduces it correctly returns one
    value per group and the same value in every group; anything else means the probe,
    not the system, is what the test would be reading.
    """
    rec = e6["queries"][key]
    assert "error" not in rec, f"{key}: {rec.get('error')}"
    vals = {v for series in rec["values"].values() for v in series}
    assert len(vals) == 1, f"{key} did not settle on one value: {rec['values']}"
    return vals.pop()


def test_as_rate_divides_the_rolled_up_value_by_the_rollup_interval(e6):
    """Not each delta by its own submission interval.

    The two readings agree whenever the submission grid equals the rollup width, which
    is why this went unmeasured for so long: every harness in this project scraped at
    the rollup interval. E6 submits every 10 s and rolls up over 30 s, which separates
    them --- 10/30 for the reading the emitter uses, 10/10 for the other.
    """
    assert _only(e6, "cnt.rollup_avg") == _approx(10.0)
    assert _only(e6, "cnt.as_rate.rollup_avg") == _approx(10.0 / 30.0)
    assert _only(e6, "cnt.as_rate.rollup_avg") != _approx(10.0 / 10.0)


def test_as_rate_applies_outside_the_space_aggregation(e6):
    """Observable only through `count:`, which does not commute with scaling.

    `count:` of three series returns 1 per group. With `.as_rate()` it returns 1/30,
    so the modifier divided the aggregated result; had it divided each series first,
    the count would have been unchanged at 1.
    """
    assert _only(e6, "grp.count") == _approx(1.0)
    assert _only(e6, "grp.count.as_rate") == _approx(1.0 / 30.0)
    # For `sum:` the placement is invisible, which is why it hid there.
    assert _only(e6, "grp.sum") == _approx(10.0)
    assert _only(e6, "grp.sum.as_rate") == _approx(10.0 / 30.0)


@pytest.mark.parametrize("modified,plain", [
    ("gau.as_rate", "gau.plain"),
    ("gau.as_count", "gau.plain"),
    ("gau.as_rate.rollup_avg", "gau.rollup_avg"),
    ("gau.as_count.rollup_avg", "gau.rollup_avg"),
])
def test_neither_modifier_does_anything_to_a_gauge(e6, modified, plain):
    """Which is why neither establishes the metric's type.

    Datadog documents `.as_count()` and `.as_rate()` for COUNT metrics, and the front
    end read their presence as evidence that a metric is one. It accepts them on a
    GAUGE and ignores them rather than rejecting the query, so a dashboard can carry
    either over a gauge and show its author something they did not ask for. The
    presence of the modifier records a belief, not a type --- so every translation
    that depends on the type carries METRIC_TYPE_UNKNOWN.
    """
    assert _only(e6, modified) == _approx(_only(e6, plain))


# ------------------- the recorded E1 measurement (the reset policy, end to end)


E1 = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "e1.json"


@pytest.fixture(scope="module")
def e1() -> dict:
    if not E1.exists():  # pragma: no cover - run scripts/reset_policy.py
        pytest.skip("e1.json not recorded; run scripts/reset_policy.py")
    return json.loads(E1.read_text())


def test_the_reset_policy_is_zero_and_nothing_else_fits(e1):
    """rho = zero, measured through a real agent rather than read off its source.

    A counter exposed at 1000 drops to 100 inside one bucket. Datadog stores 103 over
    that bucket against four points, which is the increment before the reset, the new
    value reported as the reset interval's own count, and the two increments after it.
    Clamping would store 3, discarding the interval would store 3 over three points, and
    a backend keeping the signed difference would store -897. All three are ruled out by
    the reading, not argued away.
    """
    case = e1["cases"]["to100"]
    assert case["consistent_with"] == ["zero"]
    assert case["reset_bucket_observed"] == [103.0, 4]
    assert case["reset_bucket_predicted"]["clamp"] == [3.0, 4]
    assert case["reset_bucket_predicted"]["drop"] == [3.0, 3]
    assert case["reset_bucket_predicted"]["raw"] == [-897.0, 4]


def test_a_reset_to_zero_shows_no_special_case(e1):
    # `zero` and `clamp` give the same answer when the new value is zero, so this case
    # cannot choose between them -- it is here to catch a backend that treats a reset to
    # zero differently from a reset to anything else. It does not.
    case = e1["cases"]["to0"]
    assert set(case["consistent_with"]) == {"zero", "clamp"}
    assert case["reset_bucket_observed"] == [3.0, 4]


def test_the_control_counter_rules_nothing_out_and_reads_four_increments(e1):
    """A counter that never resets: every policy fits, which is the point.

    Its value is that a bucket of it must read four increments of one. That is what
    shows the run is reading the buckets the reset is in, so a disagreement elsewhere is
    about the system rather than about the rig.
    """
    case = e1["cases"]["none"]
    assert set(case["consistent_with"]) == {"zero", "clamp", "drop", "raw"}
    full = [v for b, v in case["observed_buckets"].items()
            if float(b) in case["complete_buckets"]]
    assert full and all(v == [4.0, 4] for v in full), full


def test_datadog_and_prometheus_agree_across_the_reset(e1):
    """The claim §3.1 could previously only make about the model.

    Both systems scrape one endpoint; Datadog stores the deltas its agent computed and
    Prometheus stores the cumulative series. `increase(x[60s] anchored)` over Prometheus
    returns what Datadog stored, on every complete bucket of every case --- including
    the bucket the reset falls in, where it returns 103.
    """
    for name, case in e1["cases"].items():
        assert case["prometheus_differs_on"] == {}, (name, case["prometheus_differs_on"])
        assert len(case["prometheus_agrees_on"]) == len(case["complete_buckets"]), name
    to100 = e1["cases"]["to100"]
    at_reset = to100["prometheus_anchored"][str(float(to100["reset_bucket"]))]
    assert at_reset == 103.0


def test_the_run_met_the_warm_up_the_design_committed_to(e1):
    # The spec asks for at least ten minutes before the reset, because the agent omits
    # its first sample; a run that cut the warm-up short would be scoring a different
    # experiment from the one the prediction table was written for.
    assert e1["provenance"]["warmup"] >= 600.0
    assert sum(1 for e in e1["exposition"] if e["agent"]) > 40


# --------------- the recorded E7 measurement (what the type assumption costs)


E7 = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "e7.json"


@pytest.fixture(scope="module")
def e7() -> dict:
    if not E7.exists():  # pragma: no cover - run scripts/type_assumption.py
        pytest.skip("e7.json not recorded; run scripts/type_assumption.py")
    return json.loads(E7.read_text())


def test_the_assumption_holding_is_exact(e7):
    # A GAUGE carrying the same stored values agrees on every bucket, which is the
    # content of Theorem thm:kernel's type hypothesis when it is satisfied.
    assert e7["summary"]["gauge_max_d"] == pytest.approx(0.0, abs=1e-12)


def test_the_assumption_failing_is_unbounded_not_approximate(e7):
    """§5 calls the wrong case "not approximately right but unrelated". This is that.

    Datadog reduces the deltas it stores and reads the same value in every bucket;
    `avg_over_time` reduces the running totals and climbs with the counter's age. The
    relative difference therefore has no bound that does not mention how long the
    counter has been running -- it approaches 1 and never settles.
    """
    s = e7["summary"]
    assert s["counter_first"] > 0.4
    assert s["counter_last"] > 0.99
    assert s["counter_last"] > s["counter_first"]
    # one side constant, the other growing: that is why no fixed epsilon covers it
    assert s["promql_counter_last"] > 50 * s["promql_counter_first"]
    assert s["datadog_counter_value"] == pytest.approx(10.0)


def test_the_error_grows_monotonically_with_the_counters_age(e7):
    rows = [r for r in e7["rows"] if r["d_counter"] is not None]
    ds = [r["d_counter"] for r in rows]
    assert len(ds) > 20
    # not strictly monotone point to point, but the trend must be unmistakable
    assert ds[-1] > ds[0]
    assert ds[len(ds) // 2] > ds[0]


def _full_bucket(e6: dict, key: str) -> float:
    """The largest recorded value of a query whose edge buckets are partial.

    A summing or counting rollup reads less at the ends of the window, where the bucket
    holds one or two of its three samples rather than all three, so the full bucket is
    the maximum rather than the only value.
    """
    rec = e6["queries"][key]
    assert "error" not in rec, f"{key}: {rec.get('error')}"
    return max(v for series in rec["values"].values() for v in series)


def test_a_summing_rollup_over_a_gauge_sums_the_stored_samples(e6):
    """The third piece of type evidence, and the one whose misreading is least benign.

    `.as_count()` and `.as_rate()` are no-ops on a gauge, so a query carrying either
    over one is a mistake nobody benefits from. `.rollup(sum, D)` is not ignored: it
    sums the samples in the bucket, which is an ordinary thing to want. The front end
    reads all three as evidence of a COUNT and emits a cumulative difference, which for
    a gauge computes something else entirely.
    """
    assert _full_bucket(e6, "gau.rollup_count") == _approx(3.0)   # three samples
    assert _full_bucket(e6, "gau.rollup_sum") == _approx(150.0)   # and their sum
    assert _only(e6, "gau.rollup_avg") == _approx(50.0)           # their mean
    # Summing is not the no-op that `.as_count()` is here: it reads a third quantity.
    assert _full_bucket(e6, "gau.rollup_sum") != _approx(50.0)


# ------------- the recorded E8 measurement (how the two languages pair series)


E8 = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "e8.json"


@pytest.fixture(scope="module")
def e8() -> dict:
    if not E8.exists():  # pragma: no cover - run scripts/binop_matching.py
        pytest.skip("e8.json not recorded; run scripts/binop_matching.py")
    return json.loads(E8.read_text())


@pytest.mark.parametrize("case", [
    "both_grouped_div", "both_grouped_add", "rhs_ungrouped", "lhs_ungrouped",
    "subset_rhs_finer", "subset_lhs_finer",
])
def test_the_emitted_form_reproduces_datadogs_pairing(e8, case):
    """Theorem thm:arith's premise, read off both systems on the same four series.

    `a` carries groups x and y, `b` carries x and z, so the two sides agree on one
    group and differ on the others -- which the corpus-scale test of §7 can never
    arrange, since it instantiates both operands from the same filters.
    """
    got = e8["cases"][case]
    assert "__error__" not in got["datadog"], got["datadog"]
    assert "__error__" not in got["promql"], got["promql"]
    dd = {k: v[0] for k, v in got["datadog"].items() if v}
    assert dd == pytest.approx(got["promql"], abs=1e-6), case


def test_a_group_on_one_side_only_is_dropped_by_both(e8):
    # Both systems inner-join on the matched keys: `a` has x and y, `b` has x and z,
    # and the quotient has x alone. An outer join on either side would have made the
    # emitted expression wrong on every dashboard whose two series sets differ.
    assert set(e8["cases"]["both_grouped_div"]["datadog"]) == {"g=x"}
    assert set(e8["cases"]["both_grouped_div"]["promql"]) == {"g=x"}


def test_datadog_rejects_incomparable_groupings_rather_than_pairing_them(e8):
    """Which is why the emitter's refusal there is not a gap in PromQL.

    The error message is also the specification: group keys must match or one must be a
    subset of the other. The ungrouped and subset cases above are that rule's two
    halves, and this is the boundary.
    """
    err = e8["cases"]["different_keys"]["datadog"].get("__error__", "")
    assert "subset" in err, err


def test_default_matching_would_have_returned_nothing(e8):
    # The defect the corpus-scale test surfaced, measured directly: Datadog broadcasts
    # and PromQL's default matching pairs nothing, silently, since an empty result is
    # not an error.
    got = e8["cases"]["rhs_ungrouped_default_matching"]
    assert got["promql"] == {}
    assert got["datadog"]


# -------------- the recorded E9 measurement (what `.weighted()` computes)


E9 = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "e9.json"


@pytest.fixture(scope="module")
def e9() -> dict:
    if not E9.exists():  # pragma: no cover - run scripts/weighted_semantics.py
        pytest.skip("e9.json not recorded; run scripts/weighted_semantics.py")
    return json.loads(E9.read_text())


@pytest.mark.parametrize("plain,weighted", [
    ("by_plain", "by_weighted"),
    ("by_rollup_avg", "by_weighted_rollup_avg"),
    ("by_rollup_sum", "by_weighted_rollup_sum"),
    ("sum_plain", "sum_weighted"),
    ("count_type_plain", "count_type_weighted"),
    ("rate_type_plain", "rate_type_weighted"),
])
def test_weighted_changes_nothing(e9, plain, weighted):
    """On every metric type and query form the rig could construct.

    §7 called it "a time-weighted space average that PromQL has no operator for",
    which was wrong twice: it is not an average, and whatever it is, Datadog returns
    the same numbers without it.
    """
    assert e9["datadog"][plain] == e9["datadog"][weighted], plain


def test_the_rig_could_have_seen_a_time_weighting(e9):
    """The negative result means something only if the probe could have come out other.

    The irregular series reports at 0 s, 50 s and 55 s of each bucket. A plain mean of
    its three points is 66.7; a mean weighted by the interval each point covers is
    (0*50 + 100*5 + 100*5)/60 = 16.7. Datadog reads 66.7 with the modifier and
    without, so the weighting is not being applied rather than being invisible.
    """
    assert e9["provenance"]["irregular"] == [[0.0, 0.0], [50.0, 100.0], [55.0, 100.0]]
    got = e9["datadog"]["by_rollup_avg"]["d:irregular"]
    assert got == pytest.approx([66.6667], abs=1e-3)
    weighted_mean = (0 * 50 + 100 * 5 + 100 * 5) / 60
    assert abs(weighted_mean - 66.6667) > 40, "the two readings must be far apart"

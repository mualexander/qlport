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
    COUNT, GAUGE, RATE, RETRYABLE_STATUS, UNSPECIFIED, Credentials, QueryHTTPError,
    bucket_of, count_payload, credentials, echoed_scopes, interpret_closure,
    interpret_default_aggregator, pointlist, reported_interval, scope_term_survived,
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

# The values are the discriminating set: avg 5, sum 20, count 4, sum/60 0.333. The
# first version of this probe used 1, 2, 3, 10, whose mean and count are both 4.
E3_POINTS = [(T + 31, 1.0), (T + 32, 14.0), (T + 33, 2.0), (T + 34, 3.0)]
E3_VALUES = [1.0, 14.0, 2.0, 3.0]
E3_BUCKET = str(float(T))          # the bucket the four points fall in at 60 s


@pytest.mark.parametrize("value,expected", [
    (5.0, "avg"),
    (20.0, "sum (as if .as_count())"),
    (20.0 / 60.0, "sum/interval (as if .as_rate())"),
    (4.0, "count (as if .rollup(count))"),
])
def test_the_bare_query_aggregator_is_named_from_the_value(value, expected):
    got = interpret_default_aggregator(60, E3_POINTS, E3_VALUES, {E3_BUCKET: value})
    assert got["conclusive"], got
    assert got["behaved_as"] == [expected]


def test_count_is_among_the_hypotheses():
    """The gap that let E3 name `avg` on a reading `count` explained as well.

    With four deltas whose mean is 4, `avg` and `count` predict the same number, and the
    helper enumerated only `avg`, `sum` and `sum/interval` -- so it reported one match
    and called it conclusive. A probe is only as good as the rivals it is scored against.
    """
    got = interpret_default_aggregator(60, [(T + 31, 1.0), (T + 32, 2.0),
                                            (T + 33, 3.0), (T + 34, 10.0)],
                                       [1.0, 2.0, 3.0, 10.0], {E3_BUCKET: 4.0})
    assert set(got["behaved_as"]) == {"avg", "count (as if .rollup(count))"}
    assert not got["conclusive"], got


def test_a_reading_from_another_bucket_is_refused():
    """What the first cached E3 run did: read an earlier run's bucket and interpret it.

    The guard checked that the *submitted* points share one bucket and never that the
    bucket reported back was theirs, so three responses that predate the submission
    passed as though they were of it.
    """
    stale = str(float(T - 900))
    got = interpret_default_aggregator(60, E3_POINTS, E3_VALUES, {stale: 5.0})
    assert not got["conclusive"]
    assert "not the" in got["why"] and str(int(T)) in got["why"]


def test_no_reported_interval_is_inconclusive():
    got = interpret_default_aggregator(None, E3_POINTS, E3_VALUES, {E3_BUCKET: 20.0})
    assert not got["conclusive"] and "no interval" in got["why"]


def test_samples_straddling_two_buckets_are_inconclusive():
    # At a 2 s interval the four samples are spread over several buckets, so the value
    # is a partial sum and naming an aggregator from it would be a confident error.
    got = interpret_default_aggregator(2, E3_POINTS, E3_VALUES, {E3_BUCKET: 3.0})
    assert not got["conclusive"] and "share one" in got["why"]


def test_a_sample_on_a_bucket_edge_is_inconclusive():
    points = [(T, 1.0), (T + 1, 2.0)]
    got = interpret_default_aggregator(60, points, [1.0, 2.0], {E3_BUCKET: 3.0})
    assert not got["conclusive"] and "on_edge" in got["why"]


def test_more_than_one_non_zero_bucket_is_inconclusive():
    got = interpret_default_aggregator(60, E3_POINTS, E3_VALUES,
                                       {E3_BUCKET: 5.0, str(float(T + 60)): 5.0})
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
    would narrow to genuinely ambiguous gauges. It does not: four deltas of 1, 14, 2, 3
    come back as 5, their mean -- not 20 (their sum), not 4 (their count) and not 1.0
    (sum over the reported 20 s interval). So Datadog averages the stored deltas while
    `avg_over_time` averages the cumulative series, and the preconditions §5 reports on
    three quarters of the output are load-bearing.

    Re-derived from the recorded readings rather than read off the stored verdict, which
    is how the first version of this experiment went unnoticed for a pass: its deltas
    were 1, 2, 3, 10, whose mean and count are both 4, and its three responses were of
    an earlier run's submission 900 s before the probe.
    """
    rec = _cached("e3")
    prov = rec["provenance"]
    again = interpret_default_aggregator(
        rec["responses"]["bare"]["interval"],
        [(p[0], p[1]) for p in prov["points"]],
        [p[1] for p in prov["points"]],
        rec["responses"]["bare"]["defined"])
    assert again == rec["interpretation"], (again, rec["interpretation"])
    assert again["conclusive"], again
    assert again["behaved_as"] == ["avg"], again
    assert again["value"] == pytest.approx(5.0)
    # Four rivals, four distinct predictions: the reading names one of them.
    assert len(set(again["predictions"].values())) == len(again["predictions"]), again


def test_the_e3_run_read_its_own_bucket():
    """Isolation, the lesson E2 learned and E3 had not: a per-run tag in the query.

    Without one, the 3600 s window picked up whatever the same metric held from an
    earlier run, and `_await_points` returned on the first query because a stale bucket
    was already non-null -- so the run never waited for its own data.
    """
    rec = _cached("e3")
    run = rec["provenance"]["run"]
    assert run, rec["provenance"]
    for name, r in rec["responses"].items():
        assert f"run:{run}" in r["query"], (name, r["query"])
    first = rec["provenance"]["points"][0][0]
    (reported,) = [float(t) for t, v in rec["responses"]["bare"]["defined"].items() if v]
    assert reported == bucket_of(first, float(rec["responses"]["bare"]["interval"]),
                                 closure="left", label="start")


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


# --------- the recorded fast-scrape run (rem:flush's hypothesis, put to the agent)


E1F = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "e1_fast.json"


@pytest.fixture(scope="module")
def e1_fast() -> dict:
    if not E1F.exists():  # pragma: no cover - run scripts/reset_policy.py --step 5
        pytest.skip("e1_fast.json not recorded")
    return json.loads(E1F.read_text())


def test_the_fast_run_really_put_several_samples_in_one_flush(e1_fast):
    # The agent flushes every 15 s whatever the check interval, so a 5 s collection
    # interval is three samples per flush. Without that the run tests nothing.
    assert e1_fast["provenance"]["step"] == 5.0
    assert e1_fast["provenance"]["samples_per_flush"] == 3.0


def test_phi_still_models_the_agent_at_three_samples_per_flush(e1_fast):
    """Remark rem:flush's hypothesis turns out to be unnecessary at this ratio.

    It says phi models the agent only when at most one sample falls in each flush
    interval, and that otherwise a decrease overwrites what the interval accumulated,
    so the increase observed before a reset is lost. Here twelve 5 s samples fall in
    the 60 s bucket holding the reset: phi_zero predicts eleven increments of one plus
    the reset's own value, 111, and losing the pre-reset increment would give 110.
    Datadog stores 111.
    """
    case = e1_fast["cases"]["to100"]
    assert case["consistent_with"] == ["zero"]
    assert case["reset_bucket_observed"] == [111.0, 12]
    assert case["reset_bucket_predicted"]["zero"] == [111.0, 12]


def test_the_fast_run_agrees_with_prometheus_throughout(e1_fast):
    for name, case in e1_fast["cases"].items():
        assert case["prometheus_differs_on"] == {}, name
        assert len(case["prometheus_agrees_on"]) == len(case["complete_buckets"]), name


def test_the_two_reset_runs_differ_only_in_the_scrape_rate(e1, e1_fast):
    # Same policy, same rig, different sampling: the point of the pair is that the
    # answer does not depend on which was used.
    assert e1["cases"]["to100"]["consistent_with"] == \
        e1_fast["cases"]["to100"]["consistent_with"] == ["zero"]
    assert e1["provenance"]["step"] != e1_fast["provenance"]["step"]


# ------------- top(): the vendor's own answer, not a model of its docs ---------
#
# `tests/test_harness.py` pins the Prometheus half of paper §7's `top()` argument. This
# is the other half, which until now was a reading of the documentation -- the weaker
# kind of evidence, and the kind this paper has twice found wrong (the aggregation order
# and the bucket closure). `scripts/datadog_top.py` asks a real account.

TOP = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "top.json"


@pytest.fixture(scope="module")
def top_run() -> dict:
    if not TOP.exists():  # pragma: no cover - run scripts/datadog_top.py
        pytest.skip("top.json not recorded; run scripts/datadog_top.py (needs an account)")
    return json.loads(TOP.read_text())


def test_the_vendor_ranks_over_the_whole_queried_range(top_run):
    """Datadog's documented rule, recomputed from the cache rather than trusted.

    Every (function, n, direction) is re-predicted here through `harness/ranking.py` and
    compared against the selection the account returned, so a stored verdict cannot
    carry a result the recorded selections do not support.
    """
    from qlport.harness.ranking import model_rank

    assert top_run["cases"], "no cases recorded"
    for case in top_run["cases"]:
        lo, hi = case["indices"]
        want = model_rank(case["by"], case["n"], case["direction"], lo, hi)
        assert case["model"] == want, case
        assert case["selected"] == want, case
        assert case["matches_model"], case
    assert top_run["agreement"]["matching_model"] == top_run["agreement"]["cases"]


def test_both_directions_and_every_probed_function_were_asked_of_the_vendor(top_run):
    full = [c for c in top_run["cases"] if c["window"] == "full"]
    covered = {(c["by"], c["n"], c["direction"]) for c in full}
    for by in ("max", "mean", "min", "sum", "last"):
        for n in (2, 3):
            for direction in ("desc", "asc"):
                assert (by, n, direction) in covered, (by, n, direction)


def test_the_vendor_returns_whole_trajectories_not_winning_instants(top_run):
    """What separates `top()` from a `topk`-shaped operator, measured.

    Series `e` spikes at index 0 and sits at 1 for the other nineteen buckets. It is
    selected by `max` and it comes back with all twenty points -- an operator that ranked
    per instant would return it only where it was winning.
    """
    for case in top_run["cases"]:
        lo, hi = case["indices"]
        assert case["whole_trajectories"], case
        assert all(n == hi - lo for n in case["points_per_series"].values()), case
    spike = next(c for c in top_run["cases"] if c["window"] == "full"
                 and (c["by"], c["n"], c["direction"]) == ("max", 2, "desc"))
    assert "e" in spike["points_per_series"], spike
    assert spike["points_per_series"]["e"] == top_run["provenance"]["n_points"], spike


def test_the_selection_moves_with_the_queried_range(top_run):
    """Why §8 calls the dashboard's time range a precondition rather than a detail.

    Over all twenty buckets the top two by `max` are the riser and the spike; over the
    last ten the spike is outside the window and the answer changes. `mean` is the
    control: its selection does not move, so a run in which everything moved would be
    an apparatus fault rather than this result.
    """
    rr = top_run["range_relative"]
    assert set(rr) == {"max", "mean"}, sorted(rr)
    assert rr["max"]["moved"] and rr["max"]["model_moves"], rr["max"]
    assert not rr["mean"]["moved"] and not rr["mean"]["model_moves"], rr["mean"]
    assert all(obs["as_model_predicts"] for obs in rr.values()), rr


def test_the_vendor_stored_the_values_the_ranking_is_about(top_run):
    """A ranking over numbers the server does not hold would prove nothing."""
    assert top_run["control"]["stored_matches_input"] == \
        {k: True for k in top_run["provenance"]["series"]}
    assert {c["reported_interval"] for c in top_run["cases"]} == \
        {top_run["provenance"]["rollup_seconds"]}


def test_both_halves_of_the_comparison_saw_the_same_series(top_run):
    """The Prometheus run and this one must share a workload or compare nothing.

    `harness/ranking.py` is the single definition; `tests/test_harness.py` pins the
    Prometheus cache to it and this pins the Datadog one.
    """
    from qlport.harness import ranking

    prov = top_run["provenance"]
    assert prov["series"] == {k: ranking.values(k) for k in ranking.SERIES}
    assert (prov["n_points"], prov["step"]) == (ranking.N, ranking.STEP)


def test_the_run_was_isolated_from_earlier_runs(top_run):
    # E2's lesson: a metric name shared across runs lets one run read another's points.
    prov = top_run["provenance"]
    assert str(prov["t0"]) in prov["metric"], prov["metric"]
    for case in top_run["cases"]:
        assert prov["metric"] in case["expr"], case["expr"]


# --- the echo: whether the query that ran is the query that was asked -------------
#
# E14's first run read back points and reasoned from them. The points were right and the
# reasoning was one observation short: a scope term the parser discards returns exactly
# what a term matching everything returns. The expression Datadog reports evaluating
# separates them and was in every response all along, unread.


def _series(*expressions: str | None) -> dict:
    return {"series": [{"expression": e} if e is not None else {} for e in expressions]}


def test_the_echoed_expression_is_read_from_each_series():
    assert echoed_scopes(_series("sum:m{a:1}", "sum:m{a:2}")) == \
        ["sum:m{a:1}", "sum:m{a:2}"]


def test_a_response_with_no_series_echoes_nothing():
    assert echoed_scopes({"series": []}) == []
    assert echoed_scopes({}) == []


def test_a_series_carrying_no_expression_is_skipped_not_guessed():
    assert echoed_scopes(_series("sum:m{a:1}", None)) == ["sum:m{a:1}"]


def test_a_term_present_in_the_echo_survived_the_parse():
    assert scope_term_survived("!rig:zzz", _series("sum:m{run:r1,!rig:zzz}")) is True


def test_a_term_absent_from_the_echo_did_not():
    # The measurement E14 turns on: `!*` is not in what Datadog evaluated.
    assert scope_term_survived("!*", _series("sum:m{run:r1}")) is False


def test_a_term_must_survive_into_every_series_not_merely_one():
    grouped = _series("sum:m{k:a,!rig:zzz}", "sum:m{k:b}")
    assert scope_term_survived("!rig:zzz", grouped) is False


def test_nothing_echoed_is_inconclusive_rather_than_absent():
    """A scope that selects nothing echoes nothing, which is not evidence of dropping.

    E14's two exclusion controls return no series at all, so the only honest reading of
    their echo is that there is none -- and `False` here would have turned a working
    control into a second finding.
    """
    assert scope_term_survived("!rig:b", {"series": []}) is None


def test_the_comparison_ignores_the_case_the_echo_imposes():
    # E13's echo comes back `(aa:x OR (aa:y AND bb:1))` from a query that asked `or`.
    assert scope_term_survived("aa:x or aa:y",
                               _series("sum:m{(AA:X OR AA:Y)}")) is True


E14 = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / "e14.json"


@pytest.fixture(scope="module")
def e14() -> dict:
    if not E14.exists():  # pragma: no cover - run scripts/trial_checks.py e14
        pytest.skip("e14.json not recorded; run scripts/trial_checks.py e14")
    return json.loads(E14.read_text())


def test_the_negated_match_all_was_dropped_before_evaluation(e14):
    """What Remark 14 now rests on: `!*` is not in the expression Datadog ran.

    The first run concluded that `!*` does not exclude, from a series coming back. True,
    but inferred -- and the inference had a second branch it never ruled out, that the
    term was understood and vacuous. The echo rules it out by reading.
    """
    assert e14["term_survived_the_parse"]["match-all-and-negated"] is False
    assert e14["term_survived_the_parse"]["negated-alone"] is False
    for name in ("match-all-and-negated", "negated-alone"):
        echoed = e14["responses"][name]["echoed"]
        assert echoed and all("!*" not in e for e in echoed), echoed


def test_a_negation_datadog_does_understand_survives_it(e14):
    """Without this the finding above is about negation, not about `!*`.

    `!rig:zzz` selects the series and stays in the echoed expression; `!*` selects the
    series and does not. One contrast, two conclusions: negation is supported, and this
    spelling of it is not.
    """
    assert e14["term_survived_the_parse"]["negated-absent-value"] is True
    assert e14["selected_points"]["negated-absent-value"] == 1


def test_exclusion_was_expressible_in_the_window_that_was_measured(e14):
    """"Selects the series" means nothing until something has failed to.

    A value the series does not carry, and a negation of one it does, both return
    nothing over the same metric and window. So the two terms above returning a series
    is a fact about the terms rather than about the rig.
    """
    assert e14["selected_points"]["absent-value"] == 0
    assert e14["selected_points"]["negated-present-value"] == 0
    assert all(e14["controls"].values()), e14["controls"]


def test_the_scope_terms_still_select_the_series(e14):
    # The original measurement, unchanged: both spellings return the stored point.
    assert e14["stored_points"] == 1
    assert e14["selected_points"]["match-all-and-negated"] == 1
    assert e14["selected_points"]["negated-alone"] == 1
    assert e14["rejected"] == {}


def test_the_e14_run_was_isolated_from_earlier_runs(e14):
    # E2's lesson, applied here: a shared metric name lets one run read another's point.
    run = e14["provenance"]["run"]
    for name, entry in e14["responses"].items():
        assert f"run:{run}" in entry["query"], (name, entry["query"])


def test_every_e14_response_carries_the_expression_that_ran(e14):
    """The annotation is unconditional, so the next probe cannot repeat the omission."""
    for name, entry in e14["responses"].items():
        assert "echoed" in entry, name


# --- which HTTP statuses are answers --------------------------------------------


def test_a_query_http_error_is_still_a_runtime_error():
    """Every caller written before the status was kept must keep working."""
    exc = QueryHTTPError(431, "431 Request Header Fields Too Large: ...")
    assert isinstance(exc, RuntimeError)
    assert exc.status == 431
    assert "431" in str(exc)


def test_a_status_about_the_request_is_an_answer():
    # 431 is E17's measurement: the request was too large. Retrying cannot change it.
    assert 431 not in RETRYABLE_STATUS
    assert 400 not in RETRYABLE_STATUS


def test_a_status_about_the_server_is_not_an_answer():
    """E17 bisected to a boundary of exactly 4,096 literals on the strength of a 408.

    It had accepted 5,096 in an earlier run, so the refusal was never about length. A
    408 "request header timeout" and a socket timeout are the same non-answer, and the
    only thing that distinguishes them is which side noticed first.
    """
    assert {408, 429, 502, 503, 504} <= RETRYABLE_STATUS


# --- E17: how many literals one scope's IN list may carry ------------------------

E17 = Path(__file__).resolve().parent.parent / "data" / "raw" / "datadog" / \
    "e17_scope_length.json"
BOUND = 41 ** 2                    # the union bound §6 raised MAX_UNION to


@pytest.fixture(scope="module")
def e17() -> dict:
    if not E17.exists():  # pragma: no cover - run scripts/probe_scope_length.py
        pytest.skip("e17_scope_length.json not recorded; run probe_scope_length.py")
    return json.loads(E17.read_text())


def _rung(e17: dict, n: int) -> dict:
    rows = [r for r in e17["rows"] if r["n"] == n]
    assert len(rows) == 1, f"expected one rung at n={n}, got {len(rows)}"
    return rows[0]


def test_the_ladder_and_its_controls_asked_the_same_metric(e17):
    """The failure the second run made: controls on a metric that exists, ladder on one
    that does not. Both held and neither measured anything."""
    assert e17["metric"].startswith("qlport.")
    for r in e17["rows"]:
        for half in ("including", "excluding"):
            assert r[half]["url_bytes"] > 0
    assert all(c["holds"] for c in e17["controls"]), e17["controls"]


def test_an_accepted_rung_means_the_list_was_applied_not_merely_parsed(e17):
    """Every rung asks twice, and acceptance is the conjunction.

    A list containing the value the metric carries must return a series; the same list
    without it must not. Truncation fails the first, a dropped scope fails the second,
    and `status: ok` with no series -- what the first two runs measured -- fails both.
    """
    for r in e17["rows"]:
        if r["accepted"] is not True:
            continue
        assert r["including"]["series"] >= 1, r
        assert r["excluding"]["series"] == 0, r


def test_the_bound_the_paper_needs_is_accepted_and_applied(e17):
    r = _rung(e17, BOUND)
    assert r["accepted"] is True
    assert (r["including"]["series"], r["excluding"]["series"]) == (1, 0)


def test_the_whole_list_reached_the_engine_at_the_bound(e17):
    """The echo is the anti-truncation control: a list cut short comes back short.

    Each value is three characters and a separator, so an intact list of $41^2$ cannot
    echo in fewer than $4 \\times 1681$ bytes however the engine reformats it.
    """
    assert _rung(e17, BOUND)["including"]["echo_bytes"] >= 4 * BOUND


def test_a_limit_exists_and_it_is_the_request_line_not_the_query_language(e17):
    """`No limit is reached` was wrong, and wrong because the ladder stopped too low.

    What refuses is HTTP 431 -- the request was too large -- from a GET whose scope
    travels in the URL. Datadog's query language never gets to see it.
    """
    refused = [r for r in e17["rows"] if r["accepted"] is False]
    assert refused, "the ladder found no refusal; it did not climb far enough"
    assert all(r["including"]["status"] == 431 for r in refused), refused


def test_the_limit_is_far_above_the_bound_the_paper_relies_on(e17):
    b = e17["boundary"]
    assert b is not None
    assert "contradicted_by_accepted_above" not in b, b
    assert b["largest_accepted"] > 3 * BOUND, b
    assert b["smallest_refused"] > b["largest_accepted"]


def test_a_server_side_status_was_not_recorded_as_a_refusal(e17):
    """The artefact this probe produced before `RETRYABLE_STATUS` existed.

    A 408 at $4{,}096$ bisected to a boundary of exactly $4{,}096$, a power of two that
    looked like a finding, in a run that had already accepted $5{,}096$. Anything the
    server said about itself is now retried and then left inconclusive.
    """
    for r in e17["rows"]:
        for half in ("including", "excluding"):
            status = r[half].get("status")
            if status in RETRYABLE_STATUS:
                assert r[half]["parsed"] is None, r
                assert r[half]["attempts"] > 1, r

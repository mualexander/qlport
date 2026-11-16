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

"""Paper §3 findings as executable checks."""
import numpy as np
import pytest

from qlport.harness.workload import poisson, scrape, steady
from qlport.metrics import d, kappa, reldiff
from qlport.semantics import datadog_eval as dd
from qlport.semantics import promql_eval as pq
from qlport.semantics.ingest import cumsum_series, phi_series
from qlport.semantics.model import Series

HOUR = 3600


def counter(**kw):
    return scrape(poisson(10), 15, 6 * HOUR, start_value=5000, jitter=2, **kw)


# ---- ingestion map ------------------------------------------------------------------
def test_phi_omits_first_point_and_takes_deltas():
    s = Series({}, [15, 30, 45], [100, 130, 175], kind="counter")
    p = phi_series(s)
    assert list(p.ts) == [30, 45] and list(p.vs) == [30, 45]


def test_phi_reset_policies():
    s = Series({}, [15, 30, 45], [100, 20, 50], kind="counter")
    assert list(phi_series(s, "zero").vs) == [20, 30]
    assert list(phi_series(s, "drop").vs) == [30]


def test_phi_is_not_injective():
    a = counter()
    b = Series({}, a.ts, a.vs + 1e6, kind="counter")
    assert np.allclose(phi_series(a).vs, phi_series(b).vs)


def test_cumsum_inverts_phi_up_to_offset():
    a = counter()
    back = cumsum_series(phi_series(a), start=a.vs[0])
    assert np.allclose(back.vs, a.vs[1:])


def test_gauges_pass_through_phi():
    g = Series({}, [1, 2], [5, 3], kind="gauge")
    assert phi_series(g) is g


# ---- Corollary 2: rate() is not phi-invariant when a series starts in the window ------
def test_rate_reads_absolute_first_value():
    ts = np.arange(315, HOUR, 15.0)
    a = 1 + 10 * (ts - ts[0])
    b = a + 1e6
    assert np.allclose(np.diff(a), np.diff(b))
    assert reldiff(pq.rate(ts, a, 390, 300), pq.rate(ts, b, 390, 300)) > 0.05


# ---- Datadog -> PromQL: exact translations of .as_rate() rollups --------------------
@pytest.mark.parametrize("interval", [60, 300])
def test_offset_diff_is_exact_without_resets(interval):
    s = counter()
    p = phi_series(s)
    for t in dd.bucket_ends(600, s.ts[-1], interval):
        assert pq.offset_diff_rate(s.ts, s.vs, t, interval) == pytest.approx(
            dd.as_rate(p.ts, p.vs, t, interval), abs=1e-12)


@pytest.mark.parametrize("interval", [60, 300])
def test_anchored_model_is_exact_with_resets(interval):
    s = counter(restarts=tuple(np.arange(1800, 6 * HOUR, HOUR)))
    p = phi_series(s, "zero")
    for t in dd.bucket_ends(600, s.ts[-1], interval):
        assert pq.increase_anchored(s.ts, s.vs, t, interval) / interval == pytest.approx(
            dd.as_rate(p.ts, p.vs, t, interval), abs=1e-12)


def test_rate_is_exact_only_for_steady_aligned_counters():
    s = scrape(steady(10), 15, 6 * HOUR, start_value=5000)
    p = phi_series(s)
    for t in dd.bucket_ends(600, s.ts[-1], 300):
        assert pq.rate(s.ts, s.vs, t, 300) == pytest.approx(dd.as_rate(p.ts, p.vs, t, 300))


# ---- metrics --------------------------------------------------------------------------
def test_reldiff_is_symmetric_and_bounded():
    assert reldiff(1, 3) == reldiff(3, 1) == pytest.approx(2 / 3)
    assert reldiff(0, 5) == 1.0


def test_d_is_infinite_on_label_mismatch():
    assert d({("a",): 1.0}, {("b",): 1.0}) == float("inf")


def test_kappa_excludes_untested():
    assert kappa(["pass", "pass", "bottom", "untested"]) == pytest.approx(2 / 3)


def test_phi_clamp_keeps_the_point_and_zeroes_the_decrease():
    # The agent's non-OpenMetrics path: a decrease contributes nothing, but the interval
    # is still emitted -- which is what makes `clamp` distinguishable from `drop`.
    s = Series({}, np.array([0.0, 1.0, 2.0]), np.array([10.0, 5.0, 8.0]), kind="counter")
    clamped = phi_series(s, "clamp")
    assert list(clamped.vs) == [0.0, 3.0]
    assert list(clamped.ts) == [1.0, 2.0]
    # `drop` loses the interval entirely, so the two policies differ in length.
    assert len(phi_series(s, "drop").vs) == 1


def test_the_three_reset_policies_disagree_exactly_where_they_should():
    s = Series({}, np.array([0.0, 1.0, 2.0]), np.array([1000.0, 100.0, 101.0]),
               kind="counter")
    assert list(phi_series(s, "zero").vs) == [100.0, 1.0]
    assert list(phi_series(s, "clamp").vs) == [0.0, 1.0]
    assert list(phi_series(s, "drop").vs) == [1.0]


def test_the_policies_agree_when_no_counter_resets():
    s = Series({}, np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 6.0]), kind="counter")
    runs = [list(phi_series(s, r).vs) for r in ("zero", "clamp", "drop")]
    assert runs[0] == runs[1] == runs[2] == [2.0, 3.0]


def test_an_unknown_reset_policy_is_refused():
    s = Series({}, np.array([0.0, 1.0]), np.array([5.0, 1.0]), kind="counter")
    with pytest.raises(ValueError, match="unknown reset policy"):
        phi_series(s, "mystery")

"""The result distance d (paper §3, "Result distance") and the metric lemma.

§3's lemma "d is a metric" is the claim, and §3.1's admissibility proposition is a
triangle-inequality argument, so the claim is load-bearing rather than decorative. Its
proof splits on signs -- the interesting case being arguments that straddle zero, where
δ exceeds 1 -- so the cases are checked here as well as argued there.
"""
from __future__ import annotations

import itertools
import math
import random

import pytest

from qlport.metrics import d, kappa, reldiff

VALUES = [-8.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 8.0]


def test_delta_is_zero_only_on_equal_values():
    assert reldiff(0.0, 0.0) == 0.0
    assert reldiff(5.0, 5.0) == 0.0
    for a, b in itertools.product(VALUES, repeat=2):
        assert (reldiff(a, b) == 0.0) == (a == b), (a, b)


def test_delta_is_symmetric():
    for a, b in itertools.product(VALUES, repeat=2):
        assert reldiff(a, b) == pytest.approx(reldiff(b, a)), (a, b)


def test_delta_is_bounded_by_two_and_by_one_within_a_sign():
    for a, b in itertools.product(VALUES, repeat=2):
        assert reldiff(a, b) <= 2.0 + 1e-12, (a, b)
        if a * b > 0:
            assert reldiff(a, b) <= 1.0 + 1e-12, (a, b)


def test_a_sign_change_costs_at_least_one():
    # The step the proof uses to dispose of the mixed-sign cases.
    for a, b in itertools.product(VALUES, repeat=2):
        if a * b < 0 or (a == 0) != (b == 0):
            assert reldiff(a, b) >= 1.0 - 1e-12, (a, b)


def test_the_triangle_inequality_holds_on_a_grid_spanning_the_sign_cases():
    for a, b, c in itertools.product(VALUES, repeat=3):
        assert reldiff(a, c) <= reldiff(a, b) + reldiff(b, c) + 1e-12, (a, b, c)


def test_the_triangle_inequality_survives_random_search():
    # The grid covers the case analysis; this looks for anything it misses, including
    # the wide magnitude ratios the relative scale exists to handle.
    rng = random.Random(7)
    pools = [lambda: rng.uniform(0.0, 100.0),
             lambda: rng.uniform(-50.0, 50.0),
             lambda: rng.choice([-1e9, -1.0, -1e-9, 0.0, 1e-9, 1.0, 1e9])]
    for pool in pools:
        for _ in range(20_000):
            a, b, c = pool(), pool(), pool()
            assert reldiff(a, c) <= reldiff(a, b) + reldiff(b, c) + 1e-9, (a, b, c)


def test_mismatched_label_sets_are_infinitely_far_apart():
    # Not a large number: the translation returned the wrong series, and no tolerance
    # should absorb that.
    assert d({("k", "a"): 1.0}, {("k", "b"): 1.0}) == math.inf
    assert d({("k", "a"): 1.0}, {}) == math.inf
    assert d({}, {}) == 0.0


def test_d_is_the_worst_label_set_not_the_average():
    r1 = {("k", "a"): 1.0, ("k", "b"): 1.0}
    r2 = {("k", "a"): 1.0, ("k", "b"): 2.0}
    assert d(r1, r2) == pytest.approx(0.5)


def test_d_inherits_the_triangle_inequality_over_a_common_domain():
    keys = [("k", "a"), ("k", "b")]
    rng = random.Random(11)
    for _ in range(5_000):
        rs = [{k: rng.uniform(-10, 10) for k in keys} for _ in range(3)]
        assert d(rs[0], rs[2]) <= d(rs[0], rs[1]) + d(rs[1], rs[2]) + 1e-9


def test_kappa_here_is_the_differential_measure_not_the_one_the_paper_reports():
    # §3's remark "Two coverages": this one needs per-query outcomes from both systems.
    # §7's κ is translated-over-total and is computed in notebook 03.
    assert kappa(["pass", "pass", "bottom", "untested"]) == pytest.approx(2 / 3)
    assert math.isnan(kappa(["untested"]))

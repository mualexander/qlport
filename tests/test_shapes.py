"""Instantiating a corpus query so that it selects something (paper §7).

A differential test that compares two empty results proves nothing, so the series it
creates have to satisfy the query's own filters. These tests pin down what "satisfy"
means for each matcher kind, and that the module refuses rather than guesses where it
cannot.
"""
from __future__ import annotations

import pytest

from qlport import ir
from qlport.harness.shapes import SeriesSpec, instantiate, value_for_glob


def sel(matchers=(), metric="m", kind="gauge"):
    return ir.Select(metric, matchers, kind=kind)


def rolled(select, op="avg", keys=()):
    return ir.Agg(op, keys, "by", ir.Rollup(op, 60.0, select))


@pytest.mark.parametrize("glob,expected", [
    ("web-*", "web-x"), ("*-trace", "x-trace"), ("*mid*", "xmidx"),
    ("plain", "plain"), ("*", "x"),
])
def test_a_glob_yields_a_value_it_selects(glob, expected):
    assert value_for_glob(glob) == expected


def test_an_equality_matcher_fixes_the_label():
    got = instantiate(rolled(sel((ir.Matcher("result", "=", "miss"),))))
    assert got.ok
    assert all(s.labels["result"] == "miss" for s in got.series)


def test_presence_is_satisfied_by_any_value():
    got = instantiate(rolled(sel((ir.Matcher("host", "=~", ".+"),))))
    assert got.ok
    assert all(s.labels.get("host") for s in got.series)


def test_an_alternation_is_satisfied_by_one_branch():
    got = instantiate(rolled(sel((ir.Matcher("k", "=~", "GET|LIST"),))))
    assert got.ok
    assert {s.labels["k"] for s in got.series} == {"GET"}


def test_a_negated_matcher_is_satisfied_by_absence():
    got = instantiate(rolled(sel((ir.Matcher("env", "!=", "prod"),))))
    assert got.ok
    assert all("env" not in s.labels for s in got.series)


def test_an_unsatisfiable_regex_is_refused_not_guessed():
    got = instantiate(rolled(sel((ir.Matcher("k", "=~", "[0-9]{3}"),))))
    assert not got.ok
    assert "cannot satisfy" in got.reason


def test_grouping_produces_several_series_so_aggregation_has_work():
    got = instantiate(rolled(sel(), keys=("host",)))
    assert got.ok
    assert len(got.series) == 3
    assert {s.labels["host"] for s in got.series} == {"g0", "g1", "g2"}
    assert got.group_keys == ("host",)


def test_a_grouping_key_the_scope_pins_keeps_its_value():
    """Varying it would make the query select nothing, which is the opposite of useful."""
    got = instantiate(rolled(sel((ir.Matcher("host", "=", "h1"),)), keys=("host",)))
    assert got.ok
    assert {s.labels["host"] for s in got.series} == {"h1"}
    assert len(got.series) == 1


def test_arithmetic_instantiates_both_sides():
    lhs = ir.Rollup("sum", 60.0, sel(metric="a", kind="counter"))
    rhs = ir.Rollup("sum", 60.0, sel(metric="b", kind="counter"))
    got = instantiate(ir.BinOp("/", lhs, rhs))
    assert got.ok
    assert {s.metric for s in got.series} == {"a", "b"}


def test_duplicate_series_are_not_created_twice():
    same = ir.Rollup("sum", 60.0, sel(metric="a", kind="counter"))
    got = instantiate(ir.BinOp("/", same, same))
    assert got.ok
    assert len({s.key() for s in got.series}) == len(got.series)


def test_a_query_with_no_selector_is_refused():
    assert not instantiate(ir.Number(1.0)).ok

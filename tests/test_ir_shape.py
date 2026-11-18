"""The shape abstraction (paper §7): what a differential test actually has to cover.

The point of the abstraction is that two queries differing only in which metric they
name are one experiment. These tests pin down what counts as "only" -- in particular
that the rollup's *stated-ness* survives while its value does not, since whether the
query carried an interval is the thing `ROLLUP_DEFAULT` is about.
"""
from __future__ import annotations

import pytest

from qlport import ir
from qlport.ir import Agg, BinOp, Matcher, Number, Rollup, Scale, Select, shape, shape_slots


def q(metric="cpu", matchers=(), kind="counter", op="sum", interval=60.0, keys=()):
    return Agg(op, keys, "by", Rollup(op, interval, Select(metric, matchers, kind=kind)))


def test_queries_differing_only_in_metric_name_share_a_shape():
    assert shape(q(metric="cpu")) == shape(q(metric="mem"))


def test_queries_differing_only_in_tag_value_share_a_shape():
    a = q(matchers=(Matcher("env", "=", "prod"),))
    b = q(matchers=(Matcher("env", "=", "staging"),))
    assert shape(a) == shape(b)


def test_queries_differing_in_tag_key_still_share_a_shape():
    # The key's identity is positional too; what matters is how many there are and
    # where they appear.
    a = q(matchers=(Matcher("env", "=", "p"),))
    b = q(matchers=(Matcher("region", "=", "p"),))
    assert shape(a) == shape(b)


def test_a_different_operator_is_a_different_shape():
    assert shape(q(op="sum")) != shape(q(op="avg"))


def test_a_different_matcher_operator_is_a_different_shape():
    a = q(matchers=(Matcher("env", "=", "p"),))
    b = q(matchers=(Matcher("env", "!=", "p"),))
    assert shape(a) != shape(b)


def test_whether_the_interval_was_stated_survives_abstraction():
    """`ROLLUP_DEFAULT` is about exactly this, so it cannot be abstracted away."""
    assert shape(q(interval=60.0)) != shape(q(interval=None))


def test_the_intervals_value_survives_because_it_is_emitted():
    """The interval becomes the range-selector width, so it is not an identifier.

    This test asserted the opposite until the seventh review pass, and the behaviour it
    pinned put 70 corpus occurrences through the differential of §7 at a window width
    they do not state -- two of them a width the rig cannot test at all. A shape exists
    so that one member can stand for the class; a member that emits a different
    expression is not represented by it.
    """
    assert shape(q(interval=60.0)) != shape(q(interval=300.0))
    assert shape(q(interval=60.0)) == shape(q(interval=60.0))


def test_grouping_keys_count_but_are_not_named():
    assert shape(q(keys=("host",))) == shape(q(keys=("pod",)))
    assert shape(q(keys=("host",))) != shape(q(keys=("host", "pod")))


def test_regex_matcher_values_collapse_to_one_placeholder():
    # §6 tests the regular expressions themselves; a differential test does not need
    # one experiment per pattern.
    a = q(matchers=(Matcher("k", "=~", "a.*"),))
    b = q(matchers=(Matcher("k", "=~", "b.+"),))
    assert shape(a) == shape(b)


def test_the_metric_kind_survives_because_the_rule_depends_on_it():
    assert shape(q(kind="counter")) != shape(q(kind="unknown"))


def test_shape_is_hashable_and_stable():
    s = shape(q())
    assert hash(s) == hash(shape(q()))
    assert s == shape(q())


def test_slots_count_what_has_to_be_created():
    e = Agg("sum", ("host",), "by",
            Rollup("sum", 60.0, Select("cpu", (Matcher("env", "=", "p"),), kind="counter")))
    assert shape_slots(e) == (1, 2)      # one metric, tag keys {env, host}


def test_slots_see_both_sides_of_arithmetic():
    lhs = Rollup("sum", 60.0, Select("a", (), kind="counter"))
    rhs = Rollup("sum", 60.0, Select("b", (), kind="counter"))
    assert shape_slots(BinOp("/", lhs, rhs)) == (2, 0)


def test_an_unknown_node_is_refused_rather_than_silently_shaped():
    with pytest.raises(TypeError, match="unknown IR node"):
        shape("not an ir node")

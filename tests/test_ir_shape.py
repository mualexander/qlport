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


# --- what the abstraction keeps about a value, which is not nothing -------------------
# `test_queries_differing_only_in_tag_value_share_a_shape` is the headline: a value is
# not structure. But the canonicalisation is positional rather than blanket, and
# mutation found the asymmetry that makes it so unconstrained -- swapping which
# operator gets the blanket `*` changes no shape on any corpus query. It changes these.

def test_two_matchers_pinned_to_one_value_are_a_different_shape_from_two_pinned_apart():
    """Equality values are canonicalised by first appearance, not erased.

    `{a="X", b="X"}` asks for two labels to agree; `{a="X", b="Y"}` does not. That is a
    difference in what the query selects -- one is satisfiable by a single series only
    if both labels carry the same value -- so §7 counts them as two experiments, and
    `harness/shapes.py` instantiates them differently.

    Over the corpus the distinction never fires: swapping the two branches of this
    conditional leaves all 343 shapes exactly where they were, which is why nothing
    noticed. It is still the difference between an abstraction that forgets a value and
    one that forgets which values coincide.
    """
    same = q(matchers=(Matcher("a", "=", "X"), Matcher("b", "=", "X")))
    apart = q(matchers=(Matcher("a", "=", "X"), Matcher("b", "=", "Y")))
    assert shape(same) != shape(apart)
    # And renaming the shared value changes nothing, because only the sharing matters.
    assert shape(same) == shape(q(matchers=(Matcher("a", "=", "Z"),
                                            Matcher("b", "=", "Z"))))


def test_two_regex_matchers_share_a_shape_whether_or_not_their_patterns_agree():
    """The other branch, and the reason it is the other branch.

    A regex is not a value: `k=~"GET"` and `k=~"GET|LIST"` select different sets, and
    neither selects by equality, so there is nothing for two of them to agree *on*.
    They are abstracted to one token, and two queries carrying the same pattern twice
    are the same experiment as two carrying different patterns.
    """
    same = q(matchers=(Matcher("a", "=~", "X"), Matcher("b", "=~", "X")))
    apart = q(matchers=(Matcher("a", "=~", "X"), Matcher("b", "=~", "Y")))
    assert shape(same) == shape(apart)
    # Which is not to say the operator is forgotten: an equality is still not a regex.
    assert shape(same) != shape(q(matchers=(Matcher("a", "=", "X"),
                                            Matcher("b", "=", "X"))))


def test_a_rollup_is_not_per_second_unless_the_lowering_says_so():
    """`per_second` defaults to false, and the default is what most lowerings rely on.

    It is the `.as_rate()` normalisation flag, so a rollup that acquired it by default
    would divide by the interval on every query that never asked. The flag is part of
    the shape, because whether the expression normalises is not a detail of which
    metric it names.
    """
    plain = Rollup("avg", 60.0, Select("cpu", (), kind="counter"))
    assert plain.per_second is False
    rated = Rollup("avg", 60.0, Select("cpu", (), kind="counter"), per_second=True)
    assert shape(plain) != shape(rated)

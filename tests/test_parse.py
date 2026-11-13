"""RQ1: both grammars build as LALR(1) tables (no conflicts) and parse representative queries."""
import pytest

from qlport.parse import get_parser, parse

PROMQL = [
    'sum by (service) (rate(http_requests_total{status=~"5.."}[5m]))',
    'sum(rate(http_requests_total[5m])) by (service)',
    'http_requests_total',
    'rate(x[5m] offset 1h)',
    'max_over_time(rate(x[1m])[30m:1m])',
    'a / on(instance) group_left(job) b',
    '2 ^ 3 ^ 2',
    'x > bool 3',
    '-x + 1',
]

DATADOG = [
    'sum:http.requests{status:5*} by {service}.as_rate()',
    'avg:system.load.1{*}',
    'sum:http.requests{!env:dev,service:api} by {service,host}.as_count().rollup(sum, 60)',
    'abs(avg:system.load.1{*}) * 2',
    'sum:a{*} / sum:b{*}',
]


@pytest.mark.parametrize("lang", ["promql", "datadog"])
def test_grammar_is_lalr1(lang):
    get_parser(lang)  # raises GrammarError on any LALR conflict


@pytest.mark.parametrize("q", PROMQL)
def test_promql_parses(q):
    parse("promql", q)


@pytest.mark.parametrize("q", DATADOG)
def test_datadog_parses(q):
    parse("datadog", q)


def test_pow_is_right_associative():
    t = parse("promql", "2 ^ 3 ^ 2")
    assert t.data == "binop"
    assert t.children[-1].data == "binop"  # 2 ^ (3 ^ 2)


def test_mul_binds_tighter_than_add():
    t = parse("promql", "a + b * c")
    assert t.children[-1].data == "binop"  # a + (b * c)

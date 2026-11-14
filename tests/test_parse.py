"""RQ1: both grammars build as LALR(1) tables (no conflicts) and parse representative queries."""
import pytest

from qlport.parse import get_parser, parse

PROMQL = [
    # v2: case-insensitive keywords, signed offset, bare-dot float, raw strings,
    # UTF-8 label names, the @ modifier.
    'SUM(x) BY (job)',
    'avg(x) OR on() vector(0)',
    'x offset -1d',
    'histogram_quantile(.95, x)',
    'rate(x{integration=~`.*`}[5m])',
    'up{"otel.signal"="logs"}',
    '{"my.metric", job="x"}',
    'sum by ("otel.scope") (rate(x[5m]))',
    'rate(x[5m] @ end())',
    'x @ 1609746000',
    'a atan2 b',
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
    # v2: no space aggregator, count:/pNN:, boolean scopes, query lists, single
    # quotes, wildcard tag keys, unary signs, negative arguments.
    'system.load.1{*}',
    'count:slurm.sacct.enabled{*}',
    'p95:trace.http.request{*}',
    'avg:gpu.sm_active{cloud_provider IN (aws,gcp) AND NOT *-zone}',
    'avg:x{not status_type:completed and *}',
    'avg:x{(a:1 OR b:2) AND c:3}',
    'sum:riak.node_gets{*},sum:riak.node_puts{*}',
    "top(avg:istio.go.threads{*}, 10, 'mean', 'desc')",
    'avg:x{mountpoint:/, ! fstype:rootfs}',
    'avg:system.load.1{field:apps:argocd}',
    '-sum:tomcat.bytes_rcvd{*}',
    'timeshift(sum:a{*}, -300)',
    'min:velero.backup.last_status{*-location,*-name,*}',
    'sum.of.things{*}',
    'sum:http.requests{status:5*} by {service}.as_rate()',
    'avg:system.load.1{*}',
    'sum:http.requests{!env:dev,service:api} by {service,host}.as_count().rollup(sum, 60)',
    'abs(avg:system.load.1{*}) * 2',
    'sum:a{*} / sum:b{*}',
]


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("lang", ["promql", "datadog"])
def test_grammar_is_lalr1(lang, version):
    # RQ1: Lark raises GrammarError on any shift/reduce or reduce/reduce conflict, so
    # construction succeeding is the proof. v1 is kept conflict-free too, because the
    # gap table of paper §4 is measured against it.
    get_parser(lang, version)


def test_unknown_grammar_version_is_rejected():
    with pytest.raises(ValueError):
        get_parser("promql", "v3")


def test_scope_boolean_precedence_is_and_over_or():
    # {a AND b OR c} is ({a AND b}) OR c
    t = parse("datadog", "avg:x{a:1 AND b:2 OR c:3}")
    scope = next(t.find_data("scope"))
    assert scope.children[0].data == "scope_or"
    assert scope.children[0].children[0].data == "scope_and"


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

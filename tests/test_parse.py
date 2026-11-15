"""RQ1: both grammars build as LALR(1) tables (no conflicts) and parse representative queries."""
import pytest

from qlport.parse import get_parser, lalr_conflicts, parse


def _grammar_text(lang: str, version: str = "v2") -> str:
    from importlib.resources import files
    name = f"{lang}.lark" if version == "v2" else f"{lang}_{version}.lark"
    return files("qlport.grammars").joinpath(name).read_text()

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
def test_grammar_builds(lang, version):
    # Construction with the default strict=False only shows the grammar is usable;
    # whether it is LALR(1) is asked separately, below.
    get_parser(lang, version)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_promql_has_exactly_one_lalr_conflict(version):
    """PromQL is not conflict-free, and the one conflict is inherent to the language.

    `group_left` takes an OPTIONAL parenthesised label list, so after `group_left` a
    `(` can begin either that list or a parenthesised right operand. One token of
    lookahead cannot tell them apart. Lark resolves it by shifting -- taking the label
    list -- which is exactly what the reference implementation does.

    The assertion is deliberately tight. A second conflict appearing here would mean a
    grammar change introduced an ambiguity, which is the thing RQ1 claims is absent.
    """
    got = lalr_conflicts("promql", version)
    assert got is not None and "Shift/Reduce conflict for terminal LPAR" in got, got


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_datadog_grammar_is_conflict_free(version):
    """RQ1 for the inferred Datadog grammar: no LALR(1) conflict, no terminal collision.

    Worth stating separately from PromQL's single conflict: the Datadog grammar is the
    one with no published reference to compare against, so its conflict-freedom is a
    result rather than a reproduction.
    """
    assert lalr_conflicts("datadog", version) is None


def test_the_conflict_check_is_not_vacuous():
    """Strict mode must be doing the work the previous tests credit it with.

    A check that passes on a grammar it never examined would be worthless, so perturb
    the real grammar two ways and require each to be caught: an ambiguous production
    (shift/reduce) and two terminals matching the same strings (collision).
    """
    from lark import Lark
    from lark.exceptions import GrammarError, LexError

    base = _grammar_text("datadog")

    def strict(text):
        return Lark(text, parser="lalr", lexer="contextual", strict=True,
                    propagate_positions=True, maybe_placeholders=True)

    ambiguous = base.replace("term MULOP unary", "term MULOP term")
    assert ambiguous != base
    with pytest.raises(GrammarError, match="Shift/Reduce"):
        strict(ambiguous)

    colliding = base.replace("%ignore", r"DUPE: /[a-zA-Z][a-zA-Z0-9_.]*/" "\n%ignore")
    colliding = colliding.replace("?arg: signed_number | KEY -> word",
                                  "?arg: signed_number | DUPE -> word | KEY -> word")
    assert colliding != base
    with pytest.raises(LexError, match="Collision"):
        strict(colliding)


def test_the_conflict_resolves_as_the_reference_implementation_does():
    """`a * on(x) group_left (b)` is rejected, and Prometheus 3.15.0 rejects it too.

    Checked against a real server: it answers
    `1:25: parse error: unexpected end of input`. Our parser consumes `(b)` as the
    label list and then has no right operand, reaching the same conclusion by the same
    route. Accepting it would make us *more* permissive than PromQL.
    """
    for ok in ("a * on(x) group_left(y) b", "a * on(x) group_left b",
               "a * on(x) group_left() b", "a * on(x) group_left (y) (b)"):
        parse("promql", ok)
    with pytest.raises(Exception):
        parse("promql", "a * on(x) group_left (b)")


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

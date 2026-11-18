"""Corpus extraction, normalisation and feature detection (notebook 00)."""
import pytest

from qlport import corpus as c

# --- extraction ----------------------------------------------------------------------
DD_DASHBOARD = {
    "title": "ActiveMQ",
    "widgets": [
        {"definition": {"type": "image", "url": "/logo.svg"}},
        {"definition": {
            "type": "group",
            "title": "Queues",
            "widgets": [{"definition": {
                "type": "timeseries",
                "title": "Enqueued",
                "requests": [{
                    "response_format": "timeseries",
                    "queries": [
                        {"data_source": "metrics", "name": "query1",
                         "query": "sum:activemq.queue.enqueue_count{$host} by {queue}.as_count()"},
                        {"data_source": "logs", "name": "query2",
                         "search": {"query": "source:activemq"}},
                    ],
                    "formulas": [{"formula": "query1"}],
                }],
            }}],
        }},
        {"definition": {
            "type": "timeseries", "title": "Legacy",
            "requests": [{"q": "avg:system.load.1{*}, avg:system.load.5{*}"}],
        }},
    ],
}


def test_datadog_extraction_recurses_into_groups_and_skips_logs():
    qs = c.datadog_dashboard_queries(DD_DASHBOARD, source="ic", origin="activemq.json")
    metric = [q for q in qs if q.lang == "datadog"]
    formula = [q for q in qs if q.lang == "datadog_formula"]
    assert [q.text for q in metric] == [
        "sum:activemq.queue.enqueue_count{$host} by {queue}.as_count()",
        "avg:system.load.1{*}, avg:system.load.5{*}",
    ]
    assert [q.text for q in formula] == ["query1"]
    assert metric[0].unit == "Enqueued" and metric[0].kind == "widget"
    assert metric[0].source == "ic" and metric[0].origin == "activemq.json"


def test_datadog_requests_may_be_a_dict():
    doc = {"widgets": [{"definition": {"requests": {"fill": {"q": "avg:a{*}"}}}}]}
    assert [q.text for q in c.datadog_dashboard_queries(doc, source="s", origin="o")] == ["avg:a{*}"]


GRAFANA_DASHBOARD = {
    "title": "Node",
    "templating": {"list": [{"name": "job", "query": "label_values(up, job)"}]},
    "panels": [
        {"title": "Row", "type": "row", "panels": [
            {"title": "CPU", "targets": [{"expr": "rate(node_cpu_seconds_total[5m])"},
                                         {"expr": "   "}, {"refId": "B"}]}]},
    ],
    "rows": [{"panels": [{"title": "Old", "targets": [{"expr": "node_load1"}]}]}],
}


def test_grafana_extraction_reaches_nested_and_legacy_panels():
    qs = c.grafana_dashboard_queries(GRAFANA_DASHBOARD, source="g", origin="1860")
    assert [q.text for q in qs] == ["rate(node_cpu_seconds_total[5m])", "node_load1"]
    assert qs[0].unit == "CPU" and qs[0].kind == "panel"


def test_grafana_extraction_excludes_variable_dsl():
    texts = [q.text for q in c.grafana_dashboard_queries(GRAFANA_DASHBOARD, source="g", origin="x")]
    assert not any("label_values" in t for t in texts)


PROM_RULE = {
    "kind": "PrometheusRule",
    "spec": {"groups": [{"name": "k8s", "rules": [
        {"record": "node:cpu:ratio", "expr": "sum(rate(x[5m]))"},
        {"alert": "HighCPU", "expr": "node:cpu:ratio > 0.9", "for": "10m"},
        {"alert": "Empty", "expr": ""},
    ]}]},
}


def test_prometheus_rule_extraction_tags_alert_and_record():
    qs = c.prometheus_rule_queries(PROM_RULE, source="kp", origin="rule.yaml")
    assert [(q.kind, q.unit) for q in qs] == [("record", "node:cpu:ratio"), ("alert", "HighCPU")]


def test_non_rule_objects_yield_nothing():
    assert c.prometheus_rule_queries({"kind": "ConfigMap"}, source="s", origin="o") == []


# --- normalisation -------------------------------------------------------------------
@pytest.mark.parametrize("raw,want", [
    ("rate(x[$__rate_interval])", "rate(x[5m])"),
    ("rate(x[${interval}])", "rate(x[5m])"),
    ("rate(x[[[interval]]])", "rate(x[5m])"),
    ("max_over_time(rate(x[1m])[$__range:$__interval])", "max_over_time(rate(x[1m])[5m:1m])"),
    ("max_over_time(rate(x[1m])[30m:$__interval])", "max_over_time(rate(x[1m])[30m:1m])"),
    ("rate(x[5m])", "rate(x[5m])"),                   # literal durations survive
    ("max_over_time(x[$__range:])", "max_over_time(x[5m:])"),
    ('up{pod=~"$pod"}', 'up{pod=~"$pod"}'),          # inside a string: already parseable
    ("sum($metric)", "sum(tmplvar)"),
])
def test_promql_normalisation(raw, want):
    assert c.normalise("promql", raw) == want


@pytest.mark.parametrize("raw,want", [
    ("avg:a.b{$host}", "avg:a.b{*}"),
    ("avg:a.b{$host,svc:$s}", "avg:a.b{*,svc:*}"),
    ("sum:a{*} by {$k}", "sum:a{*} by {tmplvar}"),
    ("sum:a{*}.rollup(sum, 60)", "sum:a{*}.rollup(sum, 60)"),
])
def test_datadog_normalisation(raw, want):
    assert c.normalise("datadog", raw) == want


def test_normalisation_handles_a_variable_inside_a_duration():
    # the variable supplies only the number; the unit is literal text after it
    assert c.normalise("promql", "increase(x[${__range_s}s])") == "increase(x[5m])"
    assert c.normalise("promql", 'rate(x{a=~"[4-5].*"}[1m])') == 'rate(x{a=~"[4-5].*"}[1m])'


def test_normalisation_rejects_unknown_language():
    with pytest.raises(ValueError):
        c.normalise("logql", "x")


def test_normalised_corpus_shapes_parse():
    for lang, q in [
        ("promql", 'sum by (ns) (rate(http_requests_total{pod=~"$pod"}[$__rate_interval]))'),
        ("datadog", "sum:http.requests{$host,!env:dev} by {service}.as_count().rollup(sum, 60)"),
    ]:
        assert c.parse_outcome(lang, c.normalise(lang, q)).ok


# --- features ------------------------------------------------------------------------
def test_datadog_features():
    f = c.features("datadog", "sum:a.b{$host,!env:dev,k:v*} by {h}.as_count().rollup(sum, 60)")
    assert {"AGG_SUM", "SCOPE_VAR", "SCOPE_NEG", "SCOPE_GLOB", "GROUP_BY",
            "M_AS_COUNT", "M_ROLLUP", "M_ROLLUP_INTERVAL"} <= f
    assert "M_AS_RATE" not in f and "M_ROLLUP_AGG_ONLY" not in f


def test_datadog_percentile_and_count_aggregators_are_distinguished():
    assert "AGG_PCTL" in c.features("datadog", "p95:a.b{*}")
    assert "AGG_COUNT" in c.features("datadog", "count:a.b{*}")


def test_datadog_boolean_scope_features():
    f = c.features("datadog", "sum:a{$h AND (phase:ok OR phase:run) AND NOT k:v}")
    assert {"SCOPE_BOOL", "SCOPE_PAREN", "SCOPE_VAR", "SCOPE_EQ"} <= f
    assert "SCOPE_BARE" not in f           # $h is a variable, not a bare key
    assert "SCOPE_IN" in c.features("datadog", "sum:a{* AND origin IN (x,y)}")
    assert "SCOPE_BARE" in c.features("datadog", "sum:a{* AND origin IN (x,y)}")


def test_datadog_glob_in_scope_is_not_read_as_arithmetic():
    f = c.features("datadog", "sum:a{func:*lf*}")
    assert "SCOPE_GLOB" in f and "ARITH" not in f and "MULTI_QUERY" not in f


def test_datadog_arith_and_multi_query():
    assert "ARITH" in c.features("datadog", "sum:a{*} / sum:b{*}")
    assert "MULTI_QUERY" in c.features("datadog", "avg:a{*}, avg:b{*}")
    assert "MULTI_QUERY" not in c.features("datadog", "top(avg:a{*} by {h}, 10, 'mean', 'desc')")
    assert "STR_SINGLE_QUOTED" in c.features("datadog", "top(avg:a{*}, 10, 'mean', 'desc')")


def test_datadog_rollup_arity():
    assert "M_ROLLUP_AGG_ONLY" in c.features("datadog", "sum:a{*}.rollup(max)")
    assert "M_ROLLUP_INTERVAL" in c.features("datadog", "sum:a{*}.rollup(max, 300)")


def test_formula_features():
    assert c.features("datadog_formula", "query1") == frozenset({"FORMULA_BARE_REF"})
    f = c.features("datadog_formula", "default_zero(query1) / query2 * 100")
    assert {"FORMULA_ARITH", "FORMULA_FUNC", "FORMULA_CONST"} <= f
    assert "FORMULA_BARE_REF" not in f


def test_promql_features():
    f = c.features("promql", 'sum by (ns) (rate(x{a!~"b",c!="d"}[5m] offset 1h)) / on(ns) group_left y')
    assert {"SEL_RANGE", "OFFSET", "MATCH_NRE", "MATCH_NEQ", "AGG_BY", "AGG_APPLIED",
            "FUNC_APPLIED", "BINOP_ARITH", "MATCH_ON", "GROUP_LR", "STR_LIT"} <= f
    assert "MATCH_EQ" not in f and "SUBQUERY" not in f


def test_promql_subquery_at_modifier_and_template_range():
    assert "SUBQUERY" in c.features("promql", "max_over_time(rate(x[1m])[30m:1m])")
    assert "AT_MODIFIER" in c.features("promql", "x @ end()")
    assert {"TMPL_VAR", "TMPL_RANGE"} <= c.features("promql", "rate(x[$__rate_interval])")
    assert "TMPL_RANGE" not in c.features("promql", 'rate(x{p=~"$pod"}[5m])')


def test_promql_matcher_regex_is_not_read_as_a_comparison():
    f = c.features("promql", 'up{job=~"a|b",env!="dev"}')
    assert {"MATCH_RE", "MATCH_NEQ"} <= f
    assert "BINOP_CMP" not in f and "BINOP_SET" not in f and "BINOP_ARITH" not in f


def test_promql_nameless_selector():
    assert "SEL_NAMELESS" in c.features("promql", '{__name__=~"a|b"}')
    assert "SEL_NAMELESS" in c.features("promql", '{job="x"}')


def test_every_detectable_feature_is_documented():
    corpus = [
        ("datadog", "sum:a{$h AND (k:v OR o IN (x,y)) AND !b:c,d:e*} by {g}"
                    ".as_count().as_rate().rollup(sum, 60).fill(zero).weighted() / 2"),
        ("datadog", "top(count:a{*}, 10, 'mean', 'desc'), p95:b{*}.rollup(max)"),
        ("datadog", "max:a{*}, min:b{*}, avg:c{*}"),
        ("datadog_formula", "default_zero(query1) / query2 * 100 + $v"),
        ("datadog_formula", "query1"),
        ("promql", 'sum by (ns) (rate({__name__=~"a",b!~"c",d!="e",f="g"}[$__rate_interval] offset 1h)) '
                   '/ on(ns) group_left(j) count without (x) (y) > bool 3'),
        ("promql", "a / ignoring(x) b"),
        ("promql", "max_over_time(rate(x[1m])[30m:1m] @ end()) and z unless w"),
    ]
    seen = set()
    for lang, q in corpus:
        seen |= c.features(lang, q)
    assert seen <= set(c.FEATURE_DOC), sorted(seen - set(c.FEATURE_DOC))
    undocumented_but_reachable = {"AGG_OTHER"}   # needs an aggregator outside the doc set
    assert set(c.FEATURE_DOC) - seen == undocumented_but_reachable, sorted(set(c.FEATURE_DOC) - seen)


# --- grammar coverage ----------------------------------------------------------------
def test_parse_outcome_names_the_blocking_terminal():
    assert c.parse_outcome("promql", "rate(x[5m])").ok
    bad = c.parse_outcome("datadog", "count(v: v>=0):a.b{*}")
    assert not bad.ok and bad.blocker and bad.excerpt


def test_parse_outcome_selects_the_grammar_version():
    assert c.parse_outcome("datadog", "count:a.b{*}", "v2").ok
    assert not c.parse_outcome("datadog", "count:a.b{*}", "v1").ok


def test_parse_outcome_never_raises():
    for lang in ("promql", "datadog"):
        for q in ("", "{{{", "sum(", "\x00", "a" * 500):
            assert isinstance(c.parse_outcome(lang, q), c.ParseOutcome)


# --- storage -------------------------------------------------------------------------
def test_queries_round_trip(tmp_path):
    qs = c.datadog_dashboard_queries(DD_DASHBOARD, source="ic", origin="activemq.json")
    p = tmp_path / "queries.jsonl"
    assert c.write_queries(p, qs) == len(qs)
    assert c.read_queries(p) == qs


# --- repairs: each one is a grammar change, verified by construction -----------------
# Each row is read twice: the construct must fail under grammar v1 and be explained
# by exactly this repair, and it must parse under v2 with no repair at all. That pair
# is what makes the gap table of paper §4 a measurement rather than a classification.
REPAIR_EXAMPLES = [
    ("datadog", "100 * tokumx.ft.cachetable.size.current{*} by {host}", "IMPLICIT_AGG"),
    ("datadog", "count:a.b{*}", "AGG_SET"),
    ("datadog", "p95:a.b{*}", "AGG_SET"),
    ("datadog", "sum:a{* AND k:v}", "BOOL_SCOPE"),
    ("datadog", "sum:a{* AND o IN (x,y)}", "BOOL_SCOPE"),
    ("datadog", "avg:a{*}, avg:b{*}", "QUERY_LIST"),
    ("datadog", "top(avg:a{*}, 10, 'mean', 'desc')", "SINGLE_QUOTES"),
    ("datadog", "avg:a{field:apps:argocd}", "TAG_COLON"),
    ("datadog", "avg:a{*, mountpoint:/, ! fstype:rootfs}", "SCOPE_SPACING"),
    ("datadog", "sum:a{*,!*}", "NEGATED_WILDCARD"),
    ("datadog", "-sum:a{*}", "UNARY_SIGN"),
    ("datadog", "timeshift(sum:a{*}, -300)", "NEGATIVE_ARG"),
    ("promql", "SUM(envoy_server_memory_allocated)", "KEYWORD_CASE"),
    ("promql", "avg(x) OR on() vector(0)", "KEYWORD_CASE"),
    ("promql", 'up{"otel.signal"="logs"}', "QUOTED_LABEL"),
    ("promql", "x offset -1d", "SIGNED_OFFSET"),
    ("promql", "histogram_quantile(.95, x)", "LEADING_DOT_NUMBER"),
    ("datadog", "min:v.b{*-location,*}", "WILDCARD_TAG_KEY"),
    ("promql", "rate(x{integration=~`.*`}[5m])", "BACKTICK_STRING"),
]


@pytest.mark.parametrize("lang,text,repair", REPAIR_EXAMPLES)
def test_each_repair_explains_its_own_gap(lang, text, repair):
    assert not c.parse_outcome(lang, text, "v1").ok, "v1 already parses; the gap is not real"
    assert c.repairs_needed(lang, text, "v1") == (repair,)
    assert c.parse_outcome(lang, text, "v2").ok, "v2 does not close the gap it claims to"


def test_every_repair_has_an_example():
    # TEMPLATED_SYNTAX is a normalisation artefact, not a construct a grammar can add.
    tested = {r for _, _, r in REPAIR_EXAMPLES}
    assert set(c.REPAIR_DOC) - tested == {"TEMPLATED_SYNTAX"}


def test_repairs_needed_is_empty_for_a_parsing_query():
    assert c.repairs_needed("promql", "rate(x[5m])") == ()


def test_repairs_needed_reports_a_combination():
    got = c.repairs_needed("datadog", "count:a{* AND k:v}")
    assert got is not None and set(got) == {"AGG_SET", "BOOL_SCOPE"}


def test_repairs_needed_gives_up_rather_than_guessing():
    assert c.repairs_needed("datadog", "count(v: v>=0):a{*}") is None


def test_every_repair_is_documented():
    for lang in ("datadog", "promql"):
        for name, _ in c.REPAIRS[lang]:
            assert name in c.REPAIR_DOC


def test_logql_is_not_promql():
    assert not c.is_promql('{job="x"} | logfmt | level="warn"')
    assert not c.is_promql('{job="x"} |= "err"')
    # a bare line filter with no pipe stage: comparing a selector to a string is a
    # type error in PromQL, so this is Loki too.
    assert not c.is_promql('{app="loki",container="loki"} != "x" !~ "info"')
    assert c.is_promql('rate(x{job="y"}[5m])')
    assert c.is_promql('up{job="y"} != 0')


# ----------------------------------- name mangling on the corpus (paper §5, lem:scope)


def _corpus_identifiers():
    """Original Datadog metric names and grouping keys, before mangling."""
    from lark import Token, Tree

    from pathlib import Path

    from qlport.corpus import normalise, read_queries
    from qlport.parse import parse

    root = Path(__file__).resolve().parent.parent
    qs = [q for q in read_queries(root / "data" / "corpus" / "queries.jsonl")
          if q.lang == "datadog"]
    metrics, keys = set(), set()

    def walk(n):
        if not isinstance(n, Tree):
            return
        if str(n.data) == "query":
            for c in n.children:
                if isinstance(c, Token) and c.type in ("METRIC", "NAME", "CNAME"):
                    metrics.add(str(c))
                    break
        if str(n.data) == "grouping":
            keys.update(str(c) for c in n.children[1:] if isinstance(c, Token))
        for c in n.children:
            walk(c)

    for text in {normalise("datadog", q.text) for q in qs}:
        try:
            walk(parse("datadog", text, "v2"))
        except Exception:                                    # noqa: BLE001
            continue
    return metrics, keys


def _collisions(names, mangle):
    out = {}
    for n in names:
        out.setdefault(mangle(n), set()).add(n)
    return {k: v for k, v in out.items() if len(v) > 1}


def test_the_corpus_does_contain_colliding_names():
    """Lemma lem:scope's injectivity hypothesis is not vacuous on this corpus.

    An earlier draft discharged it by asserting that no corpus source contains a pair
    like `a.b` and `a_b`. It contains three, and the queries using the two metric pairs
    are all translated, so two distinct Datadog metrics are emitted as one PromQL
    selector. Pinned here because the claim it replaces was checked by reading.
    """
    from qlport.lower.datadog import label_name, metric_name

    metrics, keys = _corpus_identifiers()
    mc = _collisions(metrics, metric_name)
    kc = _collisions(keys, label_name)

    assert set(mc) == {"openstack_nova_hypervisor_load_1",
                       "rabbitmq_queue_messages_ready"}, mc
    assert set(kc) == {"cluster_name"}, kc
    assert mc["rabbitmq_queue_messages_ready"] == {
        "rabbitmq.queue.messages.ready", "rabbitmq.queue.messages_ready"}
    assert kc["cluster_name"] == {"cluster-name", "cluster_name"}


def test_the_collision_rate_is_low_enough_to_quote():
    # §5 quotes these denominators; a corpus change that moved them should fail here
    # rather than silently date the prose.
    metrics, keys = _corpus_identifiers()
    assert len(metrics) == 5389, len(metrics)
    assert len(keys) == 723, len(keys)


def test_a_hyphen_in_a_label_name_cannot_be_mangled_around():
    # The third collision is forced rather than incidental: PromQL label names are
    # [a-zA-Z_][a-zA-Z0-9_]*, so `cluster-name` has nowhere else to go.
    from qlport.lower.datadog import label_name

    assert label_name("cluster-name") == label_name("cluster_name") == "cluster_name"


# ---- the three ways a shipped query went missing (seventh review pass) --------


def test_a_request_carrying_its_query_as_an_object_is_extracted():
    """Distribution widgets use a singular `query` object, not `q` or a `queries` list.

    The extractor's docstring said there were two request shapes. The third cost four
    Datadog queries, which is 0.05% of the corpus and was invisible to every figure in
    the paper, because nothing counts what was never read.
    """
    doc = {
        "title": "d",
        "widgets": [{"definition": {"widgets": [{"definition": {
            "title": "Distribution of message size",
            "requests": [{"query": {"data_source": "metrics",
                                    "query": "avg:data_streams.payload_size{type:kafka}"}}],
        }}]}}],
    }
    got = c.datadog_dashboard_queries(doc, source="s", origin="o")
    assert [q.text for q in got] == ["avg:data_streams.payload_size{type:kafka}"]


def test_a_singular_query_object_of_another_language_is_still_skipped():
    doc = {"widgets": [{"definition": {"requests": [
        {"query": {"data_source": "logs", "query": "service:web"}}]}}]}
    assert c.datadog_dashboard_queries(doc, source="s", origin="o") == []


def test_a_formula_over_a_non_metric_request_is_not_a_metric_formula():
    """A formula belongs to the metrics language only when its queries do.

    The filter was applied to the queries and not to the formulas, so the formula layer
    §7 characterises was 48% log, event and security-signal formulas -- a population
    that is 90% bare references, which flattered every share in that sentence.
    """
    logs = {"widgets": [{"definition": {"requests": [{
        "queries": [{"data_source": "logs", "query": "service:web"}],
        "formulas": [{"formula": "a / b"}],
    }]}}]}
    assert c.datadog_dashboard_queries(logs, source="s", origin="o") == []

    metrics = {"widgets": [{"definition": {"requests": [{
        "queries": [{"data_source": "metrics", "query": "avg:x{*}"}],
        "formulas": [{"formula": "a / b"}],
    }]}}]}
    got = c.datadog_dashboard_queries(metrics, source="s", origin="o")
    assert [(q.lang, q.text) for q in got] == [
        ("datadog", "avg:x{*}"), ("datadog_formula", "a / b")]


def test_a_grafana_library_panel_is_visited():
    """A library panel's model lives in `__elements`; `panels` holds only a reference.

    Grafana writes that map as a list in older exports and as a UID-keyed dict in newer
    ones, and this corpus contains the list form -- so handling only the dict form fixed
    nothing, which is how the first attempt at this went.
    """
    model = {"targets": [{"expr": "idrac_cpu_count{instance=~\"$instance\"}"}],
             "title": "CPU Count"}
    for elements in ([{"model": model}], {"uid-1": {"model": model}}):
        doc = {"panels": [{"libraryPanel": {"uid": "uid-1"}, "title": "CPU Count"}],
               "__elements": elements}
        got = c.grafana_dashboard_queries(doc, source="s", origin="o")
        assert [q.text for q in got] == ["idrac_cpu_count{instance=~\"$instance\"}"], elements


def test_a_backtick_matcher_is_promql_not_logql():
    """PromQL permits a backtick raw string in a matcher; LogQL's line filter looks alike.

    Reading `!~` + backtick as a line filter wherever it appeared dropped two
    kube-prometheus alerting rules from the corpus as though they were Loki panels.
    """
    assert c.is_promql(
        'rate(alertmanager_notifications_failed_total{integration!~`.*`}[15m])')
    # A real line filter still is one: nothing precedes the `!~` but the selector.
    assert not c.is_promql('{app="loki"} !~ `error`')
    assert not c.is_promql('{app="loki"} | logfmt | level="warn"')


def test_a_conditional_aggregator_is_not_counted_as_a_function():
    """`count(v: v>=0):metric{...}` aggregates across space; §4 names it an aggregator.

    Counting its `count` among the function names put it in the paper's ten most frequent
    Datadog functions, beside `top` and `per_second`.
    """
    text = "count(v: v>=0):data_streams.latency{type:kafka}"
    assert c.function_names("datadog", text)["count"] == 0
    # A genuine call of the same name is still a function.
    assert c.function_names("datadog", "count_not_null(avg:x{*})")["count_not_null"] == 1


def test_every_promql_string_form_yields_its_matcher():
    """§6 tabulates "every `=~` and `!~` matcher", and a double-quoted-only scan did not.

    114 matcher occurrences sat in single-quoted and backtick literals, which the PromQL
    grammar accepts and two separate copies of the scanner did not.
    """
    q = ('node_filesystem_avail_bytes{instance="$node",fstype!~\'rootfs\','
         'mountpoint=~`/run/.*`,device=~"sd[a-z]"}')
    assert c.promql_matcher_values(q) == ["rootfs", "/run/.*", "sd[a-z]"]

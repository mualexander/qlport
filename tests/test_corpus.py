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
        ("datadog_formula", "autosmooth(query1) / ewma_3(query2)"),
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
        # lem:scope's hypothesis is over the tag keys a matcher can name, which is the
        # scopes' keys as well as the grouping list's -- for fourteen passes this census
        # took the grouping list alone, so the measurement was narrower than the
        # hypothesis it was cited for.
        if str(n.data) in ("tag", "member"):
            for c in n.children:
                if isinstance(c, Token) and c.type in ("KEY", "TAG", "NAME", "CNAME"):
                    keys.add(str(c))
                    break
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
    assert len(keys) == 1264, len(keys)


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


# --- what the escape decoder denotes, which no test reached --------------------------
# `scripts/mutate.py` found the whole hex-escape branch of `unquote_promql` executed by
# nothing: its three widths, its validity check, how far it advances and its fallback
# were all free. Only `promql_matcher_values` reaches the function at all, and its one
# test passed three patterns with no escape in any of them.
#
# What that costs is in the function's own docstring. §6 classifies regexes from what a
# matcher *denotes*, and five corpus patterns were once classified from the undecoded
# text instead: three two-word unions refused as inexpressible, and one alternation
# reported as having a glob that matches nothing -- the only false positive the regex
# work produced.
#
# `BS` rather than a raw string, because `r"A"` is six characters in a Python
# source file and one character after several of the tools that might rewrite this
# file, and a test whose meaning depends on that is not worth the brevity.
BS = "\\"


@pytest.mark.parametrize("body,denotes", [
    (BS + "x41", "A"),                       # two hex digits
    (BS + "u0041", "A"),                     # four
    (BS + "U00000041", "A"),                 # eight
    (BS + "x7e", "~"),
    ("a" + BS + "x41b", "aAb"),              # and the decoder resumes after it
    (BS + "n", "\n"),                        # the named escapes
    (BS + "t", "\t"),
    (BS + BS + ".", BS + "."),               # the docstring's own example
    # Not a valid escape: too few digits for the width, or none at all. Prometheus
    # rejects such a string, so the only safe reading is the literal one -- the
    # backslash stays and decoding resumes at the character after it.
    (BS + "x4", BS + "x4"),
    (BS + "xZZ", BS + "xZZ"),
    (BS + "u041", BS + "u041"),
    (BS + "d+", BS + "d+"),                  # an unknown escape, as a regex would have
    (BS, BS),                                # a trailing backslash, with nothing after
])
def test_a_quoted_matcher_denotes_the_escapes_it_carries(body, denotes):
    assert c.unquote_promql(body, raw=False) == denotes


def test_a_backtick_pattern_keeps_its_escapes_where_a_quoted_one_decodes_them():
    """The raw/cooked split, through the public reader rather than the helper.

    Both matchers carry the same six characters and they denote different regexes: the
    quoted one an `A`, the backtick one a hex escape PromQL never decodes. A reader
    that decoded both, or neither, would still pass every other test in this file.
    """
    q = 'up{a=~"' + BS + 'x41", b!~`' + BS + 'x41`}'
    assert c.promql_matcher_values(q) == ["A", BS + "x41"]


# --- the depth counter behind MULTI_QUERY --------------------------------------------

def test_a_comma_after_a_closed_call_still_separates_two_queries():
    """`MULTI_QUERY` where the first query carries a bracketed argument list.

    The increment and the clamp in `_depth0_commas` were both free: every multi-query
    form tested here put its comma before any bracket, and every bracketed form had no
    depth-0 comma at all, so a counter that never came back to zero passed. It would
    read the two queries below as one.
    """
    two = "top(avg:a{*} by {h}, 10, 'mean', 'desc'), avg:b{*}"
    assert "MULTI_QUERY" in c.features("datadog", two)
    # The control, which is the form the existing test already covers: the same call
    # on its own has three commas and is still one query.
    one = "top(avg:a{*} by {h}, 10, 'mean', 'desc')"
    assert "MULTI_QUERY" not in c.features("datadog", one)


def test_the_depth_counter_survives_an_unbalanced_closer():
    """The clamp, which only malformed input reaches.

    Tested on the helper rather than through `features`, because no well-formed query
    can express it -- `_mask` blanks scope bodies, so a stray `)` outside a scope is
    already a malformed query. Without the clamp the depth goes negative and every
    later comma is invisible, which drops `MULTI_QUERY` from the rest of the query
    rather than raising.
    """
    assert c._depth0_commas("a)b,c") == 1


# --- the rest of what mutation found unconstrained -----------------------------------
# One test per surviving mutant that was a real gap rather than an equivalent edit.
# Each names, in its docstring, what the mutant would have changed and why that is a
# claim the paper or the corpus rests on. The equivalent ones are recorded in
# `scripts/mutate.py`'s own notes rather than killed with a contrived assertion.

def test_a_longest_first_alternation_keeps_count_values_whole():
    """The aggregation-operator alternation is sorted long names first.

    `count` is a prefix of `count_values`, so a shortest-first alternation reads
    `count_values("v", x)` as the operator `count` -- which is a different operator,
    and §7's aggregation tallies are built from these names. The ordering carried a
    comment and nothing else.
    """
    assert c.aggregation_ops("promql", 'count_values("v", x)') == {"count_values": 1}
    assert c.aggregation_ops("promql", "count(x)") == {"count": 1}


def test_a_colon_inside_a_tag_value_does_not_hide_the_glob_after_it():
    """`SCOPE_GLOB` where the value itself carries a colon.

    A colon inside a tag value is a documented corpus gap -- it is what the TAG_COLON
    repair exists for -- and the split that reads the value takes only the first one.
    Splitting on every colon instead leaves `a` as the value of `k:a:*`, which reads
    a glob as an equality and moves an occurrence between two columns of §6's table.
    """
    assert "SCOPE_GLOB" in c.datadog_features("avg:m{k:a:*}")
    assert "SCOPE_EQ" not in c.datadog_features("avg:m{k:a:*}")
    # The plain forms either side of it, so the test is about the colon and not the glob.
    assert "SCOPE_EQ" in c.datadog_features("avg:m{k:a:b}")
    assert "SCOPE_GLOB" not in c.datadog_features("avg:m{k:a:b}")


def test_the_seasonal_forecast_is_the_one_with_no_promql_counterpart():
    """`forecast` is split by its second argument, and only by that.

    The comment above `_NO_PROMQL_COUNTERPART` says the `seasonal` model has no
    counterpart while the `linear` one is `predict_linear` up to the fitting window.
    Nothing read either half: the feature is what §8 counts when it says how much of
    the formula layer is out of reach, and a test that only used `anomalies` left the
    whole `forecast` clause free.
    """
    seasonal = c.datadog_formula_features("forecast(query1, 'seasonal', 1)")
    linear = c.datadog_formula_features("forecast(query1, 'linear', 1)")
    assert "FORMULA_NO_PROMQL" in seasonal
    assert "FORMULA_NO_PROMQL" not in linear
    # And the other half of the condition: a name in the list needs no argument.
    assert "FORMULA_NO_PROMQL" in c.datadog_formula_features("anomalies(query1, 'basic')")
    assert "FORMULA_NO_PROMQL" not in c.datadog_formula_features("abs(query1)")


def test_an_aggregation_operator_is_not_counted_among_the_functions():
    """Both exclusions in `function_names`, which the paper's function tallies rest on.

    `sum` is an operator and `by` is a keyword; neither is a function, and §7 lists the
    most frequent functions by name. Dropping either exclusion puts `sum` or `by` in
    that list, and the two were tested together only where neither appeared.
    """
    assert c.function_names("promql", "sum by (k) (rate(x[5m]))") == {"rate": 1}
    assert c.function_names("promql", "topk(3, irate(x[1m]))") == {"irate": 1}


def test_a_templated_subquery_step_is_a_templated_range():
    """`TMPL_RANGE` where the variable is in the step and not the range.

    The feature is read off the whole bracket, `[30m:$__interval]`, not off its first
    group -- which is the range alone and carries no variable here. This form is in the
    corpus: it is the one the normalisation test above uses.
    """
    assert "TMPL_RANGE" in c.promql_features("max_over_time(rate(x[1m])[30m:$__interval])")
    assert "TMPL_RANGE" in c.promql_features("rate(x[$__rate_interval])")
    # And a bracket with no variable at all is not a templated range.
    assert "TMPL_RANGE" not in c.promql_features("max_over_time(rate(x[1m])[30m:1m])")


def test_a_conditional_aggregator_only_excuses_its_own_occurrence():
    """One `count(` discounted, not every later one.

    `count(v: v>=0):m{*}` is an aggregator rather than a function, and the counter that
    records that is decremented once per occurrence. Decrementing it further leaves it
    truthy, so a genuine `count(` call in the same query is silently discounted too --
    which is the same defect, in the same place, that put `count` in §7's ten most
    frequent Datadog functions.
    """
    both = "count(v: v>0):m{*}, count(a{*})"
    assert c.function_names("datadog", both)["count"] == 1


def test_a_quoted_label_that_cannot_start_a_bare_name_gets_a_prefix():
    """The QUOTED_LABEL repair has to produce a name the grammar accepts.

    A quoted UTF-8 label name is quoted precisely because it uses characters a bare
    LABEL forbids, and a name starting with a digit is one of those -- substituting it
    unprefixed produces a repair that does not parse, which would report the gap as
    unexplainable rather than as this one.
    """
    assert c._prom_legal_label("123abc") == "l123abc"
    assert c._prom_legal_label("a-b") == "a_b"


def test_the_quoted_label_repair_rewrites_both_positions():
    """A quoted name in a matcher and in a label list, in one expression.

    The repair is two substitutions and the second reaches inside a `by (...)` list,
    stripping the quotes off each name it finds there. Both were free: the matcher half
    was exercised and the list half was not, and an off-by-one in either leaves a quote
    character in the name, which `_prom_legal_label` then turns into an underscore --
    a repair that parses and denotes something else.
    """
    got = c._prom_unquote_labels('sum by ("a-b") (x{"c-d"=\'v\'})')
    assert got == "sum by (a_b) (x{c_d='v'})"


def test_a_parenthesised_scope_is_dropped_like_a_boolean_one():
    """The BOOL_SCOPE repair drops a scope for three reasons, and parens is one.

    A scope the v1 grammar cannot parse is replaced by `*` so the rest of the query
    can be measured. Parentheses were the untested third of that condition, and a
    scope left in place there explains the failure as something else.
    """
    assert c._dd_drop_boolean_scope("(a)") == "*"
    assert c._dd_drop_boolean_scope("a OR b") == "*"
    assert c._dd_drop_boolean_scope("k:v") == "k:v"


def test_a_widget_with_no_title_is_cited_by_its_dashboard():
    """Provenance, which is the whole reason `Query` carries more than its text.

    A widget with no title of its own falls back to the dashboard's, in both
    extractors. The fallback is an `or` over two strings; an `and` there makes the unit
    empty for every untitled widget, and the paper cites queries by unit.
    """
    dd = {"title": "Board",
          "widgets": [{"definition": {"requests": [{"q": "avg:a{*}"}]}}]}
    assert [q.unit for q in c.datadog_dashboard_queries(dd, source="s", origin="o")] \
        == ["Board"]
    gf = {"title": "Board", "panels": [{"type": "timeseries",
                                        "targets": [{"expr": "up"}]}]}
    assert [q.unit for q in c.grafana_dashboard_queries(gf, source="s", origin="o")] \
        == ["Board"]


def test_a_blank_or_non_string_query_is_not_a_corpus_query():
    """Both halves of the guard on every text the extractors keep.

    An empty string and a number are both things a dashboard JSON can put where a
    query goes, and either entering the corpus would move every count in the paper by
    one. The guard is an `isinstance` and a `strip` joined by `and`; with `or` the
    number passes on the strip and the blank passes on the isinstance.
    """
    dd = {"widgets": [{"definition": {"requests": [
        {"q": "   "},
        {"q": 5},
        {"queries": [{"data_source": "metrics", "query": "  "}]},
        {"queries": [{"data_source": "metrics", "query": "avg:a{*}"}],
         "formulas": [{"formula": "  "}]},
    ]}}]}
    assert [q.text for q in c.datadog_dashboard_queries(dd, source="s", origin="o")] \
        == ["avg:a{*}"]


def test_a_rule_document_that_is_not_a_mapping_yields_nothing():
    """The rule extractor is handed whatever the YAML file held.

    A list, a string or `None` are all valid YAML documents, and the guard that rejects
    them is the left half of an `or` whose right half calls `.get`. Joined by `and`
    instead, a non-mapping document raises `AttributeError` mid-walk rather than being
    skipped -- during a corpus fetch, over someone else's repository.
    """
    for doc in (["a"], "a", None, 7):
        assert c.prometheus_rule_queries(doc, source="s", origin="o") == []
    assert c.prometheus_rule_queries({"kind": "ConfigMap"}, source="s", origin="o") == []


def test_the_corpus_file_is_written_in_a_stable_readable_form():
    """`queries.jsonl` is committed, so its layout is a property and not a detail.

    Sorted keys and unescaped UTF-8 are what make the file diffable and readable; with
    either reversed the corpus round-trips exactly as well and every later commit that
    rewrites it shows every line as changed. The parent directory is created too,
    which is what a fresh checkout needs.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        # Two levels deeper than anything that exists, which is what `parents=True` is
        # for and what a fresh checkout hands it.
        target = _Path(tmp) / "nested" / "deeper" / "queries.jsonl"
        q = c.Query("promql", 'up{k="café"}', "src", "origin", "unité", "panel")
        assert c.write_queries(target, [q]) == 1
        line = target.read_text(encoding="utf-8").strip()
        assert "café" in line and "unité" in line, "non-ASCII was escaped away"
        assert list(_json.loads(line)) == sorted(_json.loads(line)), "keys are not sorted"
        assert c.read_queries(target) == [q]


def test_a_formula_over_a_non_metric_request_is_counted_for_the_dilution_figure():
    """`non_metric_formulas`, which no test reached at all.

    §7 says what the formula-layer shares it used to quote were diluted with, and this
    walk is where that number comes from. Its three guards -- a metric sibling
    anywhere in the request, a dict formula, a non-blank text -- were all free, and
    each one of them changes the count in a different direction.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        tree = _Path(tmp)
        d = tree / "pkg" / "assets" / "dashboards"
        d.mkdir(parents=True)
        (d / "board.json").write_text(_json.dumps({"widgets": [{"definition": {
            "requests": [
                # a log request: its formula is not a metric expression
                {"queries": [{"data_source": "logs", "query": "source:x"}],
                 "formulas": [{"formula": "query1 * 2"}, {"formula": "  "}, "nope"]},
                # a metric request: its formula belongs to the corpus, not here
                {"queries": [{"data_source": "metrics", "query": "avg:a{*}"}],
                 "formulas": [{"formula": "query1"}]},
                # a singular query object of another language, which is still not metric
                {"query": {"data_source": "events", "query": "x"},
                 "formulas": [{"formula": "query1 / 3"}]},
            ]}}]}))
        assert sorted(c.non_metric_formulas(tree)) == ["query1 * 2", "query1 / 3"]


def test_an_unknown_language_is_reported_rather_than_raised():
    """`parse_outcome` is documented never to raise, and the branch that keeps that
    promise for an error which is not a parse error was unexecuted.

    The excerpt is bounded as well, because §4's gap table prints it.
    """
    got = c.parse_outcome("klingon", "x")
    assert got.ok is False
    assert got.blocker == "ValueError"
    assert "unknown language" in got.excerpt
    assert len(c.parse_outcome("y" * 60, "q").excerpt) == 40


def test_the_first_query_of_a_list_keeps_its_own_brackets():
    """The QUERY_LIST repair cuts at the first depth-0 comma, not the first comma.

    `_dd_first_query` carries its own copy of the depth counter and the same two
    mutants lived in it: an increment that overshoots or a clamp that never returns to
    zero cuts the query at the first comma inside its own argument list, which repairs
    `top(a, 10, 'mean', 'desc'), avg:b{*}` to `top(a` and reports the gap as
    unexplainable.
    """
    two = "top(avg:a{*}, 10, 'mean', 'desc'), avg:b{*}"
    assert c._dd_first_query(two) == "top(avg:a{*}, 10, 'mean', 'desc')"
    assert c._dd_first_query("avg:a{*}") == "avg:a{*}"


def test_a_grafana_dashboard_yields_each_panel_once():
    """`__elements` is walked at the top level only.

    The recursion passes `root=False` precisely so that a nested model does not send
    the walk back through the dashboard's `__elements` map; with `root=True` there, a
    dashboard carrying both a library panel and a nested row yields its library
    queries once per level of nesting. Duplicates in the corpus are not visible as
    failures anywhere -- they inflate every count the paper reports.
    """
    g = {"title": "G",
         "__elements": [{"model": {"type": "timeseries",
                                   "targets": [{"expr": "up"}],
                                   "panels": [{"type": "row",
                                               "targets": [{"expr": "down"}]}]}}],
         "panels": [{"type": "row",
                     "panels": [{"type": "timeseries",
                                 "targets": [{"expr": "inner"}]}]}]}
    texts = [q.text for q in c.grafana_dashboard_queries(g, source="s", origin="o")]
    assert sorted(texts) == ["down", "inner", "up"]


def test_a_masked_query_keeps_the_delimiters_it_blanks_between():
    """`_mask` is length-preserving and keeps the braces and quotes themselves.

    Operator searches run over the mask and offsets found in it index the original, so
    both properties are load-bearing; neither was asserted anywhere.
    """
    assert c._mask("avg:a{k:v}+1") == "avg:a{...}+1"
    assert c._mask("""x{a="b+c"}""") == "x{.......}"
    assert len(c._mask("""f('a,b') + g("c")""")) == len("""f('a,b') + g("c")""")


def test_a_blocked_parse_names_its_terminal_and_bounds_its_excerpt():
    """§4's gap table is one row per terminal the LALR table had no transition for, and
    it prints the offending text beside it.

    Both excerpts are bounded, by different numbers, and neither bound was exercised:
    every failing query in the suite is short enough that the truncation never fires.
    A window computed the other way round yields an empty excerpt, which would print
    the table's every row with a blocker and no text.
    """
    token = c.parse_outcome("promql", "up " + "q" * 60)
    assert token.blocker == "METRIC"
    assert len(token.excerpt) == 40

    chars = c.parse_outcome("promql", "up \xa7" + "x" * 40)
    assert chars.blocker == "<no-token>"
    assert len(chars.excerpt) == 20
    assert chars.excerpt.startswith("\xa7"), "the excerpt starts at the offending char"


def test_an_aggregation_operator_is_not_a_function_applied():
    """`FUNC_APPLIED` excludes operators and keywords, and both exclusions were free.

    This is the feature §7 counts when it says how many queries apply a function. `sum`
    is an operator and `by` is a keyword; with the two exclusions disjoined instead of
    conjoined, every aggregation in the corpus also counts as a function call, and with
    either sense inverted no real call does.
    """
    assert "FUNC_APPLIED" in c.promql_features("rate(x[5m])")
    assert "FUNC_APPLIED" not in c.promql_features("sum(x)")
    assert "FUNC_APPLIED" not in c.promql_features("sum by (k) (x)")


def test_a_nested_call_does_not_look_like_a_query_list():
    """The depth counter steps back by one bracket, not two.

    Both copies of it -- `_depth0_commas` and `_dd_first_query` -- clamp at zero, so a
    single `)` undershooting is invisible on a query that never nests. Two levels of
    brackets is where it shows: stepping back by two puts the counter at zero inside
    the outer call, and the comma separating that call's own arguments then reads as
    the separator between two queries.
    """
    nested = "top(abs(avg:a{*}), 10)"
    assert c._depth0_commas(c._mask(nested)) == 0
    assert "MULTI_QUERY" not in c.features("datadog", nested)
    assert c._dd_first_query(nested) == nested
    # And the same shape with a real second query after it, which must still split.
    assert c._dd_first_query(nested + ", avg:b{*}") == nested

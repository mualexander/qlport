"""RQ2: Datadog -> IR -> PromQL, and the Gap carried by every refusal.

Three kinds of check:
  * shape   -- the translation is the expression notebook 02 identified;
  * syntax  -- whatever we emit parses under the PromQL grammar of §4;
  * meaning -- the emitted form, read through the reference evaluators, agrees with
               Datadog's rollup on a real sample series.
"""
from __future__ import annotations

import numpy as np
import pytest

from qlport import ir
from qlport.diagnostics import Bottom, Gap
from qlport.emit.promql import emit, emit_detail
from qlport.harness.workload import scrape, steady
from qlport.lower.datadog import label_name, lower, lower_all, metric_name
from qlport.parse import get_parser, parse
from qlport.semantics import datadog_eval as dd
from qlport.semantics import promql_eval as pq
from qlport.semantics.ingest import phi_series

ROLLUP = 60.0


def translate(q: str, **kw):
    """Datadog text -> PromQL text, or the Bottom that stopped it."""
    e = lower(parse("datadog", q))
    if isinstance(e, Bottom):
        return e
    kw.setdefault("default_rollup", ROLLUP)
    return emit(e, **kw)


def caveats(q: str, **kw) -> set[Gap]:
    kw.setdefault("default_rollup", ROLLUP)
    got = emit_detail(lower(parse("datadog", q)), **kw)
    assert not isinstance(got, Bottom), got
    return set(got.caveats)


# --- the rules of §5 ------------------------------------------------------------------
@pytest.mark.parametrize("ddq,want", [
    # .as_count() is the undivided cumulative difference: the rule that matters most,
    # since the corpus uses it ~6.6x more often than .as_rate().
    ("sum:http.requests{*}.as_count().rollup(sum, 300)",
     "sum(http_requests - http_requests offset 300s)"),
    # .as_rate() is that over the interval (notebook 02, exact without a reset)
    ("sum:http.requests{*}.as_rate().rollup(sum, 60)",
     "sum((http_requests - http_requests offset 60s) / 60)"),
    # space aggregation keeps its grouping, with the rollup pushed inside it
    ("sum:http.requests{env:prod} by {service}.as_count().rollup(sum, 60)",
     'sum by (service) (http_requests{env="prod"} - http_requests{env="prod"} offset 60s)'),
    # a non-sum rollup is a range function
    ("avg:system.load.1{*}.rollup(avg, 300)", "avg(avg_over_time(system_load_1[300s]))"),
    ("max:system.load.1{*}.rollup(max, 60)", "max(max_over_time(system_load_1[60s]))"),
    # no space aggregator means avg: in the query editor
    ("system.load.1{*}.rollup(avg, 60)", "avg(avg_over_time(system_load_1[60s]))"),
    # arithmetic and constants
    ("100 * sum:a{*}.rollup(avg, 60)", "100 * sum(avg_over_time(a[60s]))"),
    ("sum:a{*}.rollup(avg, 60) / sum:b{*}.rollup(avg, 60)",
     "sum(avg_over_time(a[60s])) / sum(avg_over_time(b[60s]))"),
])
def test_translation_rules(ddq, want):
    assert translate(ddq) == want


@pytest.mark.parametrize("ddq,want", [
    # Datadog evaluates "first by time, then by space": the rollup reduces each series
    # alone and the space aggregator wraps it. These all used to be refused on the
    # mistaken grounds that the two layers had to commute.
    ("max:x{*}.rollup(avg, 60)", "max(avg_over_time(x[60s]))"),
    ("min:x{*}.rollup(avg, 60)", "min(avg_over_time(x[60s]))"),
    ("count:x{*}.rollup(avg, 60)", "count(avg_over_time(x[60s]))"),
    ("max:x{*}.as_count().rollup(sum, 60)", "max(x - x offset 60s)"),
    ("count:x{*}.as_count().rollup(sum, 60)", "count(x - x offset 60s)"),
    # the example worked through step by step in Datadog's own weighted() docs:
    # average each series over 10s first, then sum per kube_container_name
    ("sum:kubernetes.cpu.requests{*} by {kube_container_name}.rollup(avg, 10)",
     "sum by (kube_container_name) (avg_over_time(kubernetes_cpu_requests[10s]))"),
])
def test_time_aggregation_is_inside_space_aggregation(ddq, want):
    assert translate(ddq) == want


def test_as_rate_divides_per_series_not_after_space_aggregation():
    """The division is part of the time aggregation, so it sits inside the \
    aggregator. Only `count:` can tell the difference, and putting it outside would
    scale a count of series by the interval."""
    assert translate("count:x{*}.as_rate().rollup(sum, 60)") == \
        "count((x - x offset 60s) / 60)"


@pytest.mark.parametrize("term,want", [
    ("{*}", ""),
    ("{env:prod}", '{env="prod"}'),
    ("{!env:dev}", '{env!="dev"}'),
    ("{status:5*}", '{status=~"5.*"}'),
    ("{!status:5*}", '{status!~"5.*"}'),
    ("{host:web-?}", '{host=~"web\\\\-."}'),
    # a bare key, and key:*, both mean the tag is present. `.+` not `.*`: in PromQL a
    # `.*` matcher also matches series that carry no such label at all.
    ("{host}", '{host=~".+"}'),
    ("{host:*}", '{host=~".+"}'),
    ("{env:prod,!region:eu}", '{env="prod",region!="eu"}'),
])
def test_scope_terms_become_label_matchers(term, want):
    assert translate(f"avg:x{term}.rollup(avg, 60)") == f"avg(avg_over_time(x{want}[60s]))"


# --- every refusal names one gap ------------------------------------------------------
@pytest.mark.parametrize("ddq,gap", [
    ("sum:x{a:1 OR b:2}.as_count()", Gap.SCOPE_DISJUNCTION),
    ("sum:x{k IN (a,b)}.as_count()", Gap.SCOPE_DISJUNCTION),
    ("sum:x{*-zone}.as_count()", Gap.LABEL_NAME_GLOB),
    ("sum:iis.users.anon{*,!*}.as_count()", Gap.UNSUPPORTED_FUNCTION),
    ("sum:x{*}.fill(zero).as_count()", Gap.FILL_INTERPOLATION),
    ("sum:x{*}.weighted().as_count()", Gap.UNSUPPORTED_FUNCTION),
    ("p95:x{*}.as_count()", Gap.UNSUPPORTED_FUNCTION),
    ("top(avg:x{*}, 10, 'mean', 'desc')", Gap.UNSUPPORTED_FUNCTION),
    ("per_minute(sum:x{*})", Gap.UNSUPPORTED_FUNCTION),
])
def test_refusals_carry_a_gap(ddq, gap):
    got = translate(ddq)
    assert isinstance(got, Bottom) and got.gap is gap, got


def test_a_refused_operand_refuses_the_whole_expression():
    got = translate("sum:a{*}.rollup(avg, 60) + top(avg:b{*}, 5, 'mean', 'desc')")
    assert isinstance(got, Bottom) and got.gap is Gap.UNSUPPORTED_FUNCTION


def test_unknown_metric_type_refuses_a_sum_over_time():
    # Rollup(sum) built directly over a metric whose kind was never established. This is
    # not PHI_ABSOLUTE: §3 reserves that for reading absolute counter values, and here
    # the obstacle is that a dashboard never records whether a metric is a COUNT.
    e = ir.Agg("sum", (), "by", ir.Rollup("sum", 60.0, ir.Select("x")))
    got = emit_detail(e)
    assert isinstance(got, Bottom) and got.gap is Gap.METRIC_TYPE_UNKNOWN


def test_missing_rollup_interval_is_refused_not_guessed():
    e = lower(parse("datadog", "sum:x{*}.as_count()"))
    assert isinstance(emit_detail(e), Bottom)
    assert emit_detail(e).gap is Gap.ROLLUP_DEFAULT
    ok = emit_detail(e, default_rollup=60.0)
    assert not isinstance(ok, Bottom)
    assert Gap.ROLLUP_DEFAULT in ok.caveats


def test_every_rollup_reports_the_alignment_caveat():
    got = caveats("sum:x{*}.as_count().rollup(sum, 60)")
    assert Gap.ROLLUP_ALIGNMENT in got
    assert Gap.ROLLUP_DEFAULT not in got      # the query stated the interval


def test_the_exact_form_is_the_cumulative_difference_not_rate():
    out = translate("sum:x{*}.as_rate().rollup(sum, 60)")
    assert "offset" in out and "rate(" not in out


def test_the_cumulative_difference_declares_the_reset_precondition():
    assert Gap.COUNTER_RESET in caveats("sum:x{*}.as_count().rollup(sum, 60)")
    # a non-counter rollup is a range function, so no reset question arises
    assert Gap.COUNTER_RESET not in caveats("avg:x{*}.rollup(avg, 60)")


def test_anchored_form_is_opt_in_and_trades_reset_risk_for_a_version_floor():
    e = lower(parse("datadog", "sum:x{*}.as_count().rollup(sum, 60)"))
    assert "anchored" not in emit(e)
    anchored = emit_detail(e, anchored=True)
    assert "anchored" in anchored.promql
    # verified reset-exact against a real server, so the reset caveat is gone, but the
    # modifier only exists from Prometheus 3.7.0 on
    assert Gap.ANCHORED_VERSION in anchored.caveats
    assert Gap.COUNTER_RESET not in anchored.caveats


# --- names ----------------------------------------------------------------------------
@pytest.mark.parametrize("dd_name,want", [
    ("system.load.1", "system_load_1"),
    ("aws.elb.request_count", "aws_elb_request_count"),
    ("1xx.responses", "_1xx_responses"),
])
def test_metric_names_are_sanitised(dd_name, want):
    assert metric_name(dd_name) == want


def test_label_names_drop_the_colon_too():
    assert label_name("kube_app.name") == "kube_app_name"
    assert label_name("openai.request.model") == "openai_request_model"


def test_name_mapping_is_not_injective():
    # worth stating: `a.b` and `a_b` are different Datadog metrics and one PromQL name
    assert metric_name("a.b") == metric_name("a_b")


# --- query lists ----------------------------------------------------------------------
def test_lower_all_splits_a_request_into_its_queries():
    got = lower_all(parse("datadog", "sum:a{*}.as_count(),sum:b{*}.as_count()"))
    assert len(got) == 2 and all(not isinstance(g, Bottom) for g in got)
    assert [emit(g, default_rollup=60.0) for g in got] == [
        "sum(a - a offset 60s)", "sum(b - b offset 60s)"]


def test_a_query_list_is_refused_as_a_single_expression():
    got = lower(parse("datadog", "sum:a{*},sum:b{*}"))
    assert isinstance(got, Bottom) and got.gap is Gap.UNSUPPORTED_FUNCTION


# --- syntax: what we emit is PromQL ---------------------------------------------------
@pytest.mark.parametrize("ddq", [
    "sum:http.requests{env:prod,!status:5*} by {service,host}.as_count().rollup(sum, 60)",
    "sum:http.requests{*}.as_rate().rollup(sum, 300)",
    "avg:system.load.1{host:web-1} by {host}.rollup(avg, 60)",
    "100 * sum:a{*}.rollup(avg, 60) / sum:b{*}.rollup(avg, 60)",
    "-sum:a{*}.rollup(avg, 60)",
    "max:x{*}.rollup(max, 60)",
])
def test_emitted_promql_parses(ddq):
    out = translate(ddq)
    assert not isinstance(out, Bottom), out
    get_parser("promql").parse(out)          # raises if it is not PromQL


def test_emitted_promql_parses_across_the_corpus():
    from qlport import corpus as c
    qs = {c.normalise("datadog", q.text)
          for q in c.read_queries("data/corpus/queries.jsonl") if q.lang == "datadog"}
    p, emitted = get_parser("promql"), 0
    for text in sorted(qs):
        if not c.parse_outcome("datadog", text).ok:
            continue
        out = translate(text)
        if isinstance(out, Bottom):
            continue
        p.parse(out)                          # raises if any translation is not PromQL
        emitted += 1
    assert emitted > 3000, emitted            # the kernel really does carry the corpus


# --- meaning: the emitted form agrees with Datadog's rollup ---------------------------
def test_as_count_matches_datadog_on_a_real_series():
    """`x - x offset Δ` reproduces `.as_count()` bucket by bucket.

    The translation is checked through the reference evaluators rather than by eye:
    Prometheus holds the cumulative counter, Datadog holds phi of it, and the two
    readings must agree on every epoch-aligned bucket end.
    """
    s = scrape(start_value=5000, interval=15, duration=2 * 3600, seed=681,
               rate_fn=steady(10), jitter=0)
    p = phi_series(s)
    assert translate("sum:x{*}.as_count().rollup(sum, 60)") == "sum(x - x offset 60s)"
    diffs = []
    for t in dd.bucket_ends(600, s.ts[-1], 60.0):
        promql = pq.instant(s.ts, s.vs, t) - pq.instant(s.ts, s.vs, t - 60.0)
        datadog = dd.as_count(p.ts, p.vs, t, 60.0)
        diffs.append(abs(promql - datadog))
    assert diffs and np.nanmax(diffs) < 1e-9


def test_as_rate_is_as_count_over_the_interval():
    s = scrape(start_value=5000, interval=15, duration=2 * 3600, seed=681,
               rate_fn=steady(10), jitter=0)
    p = phi_series(s)
    out = translate("sum:x{*}.as_rate().rollup(sum, 60)")
    assert "/ 60)" in out      # per series, inside the space aggregation
    errs = [abs(pq.offset_diff_rate(s.ts, s.vs, t, 60.0) - dd.as_rate(p.ts, p.vs, t, 60.0))
            for t in dd.bucket_ends(600, s.ts[-1], 60.0)]
    assert errs and np.nanmax(errs) < 1e-9

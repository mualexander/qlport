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
    # `?` is not a Datadog wildcard (paper §6), so this is a literal tag value.
    ("{host:web-?}", '{host="web-?"}'),
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
    # a single-label IN now translates; a branch constraining two keys does not
    ("sum:x{(a:1 AND b:1) OR (a:2 AND b:2)}.as_count()", Gap.SCOPE_DISJUNCTION),
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
    for b in dd.bucket_starts(600, s.ts[-1] - 60.0, 60.0):
        u = dd.eval_instant(b, 60.0)
        promql = pq.instant(s.ts, s.vs, u) - pq.instant(s.ts, s.vs, u - 60.0)
        datadog = dd.as_count(p.ts, p.vs, b, 60.0)
        diffs.append(abs(promql - datadog))
    assert diffs and np.nanmax(diffs) < 1e-9


def test_as_rate_is_as_count_over_the_interval():
    s = scrape(start_value=5000, interval=15, duration=2 * 3600, seed=681,
               rate_fn=steady(10), jitter=0)
    p = phi_series(s)
    out = translate("sum:x{*}.as_rate().rollup(sum, 60)")
    assert "/ 60)" in out      # per series, inside the space aggregation
    errs = [abs(pq.offset_diff_rate(s.ts, s.vs, dd.eval_instant(b, 60.0), 60.0)
                - dd.as_rate(p.ts, p.vs, b, 60.0))
            for b in dd.bucket_starts(600, s.ts[-1] - 60.0, 60.0)]
    assert errs and np.nanmax(errs) < 1e-9


def test_lowering_never_exposes_a_bare_counter_selector():
    """§3.1's "raw selectors are inadmissible" holds by construction, so construction is
    what gets checked.

    A bare counter selector is not phi-invariant, so it can translate no Datadog query.
    The translator does not need to refuse one: lowering puts every `Select` inside a
    `Rollup`, so the emitter's `Select` branch is unreachable from a real query. That is
    a property of `lower/datadog.py`, which means it can break silently -- hence this
    test rather than a sentence in the paper. `Gap.PHI_ABSOLUTE` stays declared for the
    reverse direction and is deliberately never raised here.
    """
    from qlport import corpus as c

    def selects_are_wrapped(e, inside_rollup=False):
        if isinstance(e, ir.Select):
            return inside_rollup
        if isinstance(e, ir.Rollup):
            return selects_are_wrapped(e.expr, inside_rollup=True)
        kids = [v for v in vars(e).values() if isinstance(v, ir.Expr)]
        return all(selects_are_wrapped(k, inside_rollup) for k in kids)

    qs = {c.normalise("datadog", q.text)
          for q in c.read_queries("data/corpus/queries.jsonl") if q.lang == "datadog"}
    checked = 0
    for text in sorted(qs):
        if not c.parse_outcome("datadog", text).ok:
            continue
        for e in lower_all(parse("datadog", text)):
            if isinstance(e, Bottom):
                continue
            assert selects_are_wrapped(e), text
            checked += 1
    assert checked > 7_000, checked


# --- single-label disjunction ---------------------------------------------------------
# A PromQL selector is a conjunction of matchers, so a disjunction of scope terms has no
# selector-level form in general -- but when every branch constrains one tag key it is a
# single regex matcher. tests/test_harness.py checks the Prometheus side of that (the
# whole regex is anchored, so the alternation cannot leak into a neighbouring value).
@pytest.mark.parametrize("ddq,want", [
    # OR over one key, in both spellings Datadog allows
    ("avg:x{a:1 OR a:2}", 'a=~"1|2"'),
    ("avg:x{a:1 or a:2 or a:3}", 'a=~"1|2|3"'),
    ("avg:x{k IN (aws,azure,gcp)}", 'k=~"aws|azure|gcp"'),
    # negation distributes over the union: NOT (k:a OR k:b) is k!~"a|b"
    ("avg:x{NOT (a:1 OR a:2)}", 'a!~"1|2"'),
    ("avg:x{k NOT IN (aws,gcp)}", 'k!~"aws|gcp"'),
    ("avg:x{NOT (k NOT IN (aws,gcp))}", 'k=~"aws|gcp"'),
    # globs convert branch by branch
    ("avg:x{f:*ptl* OR f:*con*}", 'f=~".*ptl.*|.*con.*"'),
    ("avg:x{code:4* OR code:5*}", 'code=~"4.*|5.*"'),
    # a branch asking only that the tag be present subsumes every value
    ("avg:x{a:1 OR a}", 'a=~".+"'),
    ("avg:x{a:1 OR a:*}", 'a=~".+"'),
    # repeated values collapse rather than repeating in the alternation
    ("avg:x{a:1 OR a:1}", 'a=~"1"'),
    # Datadog's `or` binds looser than `,`, so the right branch here is a conjunction
    # whose match-all terms contribute nothing -- one constraint, so it still joins.
    ("avg:x{a:x or a:y,*,*}", 'a=~"x|y"'),
    ("avg:x{* AND * AND code:4* OR code:5*}", 'code=~"4.*|5.*"'),
    # a literal value is escaped before it goes into an alternation
    ("avg:x{k IN (a-1,b.2)}", r'k=~"a\\-1|b\\.2"'),
])
def test_single_label_disjunction_is_one_matcher(ddq, want):
    out = translate(f"{ddq}.rollup(avg, 60)")
    assert not isinstance(out, Bottom), out
    assert want in out, out


def test_a_match_all_branch_makes_the_disjunction_vacuous():
    # {a:1 OR *} selects everything, so the scope contributes no matcher at all.
    out = translate("avg:x{a:1 OR *}.rollup(avg, 60)")
    assert out == "avg(avg_over_time(x[60s]))", out


@pytest.mark.parametrize("ddq,detail", [
    # different keys: no selector says this, and lifting it to a union of selectors is
    # not the syntax-directed translation of a scope
    ("avg:x{a:1 OR b:2}", "disjunction across tag keys"),
    ("avg:x{(status:400 OR s) AND *}", "disjunction across tag keys"),
    # a branch that constrains two things is not one value in a union
    ("avg:x{(a:1 AND b:1) OR (a:2 AND b:2)}", "disjunction of compound terms"),
    # a negated branch cannot join a positive alternation
    ("avg:x{(!a:1) OR a:2}", "negated branch in a disjunction"),
    # De Morgan turns this into a disjunction before we see one
    ("avg:x{NOT (a:1 AND b:2)}", "negated conjunction is a disjunction"),
    # IN over a compound term would be a tag inside a tag
    ("avg:x{k IN (a:1)}", "IN (...) over a compound term"),
])
def test_disjunctions_that_are_still_refused(ddq, detail):
    out = translate(f"{ddq}.rollup(avg, 60)")
    assert isinstance(out, Bottom) and out.gap is Gap.SCOPE_DISJUNCTION, out
    assert out.detail == detail, out


def test_excluding_everything_is_still_refused():
    # NOT (a:1 OR *) is the empty set, and PromQL has no "match nothing".
    out = translate("avg:x{NOT (a:1 OR *)}.rollup(avg, 60)")
    assert isinstance(out, Bottom) and out.gap is Gap.UNSUPPORTED_FUNCTION, out


def test_a_key_glob_inside_a_disjunction_keeps_its_own_gap():
    # Label *names* are not queryable in PromQL at all, which is a different gap from
    # disjunction and must not be absorbed into it.
    out = translate("avg:x{*-zone OR *-loc}.rollup(avg, 60)")
    assert isinstance(out, Bottom) and out.gap is Gap.LABEL_NAME_GLOB, out


# --- the metric type a dashboard never states ----------------------------------------
def test_a_non_sum_rollup_reports_the_gauge_assumption_it_rests_on():
    """`avg_over_time` is exact for a gauge and wrong for a COUNT.

    Datadog's avg/min/max/count rollups reduce the points it stores: samples for a
    GAUGE, per-interval deltas for a COUNT. Averaging deltas is not averaging running
    totals, so the emission is conditional, and the condition is a metric type the
    dashboard JSON never carries.
    """
    got = emit_detail(lower(parse("datadog", "avg:x{*}.rollup(avg, 60)", "v2")))
    assert got.promql == "avg(avg_over_time(x[60s]))"
    assert Gap.METRIC_TYPE_UNKNOWN in got.caveats


@pytest.mark.parametrize("ddq", [
    "avg:x{*}.rollup(avg, 60)",
    "max:x{*}.rollup(max, 60)",
    "min:x{*}.rollup(min, 60)",
    "sum:x{*}.rollup(count, 60)",
])
def test_every_non_sum_rollup_carries_the_caveat(ddq):
    got = emit_detail(lower(parse("datadog", ddq, "v2")))
    assert Gap.METRIC_TYPE_UNKNOWN in got.caveats


def test_a_sum_rollup_does_not_carry_it_because_the_type_is_established():
    # `.as_count()` is itself the evidence that the metric is a COUNT, so for this
    # branch the type is known and the caveat would be noise.
    got = emit_detail(lower(parse("datadog", "sum:x{*}.as_count()", "v2")),
                      default_rollup=60.0)
    assert Gap.METRIC_TYPE_UNKNOWN not in got.caveats
    assert Gap.COUNTER_RESET in got.caveats


def test_the_same_code_refuses_when_no_rule_can_be_chosen_at_all():
    # A `sum` rollup over a metric of unknown type cannot be emitted either way, so
    # there the same fact is a refusal rather than a caveat. Unreachable from the
    # Datadog front end -- the only way to ask for a sum rollup is a method that also
    # establishes the type -- so it is built directly.
    from qlport import ir
    e = ir.Agg("sum", (), "by",
               ir.Rollup("sum", 60.0, ir.Select("x", (), kind="unknown")))
    got = emit_detail(e)
    assert isinstance(got, Bottom) and got.gap is Gap.METRIC_TYPE_UNKNOWN


# --- parenthesisation looks at structure, not at label values -------------------
def test_a_hyphen_in_a_label_value_is_not_an_operator():
    """`result:hit-wait` must not make the emitter parenthesise.

    The parenthesiser reads the rendered text, so anything inside a quoted literal can
    be mistaken for an operator. Two queries of identical structure then emit
    differently, which is how this was found: a corpus shape class whose members
    disagreed only in the punctuation of one tag value.
    """
    plain = translate("sum:x{result:miss}.as_rate().rollup(avg, 15)")
    hyphen = translate("sum:x{result:hit-wait}.as_rate().rollup(avg, 15)")
    assert plain == 'sum(avg_over_time(x{result="miss"}[15s]) / 15)'
    assert hyphen == 'sum(avg_over_time(x{result="hit-wait"}[15s]) / 15)'


def test_a_real_operator_still_parenthesises():
    got = translate("sum:x{*} / sum:y{*}")
    assert "(" in got and ")" in got

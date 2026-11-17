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


def test_as_rate_divides_after_space_aggregation():
    """`.as_rate()` divides the query's result, not each series.

    This test asserted the opposite until E6 measured it. Submitting three COUNT series
    of 10 every 10 s and querying `count:m{*} by {g}.rollup(avg, 30)` returns 1 per
    group; adding `.as_rate()` returns 1/30. A count of series does not depend on their
    values, so the only way the modifier can change it is from outside --- and the
    corpus-scale differential test of §7 reported this shape disagreeing by exactly
    1 - 1/60 before the measurement explained why.
    """
    assert translate("count:x{*}.as_rate().rollup(sum, 60)") == \
        "(count(x - x offset 60s)) / 60"


@pytest.mark.parametrize("ddq,want", [
    # Dividing by a positive constant commutes with these, so the emitted form keeps
    # the division inside, where it reads better. The value is the same either way.
    ("sum:x{*}.as_rate().rollup(sum, 60)", "sum((x - x offset 60s) / 60)"),
    ("avg:x{*}.as_rate().rollup(sum, 60)", "avg((x - x offset 60s) / 60)"),
    ("min:x{*}.as_rate().rollup(sum, 60)", "min((x - x offset 60s) / 60)"),
    ("max:x{*}.as_rate().rollup(sum, 60)", "max((x - x offset 60s) / 60)"),
])
def test_as_rate_stays_inside_the_aggregators_that_commute_with_it(ddq, want):
    assert translate(ddq) == want


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
    ("p95:x{*}.as_count()", Gap.PHI_QUANTILE),
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
    # Pinned, not floored. `> 3000` would have passed after losing well over half the
    # translated corpus, which is not what "the kernel carries the corpus" means.
    assert emitted == 7307, emitted


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
    assert checked == 7551, checked           # pinned for the same reason as above


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


def test_a_sum_rollup_carries_it_too_because_the_type_is_only_assumed():
    # This asserted the caveat's *absence* until E6 measured the premise it rested on.
    # The reasoning was that `.as_count()` only applies to COUNT metrics, so a query
    # using it establishes the type. Datadog does not enforce that: submitting a GAUGE
    # and querying `sum:g{*}.as_count()` returns the plain rollup, the modifier
    # silently ignored, exactly as `sum:g{*}` does. So the modifier records what the
    # dashboard's author took the metric to be, and the translation rests on an
    # assumption the query cannot settle -- which is what this caveat says.
    got = emit_detail(lower(parse("datadog", "sum:x{*}.as_count()", "v2")),
                      default_rollup=60.0)
    assert Gap.METRIC_TYPE_UNKNOWN in got.caveats
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
    # The property, not a literal: two queries of identical structure must emit
    # identical structure, so the only difference is the tag value itself.
    assert plain.replace('"miss"', "@") == hyphen.replace('"hit-wait"', "@")

    # The form above is one the fix does not change, so on its own it pins nothing:
    # reverting `_blank_strings` leaves it byte-identical. This one does change --
    # without the fix the hyphen is read as an operator and the operand is wrapped.
    scaled = translate("100 * sum:x{result:hit-wait}.rollup(avg, 60)")
    assert scaled == "100 * sum(avg_over_time(x{result=\"hit-wait\"}[60s]))", scaled
    assert "((" not in scaled


def test_a_real_operator_still_parenthesises():
    """An operand that really does contain an operator must be wrapped.

    The old form here was `sum:x{*} / sum:y{*}`, whose parentheses come from the
    aggregations either way, so it held whatever `_atom` did.
    """
    got = translate("(sum:a{*} / sum:b{*}) * 2")
    assert got == "(sum(avg_over_time(a[60s])) / sum(avg_over_time(b[60s]))) * 2", got


# --- a known counter reduced by something other than sum -------------------------
# Found by the corpus-scale differential test (paper §7): the emitter was sending
# `avg_over_time` of the cumulative series where Datadog averages the stored deltas.
# Those are not the same function, and the metric's type was known, so this was wrong
# output rather than output with an unmet precondition.

def test_the_average_of_a_counters_deltas_is_their_total_over_their_number():
    got = translate("sum:x{*}.as_count().rollup(avg, 60)")
    assert got == "sum((x - x offset 60s) / count_over_time(x[60s]))"


def test_as_rate_keeps_the_division_on_top_of_that():
    got = translate("sum:x{*}.as_rate().rollup(avg, 15)")
    assert got == "sum(((x - x offset 15s) / count_over_time(x[15s])) / 15)"


@pytest.mark.parametrize("op", ["min", "max"])
def test_min_and_max_over_a_counters_deltas_are_refused(op):
    """They need the individual deltas, which PromQL reaches only via a subquery over
    the sample grid -- and the grid spacing is not in the query."""
    got = emit(lower(parse("datadog", f"sum:x{{*}}.as_count().rollup({op}, 60)", "v2")))
    assert isinstance(got, Bottom)
    assert "individual deltas" in got.detail


def test_a_later_rollup_does_not_erase_the_type_evidence():
    """`.as_count()` applies only to COUNT metrics, so it establishes the type whatever
    aggregator a later `.rollup()` names. The kind was previously inferred from the
    final aggregator, so `.as_count().rollup(avg, 60)` came out as type-unknown."""
    e = lower(parse("datadog", "sum:x{*}.as_count().rollup(avg, 60)", "v2"))
    inner = e.expr.expr          # Agg -> Rollup -> Select
    assert inner.kind == "counter"


def test_a_gauge_rollup_is_untouched_by_the_counter_rule():
    assert translate("avg:x{*}.rollup(avg, 60)") == "avg(avg_over_time(x[60s]))"
    assert translate("max:x{*}.rollup(max, 60)") == "max(max_over_time(x[60s]))"


# --------------------------------------------------- vector matching across a binop


def test_operands_with_the_same_grouping_need_no_matching_clause():
    # PromQL's default matching is already right when both sides carry the same labels,
    # which is every binary operation in the corpus but two.
    assert translate("sum:a.b{*} by {h} / sum:c.d{*} by {h}") == \
        "sum by (h) (avg_over_time(a_b[60s])) / sum by (h) (avg_over_time(c_d[60s]))"


def test_an_ungrouped_divisor_is_broadcast_across_the_groups():
    """Datadog broadcasts; PromQL matches on labels and would return nothing.

    `sum by (k) (x) / sum(y)` pairs each group against a series with no labels. PromQL
    finds no matching pair and returns the empty vector --- no error, no result. The
    corpus-scale differential test of §7 found this as two shapes where Datadog
    returned data and Prometheus returned none.
    """
    got = translate("sum:a.b{*} by {k}.as_count() / sum:c.d{*}.as_count()")
    assert "on() group_left()" in got
    assert got == ("(sum by (k) (a_b - a_b offset 60s)) / on() group_left() "
                   "(sum(c_d - c_d offset 60s))")


def test_an_ungrouped_dividend_broadcasts_the_other_way():
    # The modifier names the side with many series, so this is group_right.
    got = translate("a.b{*} / c.d{*} by {host}")
    assert "on() group_right()" in got
    assert got == ("avg(avg_over_time(a_b[60s])) / on() group_right() "
                   "avg by (host) (avg_over_time(c_d[60s]))")


def test_two_different_groupings_are_refused_rather_than_guessed():
    # Matching would have to pick a label set, and the query does not say which. No
    # corpus query does this, so the refusal never fires on the measured corpus.
    got = emit_detail(lower(parse("datadog", "a.b{*} by {h} / c.d{*} by {k}", "v2")),
                      default_rollup=60.0)
    assert isinstance(got, Bottom)
    assert got.gap is Gap.UNSUPPORTED_FUNCTION


# ------------------------------------------- the series-start precondition (§3.1)


def test_a_counter_rule_records_that_it_needs_a_sample_before_the_bucket():
    """`x - x offset D` has nothing to subtract for a series younger than the window.

    Prometheus then returns no result at all, where Datadog returns the sum of the
    deltas it does hold -- a disagreement `d` scores as infinite rather than large,
    since the domains differ. The code naming this condition existed from the start,
    written for the reverse direction, while the counter rules emitted without it.
    """
    got = emit_detail(lower(parse("datadog", "sum:x{*}.as_count()", "v2")),
                      default_rollup=60.0)
    assert Gap.PHI_SERIES_START in got.caveats
    assert Gap.COUNTER_RESET in got.caveats


def test_the_mean_of_deltas_rule_records_it_too_and_for_its_own_reason():
    # Its denominator counts samples where Datadog counts deltas, and a series
    # beginning inside the bucket has one fewer delta than it has samples.
    got = emit_detail(lower(parse("datadog", "avg:x{*}.as_count().rollup(avg, 60)", "v2")),
                      default_rollup=60.0)
    assert Gap.PHI_SERIES_START in got.caveats


def test_a_gauge_rollup_needs_no_such_precondition():
    # `avg_over_time` reduces the samples in the window and needs no predecessor.
    got = emit_detail(lower(parse("datadog", "avg:x{*}.rollup(avg, 60)", "v2")),
                      default_rollup=60.0)
    assert Gap.PHI_SERIES_START not in got.caveats
    assert Gap.COUNTER_RESET not in got.caveats


def test_the_anchored_form_discharges_both_counter_preconditions():
    """Which is the bargain Remark rem:series-start adds to the flag's cost.

    Measured against a real server: on a bucket holding one series that began before it
    and one that began inside it, `increase(x[60s] anchored)` returns both of Datadog's
    figures while the cumulative difference returns only the first.
    """
    got = emit_detail(lower(parse("datadog", "sum:x{*}.as_count()", "v2")),
                      anchored=True, default_rollup=60.0)
    assert Gap.ANCHORED_VERSION in got.caveats
    assert Gap.PHI_SERIES_START not in got.caveats
    assert Gap.COUNTER_RESET not in got.caveats


def test_a_sum_rollup_is_a_third_piece_of_type_evidence():
    # §5 said the only ways were `.as_count()` and `.as_rate()`. `.rollup(sum, D)` is a
    # third, and the weakest: summing a gauge's samples over a window is an ordinary
    # thing to want, where `.as_count()` on a gauge is a no-op nobody benefits from.
    got = translate("sum:x{*}.rollup(sum, 60)")
    assert got == "sum(x - x offset 60s)"


# ------------------------------- vector matching: the subset rule (E8, §5)


@pytest.mark.parametrize("ddq,want", [
    # equal key sets: PromQL's default matching is already a join on those keys,
    # because aggregation drops `__name__` and leaves exactly the grouping labels
    ("sum:a.b{*} by {g} / sum:c.d{*} by {g}",
     "sum by (g) (avg_over_time(a_b[60s])) / sum by (g) (avg_over_time(c_d[60s]))"),
    # the empty set is a subset of everything: broadcast, matching on nothing
    ("sum:a.b{*} by {g} / sum:c.d{*}",
     "sum by (g) (avg_over_time(a_b[60s])) / on() group_left() "
     "sum(avg_over_time(c_d[60s]))"),
    ("sum:a.b{*} / sum:c.d{*} by {g}",
     "sum(avg_over_time(a_b[60s])) / on() group_right() "
     "sum by (g) (avg_over_time(c_d[60s]))"),
    # the general subset: match on the smaller set, key the result by the larger
    ("sum:a.b{*} by {g} / sum:c.d{*} by {g,h}",
     "sum by (g) (avg_over_time(a_b[60s])) / on(g) group_right() "
     "sum by (g, h) (avg_over_time(c_d[60s]))"),
    ("sum:a.b{*} by {g,h} / sum:c.d{*} by {g}",
     "sum by (g, h) (avg_over_time(a_b[60s])) / on(g) group_left() "
     "sum by (g) (avg_over_time(c_d[60s]))"),
])
def test_operands_pair_when_one_key_set_contains_the_other(ddq, want):
    assert translate(ddq) == want


def test_incomparable_key_sets_are_refused_because_datadog_refuses_them():
    """A third kind of refusal: neither a language gap nor unfinished work.

    Datadog rejects the query itself -- "Series to be aligned must have group keys that
    match, or where one set of group keys is a subset of the other" -- so there is no
    source denotation for a translation to preserve. Measured in E8 rather than assumed,
    which is what distinguishes this from the refusals §7 counts.
    """
    got = emit_detail(lower(parse("datadog", "a.b{*} by {g} / c.d{*} by {h}", "v2")),
                      default_rollup=60.0)
    assert isinstance(got, Bottom)
    assert got.gap is Gap.UNSUPPORTED_FUNCTION
    assert "Datadog rejects" in got.detail


def test_a_scalar_operand_matches_anything_and_keeps_the_vectors_keys():
    # `100 * q` is pointwise in both languages and needs no matching clause.
    assert translate("100 * sum:a.b{*} by {g}") == \
        "100 * sum by (g) (avg_over_time(a_b[60s]))"


# ------------------------------------- .weighted(), measured as the identity (E9)


@pytest.mark.parametrize("ddq,want", [
    # 54 of the 56 distinct corpus forms attach it to `sum:`, not to `avg:`, which is
    # the first sign it is not the "time-weighted space average" §7 called it.
    ("sum:k.x{*} by {env}.weighted()",
     "sum by (env) (avg_over_time(k_x[60s]))"),
    ("sum:k.x{*}.weighted().rollup(avg, 60)", "sum(avg_over_time(k_x[60s]))"),
    ("avg:k.x{*} by {h}.weighted()", "avg by (h) (avg_over_time(k_x[60s]))"),
    # order in the chain must not matter, since it contributes nothing
    ("sum:k.x{*}.rollup(avg, 60).weighted()", "sum(avg_over_time(k_x[60s]))"),
])
def test_weighted_translates_as_the_query_without_it(ddq, want):
    """E9 could not make the modifier change any reading, so it is dropped.

    Submitting a series whose points fall 0 s, 50 s and 55 s into each bucket -- where
    a mean weighted by the interval each point covers reads 16.7 against the plain
    66.7 -- Datadog returns the same figure with the modifier and without it, on
    GAUGE, COUNT and RATE alike and under avg, sum and count rollups.
    """
    assert translate(ddq) == want


def test_weighted_no_longer_refuses():
    got = emit_detail(lower(parse("datadog", "sum:x{*}.weighted().as_count()", "v2")),
                      default_rollup=60.0)
    assert not isinstance(got, Bottom), got


# ------------------------- percentiles: an information barrier, not a missing operator


@pytest.mark.parametrize("ddq", [
    "p50:x{*}", "p75:x{*} by {h}", "p95:trace.openai.request{*}", "p99:x{*}.as_count()",
])
def test_a_percentile_is_refused_as_an_information_barrier(ddq):
    """Not UNSUPPORTED_FUNCTION, which named the wrong obstruction.

    PromQL has `histogram_quantile`, so the operator is not what is missing. What is
    missing is the data: Datadog answers pNN from a sketch of the raw observations and
    Prometheus keeps them bucketed, and Proposition prop:quantile shows no function of
    a bucketing is the quantile. Filing it under a missing function put it in the same
    row as work that had simply not been done.
    """
    got = translate(ddq)
    assert isinstance(got, Bottom)
    assert got.gap is Gap.PHI_QUANTILE


def test_the_unsupported_function_code_is_now_only_unfinished_work():
    """The catalogue's largest row should mean one thing.

    It carried the percentiles (a real barrier) and `.weighted()` (a supposed one)
    alongside genuinely unimplemented functions, so the language-gap split had to be
    computed by matching detail strings instead of reading codes.
    """
    for ddq in ("top(avg:x{*}, 10, 'mean', 'desc')", "per_minute(sum:x{*})"):
        got = translate(ddq)
        assert isinstance(got, Bottom) and got.gap is Gap.UNSUPPORTED_FUNCTION, ddq


# --- behaviours the suite did not constrain ------------------------------------------
# Each of these was found by deliberately breaking the source and watching all 547
# tests pass. They are grouped here rather than scattered because what they have in
# common is the reason they were missed: no corpus query reaches any of them, so only
# a unit test can hold them.


def test_fill_null_stays_inside_kappa():
    """`.fill(null)` is the default and changes nothing, so it must not be refused.

    Refusing every `.fill` -- `null` included -- left the whole suite green, while
    moving the 36-occurrence FILL_INTERPOLATION row and everything it is compared
    against in the gap catalogue.
    """
    assert translate("avg:x{*}.fill(null)") == "avg(avg_over_time(x[60s]))"
    for mode in ("zero", "last", "linear", "0"):
        out = translate(f"avg:x{{*}}.fill({mode})")
        assert isinstance(out, Bottom) and out.gap is Gap.FILL_INTERPOLATION, mode


def test_a_colon_in_a_grouping_key_is_mangled_away():
    """PromQL label names forbid `:` where Datadog tag keys allow it (lem:scope).

    The test that claimed this used `kube_app.name` and `openai.request.model`,
    neither of which contains a colon.
    """
    assert label_name("a:b") == "a_b"
    assert label_name("http:status:code") == "http_status_code"


def test_an_unknown_rollup_aggregator_refuses_rather_than_raising():
    """rem:one-gap says every failure is a Bottom carrying one code, never an exception."""
    out = translate("avg:x{*}.rollup(median, 60)")
    assert isinstance(out, Bottom), out
    assert out.gap is Gap.UNSUPPORTED_FUNCTION


def test_a_star_inside_a_membership_list_is_tag_presence():
    """`k IN (v, *)` means "k is v or k is anything", i.e. presence -- not match-all.

    Mapping the `*` member to the scope-level match-all dropped the key constraint
    altogether, so the emitted selector also matched series carrying no such tag. The
    `or` spelling, which prop:disjunction calls equivalent, was already correct, so the
    two spellings disagreed.
    """
    presence = 'avg(avg_over_time(x{a=~".+"}[60s]))'
    assert translate("avg:x{a:p or a:*}") == presence
    assert translate("avg:x{a IN (p,*)}") == presence
    assert translate("avg:x{a IN (*)}") == presence
    assert translate("avg:x{a NOT IN (p,*)}") == 'avg(avg_over_time(x{a!~".+"}[60s]))'


def test_an_all_star_glob_is_presence_not_dot_star():
    """lem:scope's proof turns on presence being `.+`: `.*` also matches an absent label."""
    presence = 'avg(avg_over_time(x{a=~".+"}[60s]))'
    for q in ("avg:x{a:*}", "avg:x{a:**}", "avg:x{a IN (p,**)}"):
        assert translate(q) == presence, q


def test_a_nested_binary_operation_carries_the_finer_operands_keys():
    """`chi` of a BinOp feeds the matching clause of any operation enclosing it.

    Swapping which side's key set a BinOp reports left the suite green, because every
    binary-operator test used a single top-level operation whose own key set is never
    read by anything.
    """
    got = translate("(avg:a{*} by {g,h} / avg:b{*} by {g}) / avg:c{*} by {g}")
    assert got.count("group_left()") == 2, got
    assert "on(g)" in got


def test_the_reachable_gap_codes_are_the_ones_the_catalogue_claims():
    """The gap catalogue is organised around which codes an emitter can raise.

    `diagnostics.py` states the partition in prose and names `grep -rn 'Gap[.]'` as the
    check, which nothing ran: the docstring said ten reachable while eleven were, and
    UNTESTED_LOOKBACK appeared in neither list. A code that silently becomes reachable
    adds a row to §7's catalogue; one that stops being reachable removes a claim.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "qlport"
    referenced = {m.group(1)
                  for f in src.rglob("*.py")
                  for m in re.finditer(r"Gap\.([A-Z_]+)", f.read_text())}

    # Raised by the Datadog -> PromQL path, as caveats or as refusals.
    assert referenced == {
        "UNSUPPORTED_FUNCTION", "LABEL_NAME_GLOB", "SCOPE_DISJUNCTION", "PHI_QUANTILE",
        "FILL_INTERPOLATION", "ROLLUP_DEFAULT", "ROLLUP_ALIGNMENT",
        "METRIC_TYPE_UNKNOWN", "COUNTER_RESET", "ANCHORED_VERSION", "PHI_SERIES_START",
    }, sorted(referenced)

    # Declared because the paper argues about them; nothing raises them here.
    declared = {g.name for g in Gap}
    assert declared - referenced == {
        "PHI_ABSOLUTE", "OFFSET", "SUBQUERY", "MANY_TO_ONE", "RATE_EXTRAPOLATION",
        "REGEX_NOT_GLOB", "UNTESTED_LOOKBACK",
    }, sorted(declared - referenced)

"""Whether each experiment could have come out the other way.

Every other test about the experiments asks whether a quoted figure matches its cache.
None asked the prior question: could something other than the stated cause have produced
the observation? That is not idle. E14's first run read back points and reasoned from
them, and the reasoning was one observation short -- a scope term the parser *discards*
returns exactly what a term matching *everything* returns -- so the run was re-done with
three negation controls before its claim could stand.

So this file is the inventory of how each experiment discriminates, and the checks that
the discrimination actually happened. What it found: the designs are sound, including the
two where the weakness would have been easiest to have -- E2 probes on the grid *and*
mid-bucket because one probe cannot separate closure from labelling, and E6 submits at
10 s while rolling up over 30 s because a harness scraping at the rollup width cannot see
the order of `.as_rate()` at all. What it also found is that E4's cache had recorded
E18's second finding two weeks before anyone read it, which is the reason
`rem:rollup-spans` now says so.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# Every experiment cache, and how it rules the alternative explanation out. The kinds:
#   "predictions"  a table of what each candidate rule would give, with one match
#   "narrowing"    a `consistent_with` list the observation cuts down
#   "controls"     probes whose expected answers are checked alongside the result
#   "negative"     a case the claim says must come back empty or rejected
#   "separation"   a workload built so two candidate rules give different answers
# A cache with none of these is an experiment whose result cannot be argued with, which
# is the state E14 was in before it was re-run.
DISCRIMINATION: dict[str, tuple[str, ...]] = {
    "datadog/e1.json": ("predictions", "narrowing"),
    "datadog/e1_fast.json": ("predictions", "narrowing"),
    "datadog/e2.json": ("narrowing", "separation"),
    "datadog/e3.json": ("predictions",),
    "datadog/e4.json": ("controls",),
    "datadog/e5.json": ("controls", "negative"),
    "datadog/as_rate_order.json": ("separation", "controls"),
    "datadog/e7.json": ("controls",),
    "datadog/e8.json": ("negative", "separation"),
    "datadog/e9.json": ("separation",),
    "datadog/e12.json": ("controls", "negative"),
    "datadog/e13.json": ("negative",),
    "datadog/e14.json": ("controls", "negative"),
    # E15's discrimination is the COUNT series beside the RATE one: the same two
    # rollups asked of both, so a rescaling of the RATE metric would show as a
    # difference between them. Its `note` says which comparison to make.
    "datadog/e15.json": ("controls",),
    "datadog/e16.json": ("negative", "controls"),
    "datadog/e17_scope_length.json": ("controls", "negative"),
    "datadog/e18.json": ("controls", "separation"),
    "datadog/scope_empty_tag.json": ("predictions",),
    "datadog/top.json": ("predictions", "controls"),
    "gap_rows/e10_fill.json": ("negative", "separation"),
    "gap_rows/e11_label_name.json": ("controls", "negative"),
}


def test_every_experiment_cache_is_accounted_for_here():
    """A new experiment has to say how it discriminates, or this list stops meaning
    anything. `type_declared.json` is excluded: it joins a repository rather than
    asking a system, so there is no alternative cause to rule out."""
    on_disk = {f"{d}/{p.name}" for d in ("datadog", "gap_rows")
               for p in (RAW / d).glob("*.json")} - {"datadog/type_declared.json"}
    assert on_disk == set(DISCRIMINATION), {
        "cached but not accounted for": sorted(on_disk - set(DISCRIMINATION)),
        "accounted for but not cached": sorted(set(DISCRIMINATION) - on_disk),
    }


@pytest.mark.parametrize("cache", sorted(DISCRIMINATION))
def test_each_cache_carries_the_evidence_its_kind_implies(cache):
    d = json.loads((RAW / cache).read_text())
    kinds = DISCRIMINATION[cache]
    blob = json.dumps(d)
    if "narrowing" in kinds:
        assert "consistent_with" in blob, cache
    if "controls" in kinds:
        assert any(k in blob for k in ("control", "stored", "gauge", "e4_expected",
                                       "too_fine", "attempts")), cache
    if "predictions" in kinds:
        assert any(k in blob for k in ("predict", "model", "expected", "prediction")), \
            cache


# --- the three where the discrimination is the whole design ---------------------------

def test_the_reset_policy_narrows_to_one_only_where_it_can():
    """E1's three cases, and which of them does the work.

    A run with no reset is consistent with every policy; a reset to zero cannot
    separate `zero` from `clamp`, because both predict the same bucket; only a reset to
    a non-zero value separates all four. The paper's claim rests on the third case and
    the other two are what show the first two could not have carried it.
    """
    cases = json.loads((RAW / "datadog" / "e1.json").read_text())["cases"]
    assert set(cases["none"]["consistent_with"]) == {"zero", "clamp", "drop", "raw"}
    assert set(cases["to0"]["consistent_with"]) == {"zero", "clamp"}
    assert cases["to100"]["consistent_with"] == ["zero"]
    # And the narrowing is the policy table, not an assertion beside it.
    policies = cases["to100"]["policies"]
    assert [k for k, v in policies.items() if v["consistent"]] == ["zero"]


def test_the_bucket_convention_needed_two_probes_and_got_them():
    """E2: one probe cannot answer it, which is why there are two.

    A point exactly on the grid is reported at the same timestamp under ``right-closed,
    labelled by end'' as under ``left-closed, labelled by start''. The mid-bucket probe
    fixes the labelling and the on-grid probe then fixes the closure, so the pair cuts
    four readings to one. This is the experiment whose subject is the convention the
    corpus-scale differential was later found unable to see, so its own grid mattering
    is not a hypothetical.
    """
    d = json.loads((RAW / "datadog" / "e2.json").read_text())
    probes, interp = d["provenance"]["probes"], d["interpretation"]
    delta = d["provenance"]["delta"]
    # The two probes are half a bucket apart, which is what separates the readings.
    assert probes["midbucket"] - probes["ongrid"] == delta / 2
    assert interp["conclusive"] is True
    assert interp["consistent_with"] == [{"closure": "left", "label": "start"}]


def test_the_as_rate_order_workload_separates_the_two_orders():
    """E6: the submission grid is deliberately not the rollup width.

    `.as_rate().rollup(avg, R)` could divide each delta by its own interval and average
    the rates, or average the deltas and divide once by R. The two agree whenever the
    grid equals R, so a harness that scrapes at the rollup width cannot see the
    difference -- which is a fact about harnesses this project learned elsewhere, the
    hard way. The constants are the design.
    """
    import re

    source = (ROOT / "scripts" / "as_rate_order.py").read_text()
    step = float(re.search(r"^STEP = ([\d.]+)", source, re.M).group(1))
    rollup = float(re.search(r"^ROLLUP = ([\d.]+)", source, re.M).group(1))
    assert step != rollup, (step, rollup)
    prov = json.loads((RAW / "datadog" / "as_rate_order.json").read_text())
    assert prov["step"] == step and prov["rollup"] == rollup
    # And the gauge probe, which is the undocumented case the modifier says nothing
    # about, plus the `count:` aggregator, which is the only one that does not commute
    # with dividing by a constant and so the only one the order is visible through.
    assert "gauge_level" in prov
    assert any("count:" in q for q in json.dumps(prov["queries"]).split('"'))


# --- what a cache knew before the paper did -------------------------------------------

def test_e4_recorded_the_divergence_e18_is_credited_with_finding():
    """`rem:rollup-spans` says so, and this is why it can.

    E4 asked the automatic interval at five ranges and its cache stored both what the
    response reported and what the returned points were spaced at. They differ at two
    of the five, which is E18's second finding, recorded a fortnight before anybody read
    it. The remark now says that rather than claiming the observation as new, and this
    test is what keeps the sentence honest.
    """
    interp = json.loads((RAW / "datadog" / "e4.json").read_text())["interpretation"]
    reported, observed = interp["reported_by_range"], interp["observed_by_range"]
    diverging = {k for k in reported if reported[k] != observed[k]}
    assert diverging == {"1m", "5m"}, diverging
    assert reported["1m"] == 1 and observed["1m"] == 10.0
    remark = (ROOT / "paper" / "sections" / "moved" / "ing_rollup_spans.tex").read_text()
    assert "it was not new" in remark
    assert "two of its five ranges" in remark


def test_the_controls_of_the_two_experiments_that_were_re_run_all_hold():
    """E14 and E17 are the two whose first runs were not conclusive, so their controls
    are the evidence that the second runs were."""
    e14 = json.loads((RAW / "datadog" / "e14.json").read_text())
    assert e14["interpretation"]["conclusive"] is True
    assert e14["interpretation"]["controls_hold"] is True
    assert all(e14["controls"].values()), e14["controls"]
    e17 = json.loads((RAW / "datadog" / "e17_scope_length.json").read_text())
    assert all(c["holds"] for c in e17["controls"]), e17["controls"]
    assert {c["control"] for c in e17["controls"]} >= {
        "bare-scope-present", "bare-scope-absent"}

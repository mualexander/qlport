"""Rig A's model, and the prediction table E1 was committed to in advance."""
from __future__ import annotations

import numpy as np
import pytest

from qlport.harness.reset import (
    POLICIES, Schedule, bucket_table, consistent_policies, deltas_under, value_at,
)
from qlport.semantics.model import Series

STEP = 15.0
INTERVAL = 60.0


def ideal(reset_to: int = 100, *, k_reset: int = 5, n: int = 12) -> Schedule:
    """The schedule `docs/trial_account_checks.md` specifies for E1.

    `k_reset = 5` puts the reset at $t = 75$, the second of the four scrapes in the
    epoch-aligned bucket $[60, 120)$, which is where the spec places it.
    """
    return Schedule(t0=0.0, step=STEP, reset_at=k_reset * STEP, reset_to=reset_to)


def exposed(s: Schedule, n: int = 12):
    ts = np.arange(n) * STEP
    return ts, np.array([value_at(t, s) for t in ts], dtype=float)


def test_the_value_climbs_by_one_and_reaches_the_pre_reset_value_just_before_it():
    s = ideal()
    ts, vs = exposed(s)
    assert list(vs[:5]) == [996.0, 997.0, 998.0, 999.0, 1000.0]
    # The last step before the reset exposes exactly `pre_reset`, so the reset's signed
    # difference is a known large negative number rather than whatever the ramp reached.
    assert vs[4] == s.pre_reset
    assert vs[5] == 100.0
    assert list(np.diff(vs[5:])) == [1.0] * (len(vs) - 6)


def test_a_schedule_with_no_reset_climbs_throughout():
    s = Schedule(t0=0.0, step=STEP, reset_at=None, reset_to=0)
    _, vs = exposed(s)
    assert list(np.diff(vs)) == [1.0] * (len(vs) - 1)


def test_the_reset_needs_warm_up_before_it():
    with pytest.raises(ValueError):
        Schedule(t0=0.0, step=STEP, reset_at=0.0, reset_to=100)


# --------------------------------------------------------- the committed prediction

@pytest.mark.parametrize("policy,want", [
    ("zero", (103.0, 4)),
    ("drop", (3.0, 3)),
    ("clamp", (3.0, 4)),
    ("raw", (-897.0, 4)),
])
def test_the_predictions_e1_was_scored_against(policy, want):
    """The table in `docs/trial_account_checks.md`, recomputed from the model.

    All four are distinct in at least one column, which is what makes the run a
    discrimination rather than a confirmation: `drop` and `clamp` agree on the sum and
    part on the count, and `zero` and `raw` are separated by the sum alone.
    """
    ts, vs = exposed(ideal())
    d = deltas_under(Series({}, ts, vs, kind="counter"), policy)
    table = bucket_table(d.ts, d.vs, INTERVAL, 0.0, float(ts[-1]) + INTERVAL)
    assert table[60.0] == want


def test_every_policy_is_distinguished_from_every_other():
    ts, vs = exposed(ideal())
    seen = {}
    for policy in POLICIES:
        d = deltas_under(Series({}, ts, vs, kind="counter"), policy)
        seen[policy] = bucket_table(d.ts, d.vs, INTERVAL, 0.0,
                                    float(ts[-1]) + INTERVAL)[60.0]
    assert len(set(seen.values())) == len(POLICIES), seen


def test_a_reset_to_zero_cannot_separate_zero_from_clamp():
    """Which is why the spec asks for it as a second case rather than as the only one.

    Reporting the new value and clamping to zero are the same instruction when the new
    value *is* zero. The run includes the case anyway, to catch a special case at zero
    that the reset-to-100 case would not show.
    """
    ts, vs = exposed(ideal(reset_to=0))
    got = {}
    for policy in ("zero", "clamp"):
        d = deltas_under(Series({}, ts, vs, kind="counter"), policy)
        got[policy] = bucket_table(d.ts, d.vs, INTERVAL, 0.0,
                                   float(ts[-1]) + INTERVAL)[60.0]
    assert got["zero"] == got["clamp"]


# --------------------------------------------------------- scoring an observation

def observe(policy: str, s: Schedule, n: int = 12) -> dict:
    ts, vs = exposed(s, n)
    d = deltas_under(Series({}, ts, vs, kind="counter"), policy)
    return bucket_table(d.ts, d.vs, INTERVAL, 0.0, float(ts[-1]) + INTERVAL)


@pytest.mark.parametrize("truth", POLICIES)
def test_scoring_recovers_the_policy_that_produced_the_observation(truth):
    s = ideal()
    ts, vs = exposed(s)
    got = consistent_policies(ts, vs, observe(truth, s), INTERVAL)
    assert [p for p in POLICIES if got[p]["consistent"]] == [truth]


def test_an_observation_nothing_explains_leaves_no_policy_standing():
    # A rig fault should read as "no policy fits", not as whichever fits least badly.
    s = ideal()
    ts, vs = exposed(s)
    nonsense = {b: (v[0] + 7.0, v[1]) for b, v in observe("zero", s).items()}
    got = consistent_policies(ts, vs, nonsense, INTERVAL)
    assert not any(got[p]["consistent"] for p in POLICIES)


def test_drift_in_the_scrape_schedule_is_not_read_as_a_disagreement():
    """Two scrapes in one step give a zero increment; scoring must survive it.

    This is the reason the runner scores against the exposition as served rather than
    against the ideal schedule: the agent picks its own phase and its period drifts.
    """
    s = ideal()
    ts = np.array([0.0, 15.0, 30.0, 44.0, 46.0, 60.0, 75.0, 90.0, 105.0, 120.0])
    vs = np.array([value_at(t, s) for t in ts], dtype=float)
    d = deltas_under(Series({}, ts, vs, kind="counter"), "zero")
    observed = bucket_table(d.ts, d.vs, INTERVAL, 0.0, float(ts[-1]) + INTERVAL)
    got = consistent_policies(ts, vs, observed, INTERVAL)
    assert got["zero"]["consistent"]


def test_the_control_case_discriminates_nothing_and_is_in_the_run_anyway():
    """A counter that never decreases gives every policy nothing to do.

    So the control cannot choose between them, and listing all four as consistent is the
    right answer rather than a contradiction. What it does establish is that the rig is
    reading the right buckets: a full bucket must hold exactly four increments of one,
    and a control that failed would mean any disagreement elsewhere was the harness
    rather than the system --- which is how every earlier surprise in this project went.
    """
    s = Schedule(t0=0.0, step=STEP, reset_at=None, reset_to=0)
    ts, vs = exposed(s)
    observed = observe("zero", s)
    assert observed[60.0] == (4.0, 4)
    got = consistent_policies(ts, vs, observed, INTERVAL)
    assert all(got[p]["consistent"] for p in POLICIES)


# --- the arithmetic, at an epoch that is not zero -------------------------------------
# `scripts/mutate.py` over `harness/reset.py` found both subtractions of `t0`
# unconstrained -- `reset_index`'s and `value_at`'s. Neither is subtle: every schedule
# in this file starts at `t0 = 0.0`, where `t - t0` and `t + t0` are the same
# arithmetic, and the rig runs the model at a unix timestamp. Adding the epoch instead
# of subtracting it puts the reset index at 2.3e8 and reads every value off the end of
# the ramp, and nothing here could tell.

def test_the_model_is_the_same_model_at_a_real_epoch():
    """E1's schedule placed where the rig actually places it.

    Same numbers as the first test in this file, which uses `t0 = 0`: the reset is the
    sixth step, the ramp reaches `pre_reset` on the fifth, and the counter climbs by
    one per step on either side of it.
    """
    t0 = 1_700_000_000.0
    s = Schedule(t0=t0, step=STEP, reset_at=t0 + 5 * STEP, reset_to=100)
    assert s.reset_index == 5
    vs = [value_at(float(t0 + i * STEP), s) for i in range(12)]
    assert vs[:5] == [996, 997, 998, 999, 1000]
    assert vs[4] == s.pre_reset
    assert vs[5] == 100
    assert list(np.diff(vs[5:])) == [1] * (len(vs) - 6)
    with pytest.raises(ValueError, match="precedes the schedule's start"):
        value_at(t0 - STEP, s)


def test_a_reset_one_step_in_is_the_earliest_the_schedule_allows():
    """The warm-up guard's boundary, and the step guard beside it.

    `test_the_reset_needs_warm_up_before_it` shows a reset at index 0 is refused; it
    does not show that index 1 is allowed, so a guard rejecting one step of warm-up as
    well would pass. One step is the stated minimum -- it is what makes the reset's
    signed difference a known number rather than wherever the ramp had got to.
    """
    assert Schedule(t0=0.0, step=STEP, reset_at=STEP, reset_to=100).reset_index == 1
    with pytest.raises(ValueError, match="warm-up"):
        Schedule(t0=0.0, step=STEP, reset_at=0.0, reset_to=100)
    # And a step of zero is refused before anything divides by it -- but only zero and
    # below. "Positive" is the stated requirement, so a one-second grid is a schedule.
    for bad in (0.0, -STEP):
        with pytest.raises(ValueError, match="step must be positive"):
            Schedule(t0=0.0, step=bad, reset_at=STEP, reset_to=100)
    assert Schedule(t0=0.0, step=1.0, reset_at=5.0, reset_to=100).reset_index == 5


def test_the_scored_window_runs_one_interval_past_the_last_scrape():
    """`consistent_policies` scores the buckets the run covers, last one included.

    The window ends at the last scrape plus one interval, because the bucket holding
    that scrape is not finished at the scrape itself. Ending it at the last scrape
    drops that bucket, and with twelve scrapes fifteen seconds apart it takes the
    comparison from three buckets to two -- a third of the evidence, silently, with
    every policy still coming out consistent.
    """
    s = ideal()
    ts, vs = exposed(s)
    # What a `zero` agent would have submitted, bucketed over the same window the
    # function uses, so the observation is exactly what the prediction should match.
    d = deltas_under(Series({}, ts, vs, kind="counter"), "zero")
    observed = bucket_table(d.ts, d.vs, INTERVAL, 0.0, float(ts[-1]) + INTERVAL)
    assert sorted(observed) == [0.0, 60.0, 120.0], sorted(observed)

    got = consistent_policies(ts, vs, observed, INTERVAL)
    assert got["zero"]["buckets_scored"] == 3, "the last bucket was dropped"
    assert got["zero"]["consistent"] is True


def test_the_disagreement_list_is_capped_and_the_verdict_is_not():
    """A run long enough to disagree more than the report will print.

    `disagreements` is a sample for a reader -- five of them -- while `consistent` is
    the verdict over all of them. A cap that printed one more or one fewer would be
    invisible on the twelve-scrape schedules above, which can only disagree three
    times; this one has ten buckets and an observation that matches none of them.
    """
    s = ideal()
    n = 40
    ts = np.arange(n) * STEP
    vs = np.array([value_at(float(t), s) for t in ts], dtype=float)
    observed = {float(b): (0.0, 0)
                for b in np.arange(0.0, ts[-1] + INTERVAL, INTERVAL)}

    got = consistent_policies(ts, vs, observed, INTERVAL)
    assert got["zero"]["buckets_scored"] == 10
    assert len(got["zero"]["disagreements"]) == 5, "the printed sample is five"
    assert got["zero"]["consistent"] is False, "the verdict is over all of them"

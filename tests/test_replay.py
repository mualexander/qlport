"""The log-replay converter: parsing, counting, shifting, window selection.

No file or network access -- the parser takes lines and the rest takes arrays, so the
whole module is testable without the trace. What the tests mostly pin down is the two
things that would silently corrupt a measurement: the timezone handling in the parser,
and the half-open convention in the counter.
"""
from __future__ import annotations

import numpy as np
import pytest

from texsource import table

from qlport.harness.replay import (
    busiest_window, events_to_counter, parse_events, per_bucket_totals,
    shift_to,
)

LINE = ('199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] '
        '"GET /history/apollo/ HTTP/1.0" 200 6245')


def test_a_line_yields_only_a_time_and_a_status_class():
    (epoch, cls), = parse_events([LINE])
    assert cls == "2xx"
    # 00:00:01 at -0400 is 04:00:01 UTC on the same day.
    import calendar
    assert epoch == calendar.timegm((1995, 7, 1, 4, 0, 1, 0, 0, 0))


def test_the_host_and_request_do_not_survive_parsing():
    # The trace's publishers ask that it not be analysed beyond traffic patterns, so
    # the parser's output is the enforcement point, not a convention to remember.
    out = list(parse_events([LINE]))
    assert out == [(out[0][0], "2xx")]
    assert not any("199.72.81.55" in str(x) or "apollo" in str(x) for x in out)


def test_the_timezone_offset_is_applied():
    plus = 'h - - [01/Jul/1995:00:00:00 +0200] "GET / HTTP/1.0" 200 1'
    minus = 'h - - [01/Jul/1995:00:00:00 -0200] "GET / HTTP/1.0" 200 1'
    (a, _), = parse_events([plus])
    (b, _), = parse_events([minus])
    assert b - a == 4 * 3600


@pytest.mark.parametrize("code,cls", [("200", "2xx"), ("304", "3xx"),
                                      ("404", "4xx"), ("500", "5xx")])
def test_status_classes(code, cls):
    line = f'h - - [01/Jul/1995:00:00:00 -0400] "GET / HTTP/1.0" {code} 1'
    (_, got), = parse_events([line])
    assert got == cls


def test_unparseable_and_unknown_status_lines_are_skipped_not_raised():
    lines = ["garbage",
             'h - - [01/Jul/1995:00:00:00 -0400] "GET / HTTP/1.0" - -',   # no status
             'h - - [01/Jul/1995:00:00:00 -0400] "GET / HTTP/1.0" 100 1',  # 1xx
             LINE]
    assert len(list(parse_events(lines))) == 1


def test_the_counter_is_cumulative_and_right_closed():
    # Events at 1, 2 and 7; scrapes every 5 s from 0.
    s = events_to_counter([1, 2, 7], interval=5, start=0, end=10)
    assert list(s.ts) == [0.0, 5.0, 10.0]
    assert list(s.vs) == [0.0, 2.0, 3.0]
    assert s.kind == "counter"


def test_an_event_exactly_on_a_scrape_is_counted_by_that_scrape():
    # The half-open convention: (T_prev, T]. An event at T belongs to the scrape at T,
    # which matches what a PromQL range selector reads.
    s = events_to_counter([5], interval=5, start=0, end=10)
    assert list(s.vs) == [0.0, 1.0, 1.0]


def test_the_counter_never_decreases():
    rng = np.random.default_rng(0)
    s = events_to_counter(rng.uniform(0, 600, 5000), interval=15, start=0, end=600)
    assert np.all(np.diff(s.vs) >= 0)


def test_a_window_shorter_than_one_interval_is_refused():
    with pytest.raises(ValueError, match="shorter than one scrape"):
        events_to_counter([1], interval=60, start=0, end=10)


def test_shifting_preserves_every_gap():
    s = events_to_counter([1, 2, 7], interval=5, start=0, end=10)
    moved = shift_to(s, 1_700_000_000)
    assert moved.ts[0] == 1_700_000_000
    assert np.allclose(np.diff(moved.ts), np.diff(s.ts))
    assert list(moved.vs) == list(s.vs)


def test_the_busiest_window_finds_the_burst():
    quiet = list(range(0, 1000, 50))          # one event every 50 s
    burst = [600 + i * 0.1 for i in range(200)]   # 200 events in 20 s
    start, end = busiest_window(quiet + burst, width=60)
    assert 580 <= start <= 620, (start, end)
    assert end - start == 60


def test_the_busiest_window_needs_events():
    with pytest.raises(ValueError, match="no events"):
        busiest_window([], width=60)


# --- what the dual-write replay depends on -------------------------------------

def test_shifted_samples_stay_on_whole_seconds_when_the_shift_is_integral():
    """Datadog's intake takes integer timestamps, so the replay cannot scatter them.

    The trace's events are at second resolution and the scrape interval is a whole
    number, so an integral shift keeps every sample on a whole second; a shift taken
    straight from `time.time()` does not, and the submission is refused.
    """
    s = events_to_counter([1, 2, 7], interval=15, start=0, end=60)
    moved = shift_to(s, float(int(1_700_000_123.456)))
    assert all(float(t).is_integer() for t in moved.ts), list(moved.ts)


def test_the_counter_feeds_phi_without_losing_the_shape():
    """φ of the replayed counter is what Datadog receives; the deltas must be the
    per-interval request counts, which is the whole premise of the replay."""
    from qlport.semantics.ingest import phi_series
    events = [0.5, 1.5, 2.5, 12.0, 12.5]          # 3 in the first 10 s, 2 in the second
    s = events_to_counter(events, interval=10, start=0, end=20)
    deltas = phi_series(s, "zero")
    assert list(deltas.vs) == [3.0, 2.0]
    assert list(deltas.ts) == [10.0, 20.0]


def test_an_empty_interval_produces_a_zero_delta_not_a_gap():
    # The sparse series in the replay has buckets with no requests at all; they must
    # appear as zeroes, or the comparison silently drops them.
    from qlport.semantics.ingest import phi_series
    s = events_to_counter([1.0], interval=10, start=0, end=30)
    deltas = phi_series(s, "zero")
    assert list(deltas.vs) == [1.0, 0.0, 0.0]


# --- the recorded dual-write replay (paper §3.1) --------------------------------

import json                                                         # noqa: E402
from pathlib import Path                                            # noqa: E402

REPLAY = Path(__file__).resolve().parent.parent / "data" / "raw" / "replay"


def _run(name: str) -> dict:
    path = REPLAY / f"dualwrite_{name}.json"
    if not path.exists():  # pragma: no cover - run scripts/replay_dualwrite.py
        pytest.skip(f"{path.name} not recorded; run scripts/replay_dualwrite.py")
    return json.loads(path.read_text())


EXACT_TOL = 1e-12          # the float64 bound every exactness column in this paper uses


def _summary(name: str) -> dict:
    """The summary the tests below read, recomputed from the rows it reduces.

    `summary` is an aggregate the recording script wrote beside the rows: a share of
    buckets within tolerance, a median, a maximum. The rows hold both systems' readings
    for every bucket, so none of it needs to be taken on trust -- and the deviation is
    `metrics.reldiff` of the pair, which is the definition §3.1 states rather than
    whatever the script happened to compute.

    What this closes is the same hole the differential cache had: a stored exactness of
    1.0 would keep reading as "exact on every bucket" after a change to the comparison
    that produced it, because nothing else in the file looked at the pairs.
    """
    import collections

    from qlport.metrics import reldiff

    run = _run(name)
    assert run["rows"], name
    by: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for r in run["rows"]:
        d = reldiff(r["datadog"], r["promql"])
        assert d == pytest.approx(r["d"], abs=1e-15, rel=1e-12), r
        by[(r["instant"], r["candidate"])].append(d)
    derived = {
        f"{inst} | {cand}": {
            "buckets": len(ds),
            "exact": sum(1 for x in ds if x <= EXACT_TOL) / len(ds),
            "bit_identical": sum(1 for x in ds if x == 0) / len(ds),
            "median_d": float(np.median(ds)),
            "max_d": float(np.max(ds)),
        }
        for (inst, cand), ds in by.items()
    }
    stored = run["summary"]
    assert set(derived) == set(stored), (sorted(derived), sorted(stored))
    for key, got in derived.items():
        for field in ("buckets", "median_d", "max_d"):
            assert got[field] == pytest.approx(stored[key][field]), (name, key, field)
        # `exact` is the one field these runs predate. They were recorded before the
        # tolerance fix, so the stored share is bit-identity -- which agrees with the
        # tolerance share in eleven of the twelve cells and differs in the twelfth. A
        # re-recording with today's script stores the tolerance share under `exact` and
        # bit-identity beside it, so both shapes are accepted and both are checked.
        if "bit_identical" in stored[key]:
            assert stored[key]["exact"] == pytest.approx(got["exact"]), (name, key)
            assert stored[key]["bit_identical"] == pytest.approx(
                got["bit_identical"]), (name, key)
        else:
            assert stored[key]["exact"] == pytest.approx(
                got["bit_identical"]), (name, key)
    return derived


@pytest.mark.parametrize("name", ["unaligned", "aligned"])
def test_both_exact_forms_agree_with_real_datadog(name):
    """The dual-write test: one stream, two systems, compared bucket by bucket.

    This is the measurement §9 names as the one that would change the standing of
    everything else, and the only one in the paper taken against Datadog rather than a
    model of it. Both exact forms must be exact on every bucket, evaluated where §3.1
    says to evaluate them.
    """
    summary = _summary(name)
    for cand in ("x - x offset D", "increase(x[D] anchored)"):
        got = summary[f"inside | {cand}"]
        assert got["exact"] == 1.0, (name, cand, got)
        assert got["max_d"] == 0, (name, cand, got)


@pytest.mark.parametrize("name", ["unaligned", "aligned"])
def test_rate_is_only_epsilon_sound_on_real_traffic(name):
    """`rate()` is the translation everyone reaches for, and it is not exact.

    Synthetic steady traffic hides this -- §3.1's steady workload has it at 100%. Real
    request traffic does not.
    """
    got = _summary(name)[f"inside | rate(x[D]) * D"]
    assert got["exact"] < 0.5, got
    assert got["max_d"] > 0


def test_the_closure_convention_is_invisible_on_an_unaligned_scrape():
    """Why the wrong bucket convention survived every earlier measurement.

    Off a boundary, [b, b+D) and (b, b+D] read the same samples, so evaluating on the
    bucket edge agrees with evaluating inside it. Real scrapes are jittered, so this is
    the ordinary case -- and it is the one in which the error cannot be seen.
    """
    s = _summary("unaligned")
    for cand in ("x - x offset D", "increase(x[D] anchored)"):
        assert s[f"inside | {cand}"]["exact"] == s[f"on the edge | {cand}"]["exact"] == 1.0


def test_the_closure_convention_decides_the_answer_on_an_aligned_scrape():
    """And why it matters anyway. With samples on the boundaries, the edge is wrong."""
    s = _summary("aligned")
    for cand in ("x - x offset D", "increase(x[D] anchored)"):
        assert s[f"inside | {cand}"]["exact"] == 1.0, cand
        assert s[f"on the edge | {cand}"]["exact"] < 0.5, cand


def test_the_exactness_the_paper_states_is_the_one_the_rows_support():
    r"""The cell where bit-identity and the paper's tolerance disagree, pinned.

    Every exactness column in this paper admits a relative deviation of $10^{-12}$,
    because a float64 sum of the same numbers in a different order is not the same
    float. `replay_dualwrite.py` scored bit-identity instead when these two runs were
    recorded, and in one cell -- the aligned grid, `rate` evaluated inside the bucket --
    that counted a bucket reading $236.0$ against $235.99999999999997$ as a
    disagreement. One bucket in $84$ is $1.2$ points, which is the whole discrepancy.

    Notebook 02 re-derives from the rows, so Table 2 and §3.1 already state $20.2\%$;
    the stored summary still says $19.0\%$. Nothing joined the two until this file
    re-derived as well, and a cell that disagrees with its own rows is exactly what a
    stored verdict is for.
    """
    rows = _summary("aligned")["inside | rate(x[D]) * D"]
    stored = _run("aligned")["summary"]["inside | rate(x[D]) * D"]
    assert rows["buckets"] == 84
    assert round(rows["exact"] * 100, 1) == 20.2
    assert round(rows["bit_identical"] * 100, 1) == 19.0
    assert stored["exact"] == pytest.approx(rows["bit_identical"])
    # The figure the paper states is the derived one, in the table notebook 02 writes.
    printed = table(Path(__file__).resolve().parent.parent / "paper" / "tables"
                      / "replay_dualwrite.tex")
    assert r"20.2\%" in printed and r"19.0\%" not in printed


@pytest.mark.parametrize("name", ["unaligned", "aligned"])
def test_the_replay_kept_nothing_but_traffic_patterns(name):
    """The trace's publishers ask that it not be analysed further; the cache obeys."""
    text = (REPLAY / f"dualwrite_{name}.json").read_text()
    assert "NASA" in json.loads(text)["provenance"]["source"]
    for leak in ("GET /", "HTTP/1.0", "unicomp", ".net", ".gov"):
        assert leak not in text, leak


# --- what the one tested offset could not settle -------------------------------------
# `scripts/mutate.py` over `harness/replay.py` left most of the UTC-offset arithmetic
# alive, and the reason is in the fixture rather than in the code: the one line parsed
# here carries `-0400`, a whole number of hours with no minutes. Under that offset the
# minute term is zero whatever is done to it, `int("04")` and `int("4")` agree, and the
# slice can run past the end of the string without changing what it reads. Six mutants
# survived on one line for want of a half-hour zone.

HALF_HOUR_LINE = ('1.2.3.4 - - [01/Jul/1995:00:00:01 +0530] '
                  '"GET /history/apollo/ HTTP/1.0" 200 6245')


def test_an_offset_with_minutes_in_it_is_applied_whole():
    """A zone that is not a whole number of hours, east of UTC.

    `+0530` is the only shape that exercises every part of the conversion at once: the
    sign, the hour field, the minute field, and the two slices that separate them.
    00:00:01 on 1 July at +0530 is 18:30:01 on 30 June in UTC -- the previous day,
    which is also the only case here where the date moves backwards.
    """
    import calendar

    (epoch, cls), = parse_events([HALF_HOUR_LINE])
    assert cls == "2xx"
    assert epoch == calendar.timegm((1995, 6, 30, 18, 30, 1, 0, 0, 0))
    # And the western half-hour zone, so the sign is not read off this one case either.
    west = HALF_HOUR_LINE.replace("+0530", "-0530")
    (epoch_west, _), = parse_events([west])
    assert epoch_west == calendar.timegm((1995, 7, 1, 5, 30, 1, 0, 0, 0))
    assert epoch_west - epoch == 2 * (5 * 3600 + 30 * 60)
    # And a zone past nine hours, so the hour field is read as two digits rather than
    # as whichever one happens to be last.
    (epoch_far, _), = parse_events([HALF_HOUR_LINE.replace("+0530", "+1030")])
    assert epoch_far == calendar.timegm((1995, 6, 30, 13, 30, 1, 0, 0, 0))


def test_shifting_a_trace_that_does_not_start_at_zero():
    """`shift_to` subtracts the first sample before adding the new start.

    Every other test of it builds the series from `start=0`, where subtracting the
    first timestamp and adding it are the same thing. The rig only ever calls it on a
    1995 trace, where adding instead of subtracting lands the replay three decades in
    the future and Datadog's intake refuses the lot.
    """
    s = events_to_counter([805_000_001], interval=5, start=805_000_000, end=805_000_010)
    assert s.ts[0] == 805_000_000.0, "the fixture has to start somewhere other than 0"

    moved = shift_to(s, 1_700_000_000.0)
    assert list(moved.ts) == [1_700_000_000.0, 1_700_000_005.0, 1_700_000_010.0]
    assert np.allclose(np.diff(moved.ts), np.diff(s.ts))
    assert list(moved.vs) == list(s.vs)


def test_the_bucket_totals_read_one_candidate_at_one_instant():
    """`per_bucket_totals` filters twice, and the committed recording cannot show it.

    Every candidate in the real run reports the same Datadog side -- that is why the
    function may fix the candidate at all -- so a filter that selected the wrong one,
    or inverted either comparison, returns the same totals from `dualwrite_aligned`
    and fails nothing. The appendix's traffic figure is a derivation from this, so the
    filtering is the part worth pinning, on rows built to disagree.

    Two classes in one bucket are summed; a second bucket stays separate; the other
    candidate and the other instant are excluded.
    """
    recording = {"rows": [
        {"candidate": "wanted", "instant": "inside", "bucket": 0.0, "datadog": 1.0},
        {"candidate": "other", "instant": "inside", "bucket": 0.0, "datadog": 100.0},
        {"candidate": "wanted", "instant": "on the edge", "bucket": 0.0,
         "datadog": 1000.0},
        {"candidate": "wanted", "instant": "inside", "bucket": 60.0, "datadog": 2.0},
        {"candidate": "wanted", "instant": "inside", "bucket": 0.0, "datadog": 3.0},
    ]}
    assert per_bucket_totals(recording) == [2.0, 4.0]
    assert per_bucket_totals(recording, instant="on the edge") == [1000.0]


def test_a_window_of_exactly_one_interval_is_the_shortest_allowed():
    """The other side of `test_a_window_shorter_than_one_interval_is_refused`.

    Two scrapes is one difference, which is the least that can be compared, and the
    guard is `< 2` rather than `<= 2`. Only the refusal was tested, so a guard that
    demanded three scrapes would have passed and quietly refused the shortest window
    the replay is allowed to use.
    """
    s = events_to_counter([1.0], interval=10, start=0, end=10)
    assert list(s.ts) == [0.0, 10.0]
    assert list(s.vs) == [0.0, 1.0], "the event is counted by the scrape at 10"

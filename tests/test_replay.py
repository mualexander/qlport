"""The log-replay converter: parsing, counting, shifting, window selection.

No file or network access -- the parser takes lines and the rest takes arrays, so the
whole module is testable without the trace. What the tests mostly pin down is the two
things that would silently corrupt a measurement: the timezone handling in the parser,
and the half-open convention in the counter.
"""
from __future__ import annotations

import numpy as np
import pytest

from qlport.harness.replay import (
    busiest_window, events_to_counter, parse_events, shift_to,
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


@pytest.mark.parametrize("name", ["unaligned", "aligned"])
def test_both_exact_forms_agree_with_real_datadog(name):
    """The dual-write test: one stream, two systems, compared bucket by bucket.

    This is the measurement §9 names as the one that would change the standing of
    everything else, and the only one in the paper taken against Datadog rather than a
    model of it. Both exact forms must be exact on every bucket, evaluated where §3.1
    says to evaluate them.
    """
    summary = _run(name)["summary"]
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
    got = _run(name)["summary"][f"inside | rate(x[D]) * D"]
    assert got["exact"] < 0.5, got
    assert got["max_d"] > 0


def test_the_closure_convention_is_invisible_on_an_unaligned_scrape():
    """Why the wrong bucket convention survived every earlier measurement.

    Off a boundary, [b, b+D) and (b, b+D] read the same samples, so evaluating on the
    bucket edge agrees with evaluating inside it. Real scrapes are jittered, so this is
    the ordinary case -- and it is the one in which the error cannot be seen.
    """
    s = _run("unaligned")["summary"]
    for cand in ("x - x offset D", "increase(x[D] anchored)"):
        assert s[f"inside | {cand}"]["exact"] == s[f"on the edge | {cand}"]["exact"] == 1.0


def test_the_closure_convention_decides_the_answer_on_an_aligned_scrape():
    """And why it matters anyway. With samples on the boundaries, the edge is wrong."""
    s = _run("aligned")["summary"]
    for cand in ("x - x offset D", "increase(x[D] anchored)"):
        assert s[f"inside | {cand}"]["exact"] == 1.0, cand
        assert s[f"on the edge | {cand}"]["exact"] < 0.5, cand


@pytest.mark.parametrize("name", ["unaligned", "aligned"])
def test_the_replay_kept_nothing_but_traffic_patterns(name):
    """The trace's publishers ask that it not be analysed further; the cache obeys."""
    text = (REPLAY / f"dualwrite_{name}.json").read_text()
    assert "NASA" in json.loads(text)["provenance"]["source"]
    for leak in ("GET /", "HTTP/1.0", "unicomp", ".net", ".gov"):
        assert leak not in text, leak

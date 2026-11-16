"""Turn a real request log into counter series, for replaying into both systems.

The rollup measurements of paper §3.1 use synthetic workloads --- steady, Poisson,
bursty --- which §8 names as a threat: those shapes are chosen, not observed. A real
access log supplies observed shape for free. Counting requests into a scrape grid gives
a cumulative counter with the diurnal rhythm, the bursts and the quiet stretches that a
generator has to be told to produce.

What this module does *not* do is as deliberate as what it does. The source trace
preserves the requesting host and the full request line, and its publishers ask that it
not be analysed beyond general traffic patterns. Nothing here keeps either: the parser
yields a timestamp and an HTTP status class and discards the rest of the line, so what
reaches the experiment is a count of requests per interval per status class. That is a
traffic pattern and nothing else.

Timestamps are shifted before use. Datadog's intake refuses points more than about an
hour old, so a 1995 trace cannot be submitted where it happened; `shift_to` moves a
window to the present. The *shape* is real and the absolute times are not, which is the
right way round for measuring a rollup.
"""
from __future__ import annotations

import calendar
import re
from collections.abc import Iterable, Iterator

import numpy as np

from ..semantics.model import Series

__all__ = ["parse_events", "events_to_counter", "shift_to", "busiest_window",
           "STATUS_CLASSES"]

# Only the timestamp and the status are captured; the host and request line are matched
# but not retained, which is the point.
_LINE = re.compile(
    r"\[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})\s+([-+]\d{4})\]"   # timestamp
    r".*?\"\s+(\d{3}|-)"                                                    # status
)
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

STATUS_CLASSES = ("2xx", "3xx", "4xx", "5xx")


def _status_class(code: str) -> str | None:
    if not code.isdigit():
        return None
    cls = f"{code[0]}xx"
    return cls if cls in STATUS_CLASSES else None


def parse_events(lines: Iterable[str]) -> Iterator[tuple[int, str]]:
    """Yield `(epoch_seconds, status_class)` per parseable line, dropping everything else.

    Malformed lines are skipped rather than raised on: a thirty-year-old server log has
    truncated entries, and refusing the whole trace over a handful of them would be a
    worse failure than ignoring them. The caller counts how many were skipped.
    """
    for line in lines:
        m = _LINE.search(line)
        if m is None:
            continue
        day, mon, year, hh, mm, ss, offset, code = m.groups()
        month = _MONTHS.get(mon)
        cls = _status_class(code)
        if month is None or cls is None:
            continue
        epoch = calendar.timegm(
            (int(year), month, int(day), int(hh), int(mm), int(ss), 0, 0, 0))
        sign = -1 if offset[0] == "-" else 1
        epoch -= sign * (int(offset[1:3]) * 3600 + int(offset[3:5]) * 60)
        yield epoch, cls


def events_to_counter(times, *, interval: float, start: float, end: float,
                      labels: dict | None = None) -> Series:
    """A cumulative counter sampled every `interval` over `[start, end]`.

    The value at scrape instant `T` is the number of events at or before `T`, which is
    what a counter exported by a real server holds: a running total that a scrape reads.
    Consecutive differences are therefore the requests in `(T_prev, T]`, the same
    half-open convention a PromQL range selector uses.
    """
    times = np.sort(np.asarray(times, dtype=float))
    grid = np.arange(start, end + interval / 2, interval, dtype=float)
    if len(grid) < 2:
        raise ValueError("window is shorter than one scrape interval")
    counts = np.searchsorted(times, grid, side="right").astype(float)
    return Series(dict(labels or {}), grid, counts, kind="counter")


def shift_to(s: Series, new_start: float) -> Series:
    """The same series with its first sample moved to `new_start`.

    Datadog's intake refuses points much older than an hour, so a historical trace has
    to be replayed into the present; shifting preserves every gap and spacing.
    """
    return Series(dict(s.labels), np.asarray(s.ts, dtype=float) - float(s.ts[0]) + float(new_start),
                  np.asarray(s.vs, dtype=float), kind=s.kind)


def busiest_window(times, *, width: float) -> tuple[float, float]:
    """The `width`-second window containing the most events.

    Chosen rather than taken from the start of the trace, because the quiet hours of a
    1995 web server are nearly flat and would not exercise a rollup at all. The busiest
    window is where the burstiness the synthetic workloads were imitating actually is.
    """
    times = np.sort(np.asarray(times, dtype=float))
    if len(times) == 0:
        raise ValueError("no events")
    # For each event as a left edge, how many fall within width; take the best.
    right = np.searchsorted(times, times + width, side="right")
    best = int(np.argmax(right - np.arange(len(times))))
    return float(times[best]), float(times[best] + width)

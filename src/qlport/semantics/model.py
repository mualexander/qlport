"""Data model: a series is a label set plus strictly increasing samples.

Paper §3, "Series, database, result". Label sets are distinct within a database, which
is what makes a label set the identity of a series and lets a result be keyed by one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Series:
    labels: dict[str, str]
    ts: np.ndarray            # seconds, strictly increasing
    vs: np.ndarray            # float values
    kind: str = "unknown"     # "counter" (cumulative), "delta" (Datadog COUNT), "gauge"

    def __post_init__(self):
        self.ts = np.asarray(self.ts, dtype=float)
        self.vs = np.asarray(self.vs, dtype=float)
        if len(self.ts) != len(self.vs):
            raise ValueError("ts and vs differ in length")
        if len(self.ts) > 1 and not np.all(np.diff(self.ts) > 0):
            raise ValueError("timestamps must be strictly increasing")

    def key(self) -> tuple:
        return tuple(sorted(self.labels.items()))


@dataclass
class Database:
    series: list[Series] = field(default_factory=list)

    def __post_init__(self):
        keys = [s.key() for s in self.series]
        if len(keys) != len(set(keys)):
            raise ValueError("two series share a label set")

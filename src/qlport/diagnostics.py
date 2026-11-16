"""Named semantic gaps. Every ⊥ returned by an emitter carries exactly one Gap.

The codes double as the rows of the gap catalogue (paper §5) and the failure
breakdown for κ (paper §7), so add a code rather than reusing a vague one.

Nine of these are reachable from the Datadog → PromQL emitter; the rest name gaps the
paper argues about but no implemented path raises. PHI_ABSOLUTE, PHI_SERIES_START,
OFFSET, SUBQUERY, MANY_TO_ONE and RATE_EXTRAPOLATION belong to the reverse direction,
and REGEX_NOT_GLOB to §6's regex → glob conversion, which `regular.py` now decides;
the code stays unreachable here because the reverse-direction emitter is not built. Kept
because the paper names them, not because anything returns them: `grep -rn 'Gap[.]'` is
the check on which is which.
"""
from dataclasses import dataclass
from enum import Enum


class Gap(str, Enum):
    # information barrier (paper §3): the target system does not store what the query reads
    PHI_ABSOLUTE = "query reads absolute counter values erased by ingestion"
    METRIC_TYPE_UNKNOWN = "translation depends on a metric type the dashboard never states"
    PHI_SERIES_START = "result depends on where a series begins inside the window"
    # expressiveness gaps: data exists, target language cannot say it
    OFFSET = "time-shifted selector has no target equivalent"
    SUBQUERY = "subqueries have no target equivalent"
    MANY_TO_ONE = "group_left/group_right vector matching"
    RATE_EXTRAPOLATION = "rate extrapolation differs; only ε-sound"
    REGEX_NOT_GLOB = "regular expression has no equivalent glob"
    ROLLUP_ALIGNMENT = "rollup buckets cannot be aligned with the target's windows"
    SCOPE_DISJUNCTION = "disjunctive scope is not one label-matcher conjunction"
    LABEL_NAME_GLOB = "wildcard over tag keys, not tag values"
    FILL_INTERPOLATION = "fill() interpolation has no target equivalent"
    # Documented as a time-range x widget-type lookup table, not a formula -- but the
    # query text never records the time range, so the interval has to be assumed.
    ROLLUP_DEFAULT = "Datadog's automatic rollup interval is not carried by the query"
    COUNTER_RESET = "cumulative difference misreads a counter reset"
    ANCHORED_VERSION = "anchored range selectors need Prometheus 3.7 or newer"
    UNSUPPORTED_FUNCTION = "function not modelled yet"
    # measurement status, not a translation failure
    UNTESTED_LOOKBACK = "lookback longer than the test run; translated but not testable"


@dataclass(frozen=True)
class Bottom:
    """The ⊥ result of a translation, with the reason it was refused."""
    gap: Gap
    detail: str = ""

    def __str__(self) -> str:
        return f"⊥[{self.gap.name}] {self.detail}".rstrip()

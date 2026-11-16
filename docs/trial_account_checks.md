# Settling Datadog's undocumented behaviour (open item 5)

Three claims in the paper are stated relative to facts about Datadog that its
documentation was assumed not to settle. Paper §7.9 ("What is not measured") reports them
as open and points here for the experiments:

- §3.1 Proposition 3 holds for ρ = zero, and the restart rows of
  `rollup_translation_live.tex` compare against φ_zero.
- Σ_Δ presumes buckets are right-closed and epoch-aligned, reported as
  `ROLLUP_ALIGNMENT` on every rollup the translator emits (100% of translations).
- The rollup interval is assumed when the query does not state one, reported as
  `ROLLUP_DEFAULT` (96.9% of translations).

Reading the documentation properly turns out to answer more of this than expected, and
to contradict one thing the code currently does. **Read the "already answered" section
before running any experiment** — it removes two of them and reduces a third to a
confirmation.

## Already answered by the documentation

| Question | Status | Source |
|---|---|---|
| Order of time vs. space aggregation | **Answered — code fixed, κ 90.2% → 94.2%** | [Nested queries](https://docs.datadoghq.com/metrics/nested_queries/), [how weighted() works](https://docs.datadoghq.com/dashboards/guide/how-weighted-works/) |
| Epoch alignment of buckets | **Answered: aligned to UNIX time** | [Rollup](https://docs.datadoghq.com/dashboards/functions/rollup/) (monitors section) |
| Automatic rollup interval | **Answered as a lookup table**, not a formula | [Rollup](https://docs.datadoghq.com/dashboards/functions/rollup/) |
| Default time aggregator | Partly: `avg` is the enforced default; `.as_count()` sets it to `sum`; COUNT/RATE get `.as_count()` appended **in the UI only** | [Rollup](https://docs.datadoghq.com/dashboards/functions/rollup/) |
| Right- vs left-closure of buckets | **Not documented** | — |
| Reset policy ρ | **Not documented** | — |

### Aggregation order: a bug, not an unknown — now fixed

> "each metric query in Datadog is evaluated with two layers of aggregation: first by
> time, then by space"

The `weighted()` page works `sum:kubernetes.cpu.requests{*} by {kube_container_name}.rollup(avg, 10)`
through explicitly: step 1 averages **each timeseries** over each 10 s interval; step 2
sums those averaged values per `kube_container_name`.

`lower/datadog.py` builds `Rollup(time_agg, Δ, Agg(space_agg, keys, Select))` — rollup
*outside* the aggregation, i.e. space first. `emit/promql.py` then tries to push the
rollup inside and refuses when the two do not commute, as `AGG_ROLLUP_ORDER`.

Under the documented order there is nothing to commute: the rollup is *already* inside,
and that is the shape PromQL writes naturally.

```
max:x{*}.rollup(avg, 60)   →   max(avg_over_time(x[60s]))
min:nifi.can_connect{*}    →   min(avg_over_time(nifi_can_connect[60s]))
```

Both are currently refused. The commuting cases were emitted correctly by accident,
because with the rollup inside an aggregation that does commute the two orders coincide
— which is why the 6,931 translated queries are unaffected.

**Fixed.** The IR now builds `Agg(space, keys, Rollup(time, Δ, Select))`, the
commutation logic and the `AGG_ROLLUP_ORDER` code are gone, and `.as_rate()`'s division
sits on the inner node because it is part of the time aggregation — observable only for
`count:`, where dividing after the space aggregation would scale a count of series by
the interval. All 348 occurrences (308 distinct) now translate: **κ went from 90.2% to
94.2%** of occurrences and the gap catalogue from five rows to four. §5 gained a
subsection stating the order, because getting it backwards leaves every emitted
translation looking plausible and shows up only in the refusals.

### Automatic interval: read the table, confirm one row

The interval is set by a documented time-frame × widget-type table ("the past hour" =
20 s for the API column), driven by a 1,500-point-per-series cap, not by a formula. So
`ROLLUP_DEFAULT` can cite a documented table instead of an assumption. Two caveats
worth confirming empirically (E4 below): the table is published for widgets, and the
API column may not match what `/api/v1/query` actually returns; and a manual interval
finer than the enforced one appears to be ignored rather than honoured — the parameter
is documented as applying "if larger than the query-enforced rollup interval", which is
an implication, not a statement.

## What only you can do

I cannot create the account or handle the key. Before any of this runs:

1. Create a Datadog trial (14 days) and an API key, and an application key for the
   query API.
2. Put them in `.env` as `DD_API_KEY` / `DD_APP_KEY` and `DD_SITE` (the site matters:
   `datadoghq.com` vs `datadoghq.eu` have different API hosts). `.env` is gitignored;
   the scripts read it and never log it.
3. Decide the billing exposure. A trial on the free tier with one custom metric and a
   handful of series is well inside limits, but custom metrics are the billed axis, so
   keep the tag cardinality at one or two series.

Everything below then runs from `scripts/trial_checks.py`, caching raw API responses
under `data/raw/datadog/` so the notebooks stay offline (README rule 3).

## Step 0 — read the agent source first (free, offline)

The reset policy is the one question with real stakes for §3.1, and the agent is open
source, so the *intended* behaviour is readable without an account:

- `DataDog/datadog-agent`, `pkg/metrics` — `MonotonicCount` and how the aggregator
  computes successive differences.
- `DataDog/integrations-core`, `datadog_checks_base/datadog_checks/base/checks/openmetrics/v2/`
  — what the OpenMetrics V2 check does with a counter sample lower than the previous one,
  and what it does with the first sample after start.

The `monotonic_count` documentation hints that a lower value is read as a reset and the
sample is **dropped**, which would mean ρ = drop — and therefore that *neither* the
cumulative difference nor the anchored form matches the agent path across a reset,
because both assume ρ = zero. If the source confirms that, §3.1 Proposition 3 needs its
hypothesis changed rather than its proof, and the honest headline becomes that the two
systems disagree across a reset in a way no PromQL expression can repair. That is a
stronger result than the current one, not a weaker one, so this reading is worth doing
carefully before spending trial days.

Reading the source establishes intent; it does not establish what the backend stores.
E1 still confirms it end to end.

## The rigs

Two ingestion paths, because the questions live in different components. State in §7
which one the paper models — φ is a property of the conversion step, not of "Datadog".

**Rig A (agent path), for E1.** A controllable `/metrics` endpoint exposing one counter
whose value follows a script. Prometheus scrapes it every 15 s (the existing pinned
image). A Datadog Agent container runs an OpenMetrics V2 check on the same endpoint at
the same interval. Both systems then hold the same samples, which is the dual-write
model of §3.1. Must run in real time: the intake rejects points more than ~1 h old, and
the agent's conversion is what is under test.

**Rig B (direct submission), for E2–E4.** `POST /api/v2/series` with `type: 3` (count)
and explicit timestamps. This bypasses the agent, so it says nothing about ρ, but it
gives exact control of sample timestamps — which E2 needs and the agent cannot provide.

## E1 — the reset policy ρ (Rig A)

Expose a counter incrementing by 1 per scrape, then at one scrape drop the exposed value
from 1000 to 100, then resume incrementing by 1. Place the reset at the second of the
four scrapes in a 60 s bucket, and read both `.rollup(sum, 60)` and `.rollup(count, 60)`
for that bucket. The second reading is what separates `drop` from `clamp`:

| ρ | δ at the reset | `.rollup(sum, 60)` | `.rollup(count, 60)` |
|---|---|---|---|
| zero (PromQL's assumption) | 100 | **103** | 4 |
| drop (interval discarded) | — | **3** | **3** |
| clamp to zero | 0 | **3** | **4** |
| raw difference | −900 | **−897** | 4 |

All four predictions are distinct in at least one column. Include ≥10 min of warm-up
before the reset, since the first sample after the agent starts is omitted.

Repeat once with a reset to exactly 0 rather than 100, to check that `zero` and `clamp`
are not conflated by a special case at 0.

## E2 — are buckets right- or left-closed? (Rig B)

Submit a single COUNT point of value 1000 at a timestamp that is exactly a multiple of
60, with no other points within ±120 s. Read `.rollup(sum, 60)` over a window spanning
both neighbouring buckets.

| Closure | Bucket holding 1000 |
|---|---|
| right-closed `(t−Δ, t]` | the one **ending** at t\* |
| left-closed `[t, t+Δ)` | the one **starting** at t\* |

The returned `pointlist` timestamps also reveal the grid directly, which independently
confirms the documented UNIX alignment; the single-point probe is what disambiguates
which edge the returned timestamp labels. Repeat with t\* offset by 30 s to confirm that
alignment does not follow the query window.

## E3 — the default time aggregator for a COUNT over the API (Rig B)

The documentation says `avg` is the enforced default but that COUNT and RATE metrics get
`.as_count()` appended *in the UI*. Whether a bare API query does the same is not
documented, and `METRIC_TYPE_UNKNOWN` depends on it.

Submit four COUNT points in one 60 s bucket with deltas 1, 2, 3, 10. Query the metric
with no modifier at all, then with `.as_count()`, then with `.as_rate()`:

| Behaviour | Bare query returns |
|---|---|
| `avg` of the deltas | 4 |
| `sum` (as if `.as_count()`) | 16 |
| `sum/Δ` (as if `.as_rate()`) | 0.267 |

## E4 — confirm the automatic interval table (Rig B)

Submit a point every 10 s for two hours. Query with no `.rollup()` over ranges of 1 min,
5 min, 1 h, 4 h and 1 d, and read the spacing of the returned `pointlist`. Compare
against the API column of the documented table (e.g. 20 s for "the past hour"). Also
submit `.rollup(sum, 1)` over a 1 d range to see whether a finer-than-enforced interval
is rejected with an error or silently widened.

## What each outcome changes

| Result | Changes |
|---|---|
| ρ = zero | §3.1 Prop 3 stands as written; `COUNTER_RESET` stays a real caveat on the cumulative difference only |
| ρ = drop | §3.1 Prop 3's hypothesis becomes ρ = drop and the proof needs redoing; **neither** candidate is exact across a reset; a new gap code for the mismatch; the restart rows of `rollup_translation_live.tex` need recomputing against φ_drop (the data is cached, so this is a notebook re-run, not a new experiment) |
| right-closed | Σ_Δ in §3.1 stands; `ROLLUP_ALIGNMENT` stays a statement about the evaluation grid only |
| left-closed | Σ_Δ's half-open interval flips, and every rollup translation gains a half-bucket offset |
| bare API query behaves as `.as_count()` | `METRIC_TYPE_UNKNOWN` narrows to genuinely ambiguous gauges |
| interval table confirmed | `ROLLUP_DEFAULT` cites a documented table; `DEFAULT_ROLLUP = 60.0` in notebook 03 becomes range-dependent and κ is recomputed per range |

## Threats to validity

- **Agent version.** Pin the agent image by digest and record it in every cached
  response, as `verify_anchored.py` does for Prometheus. Reset handling is the kind of
  thing that changes between minor versions.
- **One path is not all paths.** OTLP ingest sends raw monotonic values by default,
  which is a different φ again. Rig A answers for the OpenMetrics check only; §7 should
  say so rather than generalising to "Datadog".
- **Trial tier differences.** Confirm that rollup behaviour and retention on a trial
  match a paid account; if they differ, the measured interval table may not generalise.
- **Clock skew.** The agent timestamps samples itself. E2 avoids this by submitting
  timestamps directly; E1 should record both the exposition schedule and the timestamps
  Datadog reports back, and compare.
- **Backfill rejection.** Points older than ~1 h are refused, so all of this is
  wall-clock-bound. Budget an afternoon, not a batch job.

## Code this needs

- `src/qlport/harness/expose.py` — the scripted counter endpoint. Pure and testable
  without an account; the value schedule is a function of time.
- `src/qlport/harness/datadog.py` — submit points, run queries, cache raw JSON. Mirrors
  `harness/prometheus.py`, including a `series_from_response` that keys by timestamp.
- `scripts/trial_checks.py` — orchestrate E1–E4, one subcommand each, writing
  `data/raw/datadog/<experiment>.json` with the agent digest and schedule recorded.
- A section in notebook 02 (for ρ) and notebook 05 (for the rest) reading only the
  cache, plus regression tests over it, exactly as `tests/test_harness.py` does for the
  Prometheus verification.

The first two are testable offline and could be written before the account exists.

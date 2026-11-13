"""Live experiment harness.

workload.py   deterministic synthetic traffic (seeded): steady, Poisson, deploy bursts, restarts
dualwrite.py  one generator -> Prometheus (cumulative, via exposition/remote write) AND
              Datadog (COUNT deltas via /api/v2/series), same timestamps
query.py      run q and T(q) over a time grid; every response is written to data/raw/
              as JSON before it is used, so notebooks can rerun offline
"""

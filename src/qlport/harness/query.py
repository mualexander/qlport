"""Superseded: each script queries through `harness/datadog.py` and `prometheus.py`.

Kept as a pointer rather than deleted. The grid-querying this module was going to do is
done per experiment instead, because what a run records differs by experiment -- E8
keeps whole responses, E17 keeps URL lengths and HTTP statuses, the differential keeps
per-bucket pairs -- and a common recorder would have had to store the union.
"""

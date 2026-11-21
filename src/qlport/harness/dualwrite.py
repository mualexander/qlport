"""Superseded by `scripts/replay_dualwrite.py`, which does this and is run.

Kept as a pointer rather than deleted: the paper's §3.1 replay writes one real request
trace into both systems at once, and it does so from a script with its own Prometheus
container rather than from a harness module. Datadog's default intake rejects points
more than 1 h old, which is why the replay shifts the trace forward and runs live.
"""

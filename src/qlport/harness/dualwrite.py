"""Stream one workload to Prometheus and Datadog at the same timestamps. TODO (paper §7).

Datadog's default intake rejects points more than 1 h old, so this runs live.
"""

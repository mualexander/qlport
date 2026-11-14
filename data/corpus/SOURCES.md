# Corpus sources

Written by `scripts/fetch_corpus.py`. Each entry pins the exact commit or
API snapshot the queries in `queries.jsonl` were extracted from.

## kube-prometheus

- repo: `prometheus-operator/kube-prometheus`
- ref: `main`
- commit: `d88939bfbe7ffa368a62dcc1ff477b1c19fa9674`
- files: `95`
- fetched: `2026-11-14`
- dashboards: `33`
- rule_queries: `235`
- queries_promql: `816`

## grafana

- api: `grafana.com/api/dashboards`
- filter: `dataSourceSlugIn=prometheus, orderBy=downloads desc`
- requested: `200`
- downloaded: `200`
- available: `5878`
- fetched: `2026-11-14`
- dashboards: `200`
- dashboards_with_promql: `200`
- queries_promql: `7291`

## integrations-core

- repo: `DataDog/integrations-core`
- ref: `master`
- commit: `452093828d8d91fae1a7be29b62d9197a20b1aa0`
- files: `534`
- fetched: `2026-11-14`
- dashboards: `525`
- unreadable: `9`
- queries_datadog: `8766`
- queries_datadog_formula: `11628`

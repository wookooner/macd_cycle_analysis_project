# Repository Structure

## Current Standard

The repository is organized around active code, active apps, scripts,
documentation, tests, and quarantined legacy code.

```text
macd_cycle_analysis_project/
|- configs/
|- docs/
|- src/
|  |- common/              <- shared path layer (paths.py) and utilities
|  |- dashboard_api/       <- API server, dashboard routes, query engine
|  `- services/            <- live update service
|- data_pipeline/
|  |- collectors/          <- market and futures data collection
|  |- indicators/          <- indicator recalculation for market CSV files
|  |- cycle_detectors/     <- MACD histogram cycle detection + optional JSON map
|  |- context/             <- v2.0 context builder (cycle_dim + timeframe_context)
|  |- feature_extractors/  <- cycle feature extraction
|  `- pipeline_runner.py   <- end-to-end orchestration (Steps 1-5)
|- dashboards/
|  |- chart_app/           <- real-time chart dashboard frontend
|  `- stats_app/           <- stats and feature exploration dashboard
|- trading_bot/
|- scripts/
|  |- dev/                 <- data_schema_report.py and other dev tools
|  |- analysis/
|  |- dashboard/
|  |- experiments/
|  `- legacy/
|- tests/
|- sample_data/
|- reporting/
|- legacy/
`- README.md
```

## Active Code

- `src/common/`: shared path and common utilities. `paths.py` provides
  `PROJECT_PATHS` with all canonical data paths, including `context_dir(asset)`.
- `src/dashboard_api/`: integrated API server, dashboard routes, and query engine.
- `src/services/`: runtime services such as live update.
- `data_pipeline/collectors/`: market and futures data collection (BTC, Gold).
- `data_pipeline/indicators/`: indicator calculation; writes in place to market CSV files.
- `data_pipeline/cycle_detectors/`: MACD histogram cycle detection.
  `cycle_time_mapper.py` builds the optional JSON hierarchy map.
- `data_pipeline/context/context_builder.py`: v2.0 context builder that creates
  `cycle_dim.parquet`, `timeframe_context_*.parquet`, and enriches cycle parquets
  with relationship columns (Step 5).
- `data_pipeline/pipeline_runner.py`: orchestrates Steps 1-5. The default run is
  Steps 1, 2, 3, and 5.
- `trading_bot/`: live trading and execution workflow.
- `dashboards/chart_app/`: real-time chart frontend; polls
  `/api/base-data/series` every 15 seconds.
- `dashboards/stats_app/`: stats dashboard frontend.
- `scripts/dev/data_schema_report.py`: data ecosystem inspector that reports raw
  CSV schemas, cycle parquet schemas, context file structures, and dashboard payloads.
- `docs/`: project rules and operating documentation.
- `tests/`: unit, integration, regression, fixture, and golden tests.

## Pipeline Steps

| Step | Module | Default | Output |
|------|--------|---------|--------|
| 1 collect | `collectors/` | yes | `raw/market/*.csv` updated |
| 2 indicator | `indicators/` | yes | indicator columns updated in CSV |
| 3 detect | `cycle_detectors/` | yes | `cycles_enriched/btc/*.parquet` (base) |
| 4 map | `cycle_detectors/cycle_time_mapper.py` | no | `cycle_hierarchy_map.json` (optional) |
| 5 context | `context/context_builder.py` | yes | `context/btc/` plus enriched cycle parquets |

Run with: `python -m data_pipeline.pipeline_runner --asset btc`

## Legacy Quarantine

The `legacy/` directory contains older experiments or duplicate implementations.
Active code must not import from `legacy/`.

## Data Boundary

Runtime data is outside the repository at:

```text
C:\Users\qw370\macd-cycle-data
```

New code must use `src/common/paths.py` instead of hardcoded `data/`, `outputs/`,
or `reports/` paths.

For a one-page agent overview, see `docs/AGENT_PROJECT_CONTEXT.md`.

For data-file layout, schema details, and context structure, see
`docs/DATA_LAYOUT.md` and `docs/DATA_SCHEMA.md`.

Inspect the live data ecosystem:

```powershell
python .\scripts\dev\data_schema_report.py --asset btc
```

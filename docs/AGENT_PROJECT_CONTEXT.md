# Agent Project Context

## One-Line Model

This repository is the code workspace. Runtime data, generated outputs, reports, and logs live outside the repository under `C:\Users\qw370\macd-cycle-data`.

## Repository Structure

```text
macd_cycle_analysis_project/
|- configs/
|- docs/
|- src/
|  |- common/
|  |- dashboard_api/
|  `- services/
|- data_pipeline/
|- dashboards/
|  |- chart_app/
|  `- stats_app/
|- trading_bot/
|- scripts/
|  |- analysis/
|  |- dashboard/
|  |- dev/
|  |- experiments/
|  `- legacy/
|- tests/
|- sample_data/
|- reporting/
|- legacy/
`- README.md
```

## Active Code Responsibilities

- `src/common/paths.py`: the shared path layer. New code should use this instead of guessing paths.
- `data_pipeline/collectors/`: market and futures data collection.
- `data_pipeline/indicators/`: indicator recalculation for market CSV files.
- `data_pipeline/cycle_detectors/`: MACD histogram cycle detection and hierarchy mapping.
- `data_pipeline/feature_extractors/`: cycle feature extraction.
- `data_pipeline/pipeline_runner.py`: end-to-end pipeline orchestration.
- `src/dashboard_api/`: API routes for cycle data, chart data, and dashboard query services.
- `src/services/live_update_service.py`: background live update loop.
- `dashboards/chart_app/`: real-time exchange-style chart dashboard.
- `dashboards/stats_app/`: stats and feature exploration dashboard.
- `trading_bot/`: trading and execution workflow.
- `scripts/`: user-facing runners, migration helpers, validation helpers, and dev utilities.
- `legacy/`: quarantined old experiments and duplicate implementations. Active code must not import from here.

## External Data Root

```text
C:\Users\qw370\macd-cycle-data
|- raw/
|  |- market/
|  |- hierarchy/
|  `- trades/
|- interim/
|- processed/
|  |- cycles_base/
|  |- cycles_enriched/
|  |- reversal_events/
|  |- features/
|  `- trade_positions/
|- dashboard/
|  |- candles/
|  `- meta/
|- outputs/
|- reports/
`- logs/
```

## Data Flow

```text
raw/market CSV
  -> data_pipeline/indicators
  -> data_pipeline/cycle_detectors
  -> processed/cycles_base
  -> data_pipeline/feature_extractors
  -> processed/cycles_enriched
  -> src/dashboard_api
  -> dashboards/chart_app and dashboards/stats_app
```

The live chart path is:

```text
python .\api_server.py --with-live-update
  -> src/services/live_update_service.py
  -> raw/market CSV refresh
  -> src/dashboard_api/base_data_api.py
  -> /api/base-data/series
  -> dashboards/chart_app polling every 15 seconds
```

## Path Rules

Path priority:

1. `MACD_DATA_ROOT`
2. `configs/paths.yaml`
3. fallback defaults in `src/common/paths.py`

Current standard:

```text
MACD_DATA_ROOT=C:\Users\qw370\macd-cycle-data
```

Do not hardcode:

```text
./data
./outputs
./reports
../data
```

## Commands Agents Should Know

Validate runtime paths:

```powershell
$env:PYTHONPATH='.'
python .\scripts\validate_paths.py
```

Inspect data-file composition:

```powershell
python .\scripts\dev\data_structure_inspector.py
```

Run the real-time backend:

```powershell
python .\api_server.py --with-live-update
```

Run the chart dashboard:

```powershell
npm.cmd run --prefix .\dashboards\chart_app dev
```

Run both with one helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_realtime_chart_dashboard.ps1
```

## Human-Approval Boundaries

Agents may implement code, tests, scripts, exporters, dashboards, path cleanup, and documentation drafts.

Agents must not change these without explicit human approval:

- labeling definitions
- strategy rules
- danger score meaning
- feature semantics
- lookahead or leakage policy
- schema changes with research meaning

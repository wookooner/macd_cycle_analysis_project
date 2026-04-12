# Data Layout

## Purpose

This project keeps code in the repository and runtime data outside the repository. Agents should use this document plus `src/common/paths.py` before reading, writing, or moving data files.

## Path Resolution

Runtime paths are resolved by `src/common/paths.py`.

Priority:

1. `MACD_DATA_ROOT`
2. `configs/paths.yaml`
3. fallback defaults in `src/common/paths.py`

Current operating data root:

```text
C:\Users\qw370\macd-cycle-data
```

## External Data Tree

```text
C:\Users\qw370\macd-cycle-data
|- raw
|  |- market
|  |- hierarchy
|  `- trades
|- interim
|  |- flattened
|  |- joined
|  |- temp
|  `- debug
|- processed
|  |- cycles_base
|  |- cycles_enriched
|  |- reversal_events
|  |- features
|  `- trade_positions
|- dashboard
|  |- candles
|  `- meta
|- outputs
|- reports
`- logs
```

## Data Groups

- `raw/market`: live and historical market CSV files, including OHLCV, funding, open interest, and indicator-enriched market rows.
- `raw/hierarchy`: raw hierarchy mapping inputs if they exist outside processed cycle outputs.
- `raw/trades`: imported trade records.
- `processed/cycles_base`: normalized cycle records before enrichment.
- `processed/cycles_enriched`: enriched cycle parquet files and `cycle_hierarchy_map.json`; this is the current API cycle source.
- `processed/reversal_events`: reversal/noise event tables.
- `processed/features`: feature tables for analysis and modeling.
- `processed/trade_positions`: generated trade-position tables.
- `dashboard/candles`: dashboard-ready candle payloads for lazy loading.
- `dashboard/meta`: dashboard metadata payloads.
- `outputs`: generated experiment, stats, prediction, and backtest outputs.
- `reports`: generated human-readable reports.
- `logs`: runtime logs.

## Runtime Connections

- `data_pipeline/collectors/` writes raw market/futures CSV data under `PROJECT_PATHS.base_data_dir`, which resolves to `raw/market`.
- `data_pipeline/indicators/` updates indicator columns in market CSV files.
- `data_pipeline/pipeline_runner.py` reads raw market data and writes processed cycle outputs.
- `src/dashboard_api/base_data_api.py` serves chart data from `PROJECT_PATHS.base_data_dir`.
- `src/dashboard_api/api_server.py` serves cycle dashboard data from `PROJECT_PATHS.cycle_structured_dir`.
- `src/services/live_update_service.py` runs live market, futures, and cycle syncs.
- `dashboards/chart_app/` polls `/api/base-data/series` every 15 seconds.

## Inspection Command

Use this command to inspect how the data files are composed:

```powershell
python .\scripts\dev\data_structure_inspector.py
```

Machine-readable output:

```powershell
python .\scripts\dev\data_structure_inspector.py --json --max-files 3 --sample-rows 2
```

The inspector reports path resolution, file sizes, dataframe columns, dtypes, row counts when available, nested `cycle_features` / `candle_data` structure, and sample records.

## Agent Rules

- Do not hardcode `data/`, `outputs/`, `reports/`, or local relative data paths.
- Do not treat repo-local `data/` as the source of truth.
- Do not import active code from `legacy/`.
- Do not change labeling, strategy, danger score, feature semantics, or lookahead policy without human approval.
- If a task changes runtime paths, run `scripts/validate_paths.py`.
- If a task needs to understand data composition, run `scripts/dev/data_structure_inspector.py`.

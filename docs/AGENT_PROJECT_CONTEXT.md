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
|  |- collectors/
|  |- indicators/
|  |- cycle_detectors/
|  |- context/               <- NEW v2.0: context builder (cycle_dim + timeframe_context)
|  |- feature_extractors/
|  `- pipeline_runner.py
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

- `src/common/paths.py`: the shared path layer. New code must use this instead of guessing paths. Provides `PROJECT_PATHS.context_dir(asset)` for the v2.0 context directory.
- `data_pipeline/collectors/`: market and futures data collection.
- `data_pipeline/indicators/`: indicator recalculation for market CSV files.
- `data_pipeline/cycle_detectors/`: MACD histogram cycle detection. `cycle_time_mapper.py` builds the optional JSON hierarchy map (Step 4).
- `data_pipeline/context/context_builder.py`: **v2.0 context builder** — builds `cycle_dim.parquet`, `timeframe_context_*.parquet`, enriches cycle parquets with parent/child/sibling relationship columns (Step 5).
- `data_pipeline/feature_extractors/`: cycle feature extraction.
- `data_pipeline/pipeline_runner.py`: end-to-end pipeline orchestration (Steps 1–5).
- `src/dashboard_api/`: API routes for cycle data, chart data, and dashboard query services.
- `src/services/live_update_service.py`: background live update loop.
- `dashboards/chart_app/`: real-time exchange-style chart dashboard.
- `dashboards/stats_app/`: stats and feature exploration dashboard.
- `trading_bot/`: trading and execution workflow.
- `scripts/`: user-facing runners, migration helpers, validation helpers, and dev utilities.
- `scripts/dev/data_schema_report.py`: data ecosystem inspector — reports raw CSV columns, parquet schemas, context file structure, dashboard payloads.
- `legacy/`: quarantined old experiments and duplicate implementations. Active code must not import from here.

## External Data Root

```text
C:\Users\qw370\macd-cycle-data
|- raw/
|  |- market/           <- BTCUSD_*.csv (canonical price source: BTCUSD_* not BTCUSDT_*)
|  |- hierarchy/
|  `- trades/
|- interim/
|  |- flattened/
|  |- joined/
|  |- temp/
|  `- debug/
|- processed/
|  |- cycles_enriched/
|  |  `- btc/           <- CANONICAL: 9 TF parquets (relationship-enriched)
|  |     |- cycles_1M.parquet
|  |     |- cycles_1w.parquet
|  |     |- cycles_1d.parquet
|  |     |- cycles_4h.parquet
|  |     |- cycles_1h.parquet
|  |     |- cycles_30m.parquet
|  |     |- cycles_15m.parquet
|  |     |- cycles_5m.parquet
|  |     |- cycles_1min.parquet
|  |     |- cycle_hierarchy_map.json  (optional, Step 4 output)
|  |     `- archive/    <- historical snapshots; not read by any active code
|  |- context/
|  |  `- btc/           <- NEW v2.0: surrogate-key dimension + timeframe snapshots
|  |     |- cycle_dim.parquet             (~14 MB, 604k rows, 8 cols)
|  |     |- timeframe_context_1min.parquet (~105 MB, 4.5M rows, 69 cols)
|  |     |- timeframe_context_1h.parquet  (~4 MB, 75k rows, 69 cols)
|  |     `- context_meta.json
|  |- cycles_base/
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

**Archive folders** (`archive/`) under `cycles_enriched/` hold old snapshots only. No active code reads from them.

## Data Flow (v2.0)

```text
raw/market/*.csv  (BTCUSD_*)
  |
  v Step 1: collect
  raw/market CSV updated (OHLCV + funding rate + OI)
  |
  v Step 2: indicators
  indicator columns added/updated in raw CSV (macd, ppo, rsi, cvd, oi, funding_rate)
  futures fields merged into 1h/4h/1d
  |
  v Step 3: detect
  processed/cycles_enriched/btc/cycles_*.parquet  (base — no relationship cols)
  |
  v Step 5: context  [full rebuild when Step 3 ran; enrich-only when standalone]
  |  Phase 1 -> processed/context/btc/cycle_dim.parquet
  |  Phase 2 -> processed/context/btc/timeframe_context_1min.parquet
  |          -> processed/context/btc/timeframe_context_1h.parquet
  |  Phase 3 -> processed/cycles_enriched/btc/cycles_*.parquet (enriched — adds
  |             cycle_key, parent_key/type, sibling prev_key, child_count, n_up_4, ...)
  |  Phase 4 -> validation report
  |  Phase 5 -> processed/context/btc/context_meta.json
  v
  src/dashboard_api  ->  dashboards/chart_app  and  dashboards/stats_app

Step 4 (optional):
  processed/cycles_enriched/btc/cycle_hierarchy_map.json  (JSON hierarchy map,
  kept for cross-validation with Phase 4; not required for normal operation)
```

## Pipeline Commands

Standard daily update (Steps 1–3 + context):

```powershell
python -m data_pipeline.pipeline_runner --asset btc
# equivalent: --steps 1 2 3 5
```

Rebuild context only (cycles unchanged, skip Phase 1+2 inside Step 5):

```powershell
python -m data_pipeline.pipeline_runner --asset btc --steps 5
```

Full pipeline including JSON hierarchy map:

```powershell
python -m data_pipeline.pipeline_runner --asset btc --steps 1 2 3 4 5
```

Dry-run to verify paths without writing:

```powershell
python -m data_pipeline.pipeline_runner --asset btc --dry-run
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

Inspect data-file composition and schemas (replaces `data_structure_inspector.py`):

```powershell
python .\scripts\dev\data_schema_report.py --asset btc
python .\scripts\dev\data_schema_report.py --asset btc --json
```

Run the pipeline:

```powershell
python -m data_pipeline.pipeline_runner --asset btc
python -m data_pipeline.pipeline_runner --asset btc --steps 5       # context only
python -m data_pipeline.pipeline_runner --asset btc --dry-run
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

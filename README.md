# MACD Cycle Analysis

This repository is being reorganized into a code-first research workspace.

## Current Direction

The current north star is:

- use MACD cycle data to discover, validate, and accumulate personal market theories

The working system is split into three practical roles:

- canonical data lives outside Git under `C:\Users\qw370\macd-cycle-data`
- `infra/metabase/` provides generic BI exploration through Metabase + Postgres
- `ai_analyst/` provides domain-specific validation over cycle, feature, and parent-child structures

In short:

- Metabase observes
- `ai_analyst` validates
- durable notes/theory records accumulate findings

Start here for the active AI-analysis direction:

- `ai_analyst/docs/README.md`
- `ai_analyst/docs/architecture.md`
- `ai_analyst/docs/execution_plan.md`
- `infra/metabase/README.md`

Core rules:

- Keep code, configs, docs, tests, and lightweight sample data in Git.
- Keep raw data, intermediate files, reports, outputs, and logs outside Git.
- Under `MACD_DATA_ROOT`, keep provider-original downloads in `archive/`,
  normalized source data in `raw/`, and ingestion/quality records in `metadata/`.
- Treat agents as implementers, not decision-makers for labeling or strategy semantics.

Quick start:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_workspace.ps1 -PersistUserEnv -Validate
```

This standardizes the external data root at `C:\Users\qw370\macd-cycle-data` and validates path resolution for local clones and worktrees.

Install Python dependencies before running the API server:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

If `deactivate` fails, no virtual environment is currently active, so you can
ignore that step. If `py -3.13` is not available on your machine, use
`python -m venv .venv` instead. This project needs both `binance-connector`
and `binance-futures-connector` because the collector imports
`binance.spot` and `binance.um_futures`.

Run the real-time API server. This starts the live update worker in the same process, keeps writing fresh market/futures data to the external data root, and lets the chart dashboard reflect new rows on its next refresh:

```powershell
python .\api_server.py --with-live-update
```

Run the chart dashboard frontend:

```powershell
npm.cmd run --prefix .\dashboards\chart_app dev
```

Use `npm.cmd`, not `npm`, in PowerShell on this Windows setup. The old root `chart_dashboard_app` path has been moved to `dashboards\chart_app`.

The chart dashboard polls `/api/base-data/series` every 15 seconds. With `--with-live-update`, market CSV sync defaults to 15 seconds and futures sync defaults to 60 seconds.

Or start the API live-update backend and chart dashboard together with one helper command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_realtime_chart_dashboard.ps1
```

The chart dashboard includes a local real-time footprint panel for the 15-minute
and 1-hour time-candle views. It receives only price-bin aggregates from the
API (not every raw trade): Binance USD-M `aggTrade` messages are grouped in
memory into 5-USDT price bins and published at most four times per second.
Use the `Footprint` checkbox in the chart toolbar to show or hide it. The raw
microstructure collector remains the source for durable Parquet storage and
historical research; this panel is intentionally a low-latency current-bar view.

The data manager is a separate local console for starting approved data
pipeline jobs, following their logs, and reviewing storage usage:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_data_manager.ps1
```

It opens at `http://localhost:5174`; the chart dashboard remains separate at
its normal Vite address. The API is bound to `127.0.0.1` by default, and the
console does not expose arbitrary shell commands or destructive cleanup actions.

The root `api_server.py` is a compatibility entrypoint. The implementation lives in `src/dashboard_api/`.

Legacy/general workspace references:

- `configs/paths.yaml`
- `docs/AGENT_STARTUP_CHECKLIST.md`
- `docs/AGENT_PROJECT_CONTEXT.md`
- `docs/DATA_LAYOUT.md`
- `docs/REPO_STRUCTURE.md`
- `docs/WORKSPACE_SETUP.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/AGENT_TASK_RULES.md`
- `live_update_service.py`

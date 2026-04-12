# Agent Startup Checklist

## Purpose

This is the first document an implementation agent should read before touching code, data paths, or generated artifacts.

## First 30 Seconds

1. Confirm the repository root is correct.
2. Confirm the external data root is `C:\Users\qw370\macd-cycle-data`.
3. Confirm path resolution with `scripts/validate_paths.py`.
4. Read `docs/AGENT_TASK_RULES.md`.
5. Do not edit labeling or strategy semantics unless a human explicitly approves it.

## Required Assumptions

- Code lives in this repository.
- Data lives outside the repository under `C:\Users\qw370\macd-cycle-data`.
- Generated outputs live outside the repository.
- Shared path access goes through `src/common/paths.py`.
- Path priority is:
  1. `MACD_DATA_ROOT`
  2. `configs/paths.yaml`
  3. repo fallback in `src/common/paths.py`

## Required Commands

Bootstrap the workspace if needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_workspace.ps1 -PersistUserEnv -Validate
```

Or validate the current workspace:

```powershell
$env:PYTHONPATH='.'
python .\scripts\validate_paths.py
```

Inspect data-file composition, cycle schemas, and context structure:

```powershell
python .\scripts\dev\data_schema_report.py --asset btc
python .\scripts\dev\data_schema_report.py --asset btc --json
```

Run the pipeline (full update):

```powershell
python -m data_pipeline.pipeline_runner --asset btc
# equivalent: --steps 1 2 3 5
```

Run the pipeline (context rebuild only):

```powershell
python -m data_pipeline.pipeline_runner --asset btc --steps 5
```

## Where Data Should Be

Expected external layout (v2.0):

```text
C:\Users\qw370\macd-cycle-data
|- raw/
|  |- market/                  <- BTCUSD_*.csv (canonical) + BTCUSDT_* (auxiliary)
|  |- hierarchy/
|  `- trades/
|- interim/
|  |- flattened/
|  |- joined/
|  |- temp/
|  `- debug/
|- processed/
|  |- cycles_enriched/
|  |  |- btc/                  <- CANONICAL: 9 TF enriched cycle parquets
|  |  |  `- archive/           <- historical snapshots (not active)
|  |  `- archive/              <- legacy root-level copies (not active)
|  |- context/
|  |  `- btc/                  <- v2.0 context layer
|  |     |- cycle_dim.parquet
|  |     |- timeframe_context_1min.parquet
|  |     |- timeframe_context_1h.parquet
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

**Key rules about data locations**:
- `cycles_enriched/btc/*.parquet` is the only canonical cycle source. Do not read from `cycles_enriched/*.parquet` (root-level files are archived).
- `context/btc/timeframe_context_*.parquet` is the canonical multi-TF snapshot source.
- `archive/` directories are read-only historical snapshots. No active code reads from them.

## Data Flow Summary

```
Step 1 collect  -> raw/market/*.csv updated
Step 2 indicator -> indicator cols written in-place to raw CSV
Step 3 detect   -> cycles_enriched/btc/*.parquet (base cycles, no relationships)
Step 5 context  -> context/btc/cycle_dim + timeframe_context
                -> cycles_enriched/btc/*.parquet (enriched with parent/child/sibling)
Step 4 map [opt]-> cycles_enriched/btc/cycle_hierarchy_map.json (cross-validation only)
```

## What Agents May Do

- implement code
- add tests
- improve scripts and tooling
- build exporters and payload builders
- improve dashboard wiring
- update path usage to shared config
- draft documentation

## What Agents Must Not Change Alone

- labeling definitions
- strategy rules
- danger score meaning
- feature semantics
- lookahead or leakage policy
- schema changes with research meaning

## Safe Edit Rules

- do not hardcode `./data`, `./outputs`, `./reports`, or ad-hoc relative paths in new code
- do not reintroduce repo-local generated data as a runtime dependency
- prefer `src/common/paths.py` over `Path(__file__)` path guessing
- do not read from `archive/` directories — they are historical snapshots
- do not write backup files to canonical data directories (`cycles_enriched/btc/`)
- keep issues small and single-purpose
- if a task changes runtime paths, run `scripts/validate_paths.py`

## Current Runtime Truth

- API and live-update flows use the external data root.
- Canonical cycle source: `processed/cycles_enriched/btc/cycles_*.parquet`.
- Canonical context source: `processed/context/btc/timeframe_context_*.parquet`.
- API server startup without flags is read-only; real-time chart uses `--with-live-update`.
- The real-time chart dashboard is `dashboards/chart_app/`; it polls `/api/base-data/series` every 15 seconds.
- Repo-local `data/` is not the operating source of truth.
- Active code must not import from `legacy/`.

## If Unsure

Stop and check these files in order:

1. `configs/paths.yaml`
2. `src/common/paths.py`
3. `docs/AGENT_TASK_RULES.md`
4. `docs/NO_LEAKAGE_POLICY.md`
5. `docs/AGENT_PROJECT_CONTEXT.md`
6. `docs/REPO_STRUCTURE.md`
7. `docs/DATA_LAYOUT.md`
8. `docs/DATA_SCHEMA.md`
9. `docs/WORKSPACE_SETUP.md`

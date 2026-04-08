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
- Data lives outside the repository.
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

Preview migration status if the task touches legacy outputs or data movement:

```powershell
$env:PYTHONPATH='.'
python .\scripts\migrate_data_root.py --plan-only
```

## Where Data Should Be

Expected external layout:

```text
C:\Users\qw370\macd-cycle-data
|- raw
|  |- market
|  |- hierarchy
|  `- trades
|- interim
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
- keep issues small and single-purpose
- if a task changes runtime paths, run `scripts/validate_paths.py`
- if a task changes data movement, run `scripts/migrate_data_root.py --plan-only`

## Current Runtime Truth

- API and live-update flows are expected to use the external data root.
- Repo-local `data/` is not the operating source of truth anymore.
- Legacy compatibility exists only to help controlled migration, not as the preferred target for new work.

## If Unsure

Stop and check these files in order:

1. `configs/paths.yaml`
2. `src/common/paths.py`
3. `docs/AGENT_TASK_RULES.md`
4. `docs/NO_LEAKAGE_POLICY.md`
5. `docs/WORKSPACE_SETUP.md`

# Workspace Setup

## Goal

Make every local clone or worktree use the same external data root and the same path rules.

## Standard Data Root

- `C:\Users\qw370\macd-cycle-data`

Expected layout:

```text
C:\Users\qw370\macd-cycle-data
|- raw
|- interim
|- processed
|- dashboard
|- outputs
|- reports
`- logs
```

## Path Resolution Order

1. `MACD_DATA_ROOT`
2. `configs/paths.yaml`
3. repo fallback in `src/common/paths.py`

## Fast Setup

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_workspace.ps1 -PersistUserEnv -Validate
```

What this does:

- creates the standard external data-root folders
- sets the current-shell `MACD_DATA_ROOT`
- optionally persists `MACD_DATA_ROOT` at user scope
- runs `scripts/validate_paths.py`

## Migration Check

Preview the copy plan before moving any data:

```powershell
$env:PYTHONPATH='.'
python .\scripts\migrate_data_root.py --plan-only
```

Run the first migration in safe mode:

```powershell
$env:PYTHONPATH='.'
python .\scripts\migrate_data_root.py --dry-run
```

## Agent Workflow Rules

- keep data and generated outputs outside Git
- use small issue-scoped branches or worktrees
- route new path access through `src/common/paths.py`
- do not let agents change labeling semantics or strategy rules

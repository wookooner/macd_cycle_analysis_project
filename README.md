# MACD Cycle Analysis

This repository is being reorganized into a code-first research workspace.

Core rules:

- Keep code, configs, docs, tests, and lightweight sample data in Git.
- Keep raw data, intermediate files, reports, outputs, and logs outside Git.
- Treat agents as implementers, not decision-makers for labeling or strategy semantics.

Quick start:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_workspace.ps1 -PersistUserEnv -Validate
```

This standardizes the external data root at `C:\Users\qw370\macd-cycle-data` and validates path resolution for local clones and worktrees.

Start here after bootstrap:

- `configs/paths.yaml`
- `docs/WORKSPACE_SETUP.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/AGENT_TASK_RULES.md`
- `live_update_service.py`

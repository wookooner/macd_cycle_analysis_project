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

Install Python dependencies before running the API server:

```powershell
python -m pip install -r .\requirements.txt
```

Run the integrated API server:

```powershell
python .\api_server.py --with-live-update
```

Start here after bootstrap:

- `configs/paths.yaml`
- `docs/AGENT_STARTUP_CHECKLIST.md`
- `docs/REPO_STRUCTURE.md`
- `docs/WORKSPACE_SETUP.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/AGENT_TASK_RULES.md`
- `live_update_service.py`

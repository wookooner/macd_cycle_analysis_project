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

The root `api_server.py` is a compatibility entrypoint. The implementation lives in `src/dashboard_api/`.

Start here after bootstrap:

- `configs/paths.yaml`
- `docs/AGENT_STARTUP_CHECKLIST.md`
- `docs/AGENT_PROJECT_CONTEXT.md`
- `docs/DATA_LAYOUT.md`
- `docs/REPO_STRUCTURE.md`
- `docs/WORKSPACE_SETUP.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/AGENT_TASK_RULES.md`
- `live_update_service.py`

# Git Cleanup Plan

## Why This Exists

`.gitignore` only affects new untracked files. This repository already has generated artifacts and data tracked in Git, so index cleanup must happen explicitly.

## High-Risk Tracked Paths Observed

- `data/`
- `analysis_results/`
- `pattern_discovery_results/`
- `feature_analysis_report/`
- `feature_analysis/output/`
- `__pycache__/`

## Recommended Cleanup Sequence

1. Move or copy the data you still need into the configured external data root.
2. Confirm the code can still run with `configs/paths.yaml`.
3. Remove generated artifacts from the Git index without deleting local working files.

Approved external data root:

- `C:\Users\qw370\macd-cycle-data`

Recommended first migration mode:

- `copy + verify`
- only switch to move after the copied layout is validated

## Suggested Commands

Run these only after confirming the replacement paths are valid:

```powershell
git rm -r --cached data
git rm -r --cached analysis_results
git rm -r --cached pattern_discovery_results
git rm -r --cached feature_analysis_report
git rm -r --cached __pycache__
git rm -r --cached */__pycache__
```

Then review what remains:

```powershell
git status --short
git ls-files | Select-String -Pattern '(^data/|^analysis_results/|^pattern_discovery_results/|^feature_analysis_report/|^feature_analysis/output/|__pycache__/|\.parquet$|\.png$|\.tmp$|\.pyc$)'
```

## Policy After Cleanup

- Keep only code, docs, configs, tests, and lightweight sample data in Git.
- Keep raw data, intermediate files, reports, plots, and logs outside Git.

## Follow-Up Candidates

- `cycle_detect/data/base_data/*.csv`
- `data_pipeline/data/cycle_data/structured/*.json`
- any dashboard payload dump that becomes a generated artifact instead of sample data

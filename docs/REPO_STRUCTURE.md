# Repository Structure

## Current Standard

The repository is organized around active code, active apps, scripts, documentation, tests, and quarantined legacy code.

```text
macd_cycle_analysis_project/
|- configs/
|- docs/
|- src/
|  `- common/
|- data_pipeline/
|- dashboards/
|  |- chart_app/
|  `- stats_app/
|- trading_bot/
|- scripts/
|- tests/
|- sample_data/
|- reporting/
|- legacy/
`- README.md
```

## Active Code

- `src/common/`: shared path and common utilities
- `data_pipeline/`: collectors, indicators, cycle detection, feature extractors, and pipeline runner
- `trading_bot/`: live trading and execution workflow
- `dashboards/chart_app/`: chart dashboard frontend
- `dashboards/stats_app/`: stats dashboard frontend
- `scripts/`: runnable utilities, migration, validation, dashboard, analysis, and dev tools
- `docs/`: project rules and operating documentation
- `tests/`: unit, integration, regression, fixture, and golden tests

## Legacy Quarantine

The `legacy/` directory contains older experiments or duplicate implementations. Active code should not import from `legacy/`.

Current legacy groups:

- `legacy/algo_Test/`
- `legacy/backtesting/`
- `legacy/cycle_detect/`
- `legacy/cycle_multi_analysis/`
- `legacy/data_collect/`
- `legacy/data_control/`
- `legacy/feature_develope/`
- `legacy/feature_extract/`
- `legacy/model_test/`
- `legacy/multi_cycle_analysis/`

## Data Boundary

Runtime data is outside the repository at:

```text
C:\Users\qw370\macd-cycle-data
```

New code must use `src/common/paths.py` instead of hardcoded `data/`, `outputs/`, or `reports/` paths.

# Data Schema

## Current Direction

The project is moving toward an external data root with these major layers:

- `raw/`
- `interim/`
- `processed/`
- `dashboard/`
- `outputs/`
- `reports/`
- `logs/`

## Compatibility Note

Existing code still reads some legacy paths under repo-local `data/`. New work should go through `configs/paths.yaml` and `src/common/paths.py` so migration can happen incrementally.

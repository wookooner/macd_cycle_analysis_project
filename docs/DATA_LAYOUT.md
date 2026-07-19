# Data Layout

## Purpose

This project keeps code in the repository and runtime data outside the repository.
Agents should read this document and `src/common/paths.py` before reading, writing,
or moving data files.

## Path Resolution

Runtime paths are resolved by `src/common/paths.py`.

Priority:

1. `MACD_DATA_ROOT`
2. `configs/paths.yaml`
3. fallback defaults in `src/common/paths.py`

Current operating data root:

```text
C:\Users\qw370\macd-cycle-data
```

## External Data Tree (v2.0)

```text
C:\Users\qw370\macd-cycle-data
|- raw/
|  |- market/                          <- BTCUSD_*.csv = canonical price source
|  |  |- BTCUSD_1min.csv  (35 cols, indicators included)
|  |  |- BTCUSD_5m.csv
|  |  |- BTCUSD_15m.csv
|  |  |- BTCUSD_30m.csv
|  |  |- BTCUSD_1h.csv    (+ OI + funding_rate merged)
|  |  |- BTCUSD_4h.csv    (+ OI + funding_rate merged)
|  |  |- BTCUSD_1d.csv    (+ OI + funding_rate merged)
|  |  |- BTCUSD_1w.csv
|  |  |- BTCUSD_1M.csv
|  |  |- BTCUSDT_funding_rate.csv      <- auxiliary source
|  |  |- BTCUSDT_oi_*.csv              <- auxiliary source
|  |  `- BTCUSDT_ls_*_ratio_*.csv      <- auxiliary source
|  |- hierarchy/
|  `- trades/
|- interim/
|  |- flattened/
|  |- joined/
|  |- temp/
|  `- debug/
|- processed/
|  |- cycles_enriched/
|  |  |- btc/                          <- canonical cycle parquets
|  |  |  |- cycles_1M.parquet
|  |  |  |- cycles_1w.parquet
|  |  |  |- cycles_1d.parquet
|  |  |  |- cycles_4h.parquet
|  |  |  |- cycles_1h.parquet
|  |  |  |- cycles_30m.parquet
|  |  |  |- cycles_15m.parquet
|  |  |  |- cycles_5m.parquet
|  |  |  |- cycles_1min.parquet
|  |  |  |- cycle_hierarchy_map.json   <- optional (Step 4); used by Phase 4 validation
|  |  |  `- archive/                   <- historical snapshots; not active
|  |  `- archive/                      <- legacy root-level copies; not active
|  |- context/
|  |  `- btc/                          <- v2.0 context layer
|  |     |- cycle_dim.parquet          <- surrogate-key dimension table (all TF cycles)
|  |     |- timeframe_context_1min.parquet  <- per-minute 9-TF state (69 cols)
|  |     |- timeframe_context_1h.parquet    <- hourly subset of the above (69 cols)
|  |     `- context_meta.json          <- build metadata and data-range info
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

## Data Groups

### raw/market

Live and historical market CSV files.

- **Canonical source**: `BTCUSD_*.csv` across 9 timeframes
  (`1min / 5m / 15m / 30m / 1h / 4h / 1d / 1w / 1M`).
- **Auxiliary source**: `BTCUSDT_*` for funding rate, open interest, and long-short
  ratios. These are merged into the canonical OHLCV files by `step_indicator`.
- Indicator columns (`macd`, `ppo`, `rsi`, `cvd`, `cvd_rolling`, `delta`,
  `ma_7/25/99`, `oi`, `funding_rate`) are stored in place in the CSV files by Step 2.

### processed/cycles_enriched/btc/ (canonical)

One parquet per timeframe. Each file goes through two stages:

1. **Base** (written by Step 3 detect): `cycle_id`, `timeframe`, `start_date`,
   `end_date`, `cycle_type`, `duration_candles`, `category`, `algorithm_used`,
   `candle_data` (list-struct), and `cycle_features` (nested struct).
2. **Enriched** (overwritten by Step 5 Phase 3 in the context builder): base
   columns plus the relationship columns listed below.

Relationship columns added by Step 5:

| Column | Type | Description |
|--------|------|-------------|
| cycle_key | int32 | Surrogate key from `cycle_dim` |
| prev_key / prev_type / prev_dur / prev_price_pct | int32/int8/int16/float32 | Previous sibling |
| parent_key / parent_type | int32/int8 | Primary parent cycle |
| order_in_parent / total_siblings | int16 | Position within parent |
| parent_progress_at_start / _end | float32 | Where in the parent cycle this child starts and ends |
| parent_assign_rule | int8 | `0=contained`, `1=by_start` |
| boundary_type | int8 | `0=normal`, `1=straddle`, `2=transition_trigger` |
| parent_prev_key / _type / parent_next_key / _type | int32/int8 | Adjacent parents |
| overlap_prev_ratio / overlap_next_ratio | float32 | Overlap with adjacent parents |
| n_up_4 | int8 | UP count across `1w/1d/4h/1h` at cycle start |
| combo_4 | string | Direction combo string such as `"UUDU"` |
| child_count / child_up_count / child_down_count | int16 | Child cycle counts |
| opposite_child_ratio | float32 | Fraction of children opposing parent direction |
| max_opposite_child_streak | int8 | Max consecutive opposing children |

### processed/context/btc/ (v2.0 context layer)

| File | Size | Rows | Cols | Description |
|------|------|------|------|-------------|
| cycle_dim.parquet | ~14 MB | 604,846 | 8 | All TF cycles with surrogate int32 `cycle_key` |
| timeframe_context_1min.parquet | ~105 MB | 4,547,518 | 69 | Per-minute snapshot of all 9 TF cycle states |
| timeframe_context_1h.parquet | ~4 MB | 75,792 | 69 | Hourly subset of the above |
| context_meta.json | n/a | n/a | n/a | Version, asset, `data_range`, and `timeframe_groups` |

**Context columns (69 total)**:
- `timestamp`
- Per TF (x9): `{tf}_key`, `{tf}_type`, `{tf}_time_prog`, `{tf}_candle_prog`
- Per TF (x9): `{tf}_changed`, `{tf}_chg_type`
- `major_changed_any`, `minor_changed_any`
- `n_up_4`, `n_up_8`, `major_up_count`, `minor_up_count`
- `combo_4`, `combo_8`, `minor_combo_4`
- `major_late_count`
- `1h_is_boundary`, `30m_is_boundary`, `15m_is_boundary`, `5m_is_boundary`

### processed/cycles_enriched/archive/ and btc/archive/

Historical snapshots. No active code reads from these. Do not confuse them with
the canonical files.

### dashboard/candles/ and dashboard/meta/

Dashboard-ready payloads (currently `1d/4h/1h`). Served by `src/dashboard_api/`.

## Runtime Connections

| Component | Reads from | Writes to |
|-----------|-----------|-----------|
| `data_pipeline/collectors/` | External APIs | `raw/market/*.csv` |
| `data_pipeline/indicators/` | `raw/market/*.csv` | `raw/market/*.csv` (in place) |
| `data_pipeline/cycle_detectors/` | `raw/market/*.csv` | `cycles_enriched/btc/*.parquet` (base) |
| `data_pipeline/context/context_builder.py` | `cycles_enriched/btc/*.parquet` | `context/btc/`, `cycles_enriched/btc/*.parquet` (enriched) |
| `data_pipeline/cycle_detectors/cycle_time_mapper.py` | `cycles_enriched/btc/*.parquet` | `cycles_enriched/btc/cycle_hierarchy_map.json` |
| `src/dashboard_api/` | `cycles_enriched/btc/`, `context/btc/`, `dashboard/` | varies by endpoint |
| `src/services/live_update_service.py` | External APIs | `raw/market/*.csv`, then triggers the pipeline |
| `dashboards/chart_app/` | `/api/base-data/series` (polling) | none |

## Inspection Command

Inspect data-file composition, schemas, and context structure:

```powershell
python .\scripts\dev\data_schema_report.py --asset btc
python .\scripts\dev\data_schema_report.py --asset btc --json
```

The report covers raw market CSV schemas, canonical cycle parquet schemas with
nested struct types, context file structures (`cycle_dim` timeframe distribution,
`n_up_4` distribution, and the 69-column listing), and dashboard payloads.

## Agent Rules

- Do not hardcode `data/`, `outputs/`, `reports/`, or local relative data paths.
- Do not treat repo-local `data/` as the source of truth.
- Do not import active code from `legacy/`.
- Do not read from `archive/` directories. They are historical snapshots only.
- The canonical cycle source is `processed/cycles_enriched/btc/cycles_*.parquet`.
  Do not use the root-level copies in `cycles_enriched/` (archived).
- The canonical multi-TF context source is
  `processed/context/btc/timeframe_context_*.parquet`.
- Do not change labeling, strategy, danger score, feature semantics, or lookahead
  policy without human approval.
- If a task changes runtime paths, run `scripts/validate_paths.py`.
- If a task needs to understand data composition, run `scripts/dev/data_schema_report.py`.

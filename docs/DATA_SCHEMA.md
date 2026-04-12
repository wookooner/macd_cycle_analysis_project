# Data Schema (v2.0)

## Architecture Overview

As of v2.0 the data layer has four canonical tiers:

```
raw/market/          BTCUSD_*.csv       indicator-enriched OHLCV per timeframe
cycles_enriched/btc/ cycles_*.parquet   relationship-enriched cycle records
context/btc/         cycle_dim + timeframe_context  surrogate-key dimension + snapshots
dashboard/           candles/*.parquet  dashboard-ready payloads
```

Legacy files under `cycles_enriched/archive/` and `cycles_enriched/btc/archive/` are historical snapshots and are not read by any active code.

---

## Tier 1 — raw/market/*.csv

One CSV per timeframe. Indicator columns are written in-place by Step 2.

**Timeframe naming**: `1M` (monthly) / `1w` / `1d` / `4h` / `1h` / `30m` / `15m` / `5m` / `1min` (minute).  
**Canonical symbol**: `BTCUSD_*`. The `BTCUSDT_*` files are auxiliary sources (funding rate, OI, long-short ratios) merged into the canonical CSVs during Step 2.

Common columns across all timeframes:

| Column | Type | Notes |
|--------|------|-------|
| unix | int64 | Unix timestamp (seconds) |
| date | string | Human-readable datetime |
| open / high / low / close | float | OHLC price |
| volume | float | Base asset volume |
| Volume USD | float | Quote volume |
| macd / macd_signal / macd_hist | float | MACD indicator |
| ppo / ppo_signal / ppo_hist | float | PPO indicator |
| rsi | float | RSI |
| cvd / cvd_rolling | float | Cumulative volume delta |
| volume_delta / delta | float | Volume delta |
| ma_7 / ma_25 / ma_99 | float | Moving averages |

Additional for 1h / 4h / 1d:

| Column | Type | Notes |
|--------|------|-------|
| oi / oi_usd / oi_change / oi_change_pct | float | Open interest (legacy alias) |
| oi_contracts / oi_contracts_change / oi_contracts_change_pct | float | OI in contract units |
| oi_notional / oi_notional_change / oi_notional_change_pct | float | OI in notional USD |
| funding_rate | float | Perpetual funding rate |

---

## Tier 2 — cycles_enriched/btc/cycles_{tf}.parquet

One parquet per timeframe (9 files). Written by Step 3, enriched in-place by Step 5 Phase 3.

### Base columns (written by Step 3)

| Column | Type | Notes |
|--------|------|-------|
| cycle_id | string | e.g. `cycle_1h_001` |
| timeframe | string | e.g. `1h` |
| start_date / end_date | timestamp | Cycle boundary datetimes. `end_date` = start of last candle (not end) |
| cycle_type | string | `up` or `down` |
| duration_candles | int64 | Cycle length in candles |
| category | string | Cycle category label |
| algorithm_used | string | Detection algorithm name |
| candle_data | list\<struct\> | Per-candle OHLCV + indicators for every candle in cycle |
| cycle_features | struct | Nested stats: aggregate / change / end / shape / start / strength / volatility |

### Relationship columns (added by Step 5 Phase 3)

| Column | Type | Notes |
|--------|------|-------|
| cycle_key | int32 | Surrogate key matching cycle_dim |
| prev_key / prev_type | int32 / int8 | Previous sibling key and direction |
| prev_dur | int16 | Previous sibling duration in candles |
| prev_price_pct | float32 | Previous sibling price change |
| parent_key / parent_type | int32 / int8 | Primary parent cycle |
| order_in_parent | int16 | This cycle's index within parent |
| total_siblings | int16 | Total cycles under same parent |
| parent_progress_at_start / _end | float32 | 0–1 progress within parent at this cycle's start/end |
| parent_assign_rule | int8 | 0=contained, 1=by_start |
| boundary_type | int8 | 0=normal, 1=straddle, 2=transition_trigger |
| parent_prev_key / parent_prev_type | int32 / int8 | Previous adjacent parent |
| parent_next_key / parent_next_type | int32 / int8 | Next adjacent parent |
| overlap_prev_ratio / overlap_next_ratio | float32 | Overlap fraction with adjacent parents |
| n_up_4 | int8 | Count of UP cycles across 1w/1d/4h/1h at cycle start |
| combo_4 | string | Direction combo string for 1w/1d/4h/1h (e.g. `UUDU`) |
| child_count | int16 | Total child cycles (next lower TF) |
| child_up_count / child_down_count | int16 | Children by direction |
| opposite_child_ratio | float32 | Fraction of children opposing parent direction |
| max_opposite_child_streak | int8 | Longest consecutive run of opposing children |

**Important**: `end_date` stores the **start** of the last candle, not the end. Effective cycle end = `end_date + candle_duration`. This is handled by `_CANDLE_NS` in `context_builder.py`.

---

## Tier 3 — context/btc/

### cycle_dim.parquet

Surrogate-key dimension table. One row per cycle across all timeframes.

| Column | Type | Notes |
|--------|------|-------|
| cycle_key | int32 | Sequential integer key (0-based, sorted by TF order then start_date) |
| cycle_id | string | Original string ID |
| timeframe | string | TF label |
| cycle_type | int8 | 1=up, -1=down |
| start_date / end_date | datetime64[ms] | UTC-normalized |
| duration_candles | int32 | |
| category | string | |

TF distribution (~604k rows total): 1M=12 / 1w=46 / 1d=308 / 4h=1,843 / 1h=7,336 / 30m=14,894 / 15m=29,950 / 5m=92,138 / 1min=458,319.

### timeframe_context_1min.parquet and timeframe_context_1h.parquet

69-column per-timestamp snapshot of all 9 TF cycle states. 1min version has one row per minute; 1h version is the hourly subset.

**Columns (69 total)**:

| Group | Columns | Notes |
|-------|---------|-------|
| timestamp | timestamp | UTC minute/hour |
| TF state (×9) | `{tf}_key` int32 | Surrogate key of active cycle at this timestamp |
| | `{tf}_type` int8 | 1=up, -1=down, 0=gap |
| | `{tf}_time_prog` float16 | 0–1 elapsed time fraction within current cycle |
| | `{tf}_candle_prog` float16 | 0–1 elapsed candle fraction (uses raw CSV timestamps) |
| TF transition (×9) | `{tf}_changed` bool | True if cycle changed at this row |
| | `{tf}_chg_type` int8 | 0=none, 1=UP→DOWN, 2=DOWN→UP |
| Aggregates | `major_changed_any` bool | Any of 1w/1d/4h/1h changed |
| | `minor_changed_any` bool | Any of 30m/15m/5m/1min changed |
| | `n_up_4` int8 | UP count across 1w/1d/4h/1h (0–4) |
| | `n_up_8` int8 | UP count across 1w/1d/4h/1h/30m/15m/5m/1min |
| | `major_up_count` int8 | = n_up_4 |
| | `minor_up_count` int8 | UP count across 30m/15m/5m/1min |
| | `combo_4` string | Direction string for 1w/1d/4h/1h |
| | `combo_8` string | Direction string for 1w/1d/4h/1h/30m/15m/5m/1min |
| | `minor_combo_4` string | Direction string for 30m/15m/5m/1min |
| | `major_late_count` int8 | Count of major TFs with time_prog > 0.8 |
| Boundaries | `1h_is_boundary` bool | 1h cycle unchanged while a higher TF just changed |
| | `30m_is_boundary` bool | |
| | `15m_is_boundary` bool | |
| | `5m_is_boundary` bool | |

**n_up_4 distribution** (1min table): 0=349k / 1=1.31M / 2=1.73M / 3=956k / 4=202k.

### context_meta.json

```json
{
  "version": "2.0",
  "created_at": "...",
  "asset": "btc",
  "data_range": {"start": "2017-08-17 04:03:00", "end": "..."},
  "timeframes": [...],
  "files": {...},
  "boundary_handling": {...},
  "deprecated": {...}
}
```

---

## Tier 4 — dashboard/candles/

Dashboard-ready parquets (currently 1d / 4h / 1h). Columns include start/end OHLC aggregates, aggregate indicators, and shape features. Served by `src/dashboard_api/`.

---

## cycle_hierarchy_map.json (optional, Step 4)

Located at `cycles_enriched/btc/cycle_hierarchy_map.json`. 205 MB JSON mapping parent/child cycle IDs across adjacent timeframes. Produced by Step 4 (`cycle_time_mapper.py`). Used only for cross-validation in Step 5 Phase 4. Not required for normal operation.

---

## Schema Inspection

```powershell
python .\scripts\dev\data_schema_report.py --asset btc
python .\scripts\dev\data_schema_report.py --asset btc --json
```

The report shows: raw CSV column counts and time ranges, cycle parquet column counts with nested struct descriptions (no false duplicates), context file sizes / row counts / column listings, n_up_4 distributions, context_meta content.

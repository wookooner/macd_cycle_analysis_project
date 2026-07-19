# Metabase Infra

This folder holds the generic BI layer for the MACD cycle project.

## Current Architecture

- `postgres`: durable analytics store for Metabase.
- `metabase`: dashboard and ad-hoc exploration UI.
- `scripts/load_parquet_to_postgres.py`: append-only sync from canonical parquet files into Postgres.

The current design intentionally keeps responsibilities separate:

- `ai_analyst/`: domain validation and hypothesis testing.
- `infra/metabase/`: generic BI, counts, distributions, dashboards, ad-hoc exploration.

## Source of Truth

Raw data still lives in the canonical parquet datasets under `macd-cycle-data`.

Postgres is a BI serving layer, not a replacement source of truth.

## Raw Tables vs BI Views

The sync script loads raw parquet files into Postgres tables such as:

- `cycles_1min`
- `cycles_5m`
- `cycles_15m`
- `cycles_30m`
- `cycles_1h`
- `cycles_4h`
- `cycles_1d`
- `cycles_1w`
- `cycles_1m`
- `cycle_dim`
- `timeframe_context_1min`
- `timeframe_context_1h`

These raw tables keep the original payload, including large JSON-like fields such as:

- `candle_data`
- `cycle_features`

For Metabase, the preferred exploration surface is the derived view layer:

- `cycles_1min_bi`
- `cycles_5m_bi`
- `cycles_15m_bi`
- `cycles_30m_bi`
- `cycles_1h_bi`
- `cycles_4h_bi`
- `cycles_1d_bi`
- `cycles_1w_bi`
- `cycles_1m_bi`

The `_bi` views are intentionally thinner:

- they exclude `candle_data`
- they expose common cycle metadata directly
- they extract the most useful numeric values from `cycle_features`

This gives better Metabase usability without deleting or mutating the raw cycle tables.

## Update Strategy

The sync script is designed for long-term maintenance:

- append-only by default
- no `replace` of existing cycle tables
- detects the best incremental key from:
  - `cycle_key`
  - `start_date`
  - `end_date`
- stores sync state in `metabase_ingestion_state`
- creates or refreshes `_bi` views after each table sync
- creates lightweight indexes for common exploration columns

This means later refreshes only load missing trailing data when possible.

## Typical Workflow

1. Start the stack.

```powershell
cd .\infra\metabase
docker compose up -d
```

2. Sync parquet data into Postgres.

```powershell
python .\scripts\load_parquet_to_postgres.py
```

3. Open Metabase.

- `http://localhost:3000`

4. Add the analytics database in Metabase.

- Type: `PostgreSQL`
- Host: `postgres`
- Port: `5432`
- Database: `analytics`
- Username: `metabase`
- Password: `metabase`

5. In Metabase, prefer the `_bi` views over the raw `cycles_*` tables for exploration.

## Modeling Guidance

Recommended in Metabase:

- use `cycles_*_bi` for charts, filters, and dashboards
- hide `candle_data` and `cycle_features` in the raw tables if they clutter the UI
- keep raw tables available for audit and spot checks

## Notes

- The earlier DuckDB direct-to-Metabase experiment is no longer the active path.
- The active and supported path is `parquet -> Postgres -> Metabase`.

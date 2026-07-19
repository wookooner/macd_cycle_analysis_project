from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("MACD_DATA_ROOT", str(Path.home() / "macd-cycle-data")))

CYCLE_DIR = DATA_ROOT / "processed" / "cycles_enriched" / "btc"
CONTEXT_DIR = DATA_ROOT / "processed" / "context" / "btc"

POSTGRES_URL = os.getenv(
    "METABASE_ANALYTICS_DB_URL",
    "postgresql+psycopg://metabase:metabase@localhost:5432/analytics",
)

INCREMENTAL_KEY_PREFERENCE = ("cycle_key", "start_date", "end_date")
STATE_TABLE = "metabase_ingestion_state"
SKIP_PARQUET_STEMS = frozenset({
    "cycles_1min",
    "timeframe_context_1min",
})
BI_BASE_COLUMNS = (
    "cycle_id",
    "timeframe",
    "start_date",
    "end_date",
    "cycle_type",
    "duration_candles",
    "category",
    "algorithm_used",
    "cycle_key",
    "prev_key",
    "prev_type",
    "prev_dur",
    "prev_price_pct",
    "parent_key",
    "parent_type",
    "order_in_parent",
    "total_siblings",
    "parent_progress_at_start",
    "parent_progress_at_end",
    "parent_assign_rule",
    "boundary_type",
    "parent_prev_key",
    "parent_prev_type",
    "parent_next_key",
    "parent_next_type",
    "overlap_prev_ratio",
    "overlap_next_ratio",
    "n_up_4",
    "combo_4",
    "child_count",
    "child_up_count",
    "child_down_count",
    "opposite_child_ratio",
    "max_opposite_child_streak",
    "major_1h_key",
    "major_1h_type",
    "major_4h_key",
    "major_4h_type",
)
BI_FEATURE_JSON_PATHS = (
    ("feature_change_price_pct", ("change", "price_pct")),
    ("feature_change_ppo", ("change", "ppo")),
    ("feature_change_ppo_hist", ("change", "ppo_hist")),
    ("feature_change_cvd", ("change", "cvd")),
    ("feature_strength_direction_pct", ("strength", "direction_pct")),
    ("feature_volatility_avg_true_range", ("volatility", "avg_true_range")),
    ("feature_volatility_price_change_deviation", ("volatility", "price_change_deviation")),
    ("feature_shape_core_count", ("shape", "core_count")),
    ("feature_shape_direction_change", ("shape", "direction_change")),
    ("feature_shape_duration_candles", ("shape", "duration_candles")),
    ("feature_shape_noise_count", ("shape", "noise_count")),
    ("feature_shape_peak_price_position", ("shape", "peak_price_position")),
    ("feature_shape_trough_price_position", ("shape", "trough_price_position")),
    ("feature_aggregate_area_ppo_hist", ("aggregate", "area_ppo_hist")),
    ("feature_aggregate_cvd", ("aggregate", "cvd")),
    ("feature_aggregate_taker_buy_ratio", ("aggregate", "taker_buy_ratio")),
    ("feature_aggregate_volume", ("aggregate", "volume")),
    ("feature_start_price", ("start", "price")),
    ("feature_start_rsi", ("start", "rsi")),
    ("feature_start_ppo", ("start", "ppo")),
    ("feature_start_ppo_hist", ("start", "ppo_hist")),
    ("feature_start_cvd", ("start", "cvd")),
    ("feature_end_cvd", ("end", "cvd")),
    ("feature_end_ppo", ("end", "ppo")),
    ("feature_end_ppo_hist", ("end", "ppo_hist")),
)
INDEX_COLUMN_CANDIDATES = (
    "cycle_key",
    "start_date",
    "combo_4",
    "child_count",
    "parent_key",
    "timestamp",
)


def _summarize_exception_message(exc: Exception, limit: int = 240) -> str:
    message = str(exc).replace("\n", " ").replace("\r", " ")
    if "[SQL:" in message:
        message = message.split("[SQL:", 1)[0].strip()
    if "[parameters:" in message:
        message = message.split("[parameters:", 1)[0].strip()
    if len(message) > limit:
        message = message[: limit - 3].rstrip() + "..."
    return message


def _ensure_driver() -> None:
    try:
        import psycopg  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "psycopg is required for Postgres loading. "
            "Install it in the active environment with: pip install psycopg[binary]"
        ) from exc


def _json_ready(value: Any):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _is_nested_like(value: Any) -> bool:
    return isinstance(value, (dict, list, tuple, set, np.ndarray, np.generic))


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for column in normalized.columns:
        series = normalized[column]
        non_null = series.dropna()
        if non_null.empty:
            continue

        sample = non_null.iloc[0]
        if _is_nested_like(sample):
            normalized[column] = series.map(
                lambda value: None if value is None else json.dumps(_json_ready(value), ensure_ascii=False)
            )

    normalized = normalized.astype(object).where(pd.notna(normalized), None)

    return normalized


def _ensure_state_table(engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
        table_name TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        incremental_key TEXT,
        last_loaded_value TEXT,
        loaded_rows BIGINT NOT NULL DEFAULT 0,
        source_modified_at TIMESTAMP NULL,
        last_synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _table_exists(engine, table_name: str) -> bool:
    inspector = inspect(engine)
    return inspector.has_table(table_name)


def _get_table_columns(engine, table_name: str) -> list[str]:
    inspector = inspect(engine)
    return [column["name"] for column in inspector.get_columns(table_name)]


def _detect_incremental_key(columns: list[str]) -> str | None:
    for candidate in INCREMENTAL_KEY_PREFERENCE:
        if candidate in columns:
            return candidate
    return None


def _get_existing_max(engine, table_name: str, key_column: str):
    query = text(f'SELECT MAX("{key_column}") FROM "{table_name}"')
    with engine.begin() as conn:
        return conn.execute(query).scalar()


def _coerce_incremental_value(existing_max, sample_value):
    if existing_max is None or sample_value is None:
        return existing_max

    if isinstance(sample_value, np.generic):
        sample_value = sample_value.item()

    if isinstance(sample_value, (int, np.integer)):
        return int(existing_max)
    if isinstance(sample_value, (float, np.floating)):
        return float(existing_max)
    if isinstance(sample_value, (pd.Timestamp,)):
        return pd.Timestamp(existing_max).to_pydatetime()
    return existing_max


def _prepare_scanner(parquet_path: Path, key_column: str | None, existing_max, sample_value=None):
    dataset = ds.dataset(str(parquet_path), format="parquet")
    if key_column and existing_max is not None:
        coerced_max = _coerce_incremental_value(existing_max, sample_value)
        return dataset.scanner(filter=(ds.field(key_column) > coerced_max))
    return dataset.scanner()


def _create_table_if_needed(engine, table_name: str, df: pd.DataFrame) -> None:
    if _table_exists(engine, table_name):
        return
    df.head(0).to_sql(table_name, engine, if_exists="fail", index=False)


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _jsonb_numeric_expr(path: tuple[str, ...]) -> str:
    expr = "cycle_features_json"
    for key in path[:-1]:
        expr += f" -> '{key}'"
    expr += f" ->> '{path[-1]}'"
    return f"NULLIF({expr}, '')::DOUBLE PRECISION"


def _ensure_indexes(engine, table_name: str, columns: list[str]) -> None:
    with engine.begin() as conn:
        for column in INDEX_COLUMN_CANDIDATES:
            if column not in columns:
                continue
            index_name = f"{table_name}_{column}_idx"
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {_quote_ident(index_name)} "
                    f"ON {_quote_ident(table_name)} ({_quote_ident(column)})"
                )
            )


def _create_or_replace_cycle_bi_view(engine, table_name: str, columns: list[str]) -> str | None:
    if not table_name.startswith("cycles_"):
        return None

    select_lines: list[str] = []
    for column in BI_BASE_COLUMNS:
        if column in columns:
            select_lines.append(f"    {_quote_ident(column)}")

    if "cycle_features" in columns:
        for alias, path in BI_FEATURE_JSON_PATHS:
            select_lines.append(f"    {_jsonb_numeric_expr(path)} AS {_quote_ident(alias)}")

    if not select_lines:
        return None

    view_name = f"{table_name}_bi"
    cycle_features_json = (
        "CASE "
        "WHEN cycle_features IS NULL OR cycle_features = '' THEN NULL::jsonb "
        "ELSE cycle_features::jsonb "
        "END AS cycle_features_json"
    )
    select_sql = ",\n".join(select_lines)
    sql = f"""
    CREATE OR REPLACE VIEW {_quote_ident(view_name)} AS
    WITH base AS (
        SELECT
            *,
            {cycle_features_json}
        FROM {_quote_ident(table_name)}
    )
    SELECT
{select_sql}
    FROM base
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
    return view_name


def _record_state(
    engine,
    *,
    table_name: str,
    source_path: Path,
    incremental_key: str | None,
    last_loaded_value,
    loaded_rows: int,
) -> None:
    statement = text(
        f"""
        INSERT INTO {STATE_TABLE}
            (table_name, source_path, incremental_key, last_loaded_value, loaded_rows, source_modified_at, last_synced_at)
        VALUES
            (:table_name, :source_path, :incremental_key, :last_loaded_value, :loaded_rows, :source_modified_at, CURRENT_TIMESTAMP)
        ON CONFLICT (table_name) DO UPDATE SET
            source_path = EXCLUDED.source_path,
            incremental_key = EXCLUDED.incremental_key,
            last_loaded_value = EXCLUDED.last_loaded_value,
            loaded_rows = EXCLUDED.loaded_rows,
            source_modified_at = EXCLUDED.source_modified_at,
            last_synced_at = CURRENT_TIMESTAMP
        """
    )
    with engine.begin() as conn:
        conn.execute(
            statement,
            {
                "table_name": table_name,
                "source_path": str(source_path),
                "incremental_key": incremental_key,
                "last_loaded_value": None if last_loaded_value is None else str(last_loaded_value),
                "loaded_rows": loaded_rows,
                "source_modified_at": pd.Timestamp.fromtimestamp(source_path.stat().st_mtime).to_pydatetime(),
            },
        )


def _psql_copy_method(table, conn, keys, data_iter) -> None:
    """Bulk COPY FROM STDIN for pandas to_sql via psycopg3."""
    dbapi_conn = conn.connection.driver_connection
    columns_sql = ", ".join(_quote_ident(column) for column in keys)
    table_ident = _quote_ident(table.name)
    sql = f"COPY {table_ident} ({columns_sql}) FROM STDIN"

    with dbapi_conn.cursor() as cur:
        with cur.copy(sql) as copy:
            for row in data_iter:
                copy.write_row(row)


def _load_parquet_directory(engine, directory: Path) -> list[str]:
    loaded_tables: list[str] = []
    if not directory.exists():
        return loaded_tables

    for parquet_path in sorted(directory.glob("*.parquet")):
        table_name = parquet_path.stem.lower()
        if table_name in SKIP_PARQUET_STEMS:
            print(f"Skipping {parquet_path.name} (in SKIP_PARQUET_STEMS)")
            continue
        print(f"Syncing {parquet_path.name} -> {table_name}")
        try:
            preview_df = pd.read_parquet(parquet_path, engine="pyarrow").head(5)
            if preview_df.empty:
                print("  - skipped: empty parquet")
                continue

            key_column = _detect_incremental_key(preview_df.columns.tolist())
            existing_max = _get_existing_max(engine, table_name, key_column) if key_column and _table_exists(engine, table_name) else None

            if key_column and existing_max is not None:
                print(f"  - incremental key: {key_column}, existing max: {existing_max}")
            elif key_column:
                print(f"  - incremental key: {key_column}, initial load")
            else:
                print("  - no incremental key found, loading full file")

            sample_value = preview_df[key_column].dropna().iloc[0] if key_column and key_column in preview_df.columns and not preview_df[key_column].dropna().empty else None
            scanner = _prepare_scanner(parquet_path, key_column, existing_max, sample_value=sample_value)
            total_loaded = 0
            last_loaded_value = existing_max
            table_created = _table_exists(engine, table_name)

            for batch in scanner.to_batches():
                df = batch.to_pandas()
                if df.empty:
                    continue

                df = _normalize_dataframe(df)
                if not table_created:
                    _create_table_if_needed(engine, table_name, df)
                    table_created = True

                df.to_sql(table_name, engine, if_exists="append", index=False, method=_psql_copy_method)
                total_loaded += len(df)

                if key_column and key_column in df.columns:
                    last_loaded_value = df[key_column].max()

            _record_state(
                engine,
                table_name=table_name,
                source_path=parquet_path,
                incremental_key=key_column,
                last_loaded_value=last_loaded_value,
                loaded_rows=total_loaded,
            )

            if total_loaded == 0:
                print("  - no new rows detected")
            else:
                print(f"  - loaded {total_loaded} new rows")

            actual_columns = _get_table_columns(engine, table_name)
            _ensure_indexes(engine, table_name, actual_columns)
            view_name = _create_or_replace_cycle_bi_view(engine, table_name, actual_columns)
            if view_name:
                print(f"  - refreshed BI view: {view_name}")

            loaded_tables.append(table_name)
        except (ValueError, TypeError, SQLAlchemyError) as exc:
            summary = _summarize_exception_message(exc)
            print(f"  - failed: {type(exc).__name__}: {summary}")
            raise RuntimeError(
                f"Failed while syncing {parquet_path.name} -> {table_name}. "
                f"{type(exc).__name__}: {summary}"
            ) from exc

    return loaded_tables


def main() -> None:
    _ensure_driver()

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"MACD data root not found: {DATA_ROOT}")

    engine = create_engine(POSTGRES_URL)
    _ensure_state_table(engine)

    cycle_tables = _load_parquet_directory(engine, CYCLE_DIR)
    context_tables = _load_parquet_directory(engine, CONTEXT_DIR)

    print("")
    print("Cycle tables checked:")
    for table_name in cycle_tables:
        print(f"  - {table_name}")

    print("")
    print("Context tables checked:")
    for table_name in context_tables:
        print(f"  - {table_name}")

    print("")
    print("Next steps:")
    print("  1. Open Metabase at http://localhost:3000")
    print("  2. Add a new database connection of type PostgreSQL")
    print("  3. In Metabase, use host postgres, port 5432, database analytics")
    print("  4. Login with user metabase / password metabase")


if __name__ == "__main__":
    main()

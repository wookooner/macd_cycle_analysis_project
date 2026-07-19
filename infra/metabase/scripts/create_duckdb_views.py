from __future__ import annotations

from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METABASE_ROOT = PROJECT_ROOT / "infra" / "metabase"
DB_PATH = METABASE_ROOT / "volumes" / "duckdb-data" / "analytics.duckdb"

# Host-side data root used only to discover which parquet files exist.
HOST_DATA_ROOT = Path("C:/Users/qw370/macd-cycle-data")

# Container-side mount path that Metabase/DuckDB must use inside view SQL.
CONTAINER_DATA_ROOT = "/source-data"

CYCLE_DIR = HOST_DATA_ROOT / "processed" / "cycles_enriched" / "btc"
CONTEXT_DIR = HOST_DATA_ROOT / "processed" / "context" / "btc"


def container_path(host_path: Path) -> str:
    relative = host_path.relative_to(HOST_DATA_ROOT).as_posix()
    return f"{CONTAINER_DATA_ROOT}/{relative}"


def build_cycle_view_statements() -> list[tuple[str, str]]:
    statements: list[tuple[str, str]] = []
    for parquet_path in sorted(CYCLE_DIR.glob("cycles_*.parquet")):
        view_name = parquet_path.stem
        sql = (
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM read_parquet('{container_path(parquet_path)}');"
        )
        statements.append((view_name, sql))
    return statements


def build_context_view_statements() -> list[tuple[str, str]]:
    statements: list[tuple[str, str]] = []
    for parquet_path in sorted(CONTEXT_DIR.glob("*.parquet")):
        view_name = parquet_path.stem
        sql = (
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM read_parquet('{container_path(parquet_path)}');"
        )
        statements.append((view_name, sql))
    return statements


def main() -> None:
    if not HOST_DATA_ROOT.exists():
        raise FileNotFoundError(f"Host data root not found: {HOST_DATA_ROOT}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    cycle_statements = build_cycle_view_statements()
    context_statements = build_context_view_statements()

    if not cycle_statements:
        raise FileNotFoundError(f"No cycle parquet files found under: {CYCLE_DIR}")

    con = duckdb.connect(str(DB_PATH))
    try:
        for _, sql in cycle_statements + context_statements:
            con.execute(sql)
    finally:
        con.close()

    print(f"Created/updated DuckDB views in: {DB_PATH}")
    print("")
    print("Cycle views:")
    for view_name, _ in cycle_statements:
        print(f"  - {view_name}")

    print("")
    print("Context views:")
    for view_name, _ in context_statements:
        print(f"  - {view_name}")

    print("")
    print("Next steps:")
    print("  1. Restart or keep the Metabase container running.")
    print("  2. In Metabase, add DuckDB and use file path: /data/analytics.duckdb")
    print("  3. Confirm that the views above appear as queryable tables.")


if __name__ == "__main__":
    main()

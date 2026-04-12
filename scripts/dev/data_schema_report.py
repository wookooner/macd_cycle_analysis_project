from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS


TIMEFRAME_ORDER = ["1M", "1w", "1d", "4h", "1h", "30m", "15m", "5m", "1min"]
IGNORED_FILE_MARKERS = (".tmp", ".recovery")
CORE_PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}
RELATION_COLUMNS = {"parent_cycle_ids", "child_cycle_ids"}
NESTED_COLUMNS = {"cycle_features", "candle_data", "parent_cycle_ids", "child_cycle_ids"}
INDICATOR_KEYWORDS = (
    "macd",
    "ppo",
    "rsi",
    "cvd",
    "funding",
    "fr_",
    "ma_",
    "ema",
    "atr",
    "oi",
    "ratio",
    "hist",
    "delta",
    "volatility",
    "strength_",
    "aggregate_",
    "change_",
    "start_",
    "end_",
)


def _looks_temporary(path: Path) -> bool:
    return any(marker in path.name for marker in IGNORED_FILE_MARKERS)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_PATHS.data_root))
    except ValueError:
        return str(path.resolve())


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _infer_text_value_type(value: str) -> str:
    if value == "":
        return "empty"
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return "bool"
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return "str"
    return type(parsed).__name__


def _read_csv_header_and_first_row(path: Path) -> tuple[list[str], dict[str, Any] | None]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), next(reader, None)


def _read_last_csv_row(path: Path, columns: list[str], max_bytes: int = 1024 * 256) -> dict[str, Any] | None:
    file_size = path.stat().st_size
    if file_size == 0 or not columns:
        return None

    with path.open("rb") as handle:
        read_size = min(max_bytes, file_size)
        handle.seek(-read_size, 2)
        chunk = handle.read(read_size).decode("utf-8", errors="ignore")

    lines = [line for line in chunk.splitlines() if line.strip()]
    if not lines:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            tail = deque(reader, maxlen=1)
            return tail[-1] if tail else None

    last_line = lines[-1]
    values = next(csv.reader([last_line]), None)
    if values is None:
        return None
    if len(values) < len(columns):
        values = values + [""] * (len(columns) - len(values))
    return dict(zip(columns, values))


def _extract_time_candidates(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    candidates = {}
    for key in ("date", "timestamp", "start_date", "end_date", "unix"):
        if key in row:
            candidates[key] = row[key]
    return candidates


def _detect_indicator_columns(columns: list[str]) -> list[str]:
    seen: list[str] = []
    for column in columns:
        lower = column.lower()
        if any(keyword in lower for keyword in INDICATOR_KEYWORDS):
            seen.append(column)
    return seen


def _classify_market_file(path: Path) -> dict[str, str]:
    stem = path.stem
    upper = stem.upper()

    if upper.startswith("BTCUSDT_"):
        suffix = stem[len("BTCUSDT_") :]
        if suffix.endswith("minutes"):
            tf = suffix.replace("minutes", "m")
            return {"asset": "BTCUSDT", "dataset_type": "price", "timeframe": tf}
        return {"asset": "BTCUSDT", "dataset_type": "auxiliary", "timeframe": suffix}

    if upper.startswith("BTCUSD_"):
        suffix = stem[len("BTCUSD_") :]
        return {
            "asset": "BTCUSD",
            "dataset_type": "price" if suffix in {"1min", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M", "1m_intraday"} else "auxiliary",
            "timeframe": suffix,
        }

    return {"asset": "unknown", "dataset_type": "unknown", "timeframe": stem}


def _parquet_schema_summary(path: Path) -> dict[str, Any]:
    if pq is None:
        df = pd.read_parquet(path).head(1)
        columns = list(df.columns)
        return {
            "row_count": None,
            "columns": columns,
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "duplicate_columns": [name for name, count in Counter(columns).items() if count > 1],
        }

    pf = pq.ParquetFile(path)
    schema = pf.schema
    columns = schema.names
    dtypes = {name: str(schema.column(i).physical_type) for i, name in enumerate(columns)}
    return {
        "row_count": pf.metadata.num_rows,
        "columns": columns,
        "dtypes": dtypes,
        "duplicate_columns": [name for name, count in Counter(columns).items() if count > 1],
    }


def _describe_nested_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "dict", "keys": list(value.keys())[:20]}
    if isinstance(value, (list, tuple)):
        first = value[0] if value else None
        info: dict[str, Any] = {
            "type": type(value).__name__,
            "length": len(value),
            "first_item_type": type(first).__name__ if first is not None else None,
        }
        if isinstance(first, dict):
            info["first_item_keys"] = list(first.keys())[:20]
        return info
    if hasattr(value, "dtype") and hasattr(value, "tolist"):
        seq = value.tolist()
        return _describe_nested_value(seq)
    return {"type": type(value).__name__}


def _top_level_json_keys(path: Path, limit: int = 50, chunk_size: int = 1024 * 1024) -> list[str]:
    keys: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    capture = False
    current = []
    pending_string: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            for ch in chunk:
                if in_string:
                    if escaped:
                        if capture:
                            current.append(ch)
                        escaped = False
                        continue
                    if ch == "\\":
                        escaped = True
                        if capture:
                            current.append(ch)
                        continue
                    if ch == '"':
                        in_string = False
                        if capture and depth == 1:
                            pending_string = "".join(current)
                        capture = False
                        current = []
                        continue
                    if capture:
                        current.append(ch)
                    continue

                if pending_string is not None and not ch.isspace():
                    if ch == ":":
                        keys.append(pending_string)
                        if len(keys) >= limit:
                            return keys
                    pending_string = None

                if ch == '"':
                    in_string = True
                    capture = depth == 1
                    current = []
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1

    return keys


def _read_parquet_sample(path: Path, columns: list[str]) -> pd.DataFrame:
    if pq is not None:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1, columns=columns):
            return batch.to_pandas()
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=columns).head(1)


def summarize_raw_market() -> dict[str, Any]:
    root = PROJECT_PATHS.raw_market_dir
    files = [path for path in sorted(root.glob("*.csv")) if not _looks_temporary(path)]
    summary: dict[str, Any] = {
        "root": str(root),
        "exists": root.exists(),
        "file_count": len(files),
        "price_files": [],
        "auxiliary_files": [],
    }

    for path in files:
        info = _classify_market_file(path)
        columns, first_row = _read_csv_header_and_first_row(path)
        last_row = _read_last_csv_row(path, columns=columns)
        record = {
            "path": _safe_rel(path),
            "asset": info["asset"],
            "timeframe": info["timeframe"],
            "columns": columns,
            "column_count": len(columns),
            "inferred_dtypes": {column: _infer_text_value_type(first_row.get(column, "") if first_row else "") for column in columns},
            "time_fields": {
                "first": _extract_time_candidates(first_row),
                "last": _extract_time_candidates(last_row),
            },
            "has_core_price_columns": sorted(CORE_PRICE_COLUMNS.intersection(columns)),
            "indicator_columns": _detect_indicator_columns(columns),
        }
        bucket = "price_files" if info["dataset_type"] == "price" else "auxiliary_files"
        summary[bucket].append(record)

    return summary


def summarize_cycle_parquet_dir(root: Path, asset: str | None, label: str) -> dict[str, Any]:
    scan_root = root / asset if asset and (root / asset).exists() else root
    files = sorted(scan_root.glob("cycles_*.parquet"))

    datasets: list[dict[str, Any]] = []
    for path in files:
        if _looks_temporary(path):
            continue

        schema_summary = _parquet_schema_summary(path)
        sample_columns = [column for column in ("cycle_id", "timeframe", "start_date", "end_date", *sorted(NESTED_COLUMNS)) if column in schema_summary["columns"]]
        sample = _read_parquet_sample(path, sample_columns)
        row = sample.iloc[0].to_dict() if not sample.empty else {}
        nested = {}
        for column in NESTED_COLUMNS.intersection(sample.columns):
            value = row.get(column)
            nested[column] = _describe_nested_value(value) if value is not None else {"type": "null"}

        datasets.append(
            {
                "path": _safe_rel(path),
                "timeframe_from_name": path.stem.replace("cycles_", ""),
                "row_count": schema_summary["row_count"],
                "column_count": len(schema_summary["columns"]),
                "columns": schema_summary["columns"],
                "duplicate_columns": schema_summary["duplicate_columns"],
                "indicator_columns": _detect_indicator_columns(schema_summary["columns"]),
                "relation_columns": sorted(RELATION_COLUMNS.intersection(schema_summary["columns"])),
                "nested_columns": nested,
                "timeframe_values": sorted(sample["timeframe"].dropna().astype(str).unique().tolist()) if "timeframe" in sample.columns else [],
                "date_fields": {
                    "start_date": str(row.get("start_date")) if row.get("start_date") is not None else None,
                    "end_date": str(row.get("end_date")) if row.get("end_date") is not None else None,
                },
                "cycle_id_example": row.get("cycle_id"),
            }
        )

    return {
        "label": label,
        "root": str(scan_root),
        "exists": scan_root.exists(),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def summarize_hierarchy_maps(asset: str) -> dict[str, Any]:
    candidates = [
        PROJECT_PATHS.processed_cycles_enriched_dir / asset / "cycle_hierarchy_map.json",
        PROJECT_PATHS.processed_cycles_enriched_dir / "cycle_hierarchy_map.json",
        PROJECT_PATHS.raw_hierarchy_dir / f"{asset}_cycle_hierarchy_map.json",
    ]
    maps: list[dict[str, Any]] = []

    for path in candidates:
        if not path.exists():
            continue
        maps.append(
            {
                "path": _safe_rel(path),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "top_level_keys": _top_level_json_keys(path),
            }
        )

    return {"maps": maps, "count": len(maps)}


def summarize_json_meta_dir(root: Path, label: str) -> dict[str, Any]:
    files = sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}])
    records: list[dict[str, Any]] = []

    for path in files:
        top_level_keys: list[str] = []
        try:
            top_level_keys = _top_level_json_keys(path)
        except Exception:
            top_level_keys = []
        records.append(
            {
                "path": _safe_rel(path),
                "size_kb": round(path.stat().st_size / 1024, 2),
                "top_level_keys": top_level_keys,
            }
        )

    return {
        "label": label,
        "root": str(root),
        "exists": root.exists(),
        "file_count": len(records),
        "files": records,
    }


def summarize_dashboard() -> dict[str, Any]:
    root = PROJECT_PATHS.dashboard_root
    candles = sorted((root / "candles").glob("*.parquet")) if (root / "candles").exists() else []
    candle_records: list[dict[str, Any]] = []

    for path in candles:
        schema_summary = _parquet_schema_summary(path)
        candle_records.append(
            {
                "path": _safe_rel(path),
                "row_count": schema_summary["row_count"],
                "column_count": len(schema_summary["columns"]),
                "columns": schema_summary["columns"],
                "indicator_columns": _detect_indicator_columns(schema_summary["columns"]),
            }
        )

    meta_summary = summarize_json_meta_dir(root / "meta", label="dashboard_meta")
    return {
        "root": str(root),
        "exists": root.exists(),
        "candles": candle_records,
        "meta": meta_summary,
    }


def build_report(asset: str) -> dict[str, Any]:
    return {
        "paths": PROJECT_PATHS.summary(),
        "asset": asset,
        "raw_market": summarize_raw_market(),
        "processed_cycles_enriched": summarize_cycle_parquet_dir(
            PROJECT_PATHS.processed_cycles_enriched_dir,
            asset=asset,
            label="processed_cycles_enriched",
        ),
        "processed_cycles_root_copy": summarize_cycle_parquet_dir(
            PROJECT_PATHS.processed_cycles_enriched_dir,
            asset=None,
            label="processed_cycles_root_copy",
        ),
        "processed_cycles_base": summarize_cycle_parquet_dir(
            PROJECT_PATHS.processed_cycles_base_dir,
            asset=asset if (PROJECT_PATHS.processed_cycles_base_dir / asset).exists() else None,
            label="processed_cycles_base",
        ),
        "hierarchy_maps": summarize_hierarchy_maps(asset),
        "processed_features_meta": summarize_json_meta_dir(PROJECT_PATHS.processed_features_dir, label="processed_features"),
        "dashboard": summarize_dashboard(),
    }


def print_report(report: dict[str, Any]) -> None:
    print("DATA SCHEMA REPORT")
    print("=" * 100)
    print(f"asset        : {report['asset']}")
    print(f"project_root : {report['paths']['project_root']}")
    print(f"data_root    : {report['paths']['data_root']}")
    print(f"processed    : {report['paths']['processed_root']}")
    print(f"dashboard    : {report['paths']['dashboard_root']}")

    raw = report["raw_market"]
    print("\n" + "=" * 100)
    print("RAW MARKET")
    print(f"root        : {raw['root']}")
    print(f"file_count   : {raw['file_count']}")
    print(f"price_files  : {len(raw['price_files'])}")
    print(f"aux_files    : {len(raw['auxiliary_files'])}")
    for item in raw["price_files"][:12]:
        print(f"  - {item['timeframe']:>8} | {item['path']}")
        print(f"    columns[{item['column_count']}]: {', '.join(item['columns'])}")
        if item["indicator_columns"]:
            print(f"    indicators: {', '.join(item['indicator_columns'][:20])}")
        print(f"    time: first={item['time_fields']['first']} last={item['time_fields']['last']}")
    if raw["auxiliary_files"]:
        print("  auxiliary datasets:")
        for item in raw["auxiliary_files"][:12]:
            print(f"    - {item['timeframe']} | {item['path']}")

    for section_name in ("processed_cycles_enriched", "processed_cycles_root_copy", "processed_cycles_base"):
        section = report[section_name]
        print("\n" + "=" * 100)
        print(section["label"].upper())
        print(f"root          : {section['root']}")
        print(f"dataset_count : {section['dataset_count']}")
        for item in section["datasets"]:
            print(f"  - {item['timeframe_from_name']:>8} | rows={item['row_count']} | {item['path']}")
            print(f"    columns[{item['column_count']}]: {', '.join(item['columns'][:18])}" + (" ..." if item["column_count"] > 18 else ""))
            if item["duplicate_columns"]:
                print(f"    duplicate_columns: {', '.join(item['duplicate_columns'])}")
            if item["indicator_columns"]:
                print(f"    indicators: {', '.join(item['indicator_columns'][:20])}")
            if item["relation_columns"]:
                print(f"    relations: {', '.join(item['relation_columns'])}")
            if item["nested_columns"]:
                print(f"    nested: {_safe_json(item['nested_columns'])}")
            print(f"    timeframe_values: {item['timeframe_values']} | cycle_id_example: {item['cycle_id_example']}")
            print(f"    date_fields: {item['date_fields']}")

    hierarchy = report["hierarchy_maps"]
    print("\n" + "=" * 100)
    print("HIERARCHY MAPS")
    print(f"map_count: {hierarchy['count']}")
    for item in hierarchy["maps"]:
        print(f"  - {item['path']} ({item['size_mb']} MB)")
        print(f"    top_level_keys: {item['top_level_keys']}")

    for section_name in ("processed_features_meta",):
        section = report[section_name]
        print("\n" + "=" * 100)
        print(section["label"].upper())
        print(f"root      : {section['root']}")
        print(f"file_count: {section['file_count']}")
        for item in section["files"][:20]:
            print(f"  - {item['path']} | keys={item['top_level_keys']}")

    dashboard = report["dashboard"]
    print("\n" + "=" * 100)
    print("DASHBOARD")
    print(f"root           : {dashboard['root']}")
    print(f"candle_datasets: {len(dashboard['candles'])}")
    for item in dashboard["candles"]:
        print(f"  - {item['path']} | rows={item['row_count']} | columns[{item['column_count']}]")
        print(f"    indicators: {', '.join(item['indicator_columns'][:20])}")
    print(f"meta_files     : {dashboard['meta']['file_count']}")
    for item in dashboard["meta"]["files"][:20]:
        print(f"  - {item['path']} | keys={item['top_level_keys']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the current data ecosystem without printing full datasets. "
            "It reports raw source files, parquet schemas, cycle structures, hierarchy maps, and dashboard metadata."
        )
    )
    parser.add_argument("--asset", default="btc", help="Asset namespace to inspect under processed cycle directories. Default: btc")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of the text report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(asset=args.asset)
    if args.json:
        print(_safe_json(report))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

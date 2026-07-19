from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _arrow_type_str(t: Any) -> str:
    """Return a compact string for an Arrow type, expanding struct/list one level."""
    import pyarrow as pa
    if pa.types.is_struct(t):
        fields = ", ".join(f.name for f in t)
        return f"struct<{fields}>"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        vt = t.value_type
        if pa.types.is_struct(vt):
            fields = ", ".join(f.name for f in vt)
            return f"list<struct<{fields}>>"
        return f"list<{vt}>"
    return str(t)


def _parquet_schema_summary(path: Path) -> dict[str, Any]:
    if pq is None:
        df = pd.read_parquet(path).head(1)
        columns = list(df.columns)
        return {
            "row_count": None,
            "columns": columns,
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "duplicate_columns": [],
        }

    pf = pq.ParquetFile(path)
    # Use Arrow schema so nested structs/lists appear as single logical columns,
    # not as flattened physical leaf paths (which caused false duplicate warnings).
    arrow_schema = pf.schema_arrow
    columns = arrow_schema.names
    dtypes = {field.name: _arrow_type_str(field.type) for field in arrow_schema}
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


def _process_market_file(path: Path) -> tuple[str, dict[str, Any]]:
    """Read one CSV market file; returns (bucket, record). Used for parallel execution."""
    info = _classify_market_file(path)
    columns, first_row = _read_csv_header_and_first_row(path)
    last_row = _read_last_csv_row(path, columns=columns)
    record = {
        "path": _safe_rel(path),
        "asset": info["asset"],
        "timeframe": info["timeframe"],
        "columns": columns,
        "column_count": len(columns),
        "inferred_dtypes": {col: _infer_text_value_type(first_row.get(col, "") if first_row else "") for col in columns},
        "time_fields": {
            "first": _extract_time_candidates(first_row),
            "last": _extract_time_candidates(last_row),
        },
        "has_core_price_columns": sorted(CORE_PRICE_COLUMNS.intersection(columns)),
        "indicator_columns": _detect_indicator_columns(columns),
    }
    bucket = "price_files" if info["dataset_type"] == "price" else "auxiliary_files"
    return bucket, record


def summarize_raw_market() -> dict[str, Any]:
    root = PROJECT_PATHS.raw_market_dir
    files = [path for path in sorted(root.glob("*.csv")) if not _looks_temporary(path)]
    price_files: list[dict[str, Any]] = []
    auxiliary_files: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process_market_file, path): path for path in files}
        for fut in as_completed(futures):
            try:
                bucket, record = fut.result()
                (price_files if bucket == "price_files" else auxiliary_files).append(record)
            except Exception:
                pass

    # sort by timeframe order for consistent output
    tf_rank = {tf: i for i, tf in enumerate(TIMEFRAME_ORDER)}
    price_files.sort(key=lambda r: tf_rank.get(r["timeframe"], 99))

    return {
        "root": str(root),
        "exists": root.exists(),
        "file_count": len(files),
        "price_files": price_files,
        "auxiliary_files": auxiliary_files,
    }


def _process_cycle_parquet(path: Path) -> dict[str, Any]:
    """Read one cycle parquet file; returns a record dict. Used for parallel execution."""
    schema_summary = _parquet_schema_summary(path)
    all_cols = schema_summary["columns"]
    dtypes = schema_summary["dtypes"]

    # Describe nested columns from schema (no row read needed for struct/list cols)
    nested: dict[str, Any] = {}
    for col in NESTED_COLUMNS.intersection(all_cols):
        dtype_str = dtypes.get(col, "")
        nested[col] = {"type": dtype_str}

    # Read a minimal sample only for scalar fields we actually need
    scalar_cols = [c for c in ("cycle_id", "timeframe", "start_date", "end_date") if c in all_cols]
    sample = _read_parquet_sample(path, scalar_cols)
    row = sample.iloc[0].to_dict() if not sample.empty else {}

    return {
        "path": _safe_rel(path),
        "timeframe_from_name": path.stem.replace("cycles_", ""),
        "row_count": schema_summary["row_count"],
        "column_count": len(all_cols),
        "columns": all_cols,
        "duplicate_columns": schema_summary["duplicate_columns"],
        "indicator_columns": _detect_indicator_columns(all_cols),
        "relation_columns": sorted(RELATION_COLUMNS.intersection(all_cols)),
        "nested_columns": nested,
        "timeframe_values": sorted(sample["timeframe"].dropna().astype(str).unique().tolist()) if "timeframe" in sample.columns else [],
        "date_fields": {
            "start_date": str(row.get("start_date")) if row.get("start_date") is not None else None,
            "end_date": str(row.get("end_date")) if row.get("end_date") is not None else None,
        },
        "cycle_id_example": row.get("cycle_id"),
    }


def summarize_cycle_parquet_dir(root: Path, asset: str | None, label: str) -> dict[str, Any]:
    scan_root = root / asset if asset and (root / asset).exists() else root
    files = [path for path in sorted(scan_root.glob("cycles_*.parquet")) if not _looks_temporary(path)]

    datasets: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(len(files), 9) or 1) as pool:
        future_map = {pool.submit(_process_cycle_parquet, path): path for path in files}
        results: dict[Path, dict[str, Any]] = {}
        for fut in as_completed(future_map):
            try:
                results[future_map[fut]] = fut.result()
            except Exception:
                pass
    # preserve original sorted order
    datasets = [results[path] for path in files if path in results]

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


def summarize_context(asset: str) -> dict[str, Any]:
    """Summarise context files built by CycleContextBuilder (Phase 1-5)."""
    ctx_dir = PROJECT_PATHS.context_dir(asset)
    result: dict[str, Any] = {
        "root": str(ctx_dir),
        "exists": ctx_dir.exists(),
        "cycle_dim": None,
        "timeframe_context_1min": None,
        "timeframe_context_1h": None,
        "context_meta": None,
    }
    if not ctx_dir.exists():
        return result

    # cycle_dim
    dim_path = ctx_dir / "cycle_dim.parquet"
    if dim_path.exists():
        schema = _parquet_schema_summary(dim_path)
        tf_dist: dict[str, int] = {}
        try:
            df_tf = pd.read_parquet(dim_path, columns=["timeframe"])
            tf_dist = df_tf["timeframe"].astype(str).value_counts().to_dict()
        except Exception:
            pass
        result["cycle_dim"] = {
            "path": _safe_rel(dim_path),
            "size_mb": round(dim_path.stat().st_size / 1024 / 1024, 2),
            "row_count": schema["row_count"],
            "column_count": len(schema["columns"]),
            "columns": schema["columns"],
            "timeframe_distribution": {tf: tf_dist.get(tf, 0) for tf in TIMEFRAME_ORDER if tf in tf_dist},
        }

    # timeframe_context files
    for freq in ("1min", "1h"):
        ctx_path = ctx_dir / f"timeframe_context_{freq}.parquet"
        key = f"timeframe_context_{freq}"
        if not ctx_path.exists():
            continue
        schema = _parquet_schema_summary(ctx_path)
        sample_info: dict[str, Any] = {}
        try:
            needed = [c for c in ("timestamp", "n_up_4", "n_up_8", "combo_4") if c in schema["columns"]]
            df_sample = pd.read_parquet(ctx_path, columns=needed).head(3)
            if not df_sample.empty:
                sample_info["first_timestamp"] = str(df_sample["timestamp"].iloc[0]) if "timestamp" in df_sample.columns else None
                if "n_up_4" in df_sample.columns:
                    df_full = pd.read_parquet(ctx_path, columns=["n_up_4"])
                    sample_info["n_up_4_distribution"] = df_full["n_up_4"].value_counts().sort_index().to_dict()
        except Exception:
            pass
        # list TF key + type + prog columns compactly
        tf_cols = [c for c in schema["columns"] if any(c.startswith(tf + "_") for tf in TIMEFRAME_ORDER)]
        result[key] = {
            "path": _safe_rel(ctx_path),
            "size_mb": round(ctx_path.stat().st_size / 1024 / 1024, 2),
            "row_count": schema["row_count"],
            "column_count": len(schema["columns"]),
            "all_columns": schema["columns"],
            "tf_columns": tf_cols,
            "sample": sample_info,
        }

    # context_meta.json
    meta_path = ctx_dir / "context_meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            result["context_meta"] = {
                "path": _safe_rel(meta_path),
                "size_kb": round(meta_path.stat().st_size / 1024, 2),
                "version": meta.get("version"),
                "asset": meta.get("asset"),
                "data_range": meta.get("data_range"),
                "timeframe_groups": meta.get("timeframe_groups"),
                "top_level_keys": list(meta.keys()),
            }
        except Exception:
            result["context_meta"] = {"path": _safe_rel(meta_path), "error": "parse_failed"}

    return result


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
    # Run independent sections in parallel to reduce wall-clock time
    def _enriched() -> dict[str, Any]:
        return summarize_cycle_parquet_dir(PROJECT_PATHS.processed_cycles_enriched_dir, asset=asset, label="processed_cycles_enriched")

    def _root_copy() -> dict[str, Any]:
        return summarize_cycle_parquet_dir(PROJECT_PATHS.processed_cycles_enriched_dir, asset=None, label="processed_cycles_root_copy")

    def _base() -> dict[str, Any]:
        base_asset = asset if (PROJECT_PATHS.processed_cycles_base_dir / asset).exists() else None
        return summarize_cycle_parquet_dir(PROJECT_PATHS.processed_cycles_base_dir, asset=base_asset, label="processed_cycles_base")

    def _hierarchy() -> dict[str, Any]:
        return summarize_hierarchy_maps(asset)

    def _features() -> dict[str, Any]:
        return summarize_json_meta_dir(PROJECT_PATHS.processed_features_dir, label="processed_features")

    def _dashboard() -> dict[str, Any]:
        return summarize_dashboard()

    def _context() -> dict[str, Any]:
        return summarize_context(asset)

    def _raw() -> dict[str, Any]:
        return summarize_raw_market()

    tasks = {
        "raw_market": _raw,
        "processed_cycles_enriched": _enriched,
        "processed_cycles_root_copy": _root_copy,
        "processed_cycles_base": _base,
        "hierarchy_maps": _hierarchy,
        "processed_features_meta": _features,
        "dashboard": _dashboard,
        "context": _context,
    }

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        future_map = {pool.submit(fn): key for key, fn in tasks.items()}
        for fut in as_completed(future_map):
            key = future_map[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:
                results[key] = {"error": str(exc)}

    return {
        "paths": PROJECT_PATHS.summary(),
        "asset": asset,
        **results,
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
                for nc, ninfo in item["nested_columns"].items():
                    print(f"    nested  {nc}: {ninfo.get('type','?')}")
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

    ctx = report.get("context", {})
    print("\n" + "=" * 100)
    print("CONTEXT (v2.0 architecture)")
    print(f"root   : {ctx.get('root')}")
    print(f"exists : {ctx.get('exists')}")

    dim = ctx.get("cycle_dim")
    if dim:
        print(f"\n  cycle_dim.parquet")
        print(f"    path       : {dim['path']}")
        print(f"    size       : {dim['size_mb']} MB  |  rows={dim['row_count']}  |  cols={dim['column_count']}")
        print(f"    columns    : {', '.join(dim['columns'])}")
        tf_dist = dim.get("timeframe_distribution", {})
        if tf_dist:
            dist_str = "  ".join(f"{tf}:{cnt}" for tf, cnt in tf_dist.items())
            print(f"    tf_dist    : {dist_str}")

    for freq in ("1min", "1h"):
        key = f"timeframe_context_{freq}"
        tc = ctx.get(key)
        if not tc:
            continue
        print(f"\n  timeframe_context_{freq}.parquet")
        print(f"    path       : {tc['path']}")
        print(f"    size       : {tc['size_mb']} MB  |  rows={tc['row_count']}  |  cols={tc['column_count']}")
        print(f"    all_cols   : {', '.join(tc['all_columns'])}")
        if tc.get("tf_columns"):
            print(f"    tf_cols    : {', '.join(tc['tf_columns'])}")
        sample = tc.get("sample", {})
        if sample.get("first_timestamp"):
            print(f"    first_ts   : {sample['first_timestamp']}")
        if sample.get("n_up_4_distribution"):
            dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(sample["n_up_4_distribution"].items()))
            print(f"    n_up_4_dist: {dist_str}")

    meta = ctx.get("context_meta")
    if meta:
        print(f"\n  context_meta.json")
        print(f"    path       : {meta.get('path')}")
        print(f"    version    : {meta.get('version')}  |  asset={meta.get('asset')}")
        if meta.get("data_range"):
            print(f"    data_range : {meta['data_range']}")
        if meta.get("timeframe_groups"):
            print(f"    tf_groups  : {meta['timeframe_groups']}")
        print(f"    keys       : {meta.get('top_level_keys')}")

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

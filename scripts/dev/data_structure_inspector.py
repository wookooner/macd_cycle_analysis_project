from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS


DEFAULT_SUFFIXES = {".csv", ".parquet", ".json", ".jsonl"}
DEFAULT_GROUPS = (
    "raw_market",
    "raw_hierarchy",
    "raw_trades",
    "processed_cycles_base",
    "processed_cycles_enriched",
    "processed_reversal_events",
    "processed_features",
    "processed_trade_positions",
    "dashboard",
)


def _format_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_PATHS.project_root))
    except ValueError:
        return str(path.resolve())


def _iter_data_files(root: Path, suffixes: set[str], max_files: int) -> list[Path]:
    if not root.exists():
        return []

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    files.sort(key=lambda path: str(path).lower())
    return files[:max_files]


def _sample_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            line = handle.readline().strip()
        if not line:
            return {"type": "jsonl", "empty": True}
        payload = json.loads(line)
        return {
            "type": "jsonl",
            "first_row_type": type(payload).__name__,
            "first_row_keys": list(payload.keys()) if isinstance(payload, dict) else None,
            "first_row_sample": payload,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"type": type(payload).__name__}
    if isinstance(payload, dict):
        result["top_level_keys"] = list(payload.keys())[:50]
        result["top_level_count"] = len(payload)
        first_key = next(iter(payload), None)
        if first_key is not None:
            result["first_key"] = first_key
            result["first_value_type"] = type(payload[first_key]).__name__
            if isinstance(payload[first_key], dict):
                result["first_value_keys"] = list(payload[first_key].keys())[:50]
    elif isinstance(payload, list):
        result["length"] = len(payload)
        if payload:
            result["first_item_type"] = type(payload[0]).__name__
            if isinstance(payload[0], dict):
                result["first_item_keys"] = list(payload[0].keys())[:50]
    return result


def _describe_nested_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys())[:50],
        }
    if isinstance(value, (list, tuple)):
        first = value[0] if value else None
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first_item_type": type(first).__name__ if first is not None else None,
            "first_item_keys": list(first.keys())[:50] if isinstance(first, dict) else None,
        }
    return {"type": type(value).__name__, "sample": str(value)[:300]}


def _describe_dataframe(path: Path, sample_rows: int) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=sample_rows)
        row_count: int | None = None
        try:
            row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        except OSError:
            row_count = None
    else:
        df = pd.read_parquet(path)
        row_count = len(df)
        df = df.head(sample_rows)

    nested: dict[str, Any] = {}
    for column in ("cycle_features", "candle_data", "parent_cycle_ids", "child_cycle_ids"):
        if column not in df.columns:
            continue
        non_null = df[column].dropna()
        if non_null.empty:
            nested[column] = {"type": "empty"}
        else:
            nested[column] = _describe_nested_value(non_null.iloc[0])

    return {
        "type": "dataframe",
        "row_count": row_count,
        "sampled_rows": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "nested_columns": nested,
        "sample_records": df.head(sample_rows).to_dict(orient="records"),
    }


def inspect_file(path: Path, sample_rows: int) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": _format_path(path),
        "size_bytes": stat.st_size,
        "suffix": path.suffix.lower(),
    }

    try:
        if path.suffix.lower() in {".csv", ".parquet"}:
            result.update(_describe_dataframe(path, sample_rows))
        elif path.suffix.lower() in {".json", ".jsonl"}:
            result.update(_sample_json(path))
        else:
            result["type"] = "unsupported"
    except Exception as exc:
        result["error"] = str(exc)

    return result


def _path_groups() -> dict[str, Path]:
    return {
        "data_root": PROJECT_PATHS.data_root,
        "raw_market": PROJECT_PATHS.raw_market_dir,
        "raw_hierarchy": PROJECT_PATHS.raw_hierarchy_dir,
        "raw_trades": PROJECT_PATHS.raw_trades_dir,
        "processed_cycles_base": PROJECT_PATHS.processed_cycles_base_dir,
        "processed_cycles_enriched": PROJECT_PATHS.processed_cycles_enriched_dir,
        "processed_reversal_events": PROJECT_PATHS.processed_reversal_events_dir,
        "processed_features": PROJECT_PATHS.processed_features_dir,
        "processed_trade_positions": PROJECT_PATHS.processed_trade_positions_dir,
        "dashboard": PROJECT_PATHS.dashboard_root,
        "outputs": PROJECT_PATHS.outputs_root,
        "reports": PROJECT_PATHS.reports_root,
        "logs": PROJECT_PATHS.logs_root,
    }


def build_report(groups: Iterable[str], max_files: int, sample_rows: int, suffixes: set[str]) -> dict[str, Any]:
    available_groups = _path_groups()
    selected_groups = list(groups) if groups else list(available_groups.keys())

    report: dict[str, Any] = {
        "path_summary": PROJECT_PATHS.summary(),
        "groups": {},
    }

    for group in selected_groups:
        if group not in available_groups:
            report["groups"][group] = {"error": f"unknown group: {group}"}
            continue

        root = available_groups[group]
        files = _iter_data_files(root, suffixes=suffixes, max_files=max_files)
        report["groups"][group] = {
            "root": str(root),
            "exists": root.exists(),
            "inspected_file_count": len(files),
            "files": [inspect_file(path, sample_rows=sample_rows) for path in files],
        }

    return report


def print_report(report: dict[str, Any]) -> None:
    path_summary = report["path_summary"]
    print("DATA FILE STRUCTURE")
    print("=" * 80)
    print(f"project_root : {path_summary.get('project_root')}")
    print(f"data_root    : {path_summary.get('data_root')}")
    print(f"raw_market   : {path_summary.get('raw_market_dir')}")
    print(f"processed    : {path_summary.get('processed_root')}")
    print(f"dashboard    : {path_summary.get('dashboard_root')}")

    for group_name, group in report["groups"].items():
        print("\n" + "=" * 80)
        print(f"{group_name}: {group.get('root')}")
        if "error" in group:
            print(f"  error: {group['error']}")
            continue
        print(f"  exists: {group['exists']}")
        print(f"  inspected_files: {group['inspected_file_count']}")
        if group["exists"] and not group["files"]:
            print("  no csv/parquet/json/jsonl files found")

        for file_info in group["files"]:
            size_mb = file_info["size_bytes"] / 1024 / 1024
            print("\n  " + "-" * 76)
            print(f"  {file_info['path']} ({size_mb:.2f} MB)")
            print(f"  type: {file_info.get('type')}")
            if "error" in file_info:
              print(f"  error: {file_info['error']}")
              continue

            if file_info.get("type") == "dataframe":
                print(f"  rows: {file_info.get('row_count')}")
                print(f"  columns[{file_info.get('column_count')}]: {', '.join(file_info.get('columns', []))}")
                if file_info.get("nested_columns"):
                    print(f"  nested: {json.dumps(file_info['nested_columns'], ensure_ascii=False, default=str)}")
                print(f"  sample: {json.dumps(file_info.get('sample_records', []), ensure_ascii=False, default=str)[:900]}")
            else:
                printable = {key: value for key, value in file_info.items() if key not in {"path", "size_bytes", "suffix"}}
                print(f"  structure: {json.dumps(printable, ensure_ascii=False, default=str)[:900]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the project data-file structure. "
            "Default usage: python .\\scripts\\dev\\data_structure_inspector.py"
        )
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Optional. Inspect one path group. Repeatable. Examples: raw_market, processed_cycles_enriched, dashboard.",
    )
    parser.add_argument("--all", action="store_true", help="Inspect every registered path group, including outputs/reports/logs.")
    parser.add_argument("--max-files", type=int, default=3, help="Maximum files to inspect per group.")
    parser.add_argument("--sample-rows", type=int, default=2, help="Sample rows per dataframe file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = args.group
    if not groups and not args.all:
        groups = list(DEFAULT_GROUPS)
    report = build_report(
        groups=groups,
        max_files=args.max_files,
        sample_rows=args.sample_rows,
        suffixes=DEFAULT_SUFFIXES,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

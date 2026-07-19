from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from .paths import AnalystPaths


TIMEFRAME_ORDER = ("1min", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M")
BASE_FRAME_COLUMNS = (
    "cycle_id",
    "timeframe",
    "start_date",
    "end_date",
    "cycle_type",
    "duration_candles",
    "category",
    "algorithm_used",
    "cycle_key",
    "parent_key",
    "parent_type",
    "order_in_parent",
    "total_siblings",
    "parent_progress_at_start",
    "parent_progress_at_end",
    "boundary_type",
    "overlap_prev_ratio",
    "overlap_next_ratio",
    "n_up_4",
    "combo_4",
    "child_count",
    "child_up_count",
    "child_down_count",
    "opposite_child_ratio",
    "max_opposite_child_streak",
)


class DataAccessService:
    """Read canonical cycle and context data for ai_analyst."""

    def __init__(self) -> None:
        self.paths = AnalystPaths()

    def describe_available_data(self, asset: str = "btc") -> dict[str, Any]:
        cycle_dir = self.paths.cycles_enriched_dir(asset=asset)
        context_dir = self.paths.context_dir(asset=asset)

        cycle_datasets = [
            self._describe_parquet_dataset(path, dataset_type="cycle")
            for path in self._iter_existing_cycle_files(cycle_dir)
        ]
        context_datasets = [
            self._describe_parquet_dataset(path, dataset_type="context")
            for path in self._iter_existing_context_files(context_dir)
        ]

        return {
            "status": "success",
            "tool_name": "describe_available_data",
            "summary": f"Found {len(cycle_datasets)} cycle datasets and {len(context_datasets)} context datasets for asset '{asset}'.",
            "data_preview": {
                "cycle_timeframes": [item["timeframe"] for item in cycle_datasets],
                "context_files": [item["name"] for item in context_datasets],
            },
            "artifacts": [
                {
                    "artifact_type": "schema_snapshot",
                    "name": "available_datasets",
                    "description": "Available canonical cycle and context parquet datasets.",
                    "format": "json",
                    "payload": {
                        "cycle_datasets": cycle_datasets,
                        "context_datasets": context_datasets,
                    },
                    "preview": None,
                }
            ],
            "frame_meta": None,
            "warnings": self._build_availability_warnings(cycle_datasets, context_datasets),
            "errors": [],
            "metrics": {
                "cycle_dataset_count": len(cycle_datasets),
                "context_dataset_count": len(context_datasets),
            },
        }

    def build_analysis_frame(
        self,
        timeframe: str,
        asset: str = "btc",
        columns: list[str] | None = None,
        row_cap: int = 2000,
        preview_rows: int = 5,
    ) -> dict[str, Any]:
        cycle_file = self.paths.cycle_file(timeframe=timeframe, asset=asset)
        if not cycle_file.exists():
            return {
                "status": "error",
                "tool_name": "build_analysis_frame",
                "summary": f"Cycle dataset for timeframe '{timeframe}' was not found.",
                "data_preview": None,
                "artifacts": [],
                "frame_meta": None,
                "warnings": [],
                "errors": [f"Missing file: {cycle_file}"],
                "metrics": {},
            }

        frame = self._build_frame_dataframe(cycle_file, columns=columns, row_cap=row_cap)
        available_columns = frame.columns
        preview = frame.head(max(preview_rows, 1)).to_dict(as_series=False)
        warnings: list[str] = []

        if row_cap and frame.height >= row_cap:
            warnings.append(f"Row cap of {row_cap} applied to the analysis frame.")
        if columns:
            missing_columns = [name for name in columns if name not in available_columns]
            if missing_columns:
                warnings.append(
                    "Some requested columns were not available: "
                    + ", ".join(missing_columns)
                )

        return {
            "status": "success",
            "tool_name": "build_analysis_frame",
            "summary": f"Built analysis frame for {timeframe} with {frame.height} rows and {len(available_columns)} columns.",
            "data_preview": preview,
            "artifacts": [
                {
                    "artifact_type": "frame_ref",
                    "name": f"{asset}_{timeframe}_analysis_frame",
                    "description": "In-memory analysis frame preview with flattened feature columns.",
                    "format": "json",
                    "payload": {
                        "columns": available_columns,
                        "preview_rows": preview_rows,
                    },
                    "preview": preview,
                }
            ],
            "frame_meta": {
                "frame_id": f"{asset}_{timeframe}_analysis_frame",
                "source_datasets": [cycle_file.name],
                "timeframes": [timeframe],
                "row_count": frame.height,
                "column_names": available_columns,
                "filter_history": [],
                "join_history": ["flattened cycle_features into feature__* columns"],
                "sampling_applied": bool(row_cap and frame.height >= row_cap),
                "notes": [],
            },
            "warnings": warnings,
            "errors": [],
            "metrics": {
                "row_count": frame.height,
                "column_count": len(available_columns),
            },
        }

    def filter_analysis_frame(
        self,
        timeframe: str,
        filters: list[dict[str, Any]],
        asset: str = "btc",
        columns: list[str] | None = None,
        row_cap: int = 2000,
        preview_rows: int = 5,
    ) -> dict[str, Any]:
        cycle_file = self.paths.cycle_file(timeframe=timeframe, asset=asset)
        if not cycle_file.exists():
            return {
                "status": "error",
                "tool_name": "filter_frame",
                "summary": f"Cycle dataset for timeframe '{timeframe}' was not found.",
                "data_preview": None,
                "artifacts": [],
                "frame_meta": None,
                "warnings": [],
                "errors": [f"Missing file: {cycle_file}"],
                "metrics": {},
            }

        scan = self._prepare_frame_scan(cycle_file, columns=columns)
        schema = scan.collect_schema()
        filter_errors: list[str] = []
        filter_exprs: list[pl.Expr] = []
        applied_filters: list[str] = []

        for item in filters:
            expr, description, error = self._build_filter_expr(item, schema)
            if error:
                filter_errors.append(error)
                continue
            if expr is not None and description:
                filter_exprs.append(expr)
                applied_filters.append(description)

        if filter_errors:
            return {
                "status": "error",
                "tool_name": "filter_frame",
                "summary": "One or more filter conditions were invalid.",
                "data_preview": None,
                "artifacts": [],
                "frame_meta": None,
                "warnings": [],
                "errors": filter_errors,
                "metrics": {},
            }

        for expr in filter_exprs:
            scan = scan.filter(expr)

        if row_cap > 0:
            scan = scan.limit(row_cap)

        frame = scan.collect()
        preview = frame.head(max(preview_rows, 1)).to_dict(as_series=False)
        warnings: list[str] = []
        row_cap_applied = bool(row_cap and frame.height >= row_cap)
        if frame.height == 0:
            warnings.append("The applied filters returned zero rows.")
        elif row_cap_applied:
            warnings.append(
                f"Row cap of {row_cap} applied to the filtered frame; the returned row count is a capped preview, not a guaranteed full match count."
            )

        return {
            "status": "success",
            "tool_name": "filter_frame",
            "summary": (
                f"Filtered {timeframe} frame to {frame.height} returned rows using {len(applied_filters)} condition(s). "
                "Returned rows may be capped when a row limit is applied."
                if row_cap_applied
                else f"Filtered {timeframe} frame to {frame.height} rows using {len(applied_filters)} condition(s)."
            ),
            "data_preview": preview,
            "artifacts": [
                {
                    "artifact_type": "frame_ref",
                    "name": f"{asset}_{timeframe}_filtered_frame",
                    "description": "Filtered analysis frame preview.",
                    "format": "json",
                    "payload": {
                        "columns": frame.columns,
                        "applied_filters": applied_filters,
                        "preview_rows": preview_rows,
                    },
                    "preview": preview,
                }
            ],
            "frame_meta": {
                "frame_id": f"{asset}_{timeframe}_filtered_frame",
                "source_datasets": [cycle_file.name],
                "timeframes": [timeframe],
                "row_count": frame.height,
                "column_names": frame.columns,
                "filter_history": applied_filters,
                "join_history": ["flattened cycle_features into feature__* columns"],
                "sampling_applied": row_cap_applied,
                "notes": (
                    [
                        "Returned rows are capped by row_cap; treat row_count as the returned preview size, not a guaranteed total match count."
                    ]
                    if row_cap_applied
                    else []
                ),
            },
            "warnings": warnings,
            "errors": [],
            "metrics": {
                "row_count": frame.height,
                "column_count": len(frame.columns),
                "filter_count": len(applied_filters),
                "row_cap_applied": row_cap_applied,
            },
        }

    def compare_groups(
        self,
        timeframe: str,
        group_a_filters: list[dict[str, Any]],
        group_b_filters: list[dict[str, Any]],
        metrics_columns: list[str],
        asset: str = "btc",
        row_cap: int = 5000,
        preview_rows: int = 5,
    ) -> dict[str, Any]:
        cycle_file = self.paths.cycle_file(timeframe=timeframe, asset=asset)
        if not cycle_file.exists():
            return {
                "status": "error",
                "tool_name": "compare_groups",
                "summary": f"Cycle dataset for timeframe '{timeframe}' was not found.",
                "data_preview": None,
                "artifacts": [],
                "frame_meta": None,
                "warnings": [],
                "errors": [f"Missing file: {cycle_file}"],
                "metrics": {},
            }

        required_columns = list(
            dict.fromkeys(
                [
                    *metrics_columns,
                    *(item.get("column", "") for item in group_a_filters),
                    *(item.get("column", "") for item in group_b_filters),
                ]
            )
        )
        required_columns = [name for name in required_columns if name]
        scan = self._prepare_frame_scan(cycle_file, columns=required_columns or None)
        schema = scan.collect_schema()

        group_a_exprs, group_a_desc, group_a_errors = self._validate_filter_group(
            group_a_filters,
            schema,
        )
        group_b_exprs, group_b_desc, group_b_errors = self._validate_filter_group(
            group_b_filters,
            schema,
        )
        metric_errors = [
            self._build_missing_column_error(
                missing_name=column,
                available_columns=list(schema.names()),
                context_label="Metric column",
            )
            for column in metrics_columns
            if column not in schema
        ]
        errors = [*group_a_errors, *group_b_errors, *metric_errors]
        if errors:
            return {
                "status": "error",
                "tool_name": "compare_groups",
                "summary": "Group comparison could not run because one or more inputs were invalid.",
                "data_preview": None,
                "artifacts": [],
                "frame_meta": None,
                "warnings": [],
                "errors": errors,
                "metrics": {},
            }

        base_scan = scan.limit(row_cap) if row_cap > 0 else scan
        group_a_frame = self._apply_filter_exprs(base_scan, group_a_exprs).collect()
        group_b_frame = self._apply_filter_exprs(base_scan, group_b_exprs).collect()
        metrics_summary = self._summarize_group_metrics(
            group_a_frame,
            group_b_frame,
            metrics_columns,
        )

        warnings: list[str] = []
        if group_a_frame.height == 0:
            warnings.append("Group A returned zero rows.")
        if group_b_frame.height == 0:
            warnings.append("Group B returned zero rows.")
        if group_a_frame.height < 30 or group_b_frame.height < 30:
            warnings.append(
                "At least one group has fewer than 30 rows; treat differences cautiously."
            )
        if row_cap and (group_a_frame.height >= row_cap or group_b_frame.height >= row_cap):
            warnings.append(f"Row cap of {row_cap} may have truncated one or both groups.")

        preview = {
            "group_a": group_a_frame.head(max(preview_rows, 1)).to_dict(as_series=False),
            "group_b": group_b_frame.head(max(preview_rows, 1)).to_dict(as_series=False),
            "metric_summary": metrics_summary,
        }
        return {
            "status": "success",
            "tool_name": "compare_groups",
            "summary": f"Compared 2 groups on {len(metrics_columns)} metric column(s) for {timeframe}.",
            "data_preview": preview,
            "artifacts": [
                {
                    "artifact_type": "table",
                    "name": f"{asset}_{timeframe}_group_comparison",
                    "description": "Side-by-side group metric summary.",
                    "format": "json",
                    "payload": metrics_summary,
                    "preview": metrics_summary[: min(len(metrics_summary), 5)],
                }
            ],
            "frame_meta": {
                "frame_id": f"{asset}_{timeframe}_group_comparison",
                "source_datasets": [cycle_file.name],
                "timeframes": [timeframe],
                "row_count": group_a_frame.height + group_b_frame.height,
                "column_names": required_columns,
                "filter_history": [f"group_a: {item}" for item in group_a_desc]
                + [f"group_b: {item}" for item in group_b_desc],
                "join_history": ["flattened cycle_features into feature__* columns"],
                "sampling_applied": bool(row_cap),
                "notes": [],
            },
            "warnings": warnings,
            "errors": [],
            "metrics": {
                "group_a_rows": group_a_frame.height,
                "group_b_rows": group_b_frame.height,
                "metric_count": len(metrics_columns),
            },
        }

    def rank_features(
        self,
        timeframe: str,
        focus_filters: list[dict[str, Any]],
        asset: str = "btc",
        candidate_columns: list[str] | None = None,
        top_k: int = 10,
        row_cap: int = 5000,
        preview_rows: int = 5,
    ) -> dict[str, Any]:
        cycle_file = self.paths.cycle_file(timeframe=timeframe, asset=asset)
        if not cycle_file.exists():
            return {
                "status": "error",
                "tool_name": "rank_features",
                "summary": f"Cycle dataset for timeframe '{timeframe}' was not found.",
                "data_preview": None,
                "artifacts": [],
                "frame_meta": None,
                "warnings": [],
                "errors": [f"Missing file: {cycle_file}"],
                "metrics": {},
            }

        scan = self._prepare_frame_scan(cycle_file, columns=None)
        schema = scan.collect_schema()
        focus_exprs, focus_desc, focus_errors = self._validate_filter_group(
            focus_filters,
            schema,
        )
        if focus_errors:
            return {
                "status": "error",
                "tool_name": "rank_features",
                "summary": "Feature ranking could not run because one or more filters were invalid.",
                "data_preview": None,
                "artifacts": [],
                "frame_meta": None,
                "warnings": [],
                "errors": focus_errors,
                "metrics": {},
            }

        limited_scan = scan.limit(row_cap) if row_cap > 0 else scan
        base_frame = limited_scan.collect()
        focus_frame = self._apply_filter_exprs(base_frame.lazy(), focus_exprs).collect()
        remainder_frame = base_frame.join(
            focus_frame.select("cycle_key"),
            on="cycle_key",
            how="anti",
        ) if "cycle_key" in base_frame.columns and "cycle_key" in focus_frame.columns else base_frame

        warnings: list[str] = []
        if focus_frame.height == 0:
            warnings.append("Focus subset returned zero rows.")
        if remainder_frame.height == 0:
            warnings.append("Remainder subset returned zero rows.")
        if row_cap and base_frame.height >= row_cap:
            warnings.append(
                f"Row cap of {row_cap} applied before ranking; feature ranking is based on a capped working set."
            )

        ranking = self._rank_feature_deltas(
            focus_frame=focus_frame,
            remainder_frame=remainder_frame,
            candidate_columns=candidate_columns,
            top_k=top_k,
        )

        return {
            "status": "success",
            "tool_name": "rank_features",
            "summary": f"Ranked up to {top_k} feature columns for {timeframe} using focus subset vs remainder.",
            "data_preview": {
                "top_ranked_features": ranking[: min(len(ranking), preview_rows)],
            },
            "artifacts": [
                {
                    "artifact_type": "table",
                    "name": f"{asset}_{timeframe}_feature_ranking",
                    "description": "Feature ranking by mean-delta magnitude between focus subset and remainder.",
                    "format": "json",
                    "payload": ranking,
                    "preview": ranking[: min(len(ranking), preview_rows)],
                }
            ],
            "frame_meta": {
                "frame_id": f"{asset}_{timeframe}_feature_ranking",
                "source_datasets": [cycle_file.name],
                "timeframes": [timeframe],
                "row_count": base_frame.height,
                "column_names": base_frame.columns,
                "filter_history": focus_desc,
                "join_history": ["flattened cycle_features into feature__* columns"],
                "sampling_applied": bool(row_cap and base_frame.height >= row_cap),
                "notes": (
                    [
                        "Feature ranking compares the focus subset against the capped remainder of the same timeframe."
                    ]
                    if row_cap and base_frame.height >= row_cap
                    else []
                ),
            },
            "warnings": warnings,
            "errors": [],
            "metrics": {
                "focus_rows": focus_frame.height,
                "remainder_rows": remainder_frame.height,
                "ranked_feature_count": len(ranking),
                "row_cap_applied": bool(row_cap and base_frame.height >= row_cap),
            },
        }

    def _iter_existing_cycle_files(self, cycle_dir: Path) -> list[Path]:
        return [
            cycle_dir / f"cycles_{timeframe}.parquet"
            for timeframe in TIMEFRAME_ORDER
            if (cycle_dir / f"cycles_{timeframe}.parquet").exists()
        ]

    def _iter_existing_context_files(self, context_dir: Path) -> list[Path]:
        files: list[Path] = []
        cycle_dim = context_dir / "cycle_dim.parquet"
        if cycle_dim.exists():
            files.append(cycle_dim)

        for timeframe in TIMEFRAME_ORDER:
            candidate = context_dir / f"timeframe_context_{timeframe}.parquet"
            if candidate.exists():
                files.append(candidate)

        return files

    def _describe_parquet_dataset(self, path: Path, dataset_type: str) -> dict[str, Any]:
        parquet_file = pq.ParquetFile(path)
        schema_names = parquet_file.schema_arrow.names
        timeframe = self._infer_timeframe_from_name(path.name)
        return {
            "name": path.name,
            "dataset_type": dataset_type,
            "timeframe": timeframe,
            "path": str(path),
            "rows": parquet_file.metadata.num_rows,
            "column_count": len(schema_names),
            "columns_preview": schema_names[:15],
        }

    def _build_availability_warnings(
        self,
        cycle_datasets: list[dict[str, Any]],
        context_datasets: list[dict[str, Any]],
    ) -> list[str]:
        warnings: list[str] = []
        if not cycle_datasets:
            warnings.append("No cycle datasets were found.")
        if not context_datasets:
            warnings.append("No context datasets were found.")
        return warnings

    def _infer_timeframe_from_name(self, name: str) -> str | None:
        if name.startswith("cycles_") and name.endswith(".parquet"):
            return name.replace("cycles_", "").replace(".parquet", "")
        if name.startswith("timeframe_context_") and name.endswith(".parquet"):
            return name.replace("timeframe_context_", "").replace(".parquet", "")
        return None

    def _build_frame_dataframe(
        self,
        cycle_file: Path,
        columns: list[str] | None,
        row_cap: int,
    ) -> pl.DataFrame:
        scan = self._prepare_frame_scan(cycle_file, columns=columns)
        if row_cap > 0:
            scan = scan.limit(row_cap)
        return scan.collect()

    def _prepare_frame_scan(
        self,
        cycle_file: Path,
        columns: list[str] | None,
    ) -> pl.LazyFrame:
        scan = pl.scan_parquet(cycle_file)
        schema = scan.collect_schema()
        feature_dtype = schema.get("cycle_features")

        select_exprs: list[pl.Expr] = [
            pl.col(name) for name in BASE_FRAME_COLUMNS if name in schema
        ]
        select_exprs.extend(self._feature_exprs(feature_dtype))

        if columns:
            available_exprs = {expr.meta.output_name(): expr for expr in select_exprs}
            filtered_exprs = [
                available_exprs[name] for name in columns if name in available_exprs
            ]
            if filtered_exprs:
                return scan.select(filtered_exprs)
        return scan.select(select_exprs)

    def _feature_exprs(self, dtype: pl.DataType | None) -> list[pl.Expr]:
        if dtype is None or not hasattr(dtype, "fields"):
            return []
        return self._flatten_struct_exprs(pl.col("cycle_features"), dtype, prefix=("feature",))

    def _flatten_struct_exprs(
        self,
        expr: pl.Expr,
        dtype: pl.DataType,
        prefix: tuple[str, ...],
    ) -> list[pl.Expr]:
        if not hasattr(dtype, "fields"):
            return [expr.alias("__".join(prefix))]

        flattened: list[pl.Expr] = []
        for field in dtype.fields:  # type: ignore[attr-defined]
            next_expr = expr.struct.field(field.name)
            next_prefix = prefix + (field.name,)
            flattened.extend(
                self._flatten_struct_exprs(next_expr, field.dtype, next_prefix)
            )
        return flattened

    def _build_filter_expr(
        self,
        filter_item: dict[str, Any],
        schema: Any,
    ) -> tuple[pl.Expr | None, str | None, str | None]:
        column = str(filter_item.get("column", "")).strip()
        operator = str(filter_item.get("operator", "")).strip()
        value = filter_item.get("value")

        if not column:
            return None, None, "Filter is missing a column name."
        if column not in schema:
            return (
                None,
                None,
                self._build_missing_column_error(
                    missing_name=column,
                    available_columns=list(schema.names()),
                    context_label="Filter column",
                ),
            )
        if not operator:
            return None, None, f"Filter for column '{column}' is missing an operator."

        expr = pl.col(column)
        description = f"{column} {operator} {value}"

        if operator == "==":
            return expr == value, description, None
        if operator == "!=":
            return expr != value, description, None
        if operator == ">":
            return expr > value, description, None
        if operator == ">=":
            return expr >= value, description, None
        if operator == "<":
            return expr < value, description, None
        if operator == "<=":
            return expr <= value, description, None
        if operator == "contains":
            return expr.cast(pl.String).str.contains(str(value), literal=True), description, None
        if operator == "in":
            if not isinstance(value, list):
                return None, None, f"Filter operator 'in' for column '{column}' requires a list value."
            return expr.is_in(value), description, None

        return None, None, f"Unsupported filter operator '{operator}' for column '{column}'."

    def _validate_filter_group(
        self,
        filters: list[dict[str, Any]],
        schema: Any,
    ) -> tuple[list[pl.Expr], list[str], list[str]]:
        exprs: list[pl.Expr] = []
        descriptions: list[str] = []
        errors: list[str] = []
        for item in filters:
            expr, description, error = self._build_filter_expr(item, schema)
            if error:
                errors.append(error)
                continue
            if expr is not None and description:
                exprs.append(expr)
                descriptions.append(description)
        return exprs, descriptions, errors

    def _apply_filter_exprs(
        self,
        scan: pl.LazyFrame,
        exprs: list[pl.Expr],
    ) -> pl.LazyFrame:
        for expr in exprs:
            scan = scan.filter(expr)
        return scan

    def _summarize_group_metrics(
        self,
        group_a_frame: pl.DataFrame,
        group_b_frame: pl.DataFrame,
        metrics_columns: list[str],
    ) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for column in metrics_columns:
            if column not in group_a_frame.columns or column not in group_b_frame.columns:
                continue

            a_series = group_a_frame.get_column(column).cast(pl.Float64, strict=False)
            b_series = group_b_frame.get_column(column).cast(pl.Float64, strict=False)
            a_mean = a_series.mean()
            b_mean = b_series.mean()
            a_median = a_series.median()
            b_median = b_series.median()

            summary.append(
                {
                    "column": column,
                    "group_a_count": int(a_series.len() - a_series.null_count()),
                    "group_b_count": int(b_series.len() - b_series.null_count()),
                    "group_a_mean": a_mean,
                    "group_b_mean": b_mean,
                    "group_a_median": a_median,
                    "group_b_median": b_median,
                    "mean_delta": (a_mean or 0.0) - (b_mean or 0.0),
                    "median_delta": (a_median or 0.0) - (b_median or 0.0),
                }
            )
        return summary

    def _rank_feature_deltas(
        self,
        focus_frame: pl.DataFrame,
        remainder_frame: pl.DataFrame,
        candidate_columns: list[str] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if focus_frame.height == 0 or remainder_frame.height == 0:
            return []

        if candidate_columns:
            columns = [name for name in candidate_columns if name in focus_frame.columns]
        else:
            columns = [
                name
                for name, dtype in zip(focus_frame.columns, focus_frame.dtypes, strict=False)
                if name.startswith("feature__")
                and self._is_numeric_dtype(dtype)
            ]

        ranked: list[dict[str, Any]] = []
        for column in columns:
            if column not in remainder_frame.columns:
                continue

            focus_series = focus_frame.get_column(column).cast(pl.Float64, strict=False)
            remainder_series = remainder_frame.get_column(column).cast(pl.Float64, strict=False)
            focus_count = int(focus_series.len() - focus_series.null_count())
            remainder_count = int(remainder_series.len() - remainder_series.null_count())
            if focus_count == 0 or remainder_count == 0:
                continue

            focus_mean = focus_series.mean()
            remainder_mean = remainder_series.mean()
            focus_median = focus_series.median()
            remainder_median = remainder_series.median()
            mean_delta = (focus_mean or 0.0) - (remainder_mean or 0.0)
            median_delta = (focus_median or 0.0) - (remainder_median or 0.0)

            ranked.append(
                {
                    "column": column,
                    "focus_count": focus_count,
                    "remainder_count": remainder_count,
                    "focus_mean": focus_mean,
                    "remainder_mean": remainder_mean,
                    "focus_median": focus_median,
                    "remainder_median": remainder_median,
                    "mean_delta": mean_delta,
                    "median_delta": median_delta,
                    "abs_mean_delta": abs(mean_delta),
                }
            )

        ranked.sort(key=lambda item: item["abs_mean_delta"], reverse=True)
        return ranked[:top_k]

    def _build_missing_column_error(
        self,
        missing_name: str,
        available_columns: list[str],
        context_label: str,
        preview_limit: int = 20,
    ) -> str:
        preview = ", ".join(available_columns[:preview_limit])
        remaining = max(len(available_columns) - preview_limit, 0)
        if remaining > 0:
            preview = f"{preview}, ... (+{remaining} more)"
        message = (
            f"{context_label} '{missing_name}' is not available in the frame. "
            f"Available columns include: {preview}"
        )
        suggested = self._suggest_similar_columns(missing_name, available_columns)
        if suggested:
            message += f" Suggested nearby columns: {', '.join(suggested)}"
        return message

    def _suggest_similar_columns(
        self,
        missing_name: str,
        available_columns: list[str],
        limit: int = 8,
    ) -> list[str]:
        tokens = self._column_tokens(missing_name)
        if not tokens:
            return []

        scored: list[tuple[int, str]] = []
        for column in available_columns:
            column_tokens = self._column_tokens(column)
            overlap = len(tokens.intersection(column_tokens))
            if overlap <= 0:
                continue
            prefix_bonus = 1 if "feature" in column_tokens else 0
            scored.append((overlap * 10 + prefix_bonus, column))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [column for _, column in scored[:limit]]

    def _column_tokens(self, value: str) -> set[str]:
        return {
            token
            for token in re.split(r"[^a-zA-Z0-9]+", value.lower())
            if token and len(token) >= 3
        }

    def _is_numeric_dtype(self, dtype: pl.DataType) -> bool:
        checker = getattr(dtype, "is_numeric", None)
        if callable(checker):
            return bool(checker())
        return dtype in {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        }

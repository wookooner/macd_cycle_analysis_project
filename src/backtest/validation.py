from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_cols: list[str] | None = None) -> list[dict[str, Any]]:
    required_cols = required_cols or []
    issues: list[dict[str, Any]] = []
    if not df.columns.is_unique:
        issues.append({"severity": "error", "code": "duplicate_columns", "message": "DataFrame has duplicate columns."})
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append({"severity": "error", "code": "missing_required_columns", "message": ",".join(missing)})
    all_nan = [col for col in df.columns if df[col].isna().all()]
    if all_nan:
        issues.append({"severity": "warning", "code": "all_nan_columns", "message": ",".join(all_nan[:20])})
    return issues


def validate_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    hashes = [item.get("sha256") for item in manifest if item.get("sha256")]
    for digest, count in Counter(hashes).items():
        if count > 1:
            issues.append({"severity": "error", "code": "duplicate_output_hash", "message": digest})
    return issues


def validate_decision_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if "sample_class" in df and df["sample_class"].astype(str).eq("low_sample").any():
        issues.append({"severity": "error", "code": "low_sample_in_decision_table", "message": "low_sample rows present."})
    if "rule_kind" in df and df["rule_kind"].astype(str).eq("proxy").any():
        issues.append({"severity": "error", "code": "proxy_in_decision_table", "message": "proxy rules present."})
    mdd_cols = [col for col in df.columns if "MDD" in col or "drawdown" in col]
    for col in mdd_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any() and (values <= -99.0).mean() > 0.25:
            issues.append({"severity": "warning", "code": "mdd_near_minus_100_many_rows", "message": col})
    return issues

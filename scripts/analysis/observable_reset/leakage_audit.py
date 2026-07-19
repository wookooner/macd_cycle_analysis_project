from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib import FEATURE_PATH, OBSERVABLE_FEATURES, OUT_DIR, load_features


FORBIDDEN_PATTERNS = ("cycle_id", "cycle_type", "parent_", "future_", "next_", "label_", "target_")
ALLOWED_OUTCOME_PREFIXES = ("net_return_h", "mfe_h", "mae_h")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(FEATURE_PATH))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.features)
    df = load_features(path)

    failures: list[str] = []
    warnings: list[str] = []

    for col in df.columns:
        lower = col.lower()
        if lower.startswith(ALLOWED_OUTCOME_PREFIXES):
            continue
        if any(pattern in lower for pattern in FORBIDDEN_PATTERNS):
            failures.append(f"forbidden column detected: {col}")

    missing = [col for col in OBSERVABLE_FEATURES if col not in df.columns]
    if missing:
        failures.append(f"missing observable feature columns: {missing}")

    if "feature_source_max_time" not in df.columns:
        failures.append("missing feature_source_max_time")
    else:
        src = pd.to_datetime(df["feature_source_max_time"], errors="coerce")
        entry = pd.to_datetime(df["entry_time"], errors="coerce")
        bad = (src > entry).sum()
        if bad:
            failures.append(f"{bad} rows have feature_source_max_time after entry_time")

    outcome_cols = [c for c in df.columns if c.startswith(ALLOWED_OUTCOME_PREFIXES)]
    feature_cols = [c for c in OBSERVABLE_FEATURES if c in df.columns]
    overlap = set(outcome_cols).intersection(feature_cols)
    if overlap:
        failures.append(f"outcome columns overlap observable feature list: {sorted(overlap)}")

    # Shift audit: observable features must not be exact aliases of future outcomes.
    for feat in feature_cols:
        values = pd.to_numeric(df[feat], errors="coerce")
        if values.notna().sum() < 100:
            warnings.append(f"low non-null feature count: {feat}")
            continue
        for outcome in outcome_cols[:8]:
            corr = values.corr(pd.to_numeric(df[outcome], errors="coerce").shift(-1))
            if pd.notna(corr) and abs(corr) > 0.995:
                failures.append(f"suspicious near-perfect future outcome correlation: {feat} vs {outcome}.shift(-1) corr={corr:.6f}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "feature_columns": feature_cols,
        "outcome_columns": outcome_cols,
        "failures": failures,
        "warnings": warnings,
    }
    text_lines = [
        f"LEAKAGE AUDIT: {report['status']}",
        f"rows={report['rows']}",
        f"features={', '.join(feature_cols)}",
        f"outcomes={', '.join(outcome_cols)}",
        "",
        "FAILURES:",
        *(f"- {item}" for item in failures),
        "",
        "WARNINGS:",
        *(f"- {item}" for item in warnings),
    ]
    (OUT_DIR / "leakage_audit_report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "leakage_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "failures": failures, "warnings": warnings}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

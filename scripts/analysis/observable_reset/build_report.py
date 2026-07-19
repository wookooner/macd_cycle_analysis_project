from __future__ import annotations

import json

import pandas as pd

from lib import OUT_DIR, load_json


def table(path: str, limit: int = 20) -> str:
    p = OUT_DIR / path
    if not p.exists():
        return "Missing."
    df = pd.read_csv(p)
    if df.empty:
        return "No rows."
    return df.head(limit).to_markdown(index=False)


def main() -> None:
    frozen_path = OUT_DIR / "frozen_rules.json"
    frozen = load_json(frozen_path) if frozen_path.exists() else []
    lines = [
        "# Observable Reset Report",
        "",
        "## Required Files",
        "",
        "- `univariate_scan.csv`",
        "- `candidate_rules.txt`",
        "- `frozen_rules.json`",
        "- `oos_validation.csv`",
        "- `sensitivity.csv`",
        "- `leakage_audit_report.txt`",
        "",
        "## Frozen Rules",
        "",
        f"`{len(frozen)}` rules survived robustness filtering.",
        "",
        "## OOS Validation",
        "",
        table("oos_validation.csv", 30),
        "",
        "## Sensitivity",
        "",
        table("sensitivity.csv", 30),
        "",
        "## Top Univariate Rows",
        "",
        table("univariate_scan.csv", 30),
        "",
        "## Frozen Rules JSON Preview",
        "",
        "```json",
        json.dumps(frozen[:5], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (OUT_DIR / "observable_reset_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "observable_reset_report.md"), "frozen_rules": len(frozen)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

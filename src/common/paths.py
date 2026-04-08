from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if value == "":
            current[key] = {}
            stack.append((indent, current[key]))
        else:
            current[key] = value

    return root


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    config_path: Path
    data_root: Path
    runtime_data_root: Path
    legacy_data_root: Path
    raw_root: Path
    interim_root: Path
    processed_root: Path
    dashboard_root: Path
    outputs_root: Path
    reports_root: Path
    logs_root: Path
    raw_market_dir: Path
    raw_hierarchy_dir: Path
    raw_trades_dir: Path
    interim_flattened_dir: Path
    interim_joined_dir: Path
    interim_temp_dir: Path
    interim_debug_dir: Path
    processed_cycles_base_dir: Path
    processed_cycles_enriched_dir: Path
    processed_reversal_events_dir: Path
    processed_features_dir: Path
    processed_trade_positions_dir: Path
    base_data_dir: Path
    backup_data_dir: Path
    cycle_structured_dir: Path

    @property
    def data_root_is_inside_repo(self) -> bool:
        return _is_relative_to(self.data_root, self.project_root)

    @property
    def using_external_data_root(self) -> bool:
        return not self.data_root_is_inside_repo

    def asset_cycle_dir(self, asset: str) -> Path:
        return self.cycle_structured_dir / asset

    def ensure_runtime_dirs(self) -> None:
        for directory in [
            self.data_root,
            self.raw_root,
            self.interim_root,
            self.processed_root,
            self.dashboard_root,
            self.outputs_root,
            self.reports_root,
            self.logs_root,
            self.raw_market_dir,
            self.raw_hierarchy_dir,
            self.raw_trades_dir,
            self.interim_flattened_dir,
            self.interim_joined_dir,
            self.interim_temp_dir,
            self.interim_debug_dir,
            self.processed_cycles_base_dir,
            self.processed_cycles_enriched_dir,
            self.processed_reversal_events_dir,
            self.processed_features_dir,
            self.processed_trade_positions_dir,
            self.runtime_data_root,
            self.legacy_data_root,
            self.base_data_dir,
            self.backup_data_dir,
            self.cycle_structured_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "config_path": str(self.config_path),
            "data_root": str(self.data_root),
            "runtime_data_root": str(self.runtime_data_root),
            "legacy_data_root": str(self.legacy_data_root),
            "raw_root": str(self.raw_root),
            "processed_root": str(self.processed_root),
            "dashboard_root": str(self.dashboard_root),
            "outputs_root": str(self.outputs_root),
            "reports_root": str(self.reports_root),
            "logs_root": str(self.logs_root),
            "base_data_dir": str(self.base_data_dir),
            "backup_data_dir": str(self.backup_data_dir),
            "cycle_structured_dir": str(self.cycle_structured_dir),
            "data_root_is_inside_repo": str(self.data_root_is_inside_repo),
        }

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.config_path.exists():
            issues.append(f"Missing path config: {self.config_path}")
        if self.data_root == self.project_root:
            issues.append("data_root resolves to the repository root, which is unsafe.")
        if self.data_root_is_inside_repo:
            issues.append(
                "data_root is inside the repository. This is acceptable for bootstrap, "
                "but not the recommended long-term layout."
            )
        required_parents = {
            "base_data_dir": self.base_data_dir.parent,
            "backup_data_dir": self.backup_data_dir.parent,
            "cycle_structured_dir": self.cycle_structured_dir.parent,
        }
        for label, parent in required_parents.items():
            if not parent.exists():
                issues.append(f"{label} parent directory does not exist yet: {parent}")
        return issues


def load_project_paths(config_path: Path | None = None) -> ProjectPaths:
    project_root = Path(__file__).resolve().parents[2]
    config_file = config_path or project_root / "configs" / "paths.yaml"
    if config_file.exists():
        config = _parse_simple_yaml(config_file.read_text(encoding="utf-8"))
    else:
        config = {
            "project": {
                "code_root": ".",
                "data_root_env": "MACD_DATA_ROOT",
                "default_data_root": "./runtime_data",
            },
            "data": {},
            "legacy": {
                "repo_data_root": "./data",
                "base_data_dir": "./data/base_data",
                "backup_data_dir": "./data/backup_data",
                "cycle_structured_dir": "./data/cycle_data/structured",
            },
        }

    project_cfg = config.get("project", {})
    legacy_cfg = config.get("legacy", {})
    data_cfg = config.get("data", {})

    default_data_root = _resolve_path(project_root, project_cfg.get("default_data_root", "./runtime_data"))
    data_root_env = project_cfg.get("data_root_env", "MACD_DATA_ROOT")
    data_root = _resolve_path(project_root, os.getenv(data_root_env, str(default_data_root)))

    raw_root = data_root / "raw"
    interim_root = data_root / "interim"
    processed_root = data_root / "processed"
    dashboard_root = data_root / data_cfg.get("dashboard_root", "dashboard")
    outputs_root = data_root / data_cfg.get("outputs_root", "outputs")
    reports_root = data_root / data_cfg.get("reports_root", "reports")
    logs_root = data_root / data_cfg.get("logs_root", "logs")

    repo_legacy_data_root = _resolve_path(project_root, legacy_cfg.get("repo_data_root", "./data"))
    repo_base_data_dir = _resolve_path(project_root, legacy_cfg.get("base_data_dir", "./data/base_data"))
    repo_backup_data_dir = _resolve_path(project_root, legacy_cfg.get("backup_data_dir", "./data/backup_data"))
    repo_cycle_structured_dir = _resolve_path(
        project_root,
        legacy_cfg.get("cycle_structured_dir", "./data/cycle_data/structured"),
    )

    base_data_dir = raw_root / "market"
    backup_data_dir = interim_root / "temp" / "backup_data"
    cycle_structured_dir = processed_root / "cycles_enriched"

    if data_root.resolve() == repo_legacy_data_root.resolve():
        base_data_dir = repo_base_data_dir
        backup_data_dir = repo_backup_data_dir
        cycle_structured_dir = repo_cycle_structured_dir

    return ProjectPaths(
        project_root=project_root,
        config_path=config_file,
        data_root=data_root,
        runtime_data_root=default_data_root,
        legacy_data_root=repo_legacy_data_root,
        raw_root=raw_root,
        interim_root=interim_root,
        processed_root=processed_root,
        dashboard_root=dashboard_root,
        outputs_root=outputs_root,
        reports_root=reports_root,
        logs_root=logs_root,
        raw_market_dir=data_root / data_cfg.get("raw_market", "raw/market"),
        raw_hierarchy_dir=data_root / data_cfg.get("raw_hierarchy", "raw/hierarchy"),
        raw_trades_dir=data_root / data_cfg.get("raw_trades", "raw/trades"),
        interim_flattened_dir=data_root / data_cfg.get("interim_flattened", "interim/flattened"),
        interim_joined_dir=data_root / data_cfg.get("interim_joined", "interim/joined"),
        interim_temp_dir=data_root / data_cfg.get("interim_temp", "interim/temp"),
        interim_debug_dir=data_root / data_cfg.get("interim_debug", "interim/debug"),
        processed_cycles_base_dir=data_root / data_cfg.get("processed_cycles_base", "processed/cycles_base"),
        processed_cycles_enriched_dir=data_root / data_cfg.get("processed_cycles_enriched", "processed/cycles_enriched"),
        processed_reversal_events_dir=data_root / data_cfg.get("processed_reversal_events", "processed/reversal_events"),
        processed_features_dir=data_root / data_cfg.get("processed_features", "processed/features"),
        processed_trade_positions_dir=data_root / data_cfg.get("processed_trade_positions", "processed/trade_positions"),
        base_data_dir=base_data_dir,
        backup_data_dir=backup_data_dir,
        cycle_structured_dir=cycle_structured_dir,
    )


PROJECT_PATHS = load_project_paths()

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from src.common.paths import PROJECT_PATHS


DEFAULT_TARGET = Path(r"C:\Users\qw370\macd-cycle-data")
EXCLUDED_SUFFIXES = {".tmp", ".recovery"}
MIGRATION_GROUPS = {
    "raw": [
        PROJECT_PATHS.legacy_data_root / "base_data",
    ],
    "processed": [
        PROJECT_PATHS.legacy_data_root / "cycle_data" / "structured",
    ],
    "dashboard": [
        PROJECT_PATHS.legacy_data_root / "dashboard_data",
        PROJECT_PATHS.legacy_data_root / "dashboard_meta",
    ],
    "outputs": [
        PROJECT_PATHS.project_root / "analysis_results",
        PROJECT_PATHS.project_root / "pattern_discovery_results",
        PROJECT_PATHS.project_root / "feature_analysis_report",
        PROJECT_PATHS.project_root / "feature_analysis" / "output",
    ],
    "reports": [
        PROJECT_PATHS.project_root / "reporting",
    ],
    "logs": [
        PROJECT_PATHS.project_root / "trading_bot" / "logs",
    ],
}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_files(root: Path):
    if root.is_file():
        if root.suffix not in EXCLUDED_SUFFIXES:
            yield root
        return
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix not in EXCLUDED_SUFFIXES:
            yield candidate


def _build_plan(target_root: Path) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for group, sources in MIGRATION_GROUPS.items():
        for source in sources:
            destination = _target_for_group(target_root, group, source)
            file_count = sum(1 for _ in _iter_files(source)) if source.exists() else 0
            plan.append(
                {
                    "group": group,
                    "source": str(source),
                    "destination": str(destination),
                    "exists": source.exists(),
                    "file_count": file_count,
                }
            )
    return plan


def _copy_and_verify(source: Path, destination: Path, dry_run: bool) -> tuple[int, int]:
    copied = 0
    verified = 0

    for file_path in _iter_files(source):
        relative = file_path.relative_to(source)
        target_path = destination / relative
        print(f"{'DRY' if dry_run else 'COPY'} {file_path} -> {target_path}")

        if dry_run:
            copied += 1
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_path)
        copied += 1

        if file_path.stat().st_size == target_path.stat().st_size and _sha256(file_path) == _sha256(target_path):
            verified += 1
        else:
            raise RuntimeError(f"Verification failed for {file_path}")

    return copied, verified


def _target_for_group(target_root: Path, group: str, source: Path) -> Path:
    if group == "raw":
        return target_root / "raw" / "market"
    if group == "processed":
        return target_root / "processed" / "cycles_enriched"
    if group == "dashboard":
        if source.name == "dashboard_data":
            return target_root / "dashboard" / "candles"
        if source.name == "dashboard_meta":
            return target_root / "dashboard" / "meta"
        return target_root / "dashboard" / source.name
    if group == "outputs":
        return target_root / "outputs" / source.name
    if group == "reports":
        return target_root / "reports" / source.name
    if group == "logs":
        return target_root / "logs" / source.name
    return target_root / group / source.name


def _write_manifest(
    target_root: Path,
    plan: list[dict[str, object]],
    copied: int,
    verified: int,
    dry_run: bool,
) -> Path:
    manifest_dir = target_root / "logs" / "migration"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "latest_migration_manifest.json"
    manifest = {
        "target_root": str(target_root),
        "shared_config_root": str(PROJECT_PATHS.data_root),
        "dry_run": dry_run,
        "copied": copied,
        "verified": verified,
        "plan": plan,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy legacy repo data into the external MACD data root.")
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET))
    parser.add_argument("--dry-run", action="store_true", help="Show planned copies without writing files.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the migration plan without copying files.",
    )
    args = parser.parse_args()

    target_root = Path(args.target_root)
    print(f"Target root: {target_root}")
    print(f"Shared config root: {PROJECT_PATHS.data_root}")
    print("Mode: copy + verify")

    plan = _build_plan(target_root)
    print("\nMigration plan")
    for item in plan:
        status = "OK" if item["exists"] else "MISSING"
        print(
            f"- [{status}] {item['group']}: {item['source']} -> {item['destination']} "
            f"(files={item['file_count']})"
        )

    if args.plan_only:
        print("\nPlan-only mode complete.")
        return 0

    total_copied = 0
    total_verified = 0

    for group, sources in MIGRATION_GROUPS.items():
        for source in sources:
            if not source.exists():
                print(f"SKIP missing source: {source}")
                continue

            destination = _target_for_group(target_root, group, source)
            copied, verified = _copy_and_verify(source, destination, dry_run=args.dry_run)
            total_copied += copied
            total_verified += verified

    manifest_path = _write_manifest(
        target_root=target_root,
        plan=plan,
        copied=total_copied,
        verified=total_verified,
        dry_run=args.dry_run,
    )

    print(f"\nSummary: copied={total_copied}, verified={total_verified}, dry_run={args.dry_run}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

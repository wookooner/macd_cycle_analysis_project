from __future__ import annotations

"""Copy old microstructure files into the provider-first storage layout.

The command never deletes source data. Preview first, then opt in to writes
with ``--apply`` after confirming the plan and available disk space.
"""

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.microstructure.paths import (
    LEGACY_BINANCE_PUBLIC_ARCHIVE_ROOT,
    LEGACY_MICROSTRUCTURE_ROOT,
    BINANCE_PUBLIC_ARCHIVE_ROOT,
    MICROSTRUCTURE_ROOT,
)


@dataclass(frozen=True)
class MigrationTarget:
    label: str
    source: Path
    destination: Path


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _files(root: Path):
    if root.exists():
        yield from (path for path in root.rglob("*") if path.is_file())


def _targets(include_archive: bool) -> list[MigrationTarget]:
    targets = [
        MigrationTarget(
            label="normalized microstructure parquet",
            source=LEGACY_MICROSTRUCTURE_ROOT,
            destination=MICROSTRUCTURE_ROOT,
        )
    ]
    if include_archive:
        targets.append(
            MigrationTarget(
                label="Binance public ZIP archive",
                source=LEGACY_BINANCE_PUBLIC_ARCHIVE_ROOT / "futures" / "um",
                destination=BINANCE_PUBLIC_ARCHIVE_ROOT,
            )
        )
    return targets


def _copy_target(target: MigrationTarget, apply: bool, overwrite: bool, verbose: bool) -> tuple[int, int, int]:
    copied = skipped = verified = 0
    for source_file in _files(target.source):
        destination_file = target.destination / source_file.relative_to(target.source)
        if destination_file.exists() and not overwrite:
            skipped += 1
            continue
        if verbose:
            print(f"{'COPY' if apply else 'PLAN'} {source_file} -> {destination_file}")
        if not apply:
            copied += 1
            continue
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        copied += 1
        if source_file.stat().st_size != destination_file.stat().st_size or _sha256(source_file) != _sha256(destination_file):
            raise RuntimeError(f"Verification failed: {source_file}")
        verified += 1
    return copied, skipped, verified


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate microstructure data to the canonical provider-first layout.")
    parser.add_argument("--apply", action="store_true", help="Copy files. Without this flag, only print the plan.")
    parser.add_argument("--include-archive", action="store_true", help="Also copy old data.binance.vision ZIP files to archive/.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files after copying.")
    parser.add_argument("--verbose", action="store_true", help="Print every planned or copied file.")
    args = parser.parse_args()

    totals = [0, 0, 0]
    for target in _targets(args.include_archive):
        count = sum(1 for _ in _files(target.source))
        print(f"\n{target.label}: {target.source} -> {target.destination} (files={count})")
        copied, skipped, verified = _copy_target(target, args.apply, args.overwrite, args.verbose)
        totals[0] += copied
        totals[1] += skipped
        totals[2] += verified

    if args.apply:
        print(f"\nMigration applied: copied={totals[0]}, skipped={totals[1]}, verified={totals[2]}")
    else:
        print(f"\nMigration plan only: planned={totals[0]}, already_present={totals[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

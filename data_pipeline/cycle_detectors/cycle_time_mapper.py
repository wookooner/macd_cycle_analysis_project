"""Build hierarchy links between cycle parquet files across timeframes."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.common.paths import PROJECT_PATHS


CANDLE_DURATION = {
    "1min": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
    "1M": pd.Timedelta(days=30),
}


class CycleHierarchyMapper:
    """Map parent and child relationships between adjacent timeframe cycles."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.timeframe_order = ["1M", "1w", "1d", "4h", "1h", "30m", "15m", "5m", "1min"]
        self.cycles_by_timeframe: Dict[str, pd.DataFrame] = {}
        self.hierarchy_map: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def load_all_cycles(self) -> Dict[str, pd.DataFrame]:
        """Load all available cycle parquet files and normalize timestamps."""
        print("[map] loading cycle parquet files")

        for timeframe in self.timeframe_order:
            file_path = self.data_dir / f"cycles_{timeframe}.parquet"
            if not file_path.exists():
                print(f"  [skip] {timeframe}: file not found")
                continue

            df = pd.read_parquet(file_path)
            df["start_datetime"] = self._safe_datetime_convert(df["start_date"])
            df["end_datetime"] = self._safe_datetime_convert(df["end_date"])
            duration = CANDLE_DURATION.get(timeframe, pd.Timedelta(hours=1))
            df["end_datetime_adj"] = df["end_datetime"] + duration
            df = df.sort_values("start_datetime").reset_index(drop=True)

            self.cycles_by_timeframe[timeframe] = df
            print(f"  [loaded] {timeframe}: {len(df)} cycles (duration +{duration})")

        return self.cycles_by_timeframe

    def _safe_datetime_convert(self, date_series: pd.Series) -> pd.Series:
        """Convert timestamps robustly across string, second, and millisecond inputs."""
        for kwargs in ({}, {"unit": "s"}, {"unit": "ms"}):
            try:
                return pd.to_datetime(date_series, **kwargs)
            except (ValueError, TypeError):
                continue

        sample = date_series.iloc[0] if len(date_series) > 0 else "N/A"
        print(f"[warn] datetime conversion failed: {sample}")
        return date_series

    def build_hierarchy_map(self, min_overlap: float = 0.01):
        """Build hierarchy links using adjacent timeframe pairs only."""
        _ = min_overlap
        print("\n[map] building hierarchy relationships")

        self._initialize_hierarchy_map()

        adjacent_pairs = [
            (self.timeframe_order[i], self.timeframe_order[i + 1])
            for i in range(len(self.timeframe_order) - 1)
            if self.timeframe_order[i] in self.cycles_by_timeframe
            and self.timeframe_order[i + 1] in self.cycles_by_timeframe
        ]

        total_pairs = len(adjacent_pairs)
        for pair_index, (parent_tf, child_tf) in enumerate(adjacent_pairs, start=1):
            progress_pct = pair_index / max(total_pairs, 1) * 100
            print(f"  [pair {pair_index}/{total_pairs} {progress_pct:.1f}%] {parent_tf} -> {child_tf}")
            link_count = self._map_pair(parent_tf, child_tf)
            print(f"    [pair complete] {link_count:,} links")

        print("\n[map] hierarchy mapping complete")
        self._print_statistics()
        return self.hierarchy_map

    def _initialize_hierarchy_map(self) -> None:
        self.hierarchy_map = {}
        for timeframe in self.timeframe_order:
            df = self.cycles_by_timeframe.get(timeframe)
            if df is None:
                continue

            bucket: Dict[str, Dict[str, Any]] = {}
            for row in df.itertuples(index=False):
                bucket[row.cycle_id] = {
                    "cycle_type": row.cycle_type,
                    "start_date": row.start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_date": row.end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_candles": int(row.duration_candles),
                    "parent_cycle_ids": {},
                    "child_cycle_ids": {},
                }
            self.hierarchy_map[timeframe] = bucket

    def _map_pair(self, parent_tf: str, child_tf: str) -> int:
        """Map overlaps with a two-pointer sweep over sorted intervals."""
        parent_df = self.cycles_by_timeframe[parent_tf]
        child_df = self.cycles_by_timeframe[child_tf]

        parent_starts = parent_df["start_datetime"].tolist()
        parent_ends = parent_df["end_datetime_adj"].tolist()
        parent_ids = parent_df["cycle_id"].tolist()

        child_rows = list(child_df.itertuples(index=False))
        parent_count = len(parent_ids)
        child_count = len(child_rows)
        if parent_count == 0 or child_count == 0:
            return 0

        progress_every = max(child_count // 20, 1)
        parent_left = 0
        link_count = 0

        for child_index, child_row in enumerate(child_rows, start=1):
            child_start = child_row.start_datetime
            child_end = child_row.end_datetime_adj
            child_id = child_row.cycle_id

            while parent_left < parent_count and parent_ends[parent_left] <= child_start:
                parent_left += 1

            parent_index = parent_left
            while parent_index < parent_count and parent_starts[parent_index] < child_end:
                if parent_ends[parent_index] > child_start:
                    parent_id = parent_ids[parent_index]

                    child_list = self.hierarchy_map[parent_tf][parent_id]["child_cycle_ids"]
                    child_list.setdefault(child_tf, []).append(child_id)

                    parent_list = self.hierarchy_map[child_tf][child_id]["parent_cycle_ids"]
                    parent_list.setdefault(parent_tf, []).append(parent_id)

                    link_count += 1

                parent_index += 1

            if child_index == child_count or child_index % progress_every == 0:
                progress_pct = child_index / child_count * 100
                suffix = "\r" if child_index != child_count else "\n"
                print(
                    f"    [child progress {child_index:,}/{child_count:,} {progress_pct:5.1f}%]",
                    end=suffix,
                )

        return link_count

    def _print_statistics(self) -> None:
        """Print simple hierarchy coverage statistics."""
        print("\n[map] hierarchy statistics")

        for timeframe in self.timeframe_order:
            if timeframe not in self.hierarchy_map:
                continue

            total = len(self.hierarchy_map[timeframe])
            has_parent = sum(
                1
                for cycle_info in self.hierarchy_map[timeframe].values()
                if any(cycle_info["parent_cycle_ids"].values())
            )
            has_child = sum(
                1
                for cycle_info in self.hierarchy_map[timeframe].values()
                if any(cycle_info["child_cycle_ids"].values())
            )

            print(f"\n  {timeframe}:")
            print(f"    - total: {total}")
            if total > 0:
                print(f"    - has parent: {has_parent} ({has_parent / total * 100:.1f}%)")
                print(f"    - has child: {has_child} ({has_child / total * 100:.1f}%)")

        if "1min" in self.hierarchy_map:
            total_1min = len(self.hierarchy_map["1min"])
            complete_1min = sum(
                1
                for cycle_info in self.hierarchy_map["1min"].values()
                if cycle_info["parent_cycle_ids"].get("5m")
            )
            if total_1min:
                print(
                    f"\n  minute direct-parent coverage: "
                    f"{complete_1min}/{total_1min} ({complete_1min / total_1min * 100:.1f}%)"
                )

        if "1h" in self.hierarchy_map:
            total_1h = len(self.hierarchy_map["1h"])
            complete_1h = sum(
                1
                for cycle_info in self.hierarchy_map["1h"].values()
                if cycle_info["parent_cycle_ids"].get("4h")
                and cycle_info["child_cycle_ids"].get("30m")
            )
            if total_1h:
                print(
                    f"\n  hourly adjacent coverage: "
                    f"{complete_1h}/{total_1h} ({complete_1h / total_1h * 100:.1f}%)"
                )

    def check_overlap(
        self,
        cycle1_start: datetime,
        cycle1_end: datetime,
        cycle2_start: datetime,
        cycle2_end: datetime,
    ) -> float:
        """Return overlap ratio based on the first interval."""
        if cycle1_end < cycle2_start or cycle2_end < cycle1_start:
            return 0.0

        overlap_start = max(cycle1_start, cycle2_start)
        overlap_end = min(cycle1_end, cycle2_end)
        cycle1_duration = (cycle1_end - cycle1_start).total_seconds()
        if cycle1_duration == 0:
            return 0.0

        return (overlap_end - overlap_start).total_seconds() / cycle1_duration

    def save_hierarchy_map(self, output_path: Path | None = None):
        """Save the hierarchy map to JSON."""
        if output_path is None:
            output_path = self.data_dir / "cycle_hierarchy_map.json"

        clean_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for timeframe, cycles in self.hierarchy_map.items():
            clean_map[timeframe] = {}
            for cycle_id, info in cycles.items():
                clean_map[timeframe][cycle_id] = {
                    key: value
                    for key, value in info.items()
                    if not isinstance(value, (pd.Timestamp, datetime))
                }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(clean_map, handle, indent=2, ensure_ascii=False)

        print(f"\n[map] hierarchy map saved: {output_path}")
        return output_path

    def get_cycle_family(self, cycle_id: str) -> Dict[str, Any]:
        """Return parents, children, and siblings for one cycle."""
        cycle_timeframe = None
        for timeframe, cycles in self.hierarchy_map.items():
            if cycle_id in cycles:
                cycle_timeframe = timeframe
                break

        if not cycle_timeframe:
            return {}

        cycle_info = self.hierarchy_map[cycle_timeframe][cycle_id]
        siblings: Dict[str, List[str]] = defaultdict(list)

        for parent_tf, parent_ids in cycle_info["parent_cycle_ids"].items():
            for parent_id in parent_ids:
                parent_info = self.hierarchy_map[parent_tf][parent_id]
                for sibling_id in parent_info["child_cycle_ids"].get(cycle_timeframe, []):
                    if sibling_id != cycle_id:
                        siblings[parent_tf].append(sibling_id)

        return {
            "cycle_id": cycle_id,
            "timeframe": cycle_timeframe,
            "parents": cycle_info["parent_cycle_ids"],
            "children": cycle_info["child_cycle_ids"],
            "siblings": dict(siblings),
        }

    def analyze_cycle_patterns(self) -> Dict[str, Dict[str, int]]:
        """Summarize alignment vs counter-trend counts across adjacent pairs."""
        patterns: Dict[str, Dict[str, int]] = {"alignment": {}, "counter": {}}

        for index in range(len(self.timeframe_order) - 1):
            parent_tf = self.timeframe_order[index]
            child_tf = self.timeframe_order[index + 1]
            if child_tf not in self.hierarchy_map or parent_tf not in self.hierarchy_map:
                continue

            alignment_count = 0
            counter_count = 0

            for cycle_info in self.hierarchy_map[child_tf].values():
                child_type = cycle_info["cycle_type"]
                for parent_id in cycle_info["parent_cycle_ids"].get(parent_tf, []):
                    parent_type = self.hierarchy_map[parent_tf][parent_id]["cycle_type"]
                    if child_type == parent_type:
                        alignment_count += 1
                    else:
                        counter_count += 1

            key = f"{child_tf}_in_{parent_tf}"
            patterns["alignment"][key] = alignment_count
            patterns["counter"][key] = counter_count

        return patterns


def _find_project_root() -> Path:
    """Find the project root directory."""
    if PROJECT_PATHS.project_root.exists():
        return PROJECT_PATHS.project_root

    for current in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        if (current / "data" / "base_data").exists():
            return current

    for current in [Path.cwd()] + list(Path.cwd().parents):
        if (current / "data" / "base_data").exists():
            return current

    return Path(__file__).resolve().parent.parent.parent


def main():
    """CLI entrypoint."""
    project_root = _find_project_root()
    data_dir = PROJECT_PATHS.cycle_structured_dir

    print("=" * 60)
    print("cycle hierarchy mapping start")
    print("=" * 60)
    print(f"\nproject root: {project_root}")
    print(f"data directory: {data_dir}")
    print(f"exists: {'yes' if data_dir.exists() else 'no'}")
    print("\ntimeframe hierarchy:")
    print("  1M -> 1w -> 1d -> 4h -> 1h -> 30m -> 15m -> 5m -> 1min")
    print("\ncandle duration adjustment enabled")

    mapper = CycleHierarchyMapper(data_dir)
    mapper.load_all_cycles()
    hierarchy_map = mapper.build_hierarchy_map(min_overlap=0.01)
    output_path = mapper.save_hierarchy_map()

    print("\nmap validation:")
    for sample_tf in ["1M", "1w", "1min"]:
        if sample_tf in mapper.hierarchy_map and mapper.hierarchy_map[sample_tf]:
            sample_id = next(iter(mapper.hierarchy_map[sample_tf]))
            info = mapper.hierarchy_map[sample_tf][sample_id]
            print(f"\n  sample {sample_tf}: {sample_id}")
            for parent_tf, parent_ids in sorted(info["parent_cycle_ids"].items()):
                print(f"     parent {parent_tf}: {len(parent_ids)}")
            for child_tf, child_ids in sorted(info["child_cycle_ids"].items()):
                print(f"     child {child_tf}: {len(child_ids)}")

    print("\ncycle pattern analysis:")
    patterns = mapper.analyze_cycle_patterns()
    print("\n  alignment:")
    for key, count in patterns["alignment"].items():
        total = count + patterns["counter"].get(key, 0)
        if total > 0:
            print(f"    {key}: {count} ({count / total * 100:.1f}%)")

    print("\n  counter:")
    for key, count in patterns["counter"].items():
        total = count + patterns["alignment"].get(key, 0)
        if total > 0:
            print(f"    {key}: {count} ({count / total * 100:.1f}%)")

    print(f"\nmap complete: {output_path}")
    return hierarchy_map


if __name__ == "__main__":
    main()

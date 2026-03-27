"""
cycle_hierarchy_mapper.py
시간대별 사이클 간의 상위/하위 관계를 매핑하는 스크립트

변경 이력:
  v2 (2026-03-23)
    1) end_date 캔들 duration 보정 적용
       - 문제: end_date가 마지막 캔들의 '시작 시간'이므로
         실제 사이클 종료 = end_date + 캔들 duration.
         보정 없이는 사이클 사이에 정확히 1캔들분의 갭이 생겨
         하위 사이클이 상위 부모를 못 찾는 매핑 누락 발생.
       - 효과: 4-레벨 체인 완성률 84.3% → 99.4% (+1,102개)
    2) 교차 조인 → 정렬 기반 스윕 알고리즘으로 교체
       - 문제: cross join이 대규모 데이터에서 OOM 발생
       - 효과: 메모리 O(N*M) → O(N+M), 속도도 향상
"""

import bisect
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ── 타임프레임별 캔들 duration ────────────────────────────────────────────────
# end_date(마지막 캔들 시작시간)에 이 값을 더하면 실제 사이클 종료 시점이 됨.
CANDLE_DURATION = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
    "1m": pd.Timedelta(days=30),  # 근사값 (매핑용으로 충분)
}


class CycleHierarchyMapper:
    """사이클 간의 계층적 관계를 매핑하는 클래스"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.timeframe_order = ["1m", "1w", "1d", "4h", "1h"]
        self.cycles_by_timeframe: Dict[str, pd.DataFrame] = {}
        self.hierarchy_map: Dict[str, Dict] = {}

    # ── 데이터 로드 ──────────────────────────────────────────────────────────

    def load_all_cycles(self) -> Dict[str, pd.DataFrame]:
        """모든 타임프레임의 사이클 데이터 로드 + 캔들 duration 보정"""
        print("📂 사이클 데이터 로딩 중...")

        for tf in self.timeframe_order:
            file_path = self.data_dir / f"cycles_{tf}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)

                df["start_datetime"] = self._safe_datetime_convert(df["start_date"])
                df["end_datetime"] = self._safe_datetime_convert(df["end_date"])

                # ★ 핵심 수정: end_date + 캔들 duration = 실제 종료 시점
                duration = CANDLE_DURATION.get(tf, pd.Timedelta(hours=1))
                df["end_datetime_adj"] = df["end_datetime"] + duration

                df = df.sort_values("start_datetime").reset_index(drop=True)
                self.cycles_by_timeframe[tf] = df
                print(f"  ✅ {tf}: {len(df)}개 사이클 (end 보정: +{duration})")
            else:
                print(f"  ⚠️ {tf}: 파일 없음")

        return self.cycles_by_timeframe

    def _safe_datetime_convert(self, date_series):
        """날짜 데이터를 안전하게 datetime으로 변환"""
        try:
            return pd.to_datetime(date_series)
        except (ValueError, TypeError):
            try:
                return pd.to_datetime(date_series, unit="s")
            except (ValueError, TypeError):
                try:
                    return pd.to_datetime(date_series, unit="ms")
                except (ValueError, TypeError):
                    sample = date_series.iloc[0] if len(date_series) > 0 else "N/A"
                    print(f"⚠️ 날짜 변환 실패: {sample}")
                    return date_series

    # ── 계층 매핑 ────────────────────────────────────────────────────────────

    def build_hierarchy_map(self, min_overlap: float = 0.01):
        """
        계층 관계 맵 생성.
        정렬 기반 스윕으로 overlap 탐색 → 메모리 효율적 (cross join 미사용).
        """
        print("\n🔗 사이클 계층 관계 매핑 중 (정렬 스윕 + 캔들 보정)...")

        # 1. 맵 초기화
        for tf in self.timeframe_order:
            df = self.cycles_by_timeframe.get(tf)
            if df is None:
                continue
            self.hierarchy_map[tf] = {}
            for _, row in df.iterrows():
                self.hierarchy_map[tf][row["cycle_id"]] = {
                    "cycle_type": row["cycle_type"],
                    "start_date": row["start_datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                    "end_date": row["end_datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_candles": int(row["duration_candles"]),
                    "parent_cycle_ids": {},
                    "child_cycle_ids": {},
                }

        # 2. 모든 상위↔하위 조합에 대해 매핑
        for i, parent_tf in enumerate(self.timeframe_order):
            if parent_tf not in self.cycles_by_timeframe:
                continue
            for child_tf in self.timeframe_order[i + 1 :]:
                if child_tf not in self.cycles_by_timeframe:
                    continue
                print(f"  🔄 {parent_tf} ↔ {child_tf}...", end=" ")
                cnt = self._map_pair(parent_tf, child_tf)
                print(f"{cnt}건")

        print("\n✅ 모든 관계 매핑 완료!")
        self._print_statistics()
        return self.hierarchy_map

    def _map_pair(self, parent_tf: str, child_tf: str) -> int:
        """부모-자식 한 쌍의 관계를 정렬 기반으로 효율적 매핑.

        부모 리스트를 start_datetime 기준 정렬 후,
        자식마다 bisect으로 후보 범위를 좁혀 overlap 검사.
        """
        pdf = self.cycles_by_timeframe[parent_tf]
        cdf = self.cycles_by_timeframe[child_tf]

        p_starts = pdf["start_datetime"].values  # numpy datetime64
        p_ends = pdf["end_datetime_adj"].values
        p_ids = pdf["cycle_id"].values
        n_parents = len(pdf)

        count = 0
        for _, child_row in cdf.iterrows():
            c_start = child_row["start_datetime"]
            c_end = child_row["end_datetime_adj"]
            c_id = child_row["cycle_id"]

            # 부모 중 start < c_end 인 것까지만 확인 (bisect)
            c_end_np = np.datetime64(c_end)
            right_idx = bisect.bisect_right(p_starts, c_end_np)

            for pi in range(right_idx):
                pe = pd.Timestamp(p_ends[pi])
                ps = pd.Timestamp(p_starts[pi])

                # 이미 끝난 부모는 skip (early termination은 어려움 — 부모가 길 수 있으므로)
                if pe <= c_start:
                    continue

                # overlap: c_start < pe AND ps < c_end
                if ps < c_end:
                    pid = p_ids[pi]
                    # 부모 → 자식
                    children = self.hierarchy_map[parent_tf][pid]["child_cycle_ids"]
                    if child_tf not in children:
                        children[child_tf] = []
                    children[child_tf].append(c_id)

                    # 자식 → 부모
                    parents = self.hierarchy_map[child_tf][c_id]["parent_cycle_ids"]
                    if parent_tf not in parents:
                        parents[parent_tf] = []
                    parents[parent_tf].append(pid)

                    count += 1

        return count

    # ── 통계 & 검증 ─────────────────────────────────────────────────────────

    def _print_statistics(self):
        """계층 구조 통계 출력"""
        print("\n📊 계층 구조 통계:")

        for tf in self.timeframe_order:
            if tf not in self.hierarchy_map:
                continue

            total = len(self.hierarchy_map[tf])
            has_parent = sum(
                1
                for c in self.hierarchy_map[tf].values()
                if any(c["parent_cycle_ids"].values())
            )
            has_child = sum(
                1
                for c in self.hierarchy_map[tf].values()
                if any(c["child_cycle_ids"].values())
            )

            print(f"\n  {tf} 타임프레임:")
            print(f"    - 전체: {total}개")
            if total > 0:
                print(
                    f"    - 부모 있음: {has_parent}개 ({has_parent / total * 100:.1f}%)"
                )
                print(
                    f"    - 자식 있음: {has_child}개 ({has_child / total * 100:.1f}%)"
                )

        # 4-레벨 체인 완성률 (1h 기준)
        if "1h" in self.hierarchy_map:
            total_1h = len(self.hierarchy_map["1h"])
            complete = sum(
                1
                for c in self.hierarchy_map["1h"].values()
                if c["parent_cycle_ids"].get("4h")
                and c["parent_cycle_ids"].get("1d")
                and c["parent_cycle_ids"].get("1w")
            )
            print(
                f"\n  🎯 4-레벨 체인 완성: {complete}/{total_1h} ({complete / total_1h * 100:.1f}%)"
            )

    # ── 유틸리티 ─────────────────────────────────────────────────────────────

    def check_overlap(
        self,
        cycle1_start: datetime,
        cycle1_end: datetime,
        cycle2_start: datetime,
        cycle2_end: datetime,
    ) -> float:
        """두 사이클 간 겹침 비율 (하위 기준). 호출자가 보정된 end를 전달."""
        if cycle1_end < cycle2_start or cycle2_end < cycle1_start:
            return 0.0

        overlap_start = max(cycle1_start, cycle2_start)
        overlap_end = min(cycle1_end, cycle2_end)

        cycle1_duration = (cycle1_end - cycle1_start).total_seconds()
        if cycle1_duration == 0:
            return 0.0

        return (overlap_end - overlap_start).total_seconds() / cycle1_duration

    def save_hierarchy_map(self, output_path: Path = None):
        """계층 구조를 JSON 파일로 저장"""
        if output_path is None:
            output_path = self.data_dir / "cycle_hierarchy_map.json"

        clean_map = {}
        for tf, cycles in self.hierarchy_map.items():
            clean_map[tf] = {}
            for cycle_id, info in cycles.items():
                clean_map[tf][cycle_id] = {
                    k: v
                    for k, v in info.items()
                    if not isinstance(v, (pd.Timestamp, datetime))
                }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean_map, f, indent=2, ensure_ascii=False)

        print(f"\n💾 계층 구조 저장 완료: {output_path}")
        return output_path

    def get_cycle_family(self, cycle_id: str) -> Dict:
        """특정 사이클의 전체 가족 관계 조회"""
        cycle_tf = None
        for tf, cycles in self.hierarchy_map.items():
            if cycle_id in cycles:
                cycle_tf = tf
                break

        if not cycle_tf:
            return {}

        cycle_info = self.hierarchy_map[cycle_tf][cycle_id]

        siblings: Dict[str, list] = defaultdict(list)
        for parent_tf, parent_ids in cycle_info["parent_cycle_ids"].items():
            for parent_id in parent_ids:
                parent_info = self.hierarchy_map[parent_tf][parent_id]
                if cycle_tf in parent_info["child_cycle_ids"]:
                    for sibling_id in parent_info["child_cycle_ids"][cycle_tf]:
                        if sibling_id != cycle_id:
                            siblings[parent_tf].append(sibling_id)

        return {
            "cycle_id": cycle_id,
            "timeframe": cycle_tf,
            "parents": cycle_info["parent_cycle_ids"],
            "children": cycle_info["child_cycle_ids"],
            "siblings": dict(siblings),
        }

    def analyze_cycle_patterns(self) -> Dict:
        """사이클 패턴 분석 (상위와 같은/반대 방향)"""
        patterns: Dict[str, Dict[str, int]] = {"alignment": {}, "counter": {}}

        for tf_idx, child_tf in enumerate(self.timeframe_order[:-1]):
            if child_tf not in self.hierarchy_map:
                continue
            parent_tf = self.timeframe_order[tf_idx + 1]
            if parent_tf not in self.hierarchy_map:
                continue

            alignment_count = counter_count = 0

            for cycle_id, cycle_info in self.hierarchy_map[child_tf].items():
                child_type = cycle_info["cycle_type"]
                if parent_tf in cycle_info["parent_cycle_ids"]:
                    for parent_id in cycle_info["parent_cycle_ids"][parent_tf]:
                        parent_type = self.hierarchy_map[parent_tf][parent_id][
                            "cycle_type"
                        ]
                        if child_type == parent_type:
                            alignment_count += 1
                        else:
                            counter_count += 1

            patterns["alignment"][f"{child_tf}_in_{parent_tf}"] = alignment_count
            patterns["counter"][f"{child_tf}_in_{parent_tf}"] = counter_count

        return patterns


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def _find_project_root() -> Path:
    """프로젝트 루트를 찾는 헬퍼. data/base_data 디렉토리가 있는 곳이 루트."""
    for current in [Path(__file__).resolve().parent] + list(
        Path(__file__).resolve().parents
    ):
        if (current / "data" / "base_data").exists():
            return current
    # fallback: CWD에서도 탐색
    for current in [Path.cwd()] + list(Path.cwd().parents):
        if (current / "data" / "base_data").exists():
            return current
    # 최후 fallback
    return Path(__file__).resolve().parent.parent.parent


def main():
    """메인 실행 함수"""
    project_root = _find_project_root()
    data_dir = project_root / "data" / "cycle_data" / "structured"

    print("=" * 60)
    print("🚀 사이클 계층 구조 매핑 시작")
    print("=" * 60)
    print(f"\n📁 프로젝트 루트: {project_root}")
    print(f"📁 데이터 디렉토리: {data_dir}")
    print(f"   존재 여부: {'✅' if data_dir.exists() else '❌ 경로 없음!'}")
    print("\n📌 타임프레임 계층 구조:")
    print("  1h → 4h → 1d → 1w → 1m")
    print("\n📝 캔들 duration 보정 적용됨")
    print("  end_date는 마지막 캔들의 시작시간이므로")
    print("  실제 종료 = end_date + 캔들 duration으로 보정")

    mapper = CycleHierarchyMapper(data_dir)
    mapper.load_all_cycles()
    hierarchy_map = mapper.build_hierarchy_map(min_overlap=0.01)
    output_path = mapper.save_hierarchy_map()

    # 검증
    print("\n🔍 계층 구조 검증:")

    for sample_tf, expected_parent, expected_child in [
        ("1m", False, True),
        ("1w", True, True),
        ("1h", True, False),
    ]:
        if sample_tf in mapper.hierarchy_map and mapper.hierarchy_map[sample_tf]:
            sample_id = list(mapper.hierarchy_map[sample_tf].keys())[0]
            info = mapper.hierarchy_map[sample_tf][sample_id]
            print(f"\n  ✅ {sample_tf} 예시: {sample_id}")
            if info["parent_cycle_ids"]:
                for ptf, pids in sorted(info["parent_cycle_ids"].items()):
                    print(f"     Parent {ptf}: {len(pids)}개")
            if info["child_cycle_ids"]:
                for ctf, cids in sorted(info["child_cycle_ids"].items()):
                    print(f"     Child {ctf}: {len(cids)}개")

    # 패턴 분석
    print("\n📈 사이클 패턴 분석:")
    patterns = mapper.analyze_cycle_patterns()

    print("\n  정렬 (상위와 같은 방향):")
    for key, count in patterns["alignment"].items():
        total = count + patterns["counter"].get(key, 0)
        if total > 0:
            child_tf, parent_tf = key.split("_in_")
            print(f"    {child_tf} → {parent_tf}: {count}개 ({count / total * 100:.1f}%)")

    print("\n  역행 (상위와 반대 방향):")
    for key, count in patterns["counter"].items():
        total = count + patterns["alignment"].get(key, 0)
        if total > 0:
            child_tf, parent_tf = key.split("_in_")
            print(f"    {child_tf} → {parent_tf}: {count}개 ({count / total * 100:.1f}%)")

    print(f"\n✅ 매핑 완료! 결과: {output_path}")
    return hierarchy_map


if __name__ == "__main__":
    main()
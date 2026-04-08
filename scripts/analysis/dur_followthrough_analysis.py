from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_PATHS


TIMEFRAMES = ("1d", "4h", "1h")
DEFAULT_CONFIRM_DUR = 3
DEFAULT_CONFIRM_DURS = (3, 4, 5)
DEFAULT_NOISE_POSITIONS = (4, 5, 6)
DEFAULT_PATTERN_NORMAL_COUNT = 3
DEFAULT_PATTERN_NOISE_COUNT = 6
MAX_OPPOSITE_CONSECUTIVE = 2
MIN_CYCLE_LENGTH = 3


@dataclass
class CycleInfo:
    start_pos: int
    end_pos: int
    start_idx: int
    end_idx: int
    cycle_type: str
    length: int


def find_project_root() -> Path:
    return PROJECT_PATHS.project_root


def calculate_directions(macd_hist: pd.Series) -> list[int]:
    values = macd_hist.astype(float).tolist()
    if not values:
        return []

    directions = [0]
    for idx in range(1, len(values)):
        current = values[idx]
        previous = values[idx - 1]
        if current > previous:
            directions.append(1)
        elif current < previous:
            directions.append(-1)
        else:
            directions.append(0)
    return directions


def find_cycle_end(
    directions: list[int],
    start_idx: int,
    main_direction: int,
    max_opposite: int = MAX_OPPOSITE_CONSECUTIVE,
) -> int:
    consecutive_opposite = 0
    last_valid_end = start_idx

    for idx in range(start_idx + 1, len(directions)):
        current_direction = directions[idx]
        if current_direction == main_direction:
            consecutive_opposite = 0
            last_valid_end = idx
        elif current_direction == -main_direction:
            consecutive_opposite += 1
            if consecutive_opposite > max_opposite:
                break

    return last_valid_end


def detect_cycles(
    df: pd.DataFrame,
    min_cycle_length: int = MIN_CYCLE_LENGTH,
    max_opposite: int = MAX_OPPOSITE_CONSECUTIVE,
) -> list[CycleInfo]:
    valid = df["macd_hist"].notna().copy()
    if not valid.any():
        return []

    working = df.loc[valid].reset_index(drop=False).rename(columns={"index": "original_index"})
    directions = calculate_directions(working["macd_hist"])
    cycles: list[CycleInfo] = []
    idx = 0

    while idx < len(directions):
        if directions[idx] not in (1, -1):
            idx += 1
            continue

        start_idx = idx
        main_direction = directions[idx]
        end_idx = find_cycle_end(directions, start_idx, main_direction, max_opposite=max_opposite)
        length = end_idx - start_idx + 1

        if length >= min_cycle_length:
            cycle_type = "up" if main_direction == 1 else "down"
            cycles.append(
                CycleInfo(
                    start_pos=start_idx,
                    end_pos=end_idx,
                    start_idx=int(working.loc[start_idx, "original_index"]),
                    end_idx=int(working.loc[end_idx, "original_index"]),
                    cycle_type=cycle_type,
                    length=length,
                )
            )

        idx = end_idx + 1

    return cycles


def prepare_working_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    working = df.loc[df["macd_hist"].notna()].reset_index(drop=False).rename(columns={"index": "original_index"})
    directions = calculate_directions(working["macd_hist"])
    return working, directions


def load_timeframe_data(project_root: Path, timeframe: str) -> pd.DataFrame:
    csv_path = PROJECT_PATHS.base_data_dir / f"BTCUSD_{timeframe}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing timeframe csv: {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "close" not in df.columns or "macd_hist" not in df.columns:
        raise ValueError(f"required columns missing in {csv_path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "macd_hist"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _percentile(series: pd.Series, q: float) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return round(float(clean.quantile(q)), 4)


def _round_or_none(value: Any, digits: int = 4) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.floating, float, int, np.integer)):
        return round(float(value), digits)
    return value


def summarize_cases(cases: pd.DataFrame) -> dict[str, Any]:
    if cases.empty:
        return {"count": 0}

    continuation = cases["additional_move_pct_signed"]
    remaining_candles = cases["remaining_candles"]
    mfe = cases["max_favorable_excursion_pct"]
    mae = cases["max_adverse_excursion_pct"]

    return {
        "count": int(len(cases)),
        "continuation_rate_pct": _round_or_none((continuation > 0).mean() * 100),
        "flat_or_better_rate_pct": _round_or_none((continuation >= 0).mean() * 100),
        "avg_remaining_candles": _round_or_none(remaining_candles.mean()),
        "median_remaining_candles": _round_or_none(remaining_candles.median()),
        "avg_additional_move_pct": _round_or_none(continuation.mean()),
        "median_additional_move_pct": _round_or_none(continuation.median()),
        "p25_additional_move_pct": _percentile(continuation, 0.25),
        "p75_additional_move_pct": _percentile(continuation, 0.75),
        "avg_max_favorable_excursion_pct": _round_or_none(mfe.mean()),
        "median_max_favorable_excursion_pct": _round_or_none(mfe.median()),
        "avg_max_adverse_excursion_pct": _round_or_none(mae.mean()),
        "median_max_adverse_excursion_pct": _round_or_none(mae.median()),
    }


def analyze_timeframe(df: pd.DataFrame, timeframe: str, confirm_dur: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    cycles = detect_cycles(df)
    working, _ = prepare_working_dataframe(df)
    rows: list[dict[str, Any]] = []

    for cycle_number, cycle in enumerate(cycles, start=1):
        if cycle.length < confirm_dur:
            continue

        confirm_idx = int(working.loc[cycle.start_pos + confirm_dur - 1, "original_index"])
        if confirm_idx > cycle.end_idx:
            continue

        confirm_close = df.at[confirm_idx, "close"]
        end_close = df.at[cycle.end_idx, "close"]
        if pd.isna(confirm_close) or pd.isna(end_close) or confirm_close == 0:
            continue

        future = df.iloc[confirm_idx + 1 : cycle.end_idx + 1]
        sign = 1.0 if cycle.cycle_type == "up" else -1.0
        additional_move_pct_signed = ((end_close / confirm_close) - 1.0) * 100.0 * sign

        if future.empty:
            max_favorable_excursion_pct = 0.0
            max_adverse_excursion_pct = 0.0
        elif cycle.cycle_type == "up":
            max_favorable_excursion_pct = ((future["high"].max() / confirm_close) - 1.0) * 100.0
            max_adverse_excursion_pct = ((future["low"].min() / confirm_close) - 1.0) * 100.0
        else:
            max_favorable_excursion_pct = ((confirm_close / future["low"].min()) - 1.0) * 100.0
            max_adverse_excursion_pct = ((confirm_close / future["high"].max()) - 1.0) * 100.0

        rows.append(
            {
                "timeframe": timeframe,
                "cycle_number": cycle_number,
                "cycle_type": cycle.cycle_type,
                "cycle_start_date": df.at[cycle.start_idx, "date"],
                "confirm_date": df.at[confirm_idx, "date"],
                "cycle_end_date": df.at[cycle.end_idx, "date"],
                "start_idx": cycle.start_idx,
                "confirm_idx": confirm_idx,
                "end_idx": cycle.end_idx,
                "confirm_dur": confirm_dur,
                "final_cycle_length": cycle.length,
                "remaining_candles": cycle.end_idx - confirm_idx,
                "confirm_close": confirm_close,
                "end_close": end_close,
                "additional_move_pct_signed": additional_move_pct_signed,
                "additional_move_pct_raw": ((end_close / confirm_close) - 1.0) * 100.0,
                "max_favorable_excursion_pct": max_favorable_excursion_pct,
                "max_adverse_excursion_pct": max_adverse_excursion_pct,
            }
        )

    cases = pd.DataFrame(rows)
    if not cases.empty:
        cases = cases.sort_values("confirm_date").reset_index(drop=True)

    summary = {
        "timeframe": timeframe,
        "confirm_dur": confirm_dur,
        "all": summarize_cases(cases),
        "up": summarize_cases(cases[cases["cycle_type"] == "up"]) if not cases.empty else {"count": 0},
        "down": summarize_cases(cases[cases["cycle_type"] == "down"]) if not cases.empty else {"count": 0},
    }
    return cases, summary


def summarize_duration_cases(cases: pd.DataFrame) -> dict[str, Any]:
    if cases.empty:
        return {"count": 0}

    final_dur = cases["final_cycle_length"]
    remain = cases["remaining_candles_after_noise"]
    move = cases["additional_move_from_noise_pct_signed"]

    return {
        "count": int(len(cases)),
        "avg_final_dur": _round_or_none(final_dur.mean()),
        "median_final_dur": _round_or_none(final_dur.median()),
        "p25_final_dur": _percentile(final_dur, 0.25),
        "p75_final_dur": _percentile(final_dur, 0.75),
        "avg_remaining_candles_after_noise": _round_or_none(remain.mean()),
        "median_remaining_candles_after_noise": _round_or_none(remain.median()),
        "continuation_rate_after_noise_pct": _round_or_none((move > 0).mean() * 100),
        "flat_or_better_after_noise_pct": _round_or_none((move >= 0).mean() * 100),
        "avg_additional_move_from_noise_pct": _round_or_none(move.mean()),
        "median_additional_move_from_noise_pct": _round_or_none(move.median()),
    }


def analyze_early_noise_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    noise_positions: tuple[int, ...] = DEFAULT_NOISE_POSITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working, directions = prepare_working_dataframe(df)
    cycles = detect_cycles(df)
    rows: list[dict[str, Any]] = []

    for cycle_number, cycle in enumerate(cycles, start=1):
        main_direction = 1 if cycle.cycle_type == "up" else -1
        end_close = df.at[cycle.end_idx, "close"]
        if pd.isna(end_close):
            continue

        for noise_position in noise_positions:
            if cycle.length < noise_position:
                continue

            pos_in_working = cycle.start_pos + noise_position - 1
            candle_direction = directions[pos_in_working]
            is_noise = candle_direction == -main_direction
            if not is_noise:
                continue

            candle_index = int(working.loc[pos_in_working, "original_index"])
            candle_close = df.at[candle_index, "close"]
            if pd.isna(candle_close) or candle_close == 0:
                continue

            additional_move_pct_signed = ((end_close / candle_close) - 1.0) * 100.0 * float(main_direction)

            rows.append(
                {
                    "timeframe": timeframe,
                    "cycle_number": cycle_number,
                    "cycle_type": cycle.cycle_type,
                    "noise_position": noise_position,
                    "cycle_start_date": df.at[cycle.start_idx, "date"],
                    "noise_candle_date": df.at[candle_index, "date"],
                    "cycle_end_date": df.at[cycle.end_idx, "date"],
                    "final_cycle_length": cycle.length,
                    "remaining_candles_after_noise": cycle.length - noise_position,
                    "noise_candle_close": candle_close,
                    "end_close": end_close,
                    "additional_move_from_noise_pct_signed": additional_move_pct_signed,
                }
            )

    cases = pd.DataFrame(rows)
    if not cases.empty:
        cases = cases.sort_values(["noise_position", "noise_candle_date"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for noise_position in noise_positions:
        position_cases = cases[cases["noise_position"] == noise_position] if not cases.empty else pd.DataFrame()
        stats = summarize_duration_cases(position_cases)
        summary_rows.append(
            {
                "timeframe": timeframe,
                "noise_position": noise_position,
                **stats,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return cases, summary_df


def _direction_label(value: int) -> str:
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return "0"


def find_noise_sequence_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    cycle_type: str = "up",
    normal_count: int = DEFAULT_PATTERN_NORMAL_COUNT,
    noise_count: int = DEFAULT_PATTERN_NOISE_COUNT,
) -> pd.DataFrame:
    working, directions = prepare_working_dataframe(df)
    cycles = detect_cycles(df)
    target_cycle_type = str(cycle_type).lower()
    rows: list[dict[str, Any]] = []
    required_length = normal_count + noise_count

    for cycle_number, cycle in enumerate(cycles, start=1):
        if cycle.cycle_type != target_cycle_type:
            continue
        if cycle.length < required_length:
            continue

        main_direction = 1 if cycle.cycle_type == "up" else -1
        direction_window = directions[cycle.start_pos : cycle.start_pos + required_length]
        expected_window = [main_direction] * normal_count + [-main_direction] * noise_count
        if direction_window != expected_window:
            continue

        candle_indexes = [
            int(working.loc[cycle.start_pos + offset, "original_index"])
            for offset in range(required_length)
        ]
        first_noise_index = candle_indexes[normal_count]
        last_noise_index = candle_indexes[-1]

        rows.append(
            {
                "timeframe": timeframe,
                "cycle_number": cycle_number,
                "cycle_type": cycle.cycle_type,
                "cycle_start_date": df.at[cycle.start_idx, "date"],
                "cycle_end_date": df.at[cycle.end_idx, "date"],
                "final_cycle_length": cycle.length,
                "normal_count": normal_count,
                "noise_count": noise_count,
                "pattern_length": required_length,
                "pattern_end_date": df.at[last_noise_index, "date"],
                "first_noise_date": df.at[first_noise_index, "date"],
                "last_noise_date": df.at[last_noise_index, "date"],
                "start_idx": cycle.start_idx,
                "end_idx": cycle.end_idx,
                "direction_sequence": "".join(_direction_label(value) for value in direction_window),
                "close_sequence": [float(df.at[idx, "close"]) if not pd.isna(df.at[idx, "close"]) else None for idx in candle_indexes],
                "date_sequence": [df.at[idx, "date"] for idx in candle_indexes],
            }
        )

    cases = pd.DataFrame(rows)
    if not cases.empty:
        cases = cases.sort_values(["timeframe", "cycle_start_date"]).reset_index(drop=True)
    return cases


def run_noise_sequence_search(
    cycle_type: str = "up",
    normal_count: int = DEFAULT_PATTERN_NORMAL_COUNT,
    noise_count: int = DEFAULT_PATTERN_NOISE_COUNT,
) -> dict[str, Any]:
    project_root = find_project_root()
    output_dir = PROJECT_PATHS.outputs_root / "analysis_results" / "dur_followthrough_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for timeframe in TIMEFRAMES:
        df = load_timeframe_data(project_root, timeframe)
        cases = find_noise_sequence_timeframe(
            df,
            timeframe=timeframe,
            cycle_type=cycle_type,
            normal_count=normal_count,
            noise_count=noise_count,
        )
        all_cases.append(cases)
        summary_rows.append(
            {
                "timeframe": timeframe,
                "cycle_type": cycle_type,
                "normal_count": normal_count,
                "noise_count": noise_count,
                "count": int(len(cases)),
            }
        )
        cases.to_csv(
            output_dir / f"{timeframe}_{cycle_type}_normal{normal_count}_noise{noise_count}_pattern_cases.csv",
            index=False,
            encoding="utf-8-sig",
        )

    combined_cases = pd.concat(all_cases, ignore_index=True) if all_cases else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        output_dir / f"{cycle_type}_normal{normal_count}_noise{noise_count}_pattern_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        output_dir / f"{cycle_type}_normal{normal_count}_noise{noise_count}_pattern_summary.json"
    ).write_text(
        json.dumps(
            {
                "cycle_type": cycle_type,
                "normal_count": normal_count,
                "noise_count": noise_count,
                "summary": json.loads(summary_df.to_json(orient="records", force_ascii=False)),
                "cases": json.loads(combined_cases.to_json(orient="records", date_format="iso", force_ascii=False)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "cycle_type": cycle_type,
        "normal_count": normal_count,
        "noise_count": noise_count,
        "summary": summary_rows,
        "cases_found": int(len(combined_cases)),
    }


def to_jsonable(summary: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(summary, default=str))


def build_markdown_report(all_summaries: dict[str, dict[str, Any]], confirm_dur: int) -> str:
    lines = [
        "# DUR Followthrough Analysis",
        "",
        f"- 기준: `dur >= {confirm_dur}` 확정 시점(해당 사이클의 {confirm_dur}번째 캔들 종가) 이후 추가 진행 통계",
        f"- 사이클 규칙: `min_cycle_length={MIN_CYCLE_LENGTH}`, `max_opposite_consecutive={MAX_OPPOSITE_CONSECUTIVE}`",
        "- `additional_move_pct`는 방향 보정값입니다. `up`/`down` 모두 양수일수록 사이클 방향으로 더 진행했다는 뜻입니다.",
        "- `max_adverse_excursion_pct`는 확정 이후 가장 불리했던 움직임입니다.",
        "",
    ]

    for timeframe in TIMEFRAMES:
        summary = all_summaries[timeframe]
        lines.append(f"## {timeframe}")
        lines.append("")
        lines.append("| 구분 | count | continuation_rate | avg_remaining_candles | avg_additional_move_pct | median_additional_move_pct | avg_MFE | avg_MAE |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for label in ("all", "up", "down"):
            stats = summary[label]
            lines.append(
                "| {label} | {count} | {continuation} | {remain} | {move} | {median_move} | {mfe} | {mae} |".format(
                    label=label,
                    count=stats.get("count", 0),
                    continuation=stats.get("continuation_rate_pct"),
                    remain=stats.get("avg_remaining_candles"),
                    move=stats.get("avg_additional_move_pct"),
                    median_move=stats.get("median_additional_move_pct"),
                    mfe=stats.get("avg_max_favorable_excursion_pct"),
                    mae=stats.get("avg_max_adverse_excursion_pct"),
                )
            )
        lines.append("")

    return "\n".join(lines)


def build_multi_dur_report(comparison_df: pd.DataFrame) -> str:
    lines = [
        "# Multi-DUR Followthrough Comparison",
        "",
        "- 기준: 같은 사이클 규칙으로 `dur` 확정 시점을 바꿔가며 후속 진행 통계 비교",
        "- `avg_additional_move_pct`는 방향 보정값이라 `up/down` 모두 양수일수록 더 진행한 것입니다.",
        "",
    ]

    for timeframe in TIMEFRAMES:
        subset = comparison_df[comparison_df["timeframe"] == timeframe].copy()
        lines.append(f"## {timeframe}")
        lines.append("")
        lines.append("| dur | count | continuation_rate_pct | avg_remaining_candles | avg_additional_move_pct | median_additional_move_pct | avg_MFE | avg_MAE |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in subset.iterrows():
            lines.append(
                "| {dur} | {count} | {continuation} | {remain} | {move} | {median_move} | {mfe} | {mae} |".format(
                    dur=int(row["confirm_dur"]),
                    count=int(row["count"]),
                    continuation=_round_or_none(row["continuation_rate_pct"]),
                    remain=_round_or_none(row["avg_remaining_candles"]),
                    move=_round_or_none(row["avg_additional_move_pct"]),
                    median_move=_round_or_none(row["median_additional_move_pct"]),
                    mfe=_round_or_none(row["avg_max_favorable_excursion_pct"]),
                    mae=_round_or_none(row["avg_max_adverse_excursion_pct"]),
                )
            )
        lines.append("")

    return "\n".join(lines)


def build_noise_report(summary_df: pd.DataFrame) -> str:
    lines = [
        "# Early Noise Candle DUR Analysis",
        "",
        "- 정의: 사이클의 4번째/5번째/6번째 캔들이 주방향과 반대 방향이면 해당 위치를 `초반 노이즈캔들`로 간주",
        "- 통계는 그 조건을 만족한 사이클만 모아서 `최종 dur`, `노이즈 이후 남은 캔들`, `노이즈 이후 추가 진행률`을 요약",
        "",
    ]

    for timeframe in TIMEFRAMES:
        subset = summary_df[summary_df["timeframe"] == timeframe].copy()
        lines.append(f"## {timeframe}")
        lines.append("")
        lines.append("| noise_position | count | avg_final_dur | median_final_dur | avg_remaining_after_noise | continuation_after_noise_pct | avg_move_from_noise_pct |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in subset.iterrows():
            lines.append(
                "| {pos} | {count} | {avg_dur} | {median_dur} | {remain} | {continuation} | {move} |".format(
                    pos=int(row["noise_position"]),
                    count=int(row["count"]),
                    avg_dur=_round_or_none(row["avg_final_dur"]),
                    median_dur=_round_or_none(row["median_final_dur"]),
                    remain=_round_or_none(row["avg_remaining_candles_after_noise"]),
                    continuation=_round_or_none(row["continuation_rate_after_noise_pct"]),
                    move=_round_or_none(row["avg_additional_move_from_noise_pct"]),
                )
            )
        lines.append("")

    return "\n".join(lines)


def save_outputs(
    output_dir: Path,
    timeframe_cases: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    confirm_dur: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for timeframe, cases in timeframe_cases.items():
        csv_path = output_dir / f"{timeframe}_dur{confirm_dur}_followthrough_cases.csv"
        if cases.empty:
            pd.DataFrame(
                columns=[
                    "timeframe",
                    "cycle_number",
                    "cycle_type",
                    "cycle_start_date",
                    "confirm_date",
                    "cycle_end_date",
                    "confirm_dur",
                    "final_cycle_length",
                    "remaining_candles",
                    "confirm_close",
                    "end_close",
                    "additional_move_pct_signed",
                    "max_favorable_excursion_pct",
                    "max_adverse_excursion_pct",
                ]
            ).to_csv(csv_path, index=False, encoding="utf-8-sig")
        else:
            cases.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_path = output_dir / f"dur{confirm_dur}_followthrough_summary.json"
    report_path = output_dir / f"dur{confirm_dur}_followthrough_report.md"

    payload = {
        "confirm_dur": confirm_dur,
        "timeframes": to_jsonable(summaries),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_markdown_report(summaries, confirm_dur), encoding="utf-8")


def build_comparison_dataframe(multi_summaries: dict[int, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for confirm_dur, summaries in multi_summaries.items():
        for timeframe in TIMEFRAMES:
            stats = summaries[timeframe]["all"]
            rows.append(
                {
                    "timeframe": timeframe,
                    "confirm_dur": confirm_dur,
                    "count": stats.get("count", 0),
                    "continuation_rate_pct": stats.get("continuation_rate_pct"),
                    "flat_or_better_rate_pct": stats.get("flat_or_better_rate_pct"),
                    "avg_remaining_candles": stats.get("avg_remaining_candles"),
                    "median_remaining_candles": stats.get("median_remaining_candles"),
                    "avg_additional_move_pct": stats.get("avg_additional_move_pct"),
                    "median_additional_move_pct": stats.get("median_additional_move_pct"),
                    "avg_max_favorable_excursion_pct": stats.get("avg_max_favorable_excursion_pct"),
                    "avg_max_adverse_excursion_pct": stats.get("avg_max_adverse_excursion_pct"),
                }
            )
    return pd.DataFrame(rows).sort_values(["timeframe", "confirm_dur"]).reset_index(drop=True)


def save_multi_dur_outputs(
    output_dir: Path,
    multi_summaries: dict[int, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = build_comparison_dataframe(multi_summaries)
    comparison_csv_path = output_dir / "dur_comparison_summary.csv"
    comparison_json_path = output_dir / "dur_comparison_summary.json"
    comparison_report_path = output_dir / "dur_comparison_report.md"

    comparison_df.to_csv(comparison_csv_path, index=False, encoding="utf-8-sig")
    comparison_json_path.write_text(
        json.dumps({"confirm_durs": list(multi_summaries.keys()), "results": to_jsonable(multi_summaries)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison_report_path.write_text(build_multi_dur_report(comparison_df), encoding="utf-8")
    return comparison_df


def run_early_noise_analysis(
    noise_positions: tuple[int, ...] = DEFAULT_NOISE_POSITIONS,
) -> pd.DataFrame:
    project_root = find_project_root()
    output_dir = PROJECT_PATHS.outputs_root / "analysis_results" / "dur_followthrough_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    for timeframe in TIMEFRAMES:
        df = load_timeframe_data(project_root, timeframe)
        cases, summary_df = analyze_early_noise_timeframe(df, timeframe=timeframe, noise_positions=noise_positions)
        all_cases.append(cases)
        all_summaries.append(summary_df)
        cases.to_csv(output_dir / f"{timeframe}_early_noise_dur_cases.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.concat(all_summaries, ignore_index=True)
    combined_cases = pd.concat(all_cases, ignore_index=True) if all_cases else pd.DataFrame()

    summary_df.to_csv(output_dir / "early_noise_dur_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "early_noise_dur_summary.json").write_text(
        json.dumps(
            {
                "noise_positions": list(noise_positions),
                "summary": json.loads(summary_df.to_json(orient="records", force_ascii=False)),
                "cases": json.loads(combined_cases.to_json(orient="records", date_format="iso", force_ascii=False)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "early_noise_dur_report.md").write_text(build_noise_report(summary_df), encoding="utf-8")
    return summary_df


def run(confirm_dur: int = DEFAULT_CONFIRM_DUR) -> dict[str, dict[str, Any]]:
    project_root = find_project_root()
    output_dir = PROJECT_PATHS.outputs_root / "analysis_results" / "dur_followthrough_analysis"

    timeframe_cases: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for timeframe in TIMEFRAMES:
        df = load_timeframe_data(project_root, timeframe)
        cases, summary = analyze_timeframe(df, timeframe=timeframe, confirm_dur=confirm_dur)
        timeframe_cases[timeframe] = cases
        summaries[timeframe] = summary

    save_outputs(output_dir, timeframe_cases, summaries, confirm_dur=confirm_dur)
    return summaries


def run_multi(confirm_durs: tuple[int, ...] = DEFAULT_CONFIRM_DURS) -> dict[int, dict[str, dict[str, Any]]]:
    results: dict[int, dict[str, dict[str, Any]]] = {}
    project_root = find_project_root()
    output_dir = PROJECT_PATHS.outputs_root / "analysis_results" / "dur_followthrough_analysis"

    for confirm_dur in confirm_durs:
        results[int(confirm_dur)] = run(confirm_dur=int(confirm_dur))

    save_multi_dur_outputs(output_dir, results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze post-confirmation followthrough after dur confirmation.")
    parser.add_argument("--confirm-dur", type=int, default=DEFAULT_CONFIRM_DUR, help="Confirmation dur threshold.")
    parser.add_argument(
        "--confirm-durs",
        type=int,
        nargs="+",
        help="Run multiple dur thresholds and save comparison outputs.",
    )
    parser.add_argument(
        "--early-noise",
        action="store_true",
        help="Analyze dur statistics when the 4th/5th/6th candle is an early noise candle.",
    )
    parser.add_argument(
        "--find-noise-sequence",
        action="store_true",
        help="Find cycles that start with N normal candles followed by M noise candles.",
    )
    parser.add_argument(
        "--pattern-cycle-type",
        default="up",
        choices=("up", "down"),
        help="Cycle type to search when --find-noise-sequence is used.",
    )
    parser.add_argument(
        "--pattern-normal-count",
        type=int,
        default=DEFAULT_PATTERN_NORMAL_COUNT,
        help="Leading normal candle count for --find-noise-sequence.",
    )
    parser.add_argument(
        "--pattern-noise-count",
        type=int,
        default=DEFAULT_PATTERN_NOISE_COUNT,
        help="Trailing noise candle count for --find-noise-sequence.",
    )
    args = parser.parse_args()

    if args.find_noise_sequence:
        results = run_noise_sequence_search(
            cycle_type=args.pattern_cycle_type,
            normal_count=args.pattern_normal_count,
            noise_count=args.pattern_noise_count,
        )
    elif args.early_noise:
        results = run_early_noise_analysis()
    elif args.confirm_durs:
        results = run_multi(confirm_durs=tuple(args.confirm_durs))
    else:
        results = run(confirm_dur=args.confirm_dur)
    print(json.dumps(to_jsonable(results), ensure_ascii=False, indent=2))

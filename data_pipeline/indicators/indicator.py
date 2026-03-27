# C:\Users\Administrator\Desktop\macd_cycle_analysis_project\data_control\indicator.py

import logging
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from data_pipeline.utils.io import atomic_write_csv, prune_backup_files

warnings.filterwarnings("ignore")


class IndicatorCalculator:
    def __init__(self, log_level="INFO"):
        self.setup_logging(log_level)

    def setup_logging(self, log_level):
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def check_existing_indicators(self, df, check_last_n=50):
        """기존 지표 계산 상태 확인"""
        self.logger.info("=== 기존 지표 상태 확인 ===")

        indicator_columns = [
            "macd",
            "macd_signal",
            "macd_hist",
            "ppo",
            "ppo_signal",
            "ppo_hist",
            "rsi",
            "delta",
            "cvd",
            "cvd_rolling",
            "ma_7",
            "ma_25",
            "ma_99",
        ]
        status = {}

        for col in indicator_columns:
            if col not in df.columns:
                status[col] = {
                    "exists": False,
                    "calculated_rows": 0,
                    "missing_rows": len(df),
                    "last_n_calculated": 0,
                    "last_n_missing": min(check_last_n, len(df)),
                }
                continue

            valid_count = df[col].notna().sum()
            missing_count = df[col].isna().sum()
            last_n_data = df[col].tail(check_last_n)
            last_n_valid = last_n_data.notna().sum()
            coverage = valid_count / len(df) if len(df) > 0 else 0

            status[col] = {
                "exists": True,
                "calculated_rows": valid_count,
                "missing_rows": missing_count,
                "coverage_ratio": coverage,
                "last_n_calculated": last_n_valid,
                "last_n_missing": len(last_n_data) - last_n_valid,
                "last_n_coverage": last_n_valid / len(last_n_data) if len(last_n_data) > 0 else 0,
            }

        for col, info in status.items():
            if info["exists"]:
                self.logger.info(
                    f"{col}: 전체 {info['calculated_rows']}/{len(df)} 계산됨 "
                    f"({info['coverage_ratio']:.1%}), 마지막 {check_last_n}개 중 "
                    f"{info['last_n_calculated']}개 계산됨 ({info['last_n_coverage']:.1%})"
                )
            else:
                self.logger.info(f"{col}: 컬럼이 존재하지 않음")

        return status

    def calculate_ema(self, data, period, adjust=True):
        return data.ewm(span=period, adjust=adjust).mean()

    def calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """MACD 지표 계산"""
        self.logger.info(
            f"MACD 계산 중.. (Fast:{fast_period}, Slow:{slow_period}, Signal:{signal_period})"
        )
        if "close" not in df.columns:
            self.logger.error("close 컬럼이 없습니다.")
            return df

        close_prices = df["close"].copy()
        ema_fast = self.calculate_ema(close_prices, fast_period)
        ema_slow = self.calculate_ema(close_prices, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        df["macd"] = macd_line.round(6)
        df["macd_signal"] = signal_line.round(6)
        df["macd_hist"] = histogram.round(6)

        self.logger.info(f"MACD 계산 완료: {df['macd'].notna().sum()}개 값 생성")
        return df

    def calculate_ppo(self, df, fast_period=12, slow_period=26, signal_period=9):
        """PPO 지표 계산"""
        self.logger.info(
            f"PPO 계산 중.. (Fast:{fast_period}, Slow:{slow_period}, Signal:{signal_period})"
        )
        if "close" not in df.columns:
            self.logger.error("close 컬럼이 없습니다.")
            return df

        close_prices = df["close"].copy()
        ema_fast = self.calculate_ema(close_prices, fast_period)
        ema_slow = self.calculate_ema(close_prices, slow_period)
        denominator = ema_slow.replace(0, np.nan)

        ppo_line = ((ema_fast - ema_slow) / denominator) * 100
        signal_line = self.calculate_ema(ppo_line, signal_period)
        histogram = ppo_line - signal_line

        df["ppo"] = ppo_line.round(6)
        df["ppo_signal"] = signal_line.round(6)
        df["ppo_hist"] = histogram.round(6)

        self.logger.info(f"PPO 계산 완료: {df['ppo'].notna().sum()}개 값 생성")
        return df

    def calculate_rsi(self, df, period=14):
        """RSI 지표 계산"""
        self.logger.info(f"RSI 계산 중.. (Period: {period})")
        if "close" not in df.columns:
            self.logger.error("close 컬럼이 없습니다.")
            return df

        close_prices = df["close"].copy()
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df["rsi"] = rsi.round(2)

        self.logger.info(f"RSI 계산 완료: {df['rsi'].notna().sum()}개 값 생성")
        return df

    def calculate_ma(self, df, periods=(7, 25, 99)):
        """이동평균(SMA) 계산. 이미 존재하고 마지막 값까지 유효하면 건너뜀."""
        if "close" not in df.columns:
            self.logger.error("close 컬럼이 없습니다.")
            return df

        close_prices = df["close"].copy()
        calculated = []

        for period in periods:
            col_name = f"ma_{period}"

            # 이미 존재하고 마지막 행이 유효하면 건너뜀 (새 데이터 추가 감지)
            if col_name in df.columns and len(df) >= period:
                if pd.notna(df[col_name].iloc[-1]):
                    continue

            df[col_name] = close_prices.rolling(window=period, min_periods=period).mean().round(2)
            calculated.append(col_name)

        if calculated:
            self.logger.info(f"MA 계산 완료: {', '.join(calculated)}")
        else:
            self.logger.info("MA: 기존 값 유지 (재계산 불필요)")

        return df

    def calculate_cvd(self, df, rolling_period=20):
        """delta, CVD, rolling CVD 계산"""
        self.logger.info(f"CVD 계산 중.. (rolling_period: {rolling_period})")

        has_trade_columns = "taker_buy_base" in df.columns and "volume" in df.columns
        has_legacy_delta = "volume_delta" in df.columns

        if not has_trade_columns and not has_legacy_delta:
            self.logger.warning("delta/CVD 계산을 위한 컬럼이 없습니다.")
            df["delta"] = pd.NA
            df["cvd"] = pd.NA
            df["cvd_rolling"] = pd.NA
            return df

        if has_trade_columns:
            taker_buy_base = pd.to_numeric(df["taker_buy_base"], errors="coerce")
            volume = pd.to_numeric(df["volume"], errors="coerce")
            taker_sell_base = volume - taker_buy_base
            delta_series = taker_buy_base - taker_sell_base
        else:
            delta_series = pd.to_numeric(df["volume_delta"], errors="coerce")

        df["delta"] = delta_series.round(4)
        df["volume_delta"] = df["delta"]

        valid_mask = delta_series.notna()
        cvd_series = pd.Series(np.nan, index=df.index, dtype="float64")
        if valid_mask.any():
            cvd_series.loc[valid_mask] = delta_series.loc[valid_mask].cumsum()

        cvd_rolling_series = pd.Series(np.nan, index=df.index, dtype="float64")
        if valid_mask.any():
            cvd_rolling_series.loc[valid_mask] = (
                delta_series.loc[valid_mask].rolling(window=rolling_period, min_periods=1).sum()
            )

        df["cvd"] = cvd_series.round(4)
        df["cvd_rolling"] = cvd_rolling_series.round(4)

        valid_cvd = df["cvd"].notna().sum()
        self.logger.info(f"CVD 계산 완료: {valid_cvd}개 값 생성")
        return df

    def update_indicators(self, df, force_recalculate=False, recalc_last_n=50, cvd_rolling_period=20):
        """지표 업데이트"""
        self.logger.info("=== 지표 업데이트 시작 ===")
        status = self.check_existing_indicators(df, recalc_last_n)

        full_recalc_needed = (
            force_recalculate
            or not status["macd"]["exists"]
            or not status["ppo"]["exists"]
            or not status["rsi"]["exists"]
            or status["macd"]["coverage_ratio"] < 0.5
            or status["ppo"]["coverage_ratio"] < 0.5
            or status["rsi"]["coverage_ratio"] < 0.5
        )

        cvd_recalc_needed = (
            force_recalculate
            or not status["cvd"]["exists"]
            or not status["delta"]["exists"]
            or status["cvd"]["coverage_ratio"] < 0.01
        )

        if full_recalc_needed:
            self.logger.info("전체 지표를 새로 계산합니다.")
            df = self.calculate_macd(df)
            df = self.calculate_ppo(df)
            df = self.calculate_rsi(df)
        else:
            self.logger.info(f"마지막 {recalc_last_n}개 데이터의 지표를 재계산합니다.")
            df = self.recalculate_last_indicators(df, recalc_last_n, cvd_rolling_period)

        if cvd_recalc_needed:
            df = self.calculate_cvd(df, rolling_period=cvd_rolling_period)
        else:
            self.logger.info("CVD: 기존 값 유지 (재계산 불필요)")

        # MA는 항상 계산 (내부에서 이미 존재하면 자동 skip)
        df = self.calculate_ma(df)

        return df

    def recalculate_last_indicators(self, df, last_n=50, cvd_rolling_period=20):
        """마지막 N개 데이터 지표 재계산"""
        total_rows = len(df)
        if total_rows <= last_n:
            df = self.calculate_macd(df)
            df = self.calculate_ppo(df)
            df = self.calculate_rsi(df)
            return df

        macd_recalc_start = max(0, total_rows - max(last_n, 60))
        rsi_recalc_start = max(0, total_rows - max(last_n, 30))

        self.logger.info(f"MACD/PPO 재계산 구간: {macd_recalc_start}~{total_rows}")
        self.logger.info(f"RSI 재계산 구간: {rsi_recalc_start}~{total_rows}")

        df_temp = self.calculate_macd(df.copy())
        df_temp = self.calculate_ppo(df_temp)
        df_temp = self.calculate_rsi(df_temp)

        df.loc[macd_recalc_start:, ["macd", "macd_signal", "macd_hist"]] = df_temp.loc[
            macd_recalc_start:, ["macd", "macd_signal", "macd_hist"]
        ]
        df.loc[macd_recalc_start:, ["ppo", "ppo_signal", "ppo_hist"]] = df_temp.loc[
            macd_recalc_start:, ["ppo", "ppo_signal", "ppo_hist"]
        ]
        df.loc[rsi_recalc_start:, "rsi"] = df_temp.loc[rsi_recalc_start:, "rsi"]

        df = self.calculate_cvd(df, rolling_period=cvd_rolling_period)

        self.logger.info("마지막 구간 지표 재계산 완료")
        return df

    def process_file(
        self,
        file_path,
        output_path=None,
        force_recalculate=False,
        recalc_last_n=50,
        cvd_rolling_period=20,
    ):
        """파일 단위 지표 계산 처리"""
        self.logger.info(f"파일 처리 시작: {file_path}")

        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"데이터 로드 완료: {len(df)} rows, {len(df.columns)} columns")

            if "unix" in df.columns:
                df = df.sort_values("unix").reset_index(drop=True)
                self.logger.info("데이터를 unix timestamp 기준으로 정렬했습니다.")

            df = self.update_indicators(
                df,
                force_recalculate,
                recalc_last_n,
                cvd_rolling_period=cvd_rolling_period,
            )

            if output_path is None:
                output_path = file_path

            if output_path == file_path:
                backup_dir = file_path.parent.parent / "backup_data"
                backup_dir.mkdir(parents=True, exist_ok=True)

                backup_filename = f"{file_path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = backup_dir / backup_filename

                shutil.copy2(file_path, backup_path)
                prune_backup_files(backup_dir)
                self.logger.info(f"백업 생성: {backup_path}")

            atomic_write_csv(df, output_path)
            self.logger.info(f"처리 완료: {output_path}")

            return True

        except Exception as e:
            self.logger.error(f"파일 처리 실패: {str(e)}")
            return False


def process_multiple_files(file_paths, force_recalculate=False, recalc_last_n=50, cvd_rolling_period=20):
    """여러 파일을 한 번에 처리"""
    calculator = IndicatorCalculator()
    results = {}

    calculator.logger.info(f"=== 배치 처리 시작: {len(file_paths)}개 파일 ===")

    for file_path in file_paths:
        result = calculator.process_file(
            file_path,
            force_recalculate=force_recalculate,
            recalc_last_n=recalc_last_n,
            cvd_rolling_period=cvd_rolling_period,
        )
        results[file_path] = result

    return results


if __name__ == "__main__":
    data_dir = Path("data/base_data")

    print("=== 경로 확인 ===")
    print(f"데이터 디렉터리: {data_dir.resolve()}")
    print(f"디렉터리 존재 여부: {data_dir.exists()}")

    if not data_dir.exists():
        print("데이터 디렉터리 'data/base_data'를 찾을 수 없습니다.")
        raise SystemExit(1)

    files = ["BTCUSD_1h.csv", "BTCUSD_4h.csv", "BTCUSD_1d.csv", "BTCUSD_1w.csv", "BTCUSD_1m.csv"]
    existing_files = [data_dir / f for f in files if (data_dir / f).exists()]

    print("\n--- 처리 대상 파일 ---")
    for f in existing_files:
        print(f"- {f.name}")
    if not existing_files:
        print("처리할 CSV 파일이 없습니다.")
        raise SystemExit(1)

    print("\n지표 계산 설정:")
    print("1. 마지막 구간만 업데이트")
    print("2. 전체 재계산")

    mode_choice = input("모드를 선택하세요 (1/2): ").strip()

    if mode_choice == "1":
        force_recalc = False
        recalc_n = 50
    elif mode_choice == "2":
        force_recalc = True
        recalc_n = 50
    else:
        print("잘못된 선택입니다.")
        raise SystemExit(1)

    print(f"\n지표 계산 시작... (총 {len(existing_files)}개 파일)")
    results = process_multiple_files(
        existing_files,
        force_recalculate=force_recalc,
        recalc_last_n=recalc_n,
    )

    print("\n--- 처리 결과 ---")
    for file_path, success in results.items():
        status = "성공" if success else "실패"
        print(f"  {Path(file_path).name}: {status}")

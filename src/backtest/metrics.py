from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.types import BasisConfig, MetricsRow


def sample_class(n: int, low_n: int = 100, reliable_n: int = 1000) -> str:
    if n < low_n:
        return "low_sample"
    if n < reliable_n:
        return "medium_sample"
    return "reliable_sample"


def _mdd_stats(returns_pct: pd.Series, position_size: float) -> tuple[float, int, int, float, bool]:
    returns = returns_pct.fillna(0.0).astype(float) * position_size / 100.0
    if returns.empty:
        return np.nan, 0, 0, np.nan, False
    equity = (1.0 + returns).cumprod()
    ruin = bool((equity <= 0).any())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    underwater = drawdown < 0
    max_duration = 0
    current = 0
    for value in underwater:
        current = current + 1 if value else 0
        max_duration = max(max_duration, current)
    trough = int(drawdown.argmin())
    peak_value = running_max.iloc[trough]
    after = equity.iloc[trough:]
    recovered = np.where(after >= peak_value)[0]
    recovery = int(recovered[0]) if len(recovered) else int(len(after) - 1)
    return (
        float(drawdown.min() * 100.0),
        max_duration,
        recovery,
        float(underwater.mean() * 100.0),
        ruin,
    )


def compute_metrics(
    ledger: pd.DataFrame,
    basis_config: BasisConfig | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute standardized metrics from a realized trade ledger only."""

    basis = basis_config or BasisConfig()
    group_cols = group_cols or ["rule_name", "rule_kind", "direction", "exit_policy"]
    if ledger.empty:
        return pd.DataFrame()

    required = {"gross_return", "net_return", "exit_reason", "holding_bars", "holding_hours"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger missing required metric columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    grouped = [((), ledger)] if not group_cols else ledger.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        net = group["net_return"].astype(float)
        gross = group["gross_return"].astype(float)
        wins = net[net > 0].sum()
        losses = -net[net < 0].sum()
        cum = net.cumsum()
        cum_dd = cum - cum.cummax()
        mdd025, dur, rec, tuw, ruin025 = _mdd_stats(net, 0.25)
        mdd050, _, _, _, ruin050 = _mdd_stats(net, 0.50)
        mdd100, _, _, _, ruin100 = _mdd_stats(net, 1.00)
        std = net.std(ddof=0)
        exit_dist = group["exit_reason"].astype(str).value_counts(normalize=True).mul(100).round(3).to_dict()

        row = dict(zip(group_cols, keys)) if group_cols else {}
        row.update(
            {
                "sample_class": sample_class(len(group), basis.sample_low_n, basis.sample_reliable_n),
                "n_trades": int(len(group)),
                "win_rate": float((net > 0).mean() * 100.0),
                "avg_return_gross": float(gross.mean()),
                "avg_return_net": float(net.mean()),
                "median_return_net": float(net.median()),
                "total_pnl_sum": float(net.sum()),
                "compounded_total_return": float(((1.0 + net / 100.0).prod() - 1.0) * 100.0),
                "profit_factor": float(wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan),
                "cumulative_sum_MDD_pct_point": float(cum_dd.min()),
                "compounded_equity_MDD_pct_025x": mdd025,
                "compounded_equity_MDD_pct_050x": mdd050,
                "compounded_equity_MDD_pct_100x": mdd100,
                "ruin_in_simulation_025x": ruin025,
                "ruin_in_simulation_050x": ruin050,
                "ruin_in_simulation_100x": ruin100,
                "per_trade_worst_loss": float(net.min()),
                "avg_mfe_until_exit": float(group["mfe_until_exit"].mean()) if "mfe_until_exit" in group else np.nan,
                "avg_mae_until_exit": float(group["mae_until_exit"].mean()) if "mae_until_exit" in group else np.nan,
                "avg_mfe_window_32bars": float(group["mfe_window_32bars"].mean()) if "mfe_window_32bars" in group else np.nan,
                "avg_mae_window_32bars": float(group["mae_window_32bars"].mean()) if "mae_window_32bars" in group else np.nan,
                "avg_holding_bars": float(group["holding_bars"].mean()),
                "avg_holding_hours": float(group["holding_hours"].mean()),
                "mdd_duration_bars": dur,
                "mdd_recovery_bars": rec,
                "time_underwater_pct": tuw,
                "sharpe_like_ratio": float(net.mean() / std) if std and np.isfinite(std) else np.nan,
                "tp_hit_rate": float(group["exit_reason"].astype(str).str.contains("take_profit|partial_tp", regex=True).mean() * 100.0),
                "stop_hit_rate": float(group["exit_reason"].astype(str).str.contains("stop|invalidation|swing", regex=True).mean() * 100.0),
                "exit_reason_distribution": json.dumps(exit_dist, ensure_ascii=False),
                "metric_basis": basis.metric_basis,
                "unit_basis": basis.unit_basis,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics_row(ledger: pd.DataFrame, basis_config: BasisConfig | None = None) -> MetricsRow:
    frame = compute_metrics(ledger, basis_config, group_cols=[])
    return MetricsRow(frame.iloc[0].to_dict() if not frame.empty else {})

import math

import pandas as pd

from src.backtest.metrics import compute_metrics


def test_compute_metrics_without_mdd_clipping_and_sample_class():
    ledger = pd.DataFrame(
        {
            "rule_name": ["r"] * 4,
            "rule_kind": ["production"] * 4,
            "direction": ["long"] * 4,
            "exit_policy": ["x"] * 4,
            "exit_reason": ["a", "b", "a", "b"],
            "gross_return": [10.0, -120.0, 5.0, -5.0],
            "net_return": [10.0, -120.0, 5.0, -5.0],
            "mfe_until_exit": [12.0, 1.0, 6.0, 2.0],
            "mae_until_exit": [-1.0, -121.0, -1.0, -6.0],
            "holding_bars": [1, 2, 3, 4],
            "holding_hours": [1.0, 2.0, 3.0, 4.0],
        }
    )
    row = compute_metrics(ledger).iloc[0]
    assert row["sample_class"] == "low_sample"
    assert row["win_rate"] == 50.0
    assert row["profit_factor"] == 15.0 / 125.0
    assert row["cumulative_sum_MDD_pct_point"] < -100.0
    assert row["ruin_in_simulation_100x"] is True or row["ruin_in_simulation_100x"] == True
    assert math.isfinite(row["avg_mfe_until_exit"])

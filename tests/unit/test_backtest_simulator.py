import pandas as pd

from src.backtest.simulator import simulate_trades
from src.backtest.types import Costs, EntrySignal, ExitPolicy, ExitPolicyKind, MarketContext, SizingPolicy


def _candles():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min"),
            "open": [100, 100, 100, 100, 100],
            "high": [100, 101, 103, 102, 101],
            "low": [100, 99.5, 98, 99, 100],
            "close": [100, 100.5, 102, 101, 100],
        }
    )


def test_tp_sl_stop_first_and_realized_mfe_mae():
    signal = EntrySignal(
        signal_time=pd.Timestamp("2024-01-01"),
        entry_time=pd.Timestamp("2024-01-01"),
        entry_price=100.0,
        direction="long",
        entry_tf="15m",
        bar_index=0,
        rule_name="r",
        rule_kind="production",
    )
    context = MarketContext(
        candles={"15m": _candles()},
        opposite_event_resolver=lambda tf, direction, when: (pd.Timestamp("2024-01-01 01:00"), 100.0),
    )
    ledger = simulate_trades(
        [signal],
        None,
        ExitPolicy("tp_sl", ExitPolicyKind.TP_SL, {"tp_pct": 2.0, "sl_pct": 1.0}),
        SizingPolicy(),
        Costs(fee_pct=0, slippage_pct_per_side=0),
        context,
    )
    row = ledger.iloc[0]
    assert row["exit_reason"] == "stop_loss"
    assert row["gross_return"] == -1.0
    assert round(row["mfe_until_exit"], 6) == 3.0
    assert round(row["mae_until_exit"], 6) == -2.0


def test_partial_tp_keeps_remainder_to_opposite():
    signal = EntrySignal(
        signal_time=pd.Timestamp("2024-01-01"),
        entry_time=pd.Timestamp("2024-01-01"),
        entry_price=100.0,
        direction="long",
        entry_tf="15m",
        bar_index=0,
        rule_name="r",
        rule_kind="production",
    )
    context = MarketContext(
        candles={"15m": _candles()},
        opposite_event_resolver=lambda tf, direction, when: (pd.Timestamp("2024-01-01 01:00"), 102.0),
    )
    ledger = simulate_trades(
        [signal],
        None,
        ExitPolicy("partial", ExitPolicyKind.PARTIAL_TP, {"tp_pct": 2.0, "partial_ratio": 0.5}),
        SizingPolicy(),
        Costs(fee_pct=0, slippage_pct_per_side=0),
        context,
    )
    assert ledger.iloc[0]["first_leg_hit"] is True or ledger.iloc[0]["first_leg_hit"] == True
    assert round(ledger.iloc[0]["gross_return"], 6) == 2.0

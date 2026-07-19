import pandas as pd

from data_pipeline.microstructure.binance_public_data import normalize_book_depth
from data_pipeline.microstructure.features import audit_trade_quality, build_book_depth_percent_bars, build_trade_bars
from data_pipeline.microstructure.validation import (
    add_forward_returns,
    audit_funding_boundaries,
    audit_observation_boundary,
)


def test_forward_return_starts_at_label_start_price():
    df = pd.DataFrame(
        {
            "label_start_price": [100.0, 101.0, 99.0, 103.0],
        }
    )

    out = add_forward_returns(df, [2])

    assert round(out.loc[0, "forward_return_2b"], 6) == -1.0
    assert round(out.loc[1, "forward_return_2b"], 6) == round((103.0 / 101.0 - 1.0) * 100.0, 6)


def test_observation_boundary_detects_future_feature_time():
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    df = pd.DataFrame(
        {
            "feature_source_max_time": [base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=3)],
            "label_start_time": [base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=2)],
        }
    )

    audit = audit_observation_boundary(df)

    assert audit["status"] == "fail"
    assert audit["violation_rows"] == 1


def test_funding_boundary_detects_future_funding_source():
    feature_time = pd.Timestamp("2026-01-01T07:59:00Z")
    df = pd.DataFrame(
        {
            "feature_source_max_time": [feature_time],
            "funding_rate_source_time": [pd.Timestamp("2026-01-01T08:00:00Z")],
        }
    )

    audit = audit_funding_boundaries(df, lookback_minutes=5)

    assert audit["status"] == "fail"
    assert audit["violation_rows"] == 1


def test_trade_bars_deduplicate_agg_trade_id_before_summing():
    base_ns = pd.Timestamp("2026-01-01T00:00:01Z").value
    df = pd.DataFrame(
        {
            "event_time_ns": [base_ns, base_ns, base_ns + 1_000_000_000],
            "trade_time_ns": [base_ns, base_ns, base_ns + 1_000_000_000],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "agg_trade_id": [1, 1, 2],
            "price": [100.0, 100.0, 101.0],
            "quantity": [0.5, 0.5, 0.25],
            "signed_quantity": [0.5, 0.5, -0.25],
        }
    )

    bars = build_trade_bars(df, "1min")

    assert len(bars) == 1
    assert bars.loc[0, "trade_count"] == 2
    assert bars.loc[0, "trade_volume"] == 0.75
    assert bars.loc[0, "buy_volume"] == 0.5
    assert bars.loc[0, "sell_volume"] == 0.25


def test_trade_quality_flags_one_sided_impossible_volume():
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for idx in range(20):
        rows.append(
            {
                "event_time_ns": (base + pd.Timedelta(seconds=idx)).value,
                "trade_time_ns": (base + pd.Timedelta(seconds=idx)).value,
                "symbol": "BTCUSDT",
                "agg_trade_id": idx,
                "price": 100.0,
                "quantity": 300.0,
                "signed_quantity": 300.0,
            }
        )
    df = pd.DataFrame(rows)

    audit = audit_trade_quality(df, "1min")

    assert audit["status"] == "pass"
    assert audit["max_bar_volume"] == 6000.0
    assert audit["extreme_ratio_bars"] == 1
    assert audit["warnings"]


def test_trade_bars_mark_extreme_quality_flags_without_failing():
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    df = pd.DataFrame(
        {
            "event_time_ns": [(base + pd.Timedelta(seconds=idx)).value for idx in range(20)],
            "trade_time_ns": [(base + pd.Timedelta(seconds=idx)).value for idx in range(20)],
            "symbol": ["BTCUSDT"] * 20,
            "agg_trade_id": list(range(20)),
            "price": [100.0] * 20,
            "quantity": [300.0] * 20,
            "signed_quantity": [300.0] * 20,
        }
    )

    bars = build_trade_bars(df, "1min")

    assert bool(bars.loc[0, "trade_quality_extreme_one_sided"])


def test_book_depth_percent_uses_single_cumulative_band_not_sum():
    ts = pd.Timestamp("2026-06-01T00:00:00Z").value
    df = pd.DataFrame(
        {
            "event_time_ns": [ts] * 6,
            "symbol": ["BTCUSDT"] * 6,
            "percentage": [-0.2, -1.0, -2.0, 0.2, 1.0, 2.0],
            "depth": [354.0, 2265.0, 5846.0, 100.0, 900.0, 2000.0],
            "notional": [3540.0, 22650.0, 58460.0, 1000.0, 9000.0, 20000.0],
        }
    )

    bars = build_book_depth_percent_bars(df, "1min", target_percentage=1.0)

    assert len(bars) == 1
    assert bars.loc[0, "book_depth_pct_1_bid_qty"] == 2265.0
    assert bars.loc[0, "book_depth_pct_1_ask_qty"] == 900.0
    assert bars.loc[0, "book_depth_pct_1_bid_qty"] != 354.0 + 2265.0 + 5846.0


def test_public_book_depth_normalizer_keeps_raw_percent_bands_only():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-06-01 00:00:00"],
            "percentage": [-1.0],
            "depth": [2265.0],
            "notional": [22650.0],
        }
    )

    streams = normalize_book_depth(df, "BTCUSDT")

    assert set(streams) == {"book_depth_percent"}
    assert "bid_qty_top_n" not in streams["book_depth_percent"].columns

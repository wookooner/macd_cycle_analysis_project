from src.realtime.binance_footprint_stream import FootprintAggregator, normalize_agg_trade


def test_normalize_agg_trade_preserves_aggressor_side_inputs():
    trade = normalize_agg_trade(
        {"data": {"e": "aggTrade", "s": "BTCUSDT", "E": 1_000, "T": 999, "p": "100.5", "q": "2.0", "m": True}}
    )

    assert trade == {
        "symbol": "BTCUSDT",
        "trade_time_ms": 999,
        "price": 100.5,
        "quantity": 2.0,
        "is_buyer_maker": True,
    }


def test_footprint_aggregator_groups_price_and_aggressor_volume():
    aggregator = FootprintAggregator("BTCUSDT", "15m", price_bin_size=5.0)
    aggregator.ingest({"symbol": "BTCUSDT", "trade_time_ms": 901_000, "price": 100.9, "quantity": 2.0, "is_buyer_maker": False})
    aggregator.ingest({"symbol": "BTCUSDT", "trade_time_ms": 902_000, "price": 101.7, "quantity": 0.5, "is_buyer_maker": True})

    snapshot = aggregator.snapshot(now_ms=903_000)

    assert snapshot is not None
    assert snapshot["barStartMs"] == 900_000
    assert snapshot["tradeCount"] == 2
    assert snapshot["levels"] == [
        {"price": 100.0, "buyVolume": 2.0, "sellVolume": 0.5, "delta": 1.5, "totalVolume": 2.5}
    ]

# Microstructure Pipeline

This layer is intentionally separate from the existing MACD cycle pipeline.

## 1. Collect Raw Streams

```powershell
python .\scripts\microstructure\collect_binance_microstructure.py --symbol BTCUSDT --keep-files 3
```

Default WebSocket streams:

- `btcusdt@aggTrade`
- `btcusdt@depth20@100ms`
- `btcusdt@markPrice@1s`
- `!forceOrder@arr`

The collector also polls REST snapshots for current open interest, global long/short account ratio, and top-trader long/short ratios.

Raw partitioned Parquet output:

```text
<MACD_DATA_ROOT>/raw/binance/usdm/BTCUSDT/<stream>/date=YYYY-MM-DD/*.parquet
```

Each write also creates an ingestion record below
`<MACD_DATA_ROOT>/metadata/manifests/binance/usdm/BTCUSDT/`. The record has
the row count, event-time range, columns, source and data-file path.

Existing files under the previous path
`raw/microstructure/binance_usdm/` remain readable while migrating. Preview
the copy first, then apply it (the command never deletes source data):

```powershell
python .\scripts\migrate_microstructure_layout.py --include-archive
python .\scripts\migrate_microstructure_layout.py --apply --include-archive
```

Retention is enabled by default: only the newest 3 Parquet files are kept per stream directory.

## 1b. Run From The Existing Pipeline Runner

Step `6` integrates the live microstructure collector into the existing pipeline runner. It runs until interrupted, so put it last.

Only live microstructure collection:

```powershell
python .\scripts\update_pipeline.py --asset btc --steps 6 --microstructure-keep-files 3
```

Run the normal finite pipeline first, then keep collecting microstructure:

```powershell
python .\scripts\update_pipeline.py --asset btc --steps 1 2 3 5 6 --microstructure-keep-files 3
```

Useful storage knobs:

```powershell
--microstructure-batch-rows 200000
--microstructure-flush-seconds 3600
--microstructure-keep-files 3
```

Default retention keeps only 3 raw Parquet files per stream. With the default hourly flush, that is roughly the latest 3 hours. If you lower `flush-seconds`, you also shorten the history available for validation.

## 2. Build Bar Features

```powershell
python .\scripts\microstructure\build_microstructure_features.py --symbol BTCUSDT --timeframe 1min
```

Feature Parquet output:

```text
<MACD_DATA_ROOT>/processed/features/microstructure/BTCUSDT/microstructure_features_1min.parquet
```

Feature output is overwritten per timeframe, so it does not accumulate files.

Main feature families:

- CVD from signed aggTrade volume
- top-N order book imbalance
- liquidation notional from force orders
- open interest change
- funding-rate expanding percentile bucket
- global and top-trader long/short ratios

## 3. Validate Forward Returns

```powershell
python .\scripts\microstructure\validate_microstructure_forward_returns.py --symbol BTCUSDT --timeframe 1min --horizons 5 15 60 --warmup-rows 100
```

Outputs focus on conditional forward-return distributions: quantiles, tails, and skew.

Validation/report outputs use fixed filenames per timeframe under:

```text
<MACD_DATA_ROOT>/outputs/analysis_results/microstructure_validation/BTCUSDT/
```

These outputs are overwritten on each run.

Validation also writes:

- an observation-boundary audit: `feature_source_max_time <= label_start_time`
- staleness summaries for as-of REST/mark-price joins
- a funding settlement-boundary audit
- a shuffled-feature profile for leak/artifact checks

Important interpretation notes:

- Horizons are bar counts, not minutes by name. On `1min`, `60` means 60 minutes; on `5min`, `60` means 300 minutes.
- Forward returns start from `label_start_price`, the close available at the feature bar end.
- Funding percentile is expanding-only and skips the warmup rows in validation.
- `!forceOrder@arr` is throttled by Binance, so liquidation values are lower-bound/spike proxies, not total market liquidation.

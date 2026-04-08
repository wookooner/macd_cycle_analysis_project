# Dashboard Architecture

The dashboard should consume prepared payloads, not raw parquet or raw CSV files directly.

Target payload families:

- cycles flat view
- hierarchy tree
- candle index
- trade positions
- lazy-loaded candle payloads

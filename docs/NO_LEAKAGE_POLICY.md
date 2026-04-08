# No Leakage Policy

Rules:

- No feature may depend on future candles or future labels.
- Time-series splits must preserve chronology.
- Backtest execution rules must only use information observable at decision time.
- Any new labeling or feature work must state its observation boundary explicitly.
- Export builders must not introduce future-dependent derived fields into dashboard or AI packets.

# Tool Catalog

## Purpose

This directory tracks the analysis tools that power `ai_analyst/`.

Each tool document should explain:

- what the tool does
- when the agent should use it
- what inputs it accepts
- what outputs it returns
- current limitations
- where the implementation lives

This documentation is intended to stay in sync with the real tool layer so the
agent behavior, implementation, and future UI work do not drift apart.

## Current Foundation

The current system is intentionally built on three foundational tools:

1. `describe_available_data`
2. `build_analysis_frame`
3. `filter_frame`

These tools are the base for nearly all future analysis work.

Why these tools first:

- `describe_available_data` prevents schema guessing and tells the agent what
  datasets are actually available.
- `build_analysis_frame` converts canonical cycle parquet data into a flattened
  analysis-ready frame preview that later tools can build on.
- `filter_frame` gives later analysis tools a reusable way to narrow data by
  explicit feature, context, and parent-child conditions.

## Planned Expansion

Later tools will likely build on top of the two foundation tools rather than
replace them.

Planned next-layer tools include:

- `compare_groups`
- `rank_features`
- `create_plot`
- `summarize_report`

## Tool Status Labels

Use the following statuses in tool documents:

- `implemented`: usable in the current runtime
- `partial`: present but intentionally limited
- `planned`: designed but not yet implemented

## Current Implemented Tools

- `describe_available_data`
  - Status: `implemented`
  - Implementation: `src/btc_macd_cycle_ai_analyst/tools/discovery.py`
- `build_analysis_frame`
  - Status: `implemented`
  - Implementation: `src/btc_macd_cycle_ai_analyst/tools/frame.py`
- `filter_frame`
  - Status: `implemented`
  - Implementation: `src/btc_macd_cycle_ai_analyst/tools/filtering.py`
- `compare_groups`
  - Status: `implemented`
  - Implementation: `src/btc_macd_cycle_ai_analyst/tools/analysis.py`
- `rank_features`
  - Status: `implemented`
  - Implementation: `src/btc_macd_cycle_ai_analyst/tools/analysis.py`

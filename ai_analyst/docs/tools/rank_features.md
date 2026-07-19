# `rank_features`

## Status

`implemented`

## Purpose

Rank feature columns by how strongly a focused subset differs from the
remainder of the same timeframe.

This is the current minimum ranking tool for Q4-style questions such as:

- "show the characteristics of cycles with high `child_count`"

## When To Use

Use this tool when:

- the user has already implied a subset or condition of interest
- you want to identify which `feature__*` columns differ most for that subset
- a simple ranking is more useful than a full pairwise comparison

Prefer `compare_groups` when both groups are already explicitly defined.

## Inputs

- `timeframe`
  - timeframe to analyze, such as `1h`
- `focus_filters`
  - semicolon-separated filter string for the focus subset
- `candidate_columns`
  - optional comma-separated list of candidate columns
- `top_k`
  - number of ranked features to return
- `row_cap`
  - maximum working-set rows before ranking
- `preview_rows`
  - number of ranked rows to preview
- `asset`
  - defaults to `btc`

## Outputs

Returns a standard tool result with:

- a ranking table artifact
- focus vs remainder counts
- mean and median deltas
- `abs_mean_delta` for ordering
- warnings when the working set is capped

## Current Limitations

- only ranks numeric `feature__*` columns by default
- compares one focus subset against the remainder of the capped timeframe
- does not yet perform statistical significance testing
- does not yet create charts directly

## Example Uses

- rank features for `child_count >= 3`
- rank features for `combo_4 == UUDU`
- rank features for one parent-child condition before deciding whether a plot is needed

## Implementation

- Service: `src/btc_macd_cycle_ai_analyst/services/data_access.py`
- Tool wrapper: `src/btc_macd_cycle_ai_analyst/tools/analysis.py`

# compare_groups

## Status

`implemented`

## Purpose

Compare two filtered subsets of the same timeframe frame across one or more
metric columns.

This is the first concrete analysis tool above the base frame/filter layer.

## When To Use

Use this tool when:

- the user asks how two conditions differ
- the request compares one cycle subset against another
- the agent needs side-by-side numeric summaries for candidate metric columns

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `timeframe` | string | yes | Target timeframe such as `1h`, `4h`, `1d` |
| `metrics` | string | yes | Comma-separated metric columns to compare |
| `group_a_filters` | string | yes | Semicolon-separated filters for group A |
| `group_b_filters` | string | yes | Semicolon-separated filters for group B |
| `row_cap` | integer | no | Maximum rows to include per comparison scan |
| `preview_rows` | integer | no | Number of preview rows returned per group |
| `asset` | string | no | Asset key. Current default is `btc` |

## Outputs

Returns a JSON-serialized `ToolResult`-style payload with:

- `status`
- `tool_name`
- `summary`
- `data_preview`
- `artifacts`
- `frame_meta`
- `warnings`
- `errors`
- `metrics`

## Current Comparison Metrics

For each requested metric column, the tool currently reports:

- group A count
- group B count
- group A mean
- group B mean
- group A median
- group B median
- mean delta
- median delta

## Example Use

- compare `feature__strength__direction_pct` for `combo_4 == UUDU` vs `combo_4 == DDDU`
- compare `child_count` and `opposite_child_ratio` for one boundary type vs another
- compare filtered parent/child relationship subsets within one timeframe

## Current Limitations

- compares two groups within one timeframe frame at a time
- currently provides descriptive summaries, not full statistical significance tests
- depends on columns already exposed by the frame-building layer
- does not yet support automatic metric selection

## Why It Matters

This is the first real bridge from "data access" to "analysis".

It allows the agent to move from:

- "what data exists?"
- "what does the frame look like?"

to:

- "how do these two subsets differ?"

## Implementation

`src/btc_macd_cycle_ai_analyst/tools/analysis.py`

Backed by:

`src/btc_macd_cycle_ai_analyst/services/data_access.py`

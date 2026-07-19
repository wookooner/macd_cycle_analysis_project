# build_analysis_frame

## Status

`implemented`

## Purpose

Build a flattened preview of one timeframe's cycle dataset so the agent can work
with analysis-ready columns instead of raw nested parquet structures.

This tool is the bridge between canonical cycle storage and later analysis
tools.

## When To Use

Use this tool when:

- the user asks about a specific timeframe
- the agent needs concrete columns to inspect
- the request depends on cycle features or enriched relationship fields
- later analysis requires a normalized frame preview

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `timeframe` | string | yes | Target timeframe such as `1h`, `4h`, `1d` |
| `columns` | string | no | Comma-separated requested column subset |
| `row_cap` | integer | no | Maximum rows to include in the preview frame |
| `preview_rows` | integer | no | Number of preview rows returned |
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

## Main Payload Content

The current frame includes:

- cycle identity columns
- enriched relationship/context columns already stored in canonical cycle parquet
- flattened `cycle_features` columns using `feature__*` names

Examples of included fields:

- `cycle_id`
- `timeframe`
- `start_date`
- `end_date`
- `cycle_type`
- `parent_key`
- `boundary_type`
- `n_up_4`
- `combo_4`
- `child_count`
- `feature__change__price_pct`
- `feature__strength__direction_pct`
- `feature__volatility__avg_true_range`

## Example Use

Typical questions:

- "Show me the available columns for 1h cycle analysis."
- "Build a preview frame for 4h cycles."
- "What feature columns exist in the 1d cycle dataset?"

## Current Limitations

- currently works from canonical cycle parquet only
- does not yet join context parquet dynamically
- does not yet apply filter expressions as a separate reusable step
- returns preview-oriented frame data, not a persistent frame registry

## Why It Is Foundational

This is the second base tool of the current system.

Almost every future analysis tool will depend on this step because it:

- turns nested cycle feature structures into flat analysis columns
- centralizes the canonical path and frame-building logic
- gives future tools a consistent starting point for comparison and ranking

## Implementation

`src/btc_macd_cycle_ai_analyst/tools/frame.py`

Backed by:

`src/btc_macd_cycle_ai_analyst/services/data_access.py`

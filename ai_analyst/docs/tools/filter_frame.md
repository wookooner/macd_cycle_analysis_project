# filter_frame

## Status

`implemented`

## Purpose

Filter one timeframe analysis frame with reusable condition syntax so later
analysis tools do not need to reimplement the same subset logic.

This tool is the next layer on top of the two foundation tools:

1. `describe_available_data`
2. `build_analysis_frame`

## When To Use

Use this tool when:

- the user asks for a subset such as `n_up_4 >= 3`
- a request depends on one or more explicit conditions
- a later comparison or ranking workflow needs a narrowed frame first

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `timeframe` | string | yes | Target timeframe such as `1h`, `4h`, `1d` |
| `filters` | string | yes | Semicolon-separated filter expressions |
| `columns` | string | no | Optional comma-separated column subset |
| `row_cap` | integer | no | Maximum rows to include in the filtered preview |
| `preview_rows` | integer | no | Number of preview rows returned |
| `asset` | string | no | Asset key. Current default is `btc` |

## Supported Operators

- `==`
- `!=`
- `>`
- `>=`
- `<`
- `<=`
- `contains`
- `in`

## Filter Syntax Examples

- `n_up_4 >= 3`
- `combo_4 == UUDU`
- `cycle_type == up; child_count >= 3`
- `combo_4 in [UUDU, DDDU]`

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

## Current Limitations

- currently supports simple AND-style filtering only
- does not yet support grouped boolean logic such as nested OR conditions
- currently operates on one timeframe frame at a time
- depends on columns already exposed by the frame-building layer

## Why It Matters

This tool is the reusable subset layer that later tools can depend on.

It reduces duplication because future tools like:

- `compare_groups`
- `rank_features`
- `create_plot`

can all assume that frame narrowing is already available in one consistent form.

## Implementation

`src/btc_macd_cycle_ai_analyst/tools/filtering.py`

Backed by:

`src/btc_macd_cycle_ai_analyst/services/data_access.py`

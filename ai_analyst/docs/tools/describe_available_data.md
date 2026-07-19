# describe_available_data

## Status

`implemented`

## Purpose

Return a canonical view of what cycle and context datasets are available for one
asset.

This tool exists to stop the agent from guessing:

- which timeframes exist
- which parquet datasets exist
- how many rows and columns each dataset contains

## When To Use

Use this tool when:

- a request depends on timeframe availability
- the agent needs to know what data exists before choosing an analysis path
- the user asks for a dataset overview
- the agent is unsure whether cycle or context files are present

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `asset` | string | no | Asset key. Current default is `btc`. |

## Outputs

Returns a JSON-serialized `ToolResult`-style payload with:

- `status`
- `tool_name`
- `summary`
- `data_preview`
- `artifacts`
- `warnings`
- `errors`
- `metrics`

## Main Payload Content

The current result includes:

- cycle dataset list
- context dataset list
- timeframe preview
- row count
- column count
- first columns preview

## Example Use

Typical questions:

- "What BTC datasets are available right now?"
- "Which timeframes can I analyze?"
- "Do I have context parquet files for hourly analysis?"

## Current Limitations

- focuses on dataset-level discovery, not detailed semantic interpretation
- does not yet expose a richer field taxonomy such as feature/outcome/context groups
- currently optimized for canonical parquet inspection, not arbitrary external files

## Why It Is Foundational

This is one of the two base tools of the current system.

Almost every future workflow benefits from it because it allows the agent to:

- choose a valid timeframe
- confirm that the requested dataset exists
- inspect schema-related context before deeper analysis

## Implementation

`src/btc_macd_cycle_ai_analyst/tools/discovery.py`

Backed by:

`src/btc_macd_cycle_ai_analyst/services/data_access.py`

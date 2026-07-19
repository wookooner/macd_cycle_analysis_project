# AI Analyst Contracts

## Purpose

This document defines the payload contracts that matter to the current
implemented single-agent runtime.

It intentionally focuses on what exists now, not on future request/intent
objects that are not implemented in code.

## Current Contract Scope

The current runtime meaningfully uses these contract families:

- `ToolResult`
- `AnalysisArtifact`
- `AnalysisFrameMeta`

These are the stable payloads that matter across:

- tool wrappers
- service-layer results
- agent interpretation
- future provider switches

## ToolResult

Every implemented analysis tool should return a result in this shared shape.

| Field | Type | Purpose |
|------|------|---------|
| `status` | string | `success`, `warning`, or `error` |
| `tool_name` | string | Tool that produced the result |
| `summary` | string | Short human-readable summary |
| `data_preview` | object or null | Small preview payload for the agent |
| `artifacts` | list[`AnalysisArtifact`] | Tables, frame refs, schema snapshots, future charts |
| `frame_meta` | `AnalysisFrameMeta` or null | Metadata about the analyzed frame |
| `warnings` | list[string] | Caveats such as row caps or low sample size |
| `errors` | list[string] | Error details when the result is not usable |
| `metrics` | object | Counts, deltas, or other scalar outputs |

Required behavior:

- `status` must always be present
- `warnings` must always be present, even when empty
- `errors` must always be present, even when empty
- tools should prefer useful warnings and recoverable errors over silent failure

## AnalysisArtifact

Represents a reusable structured output produced by a tool.

| Field | Type | Purpose |
|------|------|---------|
| `artifact_type` | string | `table`, `chart`, `frame_ref`, `schema_snapshot` |
| `name` | string | Short label |
| `description` | string | What the artifact contains |
| `format` | string | `json`, `markdown`, `plotly`, etc. |
| `payload` | object | Structured content or a reference descriptor |
| `preview` | object or null | Small inline preview |

## AnalysisFrameMeta

Captures where the analyzed frame came from and how it was shaped.

| Field | Type | Purpose |
|------|------|---------|
| `frame_id` | string | Stable frame identifier for the current request |
| `source_datasets` | list[string] | Source parquet dataset names |
| `timeframes` | list[string] | Timeframes represented in the frame |
| `row_count` | integer | Final returned row count |
| `column_names` | list[string] | Columns available in the frame |
| `filter_history` | list[string] | Human-readable filter trail |
| `join_history` | list[string] | Human-readable join trail |
| `sampling_applied` | bool | Whether a row cap or sampling rule was applied |
| `notes` | list[string] | Extra context such as row-cap caveats |

## Minimum Expectations By Tool Type

### Discovery tools

Should usually return:

- available dataset list
- column/schema preview
- warnings for missing datasets

### Frame tools

Should usually return:

- frame reference artifact
- frame metadata
- join/filter notes

### Analysis tools

Should usually return:

- metrics
- concise summary
- warnings for low sample size, caps, or weak interpretation
- comparison or ranking tables where relevant

## Current Practical Mapping

The current runtime maps roughly like this:

- `describe_available_data` -> discovery tool
- `build_analysis_frame` -> frame tool
- `filter_frame` -> frame/filter tool
- `compare_groups` -> analysis tool
- `rank_features` -> analysis tool

## Proposed, Not Implemented

The following concepts may become useful later, but they are not currently
implemented as stable code contracts:

- `AnalysisRequest`
- `AnalysisIntent`
- `ToolInvocation`
- `AnalysisSummary`

If the runtime later adds persistent request IDs, multi-step planning objects,
or richer API integration, those contracts can be promoted into their own
document again.

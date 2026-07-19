# AI Analyst API Exchange

## Purpose

This document explains how `ai_analyst` passes data between:

- the user request
- the model runtime
- the agent
- the tool layer
- the dataset layer

The goal is to make the system understandable to another AI, another engineer,
or a future API integration without reading the whole codebase.

## High-Level Flow

The current single-agent flow is:

1. user sends a natural-language analysis request
2. `main.py` passes the raw query into `run_query(...)`
3. the LangGraph ReAct agent decides which tool to call
4. each tool wrapper converts agent-friendly string inputs into structured arguments
5. the service layer reads canonical parquet datasets and computes deterministic results
6. the service returns a shared `ToolResult`-shaped payload
7. the agent reads that payload and writes the final user-facing answer

## Runtime Layers

### 1. User -> Agent

Input shape at runtime:

```python
{
  "messages": [
    {
      "role": "user",
      "content": "<natural language query>"
    }
  ]
}
```

Current entrypoint:

- `ai_analyst/main.py`
- `src/btc_macd_cycle_ai_analyst/agent/factory.py`

### 2. Agent -> Tool

The agent calls tools through LangChain tool wrappers.

Typical examples:

```python
filter_frame(
    timeframe="1h",
    filters="feature__change__price_pct >= 0",
    asset="btc",
    row_cap=2000,
    preview_rows=5,
)
```

```python
compare_groups(
    timeframe="1h",
    group_a_filters="combo_4 == UUDU",
    group_b_filters="combo_4 == DDDU",
    metrics="feature__strength__direction_pct, child_count",
)
```

```python
rank_features(
    timeframe="1h",
    focus_filters="child_count > 1",
    top_k=10,
)
```

### 3. Tool -> Service

Tool wrappers are intentionally thin.

They should only do:

- lightweight string parsing
- argument normalization
- JSON serialization for the agent runtime

They should not do:

- parquet loading
- metric calculation
- ranking logic
- comparison logic

Those belong in:

- `src/btc_macd_cycle_ai_analyst/services/data_access.py`

## Dataset Exchange Model

The service layer reads canonical datasets and converts them into an
analysis-friendly frame.

### Canonical sources

Primary datasets:

- `cycles_<timeframe>.parquet`
- `cycle_dim.parquet`
- `timeframe_context_<timeframe>.parquet`

Resolved through:

- `settings.py`
- `services/paths.py`

### Analysis frame concept

The agent never reads raw parquet schema directly.

Instead, the service layer exposes a flattened analysis frame with:

- identity fields
  - `cycle_id`, `cycle_key`, `timeframe`, `start_date`, `end_date`
- relationship/context fields
  - `parent_key`, `boundary_type`, `n_up_4`, `combo_4`, `child_count`
- flattened feature fields
  - `feature__change__price_pct`
  - `feature__strength__direction_pct`
  - `feature__volatility__avg_true_range`
  - etc.

This is the key abstraction between datasets and AI behavior.

## Shared Tool Result Structure

Every real tool should return the same broad result shape.

```json
{
  "status": "success",
  "tool_name": "filter_frame",
  "summary": "Filtered 1h frame to 478 rows using 1 condition(s).",
  "data_preview": {},
  "artifacts": [],
  "frame_meta": {},
  "warnings": [],
  "errors": [],
  "metrics": {}
}
```

Important fields:

- `summary`
  - short human-readable description
- `data_preview`
  - small inline preview for the agent
- `artifacts`
  - reusable structured outputs such as a table or frame reference
- `frame_meta`
  - source dataset, row count, column names, filter history, row-cap notes
- `warnings`
  - caveats the final answer should surface
- `errors`
  - explicit recovery hints when possible
- `metrics`
  - counts, deltas, ranking counts, or other scalar values

## Current Implemented Tool Exchange

### `describe_available_data`

Purpose:

- tell the agent which datasets/timeframes exist

Input style:

- small structured arguments such as `asset="btc"`

Output style:

- dataset list
- context file list
- schema snapshot artifact

### `build_analysis_frame`

Purpose:

- expose a timeframe-specific flattened frame preview

Input style:

- timeframe
- optional requested columns
- row cap

Output style:

- frame preview
- available column names
- frame metadata

### `filter_frame`

Purpose:

- narrow one timeframe frame by explicit conditions

Input style:

- semicolon-separated filter expression string

Output style:

- filtered preview
- filter history
- row-cap warnings
- recoverable column errors with suggested nearby fields

### `compare_groups`

Purpose:

- compare one filtered subset against another on selected metrics

Input style:

- group A filters
- group B filters
- metric list

Output style:

- group previews
- side-by-side metric summary
- warnings for low sample size or caps

### `rank_features`

Purpose:

- identify which `feature__*` columns distinguish a focused subset from the remainder

Input style:

- focus filter string
- optional candidate column list
- top-k size

Output style:

- ranked feature table
- focus vs remainder counts
- mean/median deltas

## AI-Facing Design Rules

When another AI system uses this project, the expected behavior is:

1. do not guess schema from memory
2. inspect data before claiming fields exist or do not exist
3. prefer `feature__*` frame columns over guessed paraphrases
4. treat warnings and row-cap notes as part of the answer, not as optional metadata
5. keep final claims grounded in returned metrics and previews

## What Is Stable vs Unstable

### Stable enough to integrate against

- `ToolResult`-style result structure
- canonical timeframe datasets
- flattened analysis-frame concept
- services/tools separation

### Likely to change later

- exact prompt wording
- local model choice
- ambiguity-handling policy
- plotting support
- future orchestration layer

## Integration Advice

If this project is later called from a paid API backend or another agent system:

- keep the service layer unchanged if possible
- wrap the same tools in the new runtime instead of rewriting logic
- re-run the 4 anchor questions before trusting the new provider

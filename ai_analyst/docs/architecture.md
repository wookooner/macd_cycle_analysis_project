# AI Analyst Architecture

## Purpose

`ai_analyst/` is the domain-analysis engine for the BTC MACD cycle dataset.
It is not meant to replace a BI platform, become a generic AI data app, or own
all visualization responsibilities.

Its job is to:

- answer domain-specific questions about cycles, features, and parent-child structure
- build analysis-ready views from canonical parquet sources
- run tool-backed hypothesis checks from natural-language requests
- return grounded summaries and evidence for theory building

## North Star

The project supports a personal research loop:

1. observe patterns in a generic BI layer
2. form a domain hypothesis
3. validate it with `ai_analyst`
4. record the result as reusable theory material

In short:

- Metabase observes
- `ai_analyst` validates
- a theory registry accumulates findings

## Product Goal

The system should let a user ask flexible questions such as:

- "Find the strongest features for 1h cycles under a bullish 4h parent context."
- "Compare the distribution of duration and strength when `combo_4` is `UUDU`."
- "Scan several timeframes and tell me which one shows the clearest separation."

The agent should not guess schema from memory. It should inspect metadata, call
analysis tools, and produce grounded conclusions from real results.

## Scope

### In Scope

- canonical cycle parquet loading
- canonical context parquet loading
- analysis-frame construction for cycle/context questions
- tool-driven filtering, comparison, and ranking
- ReAct-style single-agent orchestration
- swappable model providers through configuration

### Out of Scope

- generic BI dashboards and SQL exploration UI
- pipeline execution and data generation
- live trading logic
- portfolio management
- news, sentiment, or web-research analysis
- unbounded code generation against arbitrary local files
- multi-agent orchestration until a real bottleneck appears

## Design Principles

1. Keep the agent thin.
   The agent decides what to inspect and what tool to call next. It should not
   embed dataset logic in prompts.
2. Keep dataset knowledge in adapters and tools.
   File resolution, schema handling, and join logic belong in code.
3. Prefer stable internal contracts over direct raw-schema coupling.
   Changes in parquet shape should be absorbed by the data layer whenever possible.
4. Make the happy path explicit.
   The default path should be observe -> frame -> analyze -> summarize.
5. Separate generic BI from domain validation.
   Metabase or similar tools should handle generic exploration and dashboards.
   `ai_analyst` should stay focused on domain-specific validation.
6. Keep the first version single-agent.
   The architecture may later support supervisor/worker expansion, but current
   design decisions should optimize for a robust single-agent system first.

## System Layers

The wider working system should be viewed as four layers.

### L1. Canonical Data

Responsible for source-of-truth parquet data and schema conventions.

Examples:

- `macd-cycle-data/`
- root `docs/DATA_LAYOUT.md`

Responsibilities:

- canonical cycle parquet files
- canonical context parquet files
- shared path conventions with the main project

### L2. Domain Engine

Responsible for domain-specific analysis over MACD cycle structures.

Examples:

- `ai_analyst/src/btc_macd_cycle_ai_analyst/services/`
- `ai_analyst/src/btc_macd_cycle_ai_analyst/tools/`
- `ai_analyst/src/btc_macd_cycle_ai_analyst/agent/`

Responsibilities:

- analysis-frame construction
- filter/group/ranking logic
- hypothesis-oriented tool execution
- grounded natural-language answers

### L3. Generic BI

Responsible for broad exploration that should not be reimplemented inside
`ai_analyst`.

Current stack:

- Metabase
- Postgres

Responsibilities:

- dashboards
- ad hoc SQL
- fast row-count and distribution checks
- generic charts

Implementation notes:

- `infra/metabase/` owns the runnable BI stack.
- Canonical parquet data is synced into Postgres for Metabase.
- Raw `cycles_*` tables preserve source payloads.
- `cycles_*_bi` views expose thinner Metabase-friendly projections.

### L4. Orchestration

Responsible for future coordination only if the single-agent setup proves
insufficient.

Current status:

- intentionally not implemented

Potential future responsibilities:

- optional supervisor routing
- optional multi-agent workflow
- optional scheduled automation

## Internal Layers Inside `ai_analyst`

Within `ai_analyst`, code should still be organized into the following layers.

### 1. Entry Layer

Responsible for local execution and developer-facing smoke tests.

Examples:

- `main.py`
- `test_analyst.py`

### 2. Configuration Layer

Responsible for runtime settings and path resolution.

Responsibilities:

- model/provider selection
- dataset root discovery
- app-level defaults

### 3. Data Access Layer

Responsible for locating canonical files and converting raw storage formats into
analysis-ready in-memory structures.

Responsibilities:

- canonical path lookup
- cycle parquet loading
- context parquet loading
- schema inspection
- standardized frame construction

### 4. Tool Layer

Responsible for deterministic, testable analysis operations.

Responsibilities:

- available-data discovery
- frame building
- filtering
- group comparison
- feature ranking
- report-ready evidence generation

Tools should return structured outputs with enough metadata to support grounded
agent responses.

### 5. Agent Layer

Responsible for planning the next step from user intent and tool results.

Responsibilities:

- interpret the request
- inspect the schema before analysis
- choose which tool to call
- decide whether more analysis is needed
- summarize the final result conservatively

The agent should never invent columns, joins, or statistical findings.

## Dependency Rules

Dependencies should flow in one direction.

Allowed direction:

- entry -> configuration
- entry -> agent
- agent -> tools
- tools -> data access
- tools -> configuration
- data access -> configuration

Avoid:

- agent importing low-level parquet implementation details directly
- entrypoint embedding analysis logic
- prompts becoming the source of truth for dataset structure
- circular imports between tools and agent code

## Analysis Flow

The default execution path should follow this order:

1. understand the user request
2. inspect available data and relevant fields
3. build an analysis frame
4. apply filters or subgroup definitions
5. run analysis tools
6. summarize results with counts, limits, and warnings
7. defer generic charting/dashboard work to the BI layer unless a
   domain-specific chart is clearly needed

## Recommended Tool Categories

### Discovery Tools

- inspect available datasets
- inspect columns and types
- surface valid filter and feature candidates

### Frame Tools

- build a standard analysis frame from cycle and context data
- join context safely
- preserve metadata about source files and row counts

### Analysis Tools

- compare groups
- rank candidate features
- summarize filtered subsets
- support timeframe-aware analysis

### Visualization Tools

- deferred by default
- only add domain-specific charts that generic BI tools cannot cover well

### Report Tools

- convert results into theory-ready summaries
- keep warnings and assumptions visible

## Current State vs Target State

### Current State

- single-agent runtime exists
- shared settings and path handling exist
- foundational discovery and frame-building tools are implemented
- comparison and feature-ranking tools exist
- the project direction now treats generic BI as external infrastructure rather
  than part of `ai_analyst`

### Target State

- `ai_analyst` remains a focused domain-validation engine
- generic exploration happens in Metabase or a similar BI tool
- validated findings can be promoted into a durable theory registry
- model provider choice remains configurable without changing service logic

## Immediate Structural Priorities

Before adding more analysis features, the project should stabilize these areas:

1. clear domain-engine vs BI boundary
2. shared configuration and path handling
3. standardized data-access entrypoints
4. real tool implementations for discovery, frame building, comparison, and ranking
5. a clean single-agent orchestration layer

## Future Expansion

This design intentionally leaves room for later additions without requiring a
rewrite:

- new model providers
- richer statistical tools
- theory-registry integration
- optional supervisor or multi-agent orchestration

Those extensions should be added on top of the current layers, not by bypassing
them.

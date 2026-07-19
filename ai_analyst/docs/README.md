# AI Analyst Docs Index

## Purpose

This index is the fastest way to understand the `ai_analyst/` project.

If you are a new contributor, another AI agent, or returning after a break,
start here before diving into implementation files.

## Recommended Reading Order

1. `README.md`
   Project overview, current layout, and immediate next steps.
2. `architecture.md`
   Why the system is shaped this way: purpose, layers, and dependency rules.
3. `roadmap.md`
   High-level sequencing and scope boundaries only.
4. `execution_plan.md`
   The actual current work queue and the single source of truth for the anchor questions.
5. `baseline_runs.md`
   Captured before-state of the fixed anchor questions.
6. `contracts.md`
   Implemented payload contracts only.
7. `api_exchange.md`
   How the runtime actually passes data between user request, agent, tools, and datasets.
8. `settings.md`
   Runtime configuration and canonical path resolution.
9. `DATA_LAYOUT.md`
   Pointer to the canonical repository-level data layout.
10. `tools/README.md`
   Tool catalog and current foundation tool set.
11. `future_crew_notes.md`
   Deferred checklist for future crew expansion.

## Quick Orientation

The current project state is:

- a single-agent analysis runtime
- canonical BTC MACD cycle parquet access
- Metabase + Postgres generic BI available under `infra/metabase/`
- shared settings and path resolution
- foundational data discovery, frame-building, filtering, comparison, and ranking tools
- documented boundary between generic BI exploration and domain validation

The current foundation tools are:

1. `describe_available_data`
2. `build_analysis_frame`
3. `filter_frame`
4. `compare_groups`
5. `rank_features`

The current expansion posture is:

1. keep Metabase responsible for generic dashboards and ad hoc exploration
2. keep `ai_analyst` responsible for domain validation over cycles/features/context
3. rerun the fixed anchor questions after provider/model switch
4. add `create_plot` only if post-provider Q4 still clearly needs a domain-specific chart

## Documentation Map

- `architecture.md`
  High-level system design and dependency rules.
- `roadmap.md`
  High-level development sequencing and boundaries.
- `execution_plan.md`
  Immediate task order and anchor-question-based completion criteria.
- `baseline_runs.md`
  Captured baseline results before output-quality changes.
- `contracts.md`
  Implemented runtime contracts only.
- `api_exchange.md`
  Concrete flow of data between user request, agent, tools, and datasets.
- `settings.md`
  Environment settings and path resolution.
- `DATA_LAYOUT.md`
  Pointer to the canonical repository-level dataset layout.
- `tools/README.md`
  Tool catalog and documentation conventions.
- `future_crew_notes.md`
  Deferred checklist for future crew/supervisor expansion.
- `tools/*.md`
  Per-tool specifications.

Related infrastructure:

- `../../infra/metabase/README.md`
  Metabase + Postgres setup, parquet sync, raw table policy, and `_bi` view guidance.

## Maintenance Rules

- When a new tool is added, update `tools/README.md` and add a per-tool doc.
- When a tool changes inputs or outputs, update the matching tool doc and
  confirm `contracts.md` is still accurate.
- When the project layout changes, update `README.md` and this index.
- When the runtime flow changes, update `architecture.md` and `roadmap.md`.
- When canonical data assumptions change, update `DATA_LAYOUT.md`.

## Current Priority

The current documentation priority is not breadth. It is alignment.

The most important rule is:

- implementation, prompts, and docs should describe the same current system

If one of those drifts, fix the docs or code immediately rather than letting
the mismatch linger.

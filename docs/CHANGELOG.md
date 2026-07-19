# Changelog

## v2.0 - Context Architecture (2026-04)

### New: context layer (`data_pipeline/context/context_builder.py`)

Added Step 5 to the pipeline. It replaces `cycle_hierarchy_map.json` as the primary
relationship structure with four new files:

- `processed/context/btc/cycle_dim.parquet` - surrogate-key dimension table for all
  TF cycles (604k rows, int32 `cycle_key`, datetime64[ms] UTC-normalized).
- `processed/context/btc/timeframe_context_1min.parquet` - per-minute snapshot of all
  9 TF cycle states (4.5M rows x 69 columns; includes key, type, time_prog,
  candle_prog, changed, n_up_4, combo_4, and boundary flags).
- `processed/context/btc/timeframe_context_1h.parquet` - hourly subset of the above
  (75k rows x 69 columns).
- `processed/context/btc/context_meta.json` - build metadata and data-range details.

### New: relationship columns in cycle parquets

Step 5 Phase 3 enriches `cycles_enriched/btc/cycles_*.parquet` in place with
parent/child/sibling relationship columns such as `cycle_key`, `parent_key`,
`parent_type`, `order_in_parent`, `total_siblings`, `boundary_type`,
`child_count`, `n_up_4`, and `combo_4`. See `docs/DATA_SCHEMA.md` for the full
column list.

### Changed: canonical data location

- `processed/cycles_enriched/btc/` is now the only canonical cycle location.
- Root-level `processed/cycles_enriched/cycles_*.parquet` files and legacy
  `*_enriched.parquet` files have been moved to `processed/cycles_enriched/archive/`.
- Backup files from Phase 3 have been moved to `processed/cycles_enriched/btc/archive/`.
- Phase 3 no longer creates new backup files in the canonical directory. Step 3
  (`detect`) always regenerates the base state.

### Changed: pipeline defaults

- Default steps changed from `[1, 2, 3, 4]` to `[1, 2, 3, 5]`.
- Step 4 (JSON hierarchy map) is now optional. Run with `--steps 1 2 3 4 5` if needed.
- Step 5 run alone (`--steps 5`) automatically skips context Phases 1 and 2,
  reuses existing `cycle_dim` and `timeframe_context`, and only re-enriches cycle
  parquets.
- Step 5 preceded by Step 3 runs all phases as a full rebuild.

### Changed: `pipeline_runner.py`

- `legacy_cycle_dir=None` for BTC, so the pipeline no longer writes root-level copies.
- Step counter labels were updated from `X / 4` to `X / 5`.
- `step_context()` now accepts a `full_rebuild` parameter based on whether Step 3 ran.

### Changed: `data_schema_report.py`

- Uses Arrow schema (`schema_arrow`) instead of the Parquet physical schema, which
  eliminates false duplicate-column warnings caused by struct/list flattening.
- Nested columns (`candle_data`, `cycle_features`) are shown as compact type strings
  instead of expanded field lists.
- Added a `summarize_context()` section reporting `cycle_dim`,
  `timeframe_context_*`, and `context_meta.json`.
- All top-level report sections now run in parallel with `ThreadPoolExecutor`.
- `summarize_cycle_parquet_dir()` now parallelizes per-file reads.

### Changed: `src/common/paths.py`

- Added `context_dir(asset)`, which returns `processed_root / "context" / asset`.

### Documentation

- `docs/DATA_LAYOUT.md` - updated the data tree to v2.0, added the context layer
  description, archive directories, and a runtime connections table.
- `docs/DATA_SCHEMA.md` - rewritten to cover all four tiers with column tables.
- `docs/REPO_STRUCTURE.md` - added `data_pipeline/context/` and the pipeline step table.
- `docs/AGENT_PROJECT_CONTEXT.md` - updated the data flow, commands, and external data tree.
- `docs/AGENT_STARTUP_CHECKLIST.md` - updated the expected layout, commands, and runtime truth.

---

## Unreleased (prior)

- Added first-stage repository reorganization scaffold.
- Introduced shared path configuration in `configs/paths.yaml`.
- Added governance and architecture documentation skeleton.
- Prepared the repository for code-first and agent-safe workflows.

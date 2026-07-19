# Documentation Maintenance

## Goal

Keep `ai_analyst/` understandable and extensible over time, even when tools,
runtime flow, and data assumptions evolve.

This document exists to make future updates easier for:

- the current developer
- future collaborators
- other AI agents working in the repo

## What Must Stay In Sync

The following must not drift apart:

1. actual runtime behavior
2. system prompt assumptions
3. tool documentation
4. architecture and roadmap docs

If one changes, the matching documentation should be updated in the same round
of work whenever possible.

## Update Checklist

When changing tool behavior:

- update the tool implementation
- update the matching file in `docs/tools/`
- confirm whether `contracts.md` needs changes
- confirm whether the system prompt still describes the tool truthfully

When changing runtime flow:

- update `architecture.md`
- update `roadmap.md`
- update `README.md` if entrypoints or layout changed

When changing data assumptions:

- update `DATA_LAYOUT.md`
- update `settings.md` if path resolution or environment rules changed
- update any affected tool docs

## Documentation Priority Order

If time is limited, update docs in this order:

1. `README.md`
2. `docs/README.md`
3. affected per-tool docs
4. `contracts.md`
5. `architecture.md`
6. `roadmap.md`

## Style Rules

- prefer short, explicit descriptions over broad vague prose
- document what the system does now before describing what it may do later
- separate implemented behavior from planned behavior
- use the real file path of the implementation whenever possible
- call out limitations clearly instead of hiding them

## Definition Of Good Documentation

Good documentation means a new reader can answer these questions quickly:

- what this project is for
- what is already working
- what is still planned
- where the canonical data comes from
- which tools are foundational
- where to add the next tool or interface layer

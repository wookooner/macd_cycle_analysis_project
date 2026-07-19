# Future Crew Notes

## Purpose

This file records the main structural issues that would matter if `ai_analyst`
is later expanded from a single-agent runtime into a crew/supervisor system.

These are not current blockers. They are deferred design notes.

## Structural Gaps To Revisit

### A. Stateless request execution

Current state:

- `run_query(...)` handles each request independently

Why it matters later:

- multi-agent systems usually need shared request state, memory, and trace IDs

### B. No frame/result caching

Current state:

- repeated frame builds can reread the same parquet data

Why it matters later:

- multiple agents would multiply I/O and compute unnecessarily

### C. Tool wrappers return JSON strings

Current state:

- LangChain wrappers return `json.dumps(...)` strings

Why it matters later:

- crew-style or richer function-calling runtimes usually prefer direct dict payloads

### D. String-based filter syntax

Current state:

- filter arguments are optimized for natural-language convenience

Why it matters later:

- multi-agent handoff is often more stable with structured filter payloads

### E. Monolithic analyst prompt

Current state:

- one prompt assumes one agent does everything

Why it matters later:

- supervisor/scout/validator roles would need role-specific prompts

### F. No persistent artifact registry

Current state:

- previews and future chart outputs are returned inline without stable artifact IDs

Why it matters later:

- one agent cannot reliably hand off a large result to another agent

## Current Recommendation

Do not solve these yet.

The current priority remains:

- reliable single-agent analysis
- stable services/tools layer
- provider-agnostic behavior

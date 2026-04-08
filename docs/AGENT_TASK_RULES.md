# Agent Task Rules

Read `docs/AGENT_STARTUP_CHECKLIST.md` first when entering this repository as an implementation agent.

## Agents May Do

- implement code
- add tests
- build exporters and payload generators
- improve dashboard wiring
- draft documentation
- update CI and developer tooling

## Agents Must Not Decide Alone

- labeling semantics
- strategy selection
- feature pruning based on interpretation
- schema changes with research meaning
- rule changes justified only by outcome improvement

## Safe Working Model

- keep issues small and single-purpose
- prefer code-only PRs
- avoid touching repo-local data dumps
- route path access through shared config
- do not hardcode `./data`, `./outputs`, `./reports`, or ad-hoc relative paths in new code
- document hidden assumptions in PRs

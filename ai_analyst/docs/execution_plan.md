# AI Analyst Execution Plan

## Purpose

This document turns the roadmap into the immediate work queue.

The current goal is:

- keep `ai_analyst` focused on domain validation, not generic BI
- keep the services/tools layer model-agnostic
- preserve the anchor-question baseline for provider comparisons
- avoid unnecessary work that only serves a weak local model

## Anchor Questions

These questions define the near-term regression scope.

1. `1h에서 가격변화율이 0 이상인 데이터를 정리해줘`
2. `1h에서 combo_4가 UUDU인 데이터만 보여줘`
3. `combo_4가 UUDU인 그룹과 DDDU인 그룹을 비교해줘`
4. `child_count가 높은 cycle들의 특징을 보여주고 적절하면 시각화해줘`

ASCII intent fallback:

1. Filter 1h cycles where price-change percentage is >= 0.
2. Show only 1h cycles where `combo_4 == UUDU`.
3. Compare the `combo_4 == UUDU` group with the `combo_4 == DDDU` group.
4. Characterize cycles with high `child_count`; add visualization only if it materially helps.

## Current Status

- Q1: `partial`
- Q2: `near pass`
- Q3: `partial`
- Q4: `partial`

Current implemented tool set:

1. `describe_available_data`
2. `build_analysis_frame`
3. `filter_frame`
4. `compare_groups`
5. `rank_features`

Current deferred tool:

- `create_plot`

## Step Status

- Step 0: `done` - baseline captured
- Step 1: `mostly done` - residual audit only
- Step 2: `active next`
- Step 3: `deferred to post-provider re-evaluation`
- `create_plot`: `deferred`

## Strategic Boundary

This project is not trying to become a generic analytics UI.

Responsibility split:

- `ai_analyst`: domain validation engine for cycle/context/feature questions
- Metabase + Postgres: generic exploration, dashboards, SQL, and ad hoc BI

If a task can be solved cleanly by generic BI, it should not expand the scope of
`ai_analyst`.

## Pause Point

The project should pause after model-independent cleanup and baseline capture.

Why:

- the current local-model behavior is not a strong basis for heavy prompt tuning
- the services/tools layer is already useful and portable
- the next meaningful comparison should happen after switching to the paid API provider
- generic exploration and dashboards should be delegated to Metabase rather than expanded inside this app

At the pause point:

1. switch provider/model through `.env`
2. rerun the 4 anchor questions
3. compare results against `baseline_runs.md`
4. only then decide whether prompt tuning, ambiguity policy, or plotting is still needed

## Work Order

### Step 0: Capture the Baseline

Goal:

- preserve the current before-state of the anchor questions

Tasks:

- run all 4 anchor questions against the current code
- save the raw responses and observed tool flow in `baseline_runs.md`
- mark each answer as pass, partial, or fail

Done when:

- a baseline record exists for all 4 anchor questions

### Step 1: Lock the Boundary

Goal:

- keep computation in services and wrappers in tools
- Status: mostly complete - residual audit only

Tasks:

- audit every tool module and identify logic that still belongs in services
- move reusable calculation logic only if a real boundary violation is found
- confirm that tools keep only argument parsing, validation handoff, and JSON serialization
- avoid moving LLM-facing string parsing into services unless reuse clearly demands it

Done when:

- tool modules are thin wrappers
- service-layer functions are reusable outside LangChain

### Step 2: Upgrade Existing Output Quality

Goal:

- make the current questions produce useful analysis, not just completion text

Tasks:

- improve `compare_groups` summaries so findings appear before generic narration
- improve `filter_frame` summaries so row-cap and row-count meaning are explicit
- keep warnings consistent across filter and compare outputs
- tighten prompt guidance only where the change is model-independent

Done when:

- anchor questions 1-3 produce concise, evidence-first answers

### Step 3: Re-evaluate Q4 After Provider Switch

Goal:

- confirm whether the existing `rank_features` capability is selected and used well enough after the provider switch

Tasks:

- rerun Q4 after the provider/model switch
- verify whether the agent now prefers `rank_features` for subset-characteristics questions
- decide whether `rank_features` alone is enough before adding any plotting
- add `create_plot` only if Q4 still clearly needs it after that re-evaluation

Done when:

- anchor question 4 is either answered cleanly with ranking alone or has a clearly justified case for a minimal chart

### Step 4: Re-run the Fixed Question Set

Goal:

- verify improvements against the baseline

Tasks:

- run all 4 anchor questions manually
- capture which tools were used
- compare outputs against the baseline record
- record limitations and follow-up fixes

Done when:

- each question has an updated pass, partial, or fail outcome

### Step 5: Prepare for Repeatability

Goal:

- make future changes safer

Tasks:

- keep the 4 anchor questions as the regression checklist
- keep expected tool flow documented next to the questions
- update docs whenever a tool meaningfully changes behavior

Done when:

- a future provider or prompt change can be checked quickly against the same 4 scenarios

## Expected Tool Flow Per Anchor Question

### Question 1

Question:

- `1h에서 가격변화율이 0 이상인 데이터를 정리해줘`

Expected flow:

1. `build_analysis_frame`
2. `filter_frame`
3. final answer with counts, field used, and row-cap caveat if relevant

### Question 2

Question:

- `1h에서 combo_4가 UUDU인 데이터만 보여줘`

Expected flow:

1. `build_analysis_frame`
2. `filter_frame`
3. final answer with counts and key preview

### Question 3

Question:

- `combo_4가 UUDU인 그룹과 DDDU인 그룹을 비교해줘`

Expected flow:

1. `build_analysis_frame`
2. `compare_groups`
3. final answer with row counts, metric deltas, and warnings

### Question 4

Question:

- `child_count가 높은 cycle들의 특징을 보여주고 적절하면 시각화해줘`

Expected flow:

1. `build_analysis_frame`
2. `rank_features`
3. `create_plot` only if plotting materially helps
4. final answer with explicit caveats

## Immediate Next Tasks

These are the next concrete tasks.

1. keep docs aligned with the real implemented tool set
2. preserve the current baseline snapshot
3. avoid model-specific over-tuning before the paid API re-test
4. re-run the 4 anchor questions after provider/model switch
5. decide on `create_plot` only after that re-test

## Guardrails

- do not add CrewAI
- do not introduce a second orchestration runtime
- do not add new placeholder tools
- do not let prompts become the source of truth for schema or paths
- do not widen scope beyond the 4 anchor questions until they pass
- do not overfit prompt behavior to a weak local model if the work will be thrown away after provider switch

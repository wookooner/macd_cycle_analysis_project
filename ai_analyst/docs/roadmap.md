# AI Analyst Roadmap

## Project Definition

Build `ai_analyst/` as an extensible single-agent analysis engine for the BTC
MACD cycle dataset.

This project is not currently a CrewAI project, a multi-agent orchestration
project, or a 24/7 autonomous research loop. The current goal is to finish a
grounded single-agent system with a framework-agnostic analysis core.

## Role Of This Document

This file answers:

- why the project is shaped this way
- what phases still matter
- what is explicitly out of scope right now

For the concrete current work queue and the 4 fixed anchor questions, use:

- `execution_plan.md`

## Current Direction

The priority order is:

1. question-first validation targets
2. framework-agnostic analysis services
3. deterministic tool wrappers
4. stable single-agent orchestration
5. visualization and reporting only when justified
6. only then reassess orchestration upgrades

## Not In Scope Right Now

The following are explicitly deferred:

- CrewAI adoption
- Discord integration
- 24-hour autonomous execution loops
- agent role specialization such as separate critic or data-scout agents
- concurrent LangGraph and CrewAI runtimes in the same app

## Phase 0: Anchor-Question Discipline

Objective:

- ensure all near-term work is judged against the same fixed question set

Current source of truth:

- `execution_plan.md`

Exit criteria:

- every subsequent phase can be judged against the same question set

## Phase 1: Service and Tool Boundary Stabilization

Objective:

- keep the analysis core framework-agnostic before adding more behavior

Rules:

- actual computation belongs in `services/data_access.py` or adjacent service modules
- tool files are thin LangChain bindings only
- string parsing for LLM convenience may remain in tool wrappers, but parsing must
  hand off into deterministic service-layer inputs
- the service layer must remain usable without LangGraph or LangChain imports

Exit criteria:

- the same service functions could be wrapped by LangGraph, CrewAI, CLI, or tests
- tool modules are thin wrappers rather than analysis engines

## Phase 2: Single-Agent Analysis Quality

Objective:

- make the current single agent answer the anchor questions with grounded results

Focus areas:

- output quality
- warning clarity
- row-cap interpretation
- stable use of implemented tools

Exit criteria:

- anchor questions 1-3 can be answered end-to-end
- anchor question 4 is either answered directly or clearly blocked by one missing tool

## Phase 3: Minimum Missing Capability

Objective:

- add only the smallest missing feature needed to satisfy the anchor questions

Current candidate:

- `create_plot`, but only if post-provider re-evaluation shows that ranking alone is not enough

Exit criteria:

- the fixed question set can be answered fully, with charting only if truly needed

## Phase 4: Regression Validation

Objective:

- rerun the fixed question set after meaningful changes

Exit criteria:

- the team can tell whether a prompt, tool, provider, or service change improved or degraded behavior

## Phase 5: Interface Layer

Objective:

- make the validated engine easier to use without changing its core logic

Possible deliverables:

- cleaned-up CLI experience
- optional chat-oriented or analysis-oriented interface shell
- artifact presentation for tables, warnings, and plots

## Phase 6: Orchestration Reassessment

Objective:

- decide whether more orchestration is justified after the single-agent engine works

Decision rule:

- first test whether a LangGraph supervisor pattern adds value on top of the
  existing services and tools
- only consider CrewAI if a clear problem remains that LangGraph does not solve
- do not introduce a second runtime for novelty alone

## Definition of Success

The project is on track when a single agent can:

- interpret the anchor questions correctly
- inspect valid datasets and fields
- build and filter analysis frames from canonical sources
- compare groups with grounded metrics
- identify important features when required
- create a useful chart when the question calls for one
- explain results with counts, warnings, and limitations

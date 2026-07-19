# AI Analyst Baseline Runs

## Purpose

This document captures the current before-state of the fixed anchor questions
before more prompt or tool-quality work continues.

## Environment

- Date: 2026-04-19
- Runtime: `ai_analyst/.venv`
- Entry path used: `run_query(...)` via the current LangGraph single-agent runtime
- Result status: runtime is available, but answer quality is still the active blocker

## Baseline Summary

The original runtime blocker has been resolved.

- The model endpoint is reachable.
- The current priority issues are tool discipline, ambiguity handling, and output robustness.
- Q1 is now a recovered partial rather than a hard fail.
- Q2 is close to a pass.
- Q3 remains conservative because timeframe ambiguity blocks completion.
- Q4 now completes end-to-end, but still uses the wrong analysis path for the question.

This means the baseline is now an answer-quality baseline rather than a
runtime-only baseline.

## Anchor Question Results

### Q1

Question:

- `1h에서 가격변화율이 0 이상인 데이터를 정리해줘`

Observed result:

- Status: `partial recovery`
- Initial failure: the agent first tried a guessed field such as `price_change_rate`
- Recovery: after the tool error exposed nearby columns, the agent retried with `feature__change__price_pct >= 0`
- Final answer: returned a filtered result and acknowledged the correct frame column

Assessment:

- `partial`

Notes:

- This is no longer a pure hallucination failure because the agent now self-corrects.
- However, the first tool choice still violates the intended flow because it guesses a field before verifying the frame columns.
- The final summary improved after the row-cap warning change, but the overall behavior is still less disciplined than desired.

### Q2

Question:

- `1h에서 combo_4가 UUDU인 데이터만 보여줘`

Observed result:

- Status: `successful filtering with minor overreach`
- Tool sequence observed:
  1. `filter_frame(timeframe='1h', filters="combo_4 == 'UUDU'")`
- Final answer returned the filtered subset and a useful preview

Assessment:

- `near pass`

Notes:

- The filtering path was correct and end-to-end completion was strong.
- The answer still included some interpretation that went beyond strict tool evidence.
- This is a good sign that explicit field names are handled much better than natural-language field descriptions.

### Q3

Question:

- `combo_4가 UUDU인 그룹과 DDDU인 그룹을 비교해줘`

Observed result:

- Status: `asked for missing timeframe`
- Tool sequence observed:
  1. `describe_available_data`
- Final answer asked the user to specify a timeframe before proceeding

Assessment:

- `partial`

Notes:

- This is not a hallucination failure.
- The model correctly recognized `combo_4` as a valid relationship field.
- However, the anchor question did not complete end-to-end because no comparison was executed.
- Whether this is acceptable depends on the product rule for ambiguous timeframe requests.

### Q4

Question:

- `child_count가 높은 cycle들의 특징을 보여주고 적절하면 시각화해줘`

Observed result:

- Status: `end-to-end success with limitations`
- Tool sequence observed:
  1. `describe_available_data`
  2. `build_analysis_frame(timeframe='1h', columns='child_count,cycle_key,cycle_type,duration_candles')`
  3. `compare_groups(group_a='child_count > 1', group_b='child_count <= 1', metrics='duration_candles,cycle_type')`
- Final answer completed without asking for more input, but it did not use `rank_features`
- The answer also leaked a `<|channel>` tag into the visible output

Assessment:

- `partial`

Notes:

- This is an end-to-end completion, which is better than the earlier blocked behavior.
- However, it is still only a partial because the question asked for "characteristics" and the agent used `compare_groups` instead of `rank_features`.
- Visualization is still not implemented, but the agent did not fabricate a chart.
- The visible `<|channel>` tag appears to be a model-output issue rather than a tool-layer issue.
- Treat the tag leak as low code-fix priority unless it persists after the provider switch.

## Immediate Implication

The next work should be driven by the full pattern, not by Q1 alone.

Current themes are:

- Q1: recovery works, but first-call discipline is weak
- Q2: explicit field names produce much stronger behavior than natural-language field descriptions
- Q3: ambiguity handling is conservative and blocks completion
- Q4: end-to-end completion is possible, but tool choice still needs steering

The next decision should be based on these combined patterns:

- prompt adjustment
- tool output improvement
- model upgrade
- or a small default-scope policy for ambiguous timeframe requests

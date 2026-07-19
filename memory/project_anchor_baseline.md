---
name: Anchor question baseline
description: Status of the 4 anchor questions captured in baseline_runs.md on 2026-04-19
type: project
---

Baseline captured 2026-04-19 in `ai_analyst/docs/baseline_runs.md`:

- Q1 (`가격변화율 >= 0` on 1h): `partial` — agent first guesses field name, recovers after tool error exposes columns.
- Q2 (`combo_4 == UUDU` on 1h): `near pass` — filtering correct, minor interpretive overreach.
- Q3 (UUDU vs DDDU compare): `partial` — agent asks for timeframe and stops instead of comparing.
- Q4 (child_count high cycles): `partial` — completes end-to-end but uses `compare_groups` instead of `rank_features`; `<|channel>` tag leaks into output.

**Why:** This is the before-state for the answer-quality (not runtime) phase of work. Any change to prompts/tools/model must be judged against this snapshot.

**How to apply:** Before claiming improvement, re-run all 4 questions and compare. Do not declare a fix from Q1 alone — the themes (first-call discipline, ambiguity handling, tool-choice steering) span all four.

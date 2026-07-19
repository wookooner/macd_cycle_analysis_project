---
name: ai_analyst project scope
description: Purpose, boundaries, and pause-point policy for the ai_analyst/ subproject
type: project
---

`ai_analyst/` is a single-agent domain-analysis engine over BTC MACD cycle parquet data. Not a BI tool, not a multi-agent system, not a trading app.

Five foundation tools exist: `describe_available_data`, `build_analysis_frame`, `filter_frame`, `compare_groups`, `rank_features`. `create_plot` is explicitly deferred.

Near-term scope is locked to 4 Korean anchor questions defined in `ai_analyst/docs/execution_plan.md`.

**Why:** The current local model is weak, so the plan is to do only model-independent cleanup (service/tool boundary, output quality), then pause and re-test after switching to a paid API. This avoids throwing away prompt tuning done for a weak model.

**How to apply:** Before adding any feature, check whether it (a) helps answer the 4 anchor questions, (b) is model-independent, and (c) belongs in `ai_analyst` vs Metabase. If any answer is no, defer. Do not add CrewAI, Discord, autonomous loops, or a second orchestration runtime.

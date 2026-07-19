---
name: BI boundary rule
description: Generic BI work belongs in Metabase+DuckDB, not inside ai_analyst
type: feedback
---

Do not expand `ai_analyst` to cover dashboards, ad-hoc SQL, generic charts, or broad distribution exploration. Those belong in Metabase + DuckDB.

**Why:** The user explicitly split responsibility in `architecture.md` and `execution_plan.md` — `ai_analyst` is for domain validation (cycle/context/feature hypotheses), Metabase is for observation. Rebuilding BI inside `ai_analyst` duplicates work and dilutes focus on the 4 anchor questions.

**How to apply:** When a task request sounds like "show me a dashboard", "let me browse the data", or "generic chart of X", suggest Metabase instead of adding it to `ai_analyst`. Only add features to `ai_analyst` that serve domain-specific validation the BI layer cannot do cleanly.

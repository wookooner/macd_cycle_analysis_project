# BTC MACD Cycle AI Analyst

Lightweight domain-analysis engine for the BTC MACD cycle dataset.

This repository is not meant to become a full BI platform, dashboard product,
or multi-agent orchestration system.

Its job is narrower:

- load canonical cycle/context parquet data
- expose domain-specific analysis tools for cycle/context questions
- answer theory-oriented questions that generic BI tools do not understand

## Current Scope

This is an early single-agent domain engine. The structure is in place, the
base tools are real, and the current focus is to make a fixed set of user
questions pass reliably before expanding scope.

Planned operating model:

- `ai_analyst` handles domain questions and theory validation
- `Metabase + DuckDB` will later handle generic BI, dashboarding, and ad-hoc exploration
- orchestration beyond the current single-agent runtime is intentionally deferred

Architecture references:

- `docs/README.md`
- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/execution_plan.md`
- `docs/contracts.md`
- `docs/settings.md`
- `docs/maintenance.md`
- `docs/tools/README.md`

Current foundation tools:

1. `describe_available_data`
2. `build_analysis_frame`
3. `filter_frame`
4. `compare_groups`
5. `rank_features`

Planned next-layer tools:

1. `create_plot`

## Current Layout

```text
ai_analyst/
|- README.md
|- requirements.txt
|- .env
|- main.py
|- test_analyst.py
|- configs/
|  `- paths.yaml
|- docs/
|  |- architecture.md
|  |- contracts.md
|  |- DATA_LAYOUT.md
|  |- roadmap.md
|  |- settings.md
|  `- tools/
|     |- README.md
|     |- describe_available_data.md
|     |- build_analysis_frame.md
|     |- filter_frame.md
|     `- compare_groups.md
`- src/
   `- btc_macd_cycle_ai_analyst/
      |- __init__.py
      |- settings.py
      |- prompts/
      |  |- __init__.py
      |  `- system_prompt.py
      |- agent/
      |  |- __init__.py
      |  `- factory.py
      |- tools/
      |  |- __init__.py
      |  |- analysis.py
      |  |- discovery.py
      |  |- filtering.py
      |  `- frame.py
      `- services/
         |- __init__.py
         |- data_access.py
         `- paths.py
```

## Design Notes

- Keep the agent layer thin.
- Put real dataset knowledge into tools, not prompts.
- Treat this repository as a focused domain-analysis service, not a general trading system or generic BI app.
- Reuse canonical data conventions from the MACD project rather than duplicating them.
- Let generic BI concerns move to Metabase instead of rebuilding them here.

## Next Steps

1. Keep docs aligned with the real implemented tool set.
2. Preserve the current baseline snapshot for the 4 anchor questions.
3. Re-test the engine after switching to the paid API provider.
4. Decide whether `create_plot` is still needed after that re-test.
5. Integrate generic exploration through Metabase rather than growing a custom BI layer here.

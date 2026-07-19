# AI Analyst Settings And Paths

## Purpose

This document describes the first shared runtime layer for `ai_analyst/`.
The goal is to remove hardcoded local assumptions from `test_analyst.py` and
replace them with reusable settings and canonical path resolution.

## Current Runtime Modules

- `src/btc_macd_cycle_ai_analyst/settings.py`
- `src/btc_macd_cycle_ai_analyst/services/paths.py`
- `src/btc_macd_cycle_ai_analyst/agent/factory.py`

## Settings Responsibilities

`settings.py` is responsible for:

- loading `.env`
- reading default values
- exposing LLM connection settings
- exposing the default asset
- exposing the path-config location

Supported environment variables now:

- `AI_ANALYST_LLM_MODEL`
- `AI_ANALYST_LLM_BASE_URL`
- `AI_ANALYST_LLM_API_KEY`
- `AI_ANALYST_LLM_TEMPERATURE`
- `AI_ANALYST_LLM_MAX_TOKENS`
- `AI_ANALYST_DEFAULT_ASSET`

## Path Resolution

`services/paths.py` resolves the canonical data root in this order:

1. environment variable named by `configs/paths.yaml`
2. `project.default_data_root` from `configs/paths.yaml`

From there it exposes helper methods for:

- processed root
- enriched cycle directory
- context directory
- timeframe-specific cycle files
- timeframe-specific context files

## Immediate Benefit

This is still a small step, but it gives the app one shared source of truth for:

- where data lives
- which model endpoint is used
- which asset is assumed by default

That keeps future tools from repeating local path logic in test scripts.

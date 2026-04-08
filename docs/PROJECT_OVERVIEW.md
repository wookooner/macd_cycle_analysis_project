# Project Overview

## Goal

Separate code, data, outputs, documentation, and agent working rules so the repository stays small, reviewable, and safe for parallel agent workflows.

## Repository Role

This repository should contain:

- reproducible code
- configuration
- documentation
- tests
- dashboard UI code
- lightweight sample data only

This repository should not be the long-term home for:

- raw market files
- intermediate processing outputs
- dashboard payload dumps
- backtest result dumps
- large reports
- operational logs

## Path Standard

- Shared path resolution lives in `configs/paths.yaml` and `src/common/paths.py`.
- New code should not hardcode `data/`, `outputs/`, or `reports/`.
- Legacy repo-local paths still exist for compatibility during migration.

## Human vs Agent

- Humans decide labeling semantics, feature meaning, interpretation, and strategy adoption.
- Agents implement code, tests, exporters, dashboards, CI, and draft documentation.

"""Collectors used by the data pipeline."""
"""Active market-data collectors.

Use :func:`data_pipeline.collectors.registry.create_collector` from pipeline
code. Historical module names stay available only for compatibility.
"""

from data_pipeline.collectors.registry import create_collector

__all__ = ["create_collector"]

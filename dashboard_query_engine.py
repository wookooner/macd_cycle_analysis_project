from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Callable


SUPPORTED_QUERY_ENGINES = ("pandas", "duckdb")


def get_configured_query_engine() -> str:
  engine = os.getenv("DASHBOARD_QUERY_ENGINE", "pandas").strip().lower()
  return engine if engine in SUPPORTED_QUERY_ENGINES else "pandas"


class DashboardQueryEngine(ABC):
  name: str

  @abstractmethod
  def list_datasets(self) -> dict[str, Any]:
    raise NotImplementedError

  @abstractmethod
  def get_dataset_features(self, dataset: str) -> dict[str, Any]:
    raise NotImplementedError

  @abstractmethod
  def preview_dataset(self, dataset: str, filters: list[Any], limit: int) -> dict[str, Any]:
    raise NotImplementedError

  @abstractmethod
  def build_chart(
    self,
    dataset: str,
    filters: list[Any],
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    group_by: str | None = None,
    metric: str | None = "count",
    bins: int | None = 20,
    limit: int | None = 100,
  ) -> dict[str, Any]:
    raise NotImplementedError


class PandasDashboardQueryEngine(DashboardQueryEngine):
  name = "pandas"

  def __init__(
    self,
    *,
    list_datasets_fn: Callable[[], dict[str, Any]],
    get_dataset_features_fn: Callable[[str], dict[str, Any]],
    preview_dataset_fn: Callable[[str, list[Any], int], dict[str, Any]],
    build_chart_fn: Callable[..., dict[str, Any]],
  ) -> None:
    self._list_datasets_fn = list_datasets_fn
    self._get_dataset_features_fn = get_dataset_features_fn
    self._preview_dataset_fn = preview_dataset_fn
    self._build_chart_fn = build_chart_fn

  def list_datasets(self) -> dict[str, Any]:
    return self._list_datasets_fn()

  def get_dataset_features(self, dataset: str) -> dict[str, Any]:
    return self._get_dataset_features_fn(dataset)

  def preview_dataset(self, dataset: str, filters: list[Any], limit: int) -> dict[str, Any]:
    return self._preview_dataset_fn(dataset, filters, limit)

  def build_chart(
    self,
    dataset: str,
    filters: list[Any],
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    group_by: str | None = None,
    metric: str | None = "count",
    bins: int | None = 20,
    limit: int | None = 100,
  ) -> dict[str, Any]:
    return self._build_chart_fn(
      dataset=dataset,
      filters=filters,
      chart_type=chart_type,
      x=x,
      y=y,
      color=color,
      group_by=group_by,
      metric=metric,
      bins=bins,
      limit=limit,
    )


class DuckDBDashboardQueryEngine(DashboardQueryEngine):
  name = "duckdb"

  def __init__(self) -> None:
    self._message = "DuckDB engine is not wired yet. Switch DASHBOARD_QUERY_ENGINE=pandas for current behavior."

  def list_datasets(self) -> dict[str, Any]:
    raise NotImplementedError(self._message)

  def get_dataset_features(self, dataset: str) -> dict[str, Any]:
    raise NotImplementedError(self._message)

  def preview_dataset(self, dataset: str, filters: list[Any], limit: int) -> dict[str, Any]:
    raise NotImplementedError(self._message)

  def build_chart(
    self,
    dataset: str,
    filters: list[Any],
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    group_by: str | None = None,
    metric: str | None = "count",
    bins: int | None = 20,
    limit: int | None = 100,
  ) -> dict[str, Any]:
    raise NotImplementedError(self._message)

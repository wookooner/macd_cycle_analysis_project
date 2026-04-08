from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import dashboard_query_service as query_service


router = APIRouter()


class FilterCondition(BaseModel):
    field: str
    op: str
    value: Any


class PreviewRequest(BaseModel):
    dataset: str
    filters: list[FilterCondition] = Field(default_factory=list)
    limit: int = 50


class ChartRequest(BaseModel):
    dataset: str
    filters: list[FilterCondition] = Field(default_factory=list)
    chart_type: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    group_by: str | None = None
    metric: str | None = "count"
    bins: int | None = 20
    limit: int | None = 100


@router.get("/api/dashboard/datasets")
def list_datasets() -> dict[str, Any]:
    return query_service.get_query_engine().list_datasets()


@router.get("/api/dashboard/features/{dataset}")
def get_dataset_features(dataset: str) -> dict[str, Any]:
    return query_service.get_query_engine().get_dataset_features(dataset)


@router.post("/api/dashboard/preview")
def preview_dataset(request: PreviewRequest) -> dict[str, Any]:
    return query_service.get_query_engine().preview_dataset(
        dataset=request.dataset,
        filters=request.filters,
        limit=request.limit,
    )


@router.post("/api/dashboard/chart")
def build_chart(request: ChartRequest) -> dict[str, Any]:
    return query_service.get_query_engine().build_chart(
        dataset=request.dataset,
        filters=request.filters,
        chart_type=request.chart_type,
        x=request.x,
        y=request.y,
        color=request.color,
        group_by=request.group_by,
        metric=request.metric,
        bins=request.bins,
        limit=request.limit,
    )


app = FastAPI(title="Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

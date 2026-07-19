"""Tool exports for ai_analyst."""

from .analysis import analyze_feature_combinations, compare_groups, rank_features
from .discovery import describe_available_data
from .frame import build_analysis_frame
from .filtering import filter_frame


def get_default_tools():
    return [
        describe_available_data,
        build_analysis_frame,
        filter_frame,
        compare_groups,
        rank_features,
        analyze_feature_combinations,
    ]


__all__ = [
    "analyze_feature_combinations",
    "build_analysis_frame",
    "compare_groups",
    "describe_available_data",
    "filter_frame",
    "get_default_tools",
    "rank_features",
]

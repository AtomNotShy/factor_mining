"""
策略组件库
提供可复用的评分器、过滤器等组件
"""

from .scorers import (
    ScoreCalculator,
    MomentumScorer,
    SimpleReturnScorer,
    RSIScorer,
    ATRScorer,
    CompositeScorer,
)

from .filters import (
    Filter,
    RangeFilter,
    DrawdownFilter,
    ConsecutiveDeclineFilter,
    VolumeFilter,
    VolatilityFilter,
    CompositeFilter,
)

__all__ = [
    # 评分器
    "ScoreCalculator",
    "MomentumScorer",
    "SimpleReturnScorer",
    "RSIScorer",
    "ATRScorer",
    "CompositeScorer",
    # 过滤器
    "Filter",
    "RangeFilter",
    "DrawdownFilter",
    "ConsecutiveDeclineFilter",
    "VolumeFilter",
    "VolatilityFilter",
    "CompositeFilter",
]

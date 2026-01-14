"""
数据源 providers 模块
提供 DataFeed 抽象接口和实现
"""

from src.data.providers.base import (
    DataFeed,
    HistoricalDataFeed,
    DataFeedFactory,
)

__all__ = [
    "DataFeed",
    "HistoricalDataFeed",
    "DataFeedFactory",
]

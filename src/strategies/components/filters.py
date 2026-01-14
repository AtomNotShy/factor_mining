"""
过滤器组件
提供可复用的信号过滤逻辑
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np


class Filter(ABC):
    """过滤器基类"""

    @abstractmethod
    def apply(self, df: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """应用过滤逻辑"""
        pass


class RangeFilter(Filter):
    """范围过滤器"""

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, df: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """应用范围过滤"""
        filtered = scores.copy()
        if self.min_val is not None:
            filtered = filtered.where(filtered >= self.min_val, 0)
        if self.max_val is not None:
            filtered = filtered.where(filtered <= self.max_val, 0)
        return filtered


class DrawdownFilter(Filter):
    """回撤过滤器"""

    def __init__(
        self,
        max_drawdown: float = 0.05,
        lookback: int = 20,
    ):
        self.max_drawdown = max_drawdown
        self.lookback = lookback

    def apply(self, df: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """应用回撤过滤"""
        if "close" not in df.columns:
            return scores

        close = df["close"]

        # 计算滚动最高价
        rolling_high = close.rolling(window=self.lookback).max()

        # 计算回撤
        drawdown = (close - rolling_high) / rolling_high

        # 过滤：近期最大回撤超过阈值的得分为0
        recent_drawdown = drawdown.rolling(window=5).min()

        filtered = scores.copy()
        filtered = filtered.where(recent_drawdown >= -self.max_drawdown, 0)

        return filtered


class ConsecutiveDeclineFilter(Filter):
    """连续下跌过滤器"""

    def __init__(self, min_consecutive: int = 3, decline_threshold: float = 0.05):
        self.min_consecutive = min_consecutive
        self.decline_threshold = decline_threshold

    def apply(self, df: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """应用连续下跌过滤"""
        if "close" not in df.columns:
            return scores

        close = df["close"].values
        n = len(close)

        if n < self.min_consecutive + 1:
            return scores

        filtered = scores.copy()

        for i in range(self.min_consecutive, n):
            # 检查最近 N+1 天
            prices = close[i - self.min_consecutive : i + 1]

            # 连续下跌检查
            consecutive_declines = 0
            for j in range(len(prices) - 1):
                if prices[j + 1] < prices[j]:
                    consecutive_declines += 1

            # 累计下跌检查
            total_decline = prices[-1] / prices[0] - 1

            if (
                consecutive_declines >= self.min_consecutive
                and total_decline < -self.decline_threshold
            ):
                filtered.iloc[i] = 0

        return filtered


class VolumeFilter(Filter):
    """成交量过滤器"""

    def __init__(
        self,
        min_volume_ratio: float = 0.5,
        lookback: int = 20,
    ):
        self.min_volume_ratio = min_volume_ratio
        self.lookback = lookback

    def apply(self, df: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """应用成交量过滤"""
        if "volume" not in df.columns:
            return scores

        volume = df["volume"]

        # 计算平均成交量
        avg_volume = volume.rolling(window=self.lookback).mean()

        # 成交量比率
        volume_ratio = volume / avg_volume

        # 过滤：成交量低于阈值的得分为0
        filtered = scores.copy()
        filtered = filtered.where(volume_ratio >= self.min_volume_ratio, 0)

        return filtered


class VolatilityFilter(Filter):
    """波动率过滤器"""

    def __init__(
        self,
        max_volatility: float = 0.5,
        lookback: int = 20,
    ):
        self.max_volatility = max_volatility
        self.lookback = lookback

    def apply(self, df: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """应用波动率过滤"""
        if "close" not in df.columns:
            return scores

        close = df["close"]

        # 计算收益率
        returns = close.pct_change()

        # 计算滚动波动率
        volatility = returns.rolling(window=self.lookback).std()

        # 年化
        annualized_vol = volatility * np.sqrt(250)

        # 过滤：波动率超过阈值的得分为0
        filtered = scores.copy()
        filtered = filtered.where(annualized_vol <= self.max_volatility, 0)

        return filtered


class CompositeFilter(Filter):
    """复合过滤器"""

    def __init__(self, filters: list[Filter]):
        self.filters = filters

    def apply(self, df: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """应用所有过滤器"""
        result = scores.copy()
        for filter_obj in self.filters:
            result = filter_obj.apply(df, result)
        return result

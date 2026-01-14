"""
评分器组件
提供可复用的得分计算逻辑
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np
import math


class ScoreCalculator(ABC):
    """评分器基类"""

    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """计算得分"""
        pass


class MomentumScorer(ScoreCalculator):
    """
    动量评分器
    基于加权线性回归计算动量得分
    """

    def __init__(
        self,
        lookback: int = 20,
        weighting: str = "linear",
        annualization_days: int = 250,
    ):
        self.lookback = lookback
        self.weighting = weighting
        self.annualization_days = annualization_days

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算动量得分"""
        if "close" not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        close = df["close"].values
        n = len(close)
        lookback = min(self.lookback, n)

        if lookback < 5:
            return pd.Series(index=df.index, dtype=float)

        scores = np.full(n, np.nan)

        for i in range(lookback - 1, n):
            prices = close[i - lookback + 1 : i + 1]
            if len(prices) != lookback:
                continue

            y_log = np.log(prices)
            if not np.isfinite(y_log).all():
                continue

            x = np.arange(len(y_log))

            # 权重
            if self.weighting == "linear":
                weights = np.linspace(1, 2, len(y_log))
            else:
                weights = np.ones(len(y_log))

            # 线性回归
            slope, intercept = np.polyfit(x, y_log, 1, w=weights)

            # 年化收益率
            ann_ret = math.exp(slope * self.annualization_days) - 1

            # R²
            y_pred = slope * x + intercept
            ss_res = np.sum(weights * (y_log - y_pred) ** 2)
            y_mean = np.mean(y_log)
            ss_tot = np.sum(weights * (y_log - y_mean) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot else 0

            # 得分
            scores[i] = ann_ret * r2

        return pd.Series(scores, index=df.index, dtype=float)


class SimpleReturnScorer(ScoreCalculator):
    """
    简单收益率评分器
    基于指定回溯期的收益率
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算收益率得分"""
        if "close" not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        returns = df["close"].pct_change(self.lookback)
        return returns


class RSIScorer(ScoreCalculator):
    """
    RSI 评分器
    基于 RSI 值的得分
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算 RSI 得分"""
        if "close" not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        # 计算 RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # 转换为得分：oversold 时高得分，overbought 时低得分
        score = (rsi - self.oversold) / (self.overbought - self.oversold)
        score = score.clip(0, 1)

        return score


class ATRScorer(ScoreCalculator):
    """
    ATR 波动率评分器
    基于 ATR 百分比
    """

    def __init__(self, period: int = 14, annualized: bool = True):
        self.period = period
        self.annualized = annualized

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算 ATR 得分"""
        if not {"high", "low", "close"}.issubset(df.columns):
            return pd.Series(index=df.index, dtype=float)

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 计算 ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        if self.annualized:
            # ATR 年化百分比
            atr_pct = atr / close * np.sqrt(250)
        else:
            atr_pct = atr / close

        return atr_pct


class CompositeScorer(ScoreCalculator):
    """
    复合评分器
    组合多个评分器
    """

    def __init__(
        self,
        scorers: list[ScoreCalculator],
        weights: Optional[list[float]] = None,
    ):
        self.scorers = scorers
        self.weights = weights if weights else [1.0] * len(scorers)

        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复合得分"""
        scores = []
        for scorer in self.scorers:
            score = scorer.calculate(df)
            scores.append(score)

        # 加权平均
        result = scores[0] * self.weights[0]
        for i in range(1, len(scores)):
            result = result + scores[i] * self.weights[i]

        return result

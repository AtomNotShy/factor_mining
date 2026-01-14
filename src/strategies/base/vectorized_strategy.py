"""
向量化策略基类
提供 DataFrame 向量化处理的统一接口
"""

from abc import ABC
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .strategy import Strategy
from .parameters import Parameter


class VectorizedMixin:
    """向量化功能混入类"""

    def _validate_dataframe_columns(
        self, df: pd.DataFrame, required_cols: list
    ) -> None:
        """验证 DataFrame 是否有必需列"""
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _add_signal_columns(self, df: pd.DataFrame, signals: dict) -> pd.DataFrame:
        """统一添加信号列"""
        for signal_type, values in signals.items():
            if signal_type not in df.columns:
                df[signal_type] = values
            else:
                df[signal_type] = values
        return df

    def _extract_signals_from_dataframe(
        self, df: pd.DataFrame, signal_col: str = "enter_long"
    ) -> List[Dict]:
        """从 DataFrame 提取信号"""
        signals = []
        if signal_col in df.columns:
            for idx, row in df.iterrows():
                if row.get(signal_col, 0) == 1:
                    signals.append(
                        {
                            "timestamp": idx,
                            "symbol": row.get("symbol", ""),
                            "action": "long",
                            "price": row.get("close", 0),
                        }
                    )
        return signals


class VectorizedStrategy(Strategy, VectorizedMixin, ABC):
    """
    向量化策略基类
    优先使用 populate_* 方法进行 DataFrame 向量化处理
    """

    def __init__(self, config=None):
        super().__init__(config)

    def process_dataframes(
        self, dataframes: Dict[str, pd.DataFrame], metadata: Dict
    ) -> Dict[str, pd.DataFrame]:
        """
        处理所有时间框架的数据
        主时间框架完整处理，其他仅计算指标
        """
        processed = {}

        for timeframe, df in dataframes.items():
            if df is None or df.empty:
                continue

            if timeframe == self.timeframe:
                # 主时间框架：完整处理
                df = self.populate_indicators(df, metadata)
                df = self.populate_entry_trend(df, metadata)
                df = self.populate_exit_trend(df, metadata)
            else:
                # 信息时间框架：仅指标
                df = self.populate_indicators(df, metadata)

            processed[timeframe] = df

        return processed

    def get_vectorized_signals(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> Dict[str, pd.Series]:
        """
        获取向量化信号
        返回包含 enter_long, exit_long 等信号的字典
        """
        signals = {}

        # 处理指标
        df = self.populate_indicators(dataframe.copy(), metadata)

        # 提取信号
        if "enter_long" in df.columns:
            signals["enter_long"] = df["enter_long"]
        if "enter_short" in df.columns:
            signals["enter_short"] = df["enter_short"]
        if "exit_long" in df.columns:
            signals["exit_long"] = df["exit_long"]
        if "exit_short" in df.columns:
            signals["exit_short"] = df["exit_short"]

        # 保留指标列
        for col in df.columns:
            if col.startswith(("momentum_", "score_", "rsi_", "ema_", "atr_")):
                signals[col] = df[col]

        return signals

    def should_run_vectorized(self) -> bool:
        """
        判断是否应该使用向量化执行
        子类可 override 此方法
        """
        return self._has_vectorized_methods()


class CrossSectionalStrategy(VectorizedStrategy, ABC):
    """
    横截面策略基类
    专门处理多标的选择和轮动
    """

    def __init__(self, config=None):
        super().__init__(config)

    def calculate_cross_sectional_scores(
        self, dataframes: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.Series]:
        """
        计算横截面得分
        每个标的返回一个得分 Series
        """
        scores = {}
        for symbol, df in dataframes.items():
            if df is None or df.empty:
                continue
            score = self._calculate_single_score(df)
            if score is not None:
                scores[symbol] = score
        return scores

    def _calculate_single_score(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        计算单个标的的得分
        子类必须实现此方法
        """
        raise NotImplementedError

    def select_top_n(
        self, scores: Dict[str, pd.Series], n: int = 3
    ) -> List[str]:
        """
        选出得分最高的 N 个标的
        """
        if not scores:
            return []

        # 计算最新得分
        latest_scores = {symbol: score.iloc[-1] for symbol, score in scores.items()}

        # 排序并返回 Top N
        sorted_symbols = sorted(latest_scores.items(), key=lambda x: x[1], reverse=True)

        return [symbol for symbol, score in sorted_symbols[:n]]

    def rank_symbols(
        self, scores: Dict[str, pd.Series], min_score: float = 0.0
    ) -> List[str]:
        """
        对标的进行排名
        返回按得分降序排列的列表
        """
        if not scores:
            return []

        # 过滤低分标的
        valid_scores = {
            symbol: score for symbol, score in scores.items() if score.iloc[-1] > min_score
        }

        # 排序
        sorted_symbols = sorted(
            valid_scores.items(), key=lambda x: x[1].iloc[-1], reverse=True
        )

        return [symbol for symbol, score in sorted_symbols]


class MomentumCrossSectionalStrategy(CrossSectionalStrategy, ABC):
    """
    动量横截面策略基类
    提供通用的动量得分计算逻辑
    """

    # 动量参数（可在子类中 override）
    momentum_lookback: int = 20
    annualization_days: int = 250

    def _calculate_single_score(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """计算动量得分"""
        if "close" not in df.columns or len(df) < self.momentum_lookback:
            return None

        close = df["close"].values
        lookback = min(self.momentum_lookback, len(close))

        if lookback < 5:
            return None

        prices = close[-lookback:]
        y_log = np.log(prices)
        x = np.arange(len(y_log))
        weights = np.linspace(1, 2, len(y_log))

        # 线性回归
        slope, intercept = np.polyfit(x, y_log, 1, w=weights)

        # 年化收益率
        ann_ret = np.exp(slope * self.annualization_days) - 1

        # R²
        y_pred = slope * x + intercept
        ss_res = np.sum(weights * (y_log - y_pred) ** 2)
        y_mean = np.mean(y_log)
        ss_tot = np.sum(weights * (y_log - y_mean) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else 0

        # 得分
        score = ann_ret * r2

        # 风控过滤：近期大幅下跌得分为0
        if lookback >= 4:
            if min(prices[-1] / prices[-2], prices[-2] / prices[-3], prices[-3] / prices[-4]) < 0.95:
                score = 0

        # 创建得分 Series
        score_series = pd.Series(index=df.index, dtype=float)
        score_series.iloc[-1] = score

        return score_series

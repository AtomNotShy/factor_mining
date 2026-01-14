"""
简化的 ETF 动量策略示例
展示如何使用新的策略框架编写简洁的策略

对比：原始实现 455 行 -> 新实现 ~80 行
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from src.strategies.base.vectorized_strategy import VectorizedStrategy
from src.strategies.base.strategy import StrategyConfig
from src.strategies.base.parameters import IntParameter, DecimalParameter
from src.strategies.components.scorers import MomentumScorer
from src.strategies.components.filters import (
    RangeFilter,
    DrawdownFilter,
    ConsecutiveDeclineFilter,
    CompositeFilter,
)
from src.core.types import (
    Signal,
    OrderIntent,
    PortfolioState,
    RiskState,
    MarketData,
    ActionType,
    OrderSide,
    OrderType,
)
from src.core.context import RunContext


class SimpleETFMomentumStrategy(VectorizedStrategy):
    """
    简化的 ETF 动量策略

    特点：
    - 使用组件化设计
    - 参数自动收集
    - 向量化处理
    - 代码简洁
    """

    # ===== 参数定义 =====
    lookback_days = IntParameter("lookback_days", 20, 60, default=25)
    min_score = DecimalParameter("min_score", 0.0, 2.0, default=0.0)
    max_score = DecimalParameter("max_score", 4.0, 8.0, default=6.0)
    max_drawdown = DecimalParameter("max_drawdown", 0.02, 0.10, default=0.05)
    target_positions = IntParameter("target_positions", 1, 5, default=3)

    # ===== 默认配置 =====
    DEFAULT_ETF_POOL = [
        "QQQ",  # 纳斯达克100
        "SPY",  # 标普500
        "TLT",  # 20年期国债
        "GLD",  # 黄金
        "USO",  # 原油
    ]

    def __init__(self, config=None):
        if config is None:
            config = StrategyConfig(
                strategy_id="simple_etf_momentum",
                timeframe="1d",
                params={
                    "etf_pool": self.DEFAULT_ETF_POOL.copy(),
                },
            )
        super().__init__(config)

        # ===== 初始化组件 =====
        self.scorer = MomentumScorer(
            lookback=int(self.lookback_days.value),
            weighting="linear",
            annualization_days=250,
        )

        self.filter = CompositeFilter(
            [
                RangeFilter(
                    min_val=self.min_score.value,
                    max_val=self.max_score.value,
                ),
                DrawdownFilter(max_drawdown=self.max_drawdown.value),
                ConsecutiveDeclineFilter(
                    min_consecutive=3,
                    decline_threshold=0.05,
                ),
            ]
        )

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        """
        计算动量指标
        向量化实现：一次性处理所有数据
        """
        # 计算动量得分
        scores = self.scorer.calculate(dataframe)

        # 应用过滤器
        filtered_scores = self.filter.apply(dataframe, scores)

        # 添加到 DataFrame
        dataframe["momentum_score"] = filtered_scores
        dataframe["raw_score"] = scores

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        """
        进场信号
        得分大于 0 且有效的标的进入
        """
        dataframe["enter_long"] = np.where((dataframe["momentum_score"] > 0)
            & (dataframe["momentum_score"] < self.max_score.value),1,0,)
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        """
        离场信号
        得分变为负数或低于最小阈值时退出
        """
        dataframe["exit_long"] = np.where( dataframe["momentum_score"] <= self.min_score.value,1,0,)
        return dataframe

    def generate_signals(
        self, md: MarketData, ctx: RunContext
    ) -> List[Signal]:
        """
        生成交易信号
        从 DataFrame 提取信号
        """
        signals = []

        if md.bars_all is None or md.bars_all.empty:
            return signals

        # 按标的分组处理
        for symbol in self.config.params.get("etf_pool", []):
            symbol_data = md.bars_all[md.bars_all["symbol"] == symbol]

            if symbol_data.empty:
                continue

            latest = symbol_data.iloc[-1]

            # 检查进场信号
            if latest.get("enter_long", 0) == 1:
                signals.append(
                    Signal(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        strategy_id=self.strategy_id,
                        action=ActionType.LONG,
                        strength=latest.get("momentum_score", 0),
                        metadata={
                            "score": latest.get("momentum_score", 0),
                            "price": latest.get("close", 0),
                        },
                    )
                )

            # 检查离场信号
            elif latest.get("exit_long", 0) == 1:
                signals.append(
                    Signal(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        strategy_id=self.strategy_id,
                        action=ActionType.FLAT,
                        strength=0,
                        metadata={
                            "score": latest.get("momentum_score", 0),
                            "price": latest.get("close", 0),
                        },
                    )
                )

        return signals

    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        """
        仓位计算
        简化的等权配置
        """
        orders = []

        if not signals:
            return orders

        target_positions = self.target_positions.value

        # 筛选有效的 LONG 信号
        valid_signals = [
            s for s in signals if s.action == ActionType.LONG and s.strength > 0
        ]

        if not valid_signals:
            # 没有有效信号，清空持仓
            for symbol, qty in portfolio.positions.items():
                if abs(qty) > 1e-6:
                    orders.append(
                        OrderIntent(
                            ts_utc=ctx.now_utc,
                            symbol=symbol,
                            side=OrderSide.SELL,
                            qty=abs(qty),
                            order_type=OrderType.MKT,
                            strategy_id=self.strategy_id,
                            metadata={"reason": "no_valid_signal"},
                        )
                    )
            return orders

        # 按得分排序
        valid_signals.sort(key=lambda x: x.strength, reverse=True)

        # 选出 Top N
        top_signals = valid_signals[:target_positions]
        target_symbols = {s.symbol for s in top_signals}

        # 卖出不在目标中的持仓
        for symbol, qty in portfolio.positions.items():
            if abs(qty) > 1e-6 and symbol not in target_symbols:
                orders.append(
                    OrderIntent(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=abs(qty),
                        order_type=OrderType.MKT,
                        strategy_id=self.strategy_id,
                        metadata={"reason": "rank_change"},
                    )
                )

        # 买入目标标的
        held_symbols = set(portfolio.positions.keys())
        slots = target_positions - len(held_symbols & target_symbols)

        if slots > 0 and top_signals:
            cash_per_position = portfolio.cash / slots

            for signal in top_signals:
                if signal.symbol in held_symbols:
                    continue

                price = signal.metadata.get("price", 0)
                if price <= 0:
                    continue

                qty = cash_per_position / price

                if qty * price < 100:  # 最小交易金额
                    continue

                orders.append(
                    OrderIntent(
                        ts_utc=ctx.now_utc,
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        qty=qty,
                        order_type=OrderType.MKT,
                        strategy_id=self.strategy_id,
                        metadata={
                            "reason": "rank_change",
                            "score": signal.strength,
                        },
                    )
                )

        return orders


# ===== 使用示例 =====
if __name__ == "__main__":
    # 创建策略实例
    strategy = SimpleETFMomentumStrategy()

    # 获取参数
    print("策略参数:")
    for name, param in strategy.get_parameters().items():
        print(f"  {name}: {param.value} (范围: {param.min_val} - {param.max_val})")

    # 修改参数
    strategy.set_parameter_values({"lookback_days": 30, "target_positions": 2})
    print(f"\n修改后 lookback_days: {strategy.get_parameter_value('lookback_days')}")

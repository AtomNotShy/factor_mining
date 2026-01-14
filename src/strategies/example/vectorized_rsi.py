"""
向量化均衡策略示例
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.core.types import Signal, OrderIntent, MarketData, PortfolioState, RiskState, ActionType, OrderSide, OrderType
from src.core.context import RunContext

class VectorizedRSISignal(Strategy):
    """
    基于 RSI 的向量化策略示例
    """
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        # 使用 pandas-ta 计算 RSI (向量化)
        dataframe['rsi'] = self.ta.rsi(dataframe['close'], length=self.config.params.get('rsi_period', 14))
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        # RSI < 30 进入多头 (向量化)
        dataframe['enter_long'] = np.where(dataframe['rsi'] < 30, 1, 0)
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        # RSI > 70 退出多头 (向量化)
        dataframe['exit_long'] = (dataframe['rsi'] > self.config.params.get('rsi_upper', 70)).astype(int)
        return dataframe

    def hyperopt_space(self) -> Dict[str, Any]:
        return {
            "rsi_period": ("int", 7, 30),
            "rsi_lower": ("int", 20, 40),
            "rsi_upper": ("int", 60, 80)
        }

    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        # 虽然引擎现在会优先使用向量化信号，但为了兼容性，我们保留这个方法
        # 如果是纯向量化策略，可以直接返回空列表
        return []

    def size_positions(self, signals: List[Signal], portfolio: PortfolioState, risk: RiskState, ctx: RunContext) -> List[OrderIntent]:
        # 简单的调仓逻辑
        intents = []
        for signal in signals:
            if signal.action == ActionType.LONG:
                intents.append(OrderIntent(
                    symbol=signal.symbol,
                    qty=100, # 简化处理：固定买入100股
                    side=OrderSide.BUY,
                    order_type=OrderType.MKT,
                    ts_utc=signal.ts_utc,
                    strategy_id=self.strategy_id
                ))
            elif signal.action == ActionType.FLAT:
                current_pos = portfolio.positions.get(signal.symbol, 0)
                if current_pos > 0:
                    intents.append(OrderIntent(
                        symbol=signal.symbol,
                        qty=current_pos,
                        side=OrderSide.SELL,
                        order_type=OrderType.MKT,
                        ts_utc=signal.ts_utc,
                        strategy_id=self.strategy_id
                    ))
        return intents

"""
多时间框架动量策略示例
- 1小时时间框架作为主时间框架
- 4小时时间框架用于判断大趋势
- 15分钟时间框架用于寻找入场点

使用@informative装饰器演示多时间框架策略开发
"""

from typing import List
import pandas_ta as ta
import pandas as pd
import numpy as np

from src.strategies.base.strategy import Strategy, StrategyConfig
from src.strategies.base.informative import informative, merge_informative_pair
from src.core.types import Signal, OrderIntent, MarketData, PortfolioState, RiskState, ActionType, OrderSide, OrderType
from src.core.context import RunContext

class MultiTfMomentumStrategy(Strategy):
    """多时间框架动量策略"""
    
    def __init__(self):
        super().__init__(StrategyConfig(
            strategy_id="multi_tf_momentum",
            timeframe="1h",  # 主时间框架
            params={
                "symbols": ["SPY", "QQQ", "TLT"],
                "lookback": 20,
            }
        ))
        
        # 声明需要的informative时间框架
        self.informative_timeframes = ['4h', '15m']
    
    @informative('4h')
    def populate_indicators_4h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """4小时时间框架：判断大趋势"""
        dataframe['ema_20'] = ta.ema(dataframe['close'], length=20)
        dataframe['ema_50'] = ta.ema(dataframe['close'], length=50)
        dataframe['trend'] = np.where(
            dataframe['ema_20'] > dataframe['ema_50'], 1, -1
        )
        return dataframe
    
    @informative('15m')
    def populate_indicators_15m(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """15分钟时间框架：寻找入场信号"""
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        dataframe['bb_upper'] = ta.bbands(dataframe['close'], length=20, std=2)['BBU_20_2.0']
        dataframe['bb_lower'] = ta.bbands(dataframe['close'], length=20, std=2)['BBL_20_2.0']
        return dataframe
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """主时间框架（1h）指标计算
        
        可以通过以下方式访问informative数据：
        - self.get_informative_pair('4h', 'trend') -> 4小时趋势
        - self.get_informative_pair('15m', 'rsi') -> 15分钟RSI
        - self.get_informative_pair('15m', 'bb_upper') -> 15分钟布林带上轨
        - self.get_informative_pair('15m', 'bb_lower') -> 15分钟布林带下轨
        """
        # 1h指标
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], 
                                   dataframe['close'], length=14)
        
        # 获取4小时趋势（已通过merge_informative_pair自动ffill到每根1h bar）
        dataframe['trend_4h'] = self.get_informative_pair('4h', 'trend')
        
        # 获取15分钟数据（已自动对齐到1h）
        dataframe['rsi_15m'] = self.get_informative_pair('15m', 'rsi')
        dataframe['bb_upper_15m'] = self.get_informative_pair('15m', 'bb_upper')
        dataframe['bb_lower_15m'] = self.get_informative_pair('15m', 'bb_lower')
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """进场信号：多时间框架过滤
        
        条件：
        1. 4h趋势向上
        2. 15分钟RSI超卖（< 30）
        3. 15分钟价格接近布林带下轨
        """
        # 条件1: 4h趋势向上
        condition_trend = dataframe['trend_4h'] > 0
        
        # 条件2: 15分钟RSI超卖
        condition_rsi = dataframe['rsi_15m'] < 30
        
        # 条件3: 15分钟价格接近布林带下轨
        condition_bb = dataframe['close'] <= dataframe['bb_lower_15m'] * 1.02
        
        dataframe['enter_long'] = np.where(
            condition_trend & condition_rsi & condition_bb,
            1, 0
        )
        
        return dataframe
    
    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        """使用向量化结果生成信号"""
        return []
    
    def size_positions(self, signals: List[Signal], portfolio: PortfolioState, 
                     risk: RiskState, ctx: RunContext) -> List[OrderIntent]:
        """仓位管理"""
        orders = []
        
        # 等权分配示例
        if signals:
            weight = 1.0 / len(signals)
            for signal in signals:
                target_value = portfolio.equity * weight
                
                if signal.action == ActionType.LONG:
                    qty = target_value / md.latest_price(signal.symbol)
                    orders.append(OrderIntent(
                        ts_utc=signal.ts_utc,
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        qty=qty,
                        order_type=OrderType.LMT,
                        limit_price=md.latest_price(signal.symbol),
                        strategy_id=self.strategy_id,
                    ))
        
        return orders

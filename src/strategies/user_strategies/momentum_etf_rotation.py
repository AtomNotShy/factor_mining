"""
Momentum ETF Rotation Strategy (Vectorized)
============================================

A clean, vectorized ETF rotation strategy based on 60-day risk-adjusted momentum.

Logic:
1. At EACH month's end, rank ETFs by RAMOM (risk-adjusted momentum)
2. Hold only the top 1 ETF with positive momentum
3. Otherwise hold TLT (20+ year Treasury bonds)

Vectorized implementation for proper backtesting.

Author: Factor Mining System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.utils.logger import get_logger

logger = get_logger("strategy.momentum_etf_rotation")


class MomentumETFRotationStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """
    动量 ETF 轮动策略 (向量化版本)
    
    核心逻辑:
    - 每月末计算所有 ETF 的风险调整动量
    - 持有动量最高的 1 只 ETF
    - 动量为负则持有 TLT (国债)
    """
    
    strategy_name = "Momentum ETF Rotation"
    strategy_id = "momentum_etf_rotation"
    timeframe = "1d"
    startup_candle_count = 200
    
    # 策略参数
    minimal_roi = {0: float('inf')}
    stoploss = -0.15
    trailing_stop = False
    
    # ETF 池
    etf_pool = [
        'SPY', 'QQQ', 'VTV', 'SCHD', 'XLV', 'XLU', 'TLT', 'AGG'
    ]
    
    def __init__(
        self,
        momentum_lookback: int = 60,
        vol_window: int = 20,
    ):
        super().__init__()
        
        self.momentum_lookback = momentum_lookback
        self.vol_window = vol_window
        
        self.logger = get_logger(f"strategy.{self.strategy_id}")
    
    def populate_indicators(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict = None
    ) -> pd.DataFrame:
        """计算技术指标"""
        close = dataframe['close']
        
        # 60日动量
        dataframe['momentum_60'] = close / close.shift(self.momentum_lookback) - 1
        
        # 20日波动率
        returns = close.pct_change()
        dataframe['volatility_20'] = returns.rolling(self.vol_window).std() * np.sqrt(252)
        
        # 风险调整动量
        dataframe['ramom'] = dataframe['momentum_60'] / dataframe['volatility_20']
        dataframe['ramom'] = dataframe['ramom'].fillna(0)
        
        return dataframe
    
    def populate_entry_trend(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict = None
    ) -> pd.DataFrame:
        """生成进场信号 - 每月末动量正向则进场"""
        dataframe['enter_long'] = 0
        dataframe['enter_tag'] = ""
        
        # 标记月末
        dates = dataframe.index
        is_month_end = pd.Series(False, index=dates)
        
        for i in range(len(dates) - 1):
            if dates[i].month != dates[i + 1].month:
                is_month_end.iloc[i] = True
        is_month_end.iloc[-1] = True  # 最后一天也视为调仓日
        
        # 月末且动量正向则进场
        entry_condition = is_month_end & (dataframe['ramom'] > 0)
        dataframe.loc[entry_condition, 'enter_long'] = 1
        dataframe.loc[entry_condition, 'enter_tag'] = "momentum_rotation"
        
        return dataframe
    
    def populate_exit_trend(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict = None
    ) -> pd.DataFrame:
        """生成离场信号 - 每月末动量负向则离场"""
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ""
        
        # 标记月末
        dates = dataframe.index
        is_month_end = pd.Series(False, index=dates)
        
        for i in range(len(dates) - 1):
            if dates[i].month != dates[i + 1].month:
                is_month_end.iloc[i] = True
        is_month_end.iloc[-1] = True
        
        # 月末且动量负向则离场
        exit_condition = is_month_end & (dataframe['ramom'] <= 0)
        dataframe.loc[exit_condition, 'exit_long'] = 1
        dataframe.loc[exit_condition, 'exit_tag'] = "momentum_weak"
        
        return dataframe
    
    async def bot_start(self, **kwargs) -> None:
        self.logger.info(f"🚀 启动动量ETF轮动策略")
        self.logger.info(f"   ETF池: {self.etf_pool}")
    
    async def botShutdown(self, **kwargs) -> None:
        self.logger.info(f"🛑 动量ETF轮动策略已停止")

"""
Simplified Robust ETF Rotation Strategy
========================================
A simple, robust momentum-based rotation strategy.

Core Logic:
- Monthly rebalancing on the last trading day of each month
- Rank ETFs by 60-day risk-adjusted momentum
- Hold only the top 1 ETF with positive momentum
- Otherwise hold TLT (bonds) or SHY (cash)

Target: 
- Annual volatility: ~10-12%
- Max drawdown: <15%
- Sharpe ratio: >0.6

Author: Factor Mining System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.utils.logger import get_logger

logger = get_logger("strategy.simple_etf_rotation")


@dataclass
class StrategyConfig:
    """策略配置"""
    momentum_lookback: int = 60      # 动量回溯天数
    vol_window: int = 20             # 波动率计算窗口
    holding_period: int = 21         # 持仓天数 (月度)
    score_threshold: float = 0.0     # 动量阈值
    transaction_cost: float = 0.0005  # 5bps 交易成本


class SimpleETFRotationStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """
    简化版 ETF 轮动策略
    
    特点:
    - 每月末调仓
    - 按风险调整动量排名
    - 只持有最优的 1 只 ETF
    - 负动量则持有债券/现金
    """
    
    strategy_name = "Simple ETF Rotation"
    strategy_id = "simple_etf_rotation"
    timeframe = "1d"
    startup_candle_count = 200
    
    # 策略参数
    minimal_roi = {0: float('inf')}
    stoploss = -0.10
    trailing_stop = False
    
    # ETF 池
    etf_pool = [
        'SPY', 'QQQ',       # 核心
        'VTV', 'SCHD',      # 价值
        'XLV', 'XLU',       # 防御
        'TLT', 'AGG',       # 债券
    ]
    
    def __init__(
        self,
        momentum_lookback: int = 60,
        vol_window: int = 20,
        score_threshold: float = 0.0,
    ):
        super().__init__()
        
        self.config = StrategyConfig(
            momentum_lookback=momentum_lookback,
            vol_window=vol_window,
            score_threshold=score_threshold,
        )
        
        self.logger = get_logger(f"strategy.{self.strategy_id}")
        
        # 缓存每日的动量排名
        self._momentum_rank_cache = {}
    
    def calculate_momentum_score(
        self,
        close: pd.Series,
        lookback: int = 60,
        vol_window: int = 20,
    ) -> Tuple[float, float, float]:
        """
        计算单只 ETF 的动量分数
        
        Returns:
            (momentum, volatility, risk_adjusted_score)
        """
        returns = close.pct_change()
        
        # 累计收益率
        cum_ret = close.iloc[-1] / close.iloc[-lookback] - 1 if len(close) >= lookback else 0
        
        # 年化波动率
        vol = returns.iloc[-vol_window:].std() * np.sqrt(252) if len(returns) >= vol_window else 0.1
        
        # 风险调整动量
        score = cum_ret / vol if vol > 0 else 0
        
        return cum_ret, vol, score
    
    def populate_indicators(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict = None
    ) -> pd.DataFrame:
        """计算技术指标"""
        # 简化：只用简单动量
        close = dataframe['close']
        
        # 60日动量
        dataframe['momentum_60'] = close / close.shift(60) - 1
        
        # 20日波动率
        returns = close.pct_change()
        dataframe['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        # 风险调整动量
        dataframe['ramom'] = dataframe['momentum_60'] / dataframe['volatility_20']
        
        # 填充NaN
        dataframe['ramom'] = dataframe['ramom'].fillna(0)
        
        return dataframe
    
    def populate_entry_trend(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict = None
    ) -> pd.DataFrame:
        """
        生成进场信号
        
        注意：此策略的实际选股逻辑在策略层面统一处理，
        这里只基于当前 ETF 的动量打分生成信号
        """
        dataframe['enter_long'] = 0
        dataframe['enter_tag'] = ""
        
        # 获取最近一次有效的 RAMOM 值
        ramom = dataframe['ramom'].iloc[-1] if len(dataframe) > 0 else 0
        
        # 动量正向则标记为潜在候选
        if ramom > self.config.score_threshold:
            dataframe.iloc[-1, dataframe.columns.get_loc('enter_long')] = 1
            dataframe.iloc[-1, dataframe.columns.get_loc('enter_tag')] = "momentum_positive"
        
        return dataframe
    
    def populate_exit_trend(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict = None
    ) -> pd.DataFrame:
        """生成离场信号"""
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ""
        
        ramom = dataframe['ramom'].iloc[-1] if len(dataframe) > 0 else 0
        
        # 动量变负或过低则离场
        if ramom < self.config.score_threshold:
            dataframe.iloc[-1, dataframe.columns.get_loc('exit_long')] = 1
            dataframe.iloc[-1, dataframe.columns.get_loc('exit_tag')] = "momentum_weak"
        
        return dataframe
    
    async def bot_start(self, **kwargs) -> None:
        """策略启动"""
        self.logger.info(f"🚀 启动简单ETF轮动策略")
        self.logger.info(f"   ETF池: {self.etf_pool}")
        self.logger.info(f"   动量回溯: {self.config.momentum_lookback}天")
    
    async def botShutdown(self, **kwargs) -> None:
        """策略停止"""
        self.logger.info(f"🛑 简单ETF轮动策略已停止")

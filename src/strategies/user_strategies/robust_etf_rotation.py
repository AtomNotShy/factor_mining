"""
Robust ETF Rotation Strategy
============================
A risk-adjusted momentum strategy with defensive rotation for US equity ETFs.

Signals:
1. RAMOM: Risk-Adjusted Momentum (risk-adjusted returns)
2. DTTF: Dual-Threshold Trend Filter (trend confirmation)
3. DRS: Defensive Rotation Score (volatility-adaptive)

Target: 
- Annual volatility: ~10%
- Max drawdown: <15%
- Sharpe ratio: >0.7

Author: Factor Mining System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.utils.logger import get_logger

logger = get_logger("strategy.robust_etf_rotation")


@dataclass
class StrategyConfig:
    """策略配置参数"""
    # 信号参数
    momentum_lookback: int = 60  # 动量回溯天数
    vol_window: int = 20  # 波动率计算窗口
    short_ma: int = 50  # 短期均线
    long_ma: int = 200  # 长期均线
    atr_window: int = 20  # ATR窗口
    atr_threshold: float = 0.02  # ATR阈值
    
    # 组合参数
    score_threshold: float = 0.0  # 信号得分阈值
    max_weight: float = 0.30  # 单标最大权重
    min_weight: float = 0.05  # 单标最小权重
    defensive_min: float = 0.10  # 防御类最小权重
    
    # 交易参数
    monthly_turnover_limit: float = 0.15  # 月度换手率限制
    transaction_cost: float = 0.0005  # 5bps 交易成本


class RobustETFRotationStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """
    稳健ETF轮动策略
    
    特点:
    - 风险调整后动量信号
    - 双均线趋势过滤
    - 波动率自适应轮动
    - 月度再平衡
    """
    
    strategy_name = "Robust ETF Rotation"
    strategy_id = "robust_etf_rotation"
    timeframe = "1d"
    startup_candle_count = 250  # 长期均线需要更多数据
    
    # 默认配置
    minimal_roi = {0: float('inf')}  # 不使用自动ROI
    stoploss = -0.10  # 10% 止损
    trailing_stop = False
    
    # ETF 扁平列表
    etf_pool = [
        'SPY', 'QQQ',      # 核心宽基
        'VTV', 'SCHD',     # 价值/红利
        'XLV', 'XLU',      # 防御性
        'TLT', 'AGG',      # 债券
        'SHY',             # 现金
    ]
    
    # 用于分类的字典（用于权重约束）
    etf_categories = {
        'core': ['SPY', 'QQQ'],
        'value': ['VTV', 'SCHD'],
        'defensive': ['XLV', 'XLU'],
        'bonds': ['TLT', 'AGG'],
        'cash': ['SHY'],
    }
    
    def __init__(
        self,
        momentum_lookback: int = 60,
        vol_window: int = 20,
        short_ma: int = 50,
        long_ma: int = 200,
        atr_window: int = 20,
        atr_threshold: float = 0.02,
        score_threshold: float = 0.0,
        max_weight: float = 0.30,
        min_weight: float = 0.05,
        defensive_min: float = 0.10,
        monthly_turnover_limit: float = 0.15,
        transaction_cost: float = 0.0005,
    ):
        super().__init__()
        
        self.config = StrategyConfig(
            momentum_lookback=momentum_lookback,
            vol_window=vol_window,
            short_ma=short_ma,
            long_ma=long_ma,
            atr_window=atr_window,
            atr_threshold=atr_threshold,
            score_threshold=score_threshold,
            max_weight=max_weight,
            min_weight=min_weight,
            defensive_min=defensive_min,
            monthly_turnover_limit=monthly_turnover_limit,
            transaction_cost=transaction_cost,
        )
        
        self.logger = get_logger(f"strategy.{self.strategy_id}")
    
    # =====================================================================
    # 信号计算方法
    # =====================================================================
    
    def calculate_ramom(
        self,
        close: pd.Series,
        lookback: int = 60,
        vol_window: int = 20,
    ) -> pd.Series:
        """
        计算风险调整后动量 (Risk-Adjusted Momentum)
        
        RAMOM = 累计收益率 / (波动率 * sqrt(N))
        
        Args:
            close: 价格序列
            lookback: 动量回溯期
            vol_window: 波动率计算窗口
            
        Returns:
            风险调整后动量分数
        """
        # 日收益率
        returns = close.pct_change()
        
        # 累计收益率 (使用 log 返回的累加)
        log_ret = np.log(1 + returns)
        cum_log_ret = log_ret.rolling(lookback).sum()
        cum_ret = np.exp(cum_log_ret) - 1
        
        # 年化波动率
        vol = returns.rolling(vol_window).std() * np.sqrt(252)
        
        # 风险调整动量
        ramom = cum_ret / vol.replace(0, np.nan)
        
        return ramom.fillna(0)
    
    def calculate_dttf(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        short_ma: int = 50,
        long_ma: int = 200,
        atr_window: int = 20,
        atr_threshold: float = 0.02,
    ) -> pd.Series:
        """
        计算双阈值趋势过滤 (Dual-Threshold Trend Filter)
        
        基于均线金叉/死叉和ATR确认趋势
        
        Args:
            close: 收盘价
            high: 最高价
            low: 最低价
            short_ma: 短期均线周期
            long_ma: 长期均线周期
            atr_window: ATR窗口
            atr_threshold: ATR阈值
            
        Returns:
            趋势分数 (-1 到 1)
        """
        # 均线
        sma_short = close.rolling(short_ma).mean()
        sma_long = close.rolling(long_ma).mean()
        
        # ATR 归一化
        atr = (high - low).rolling(atr_window).mean()
        atr_normalized = atr / close.replace(0, np.nan)
        
        # 均线差值
        ma_diff = sma_short - sma_long
        
        # 趋势分数
        trend = np.sign(ma_diff) * np.minimum(
            1.0, 
            np.abs(ma_diff) / (2 * atr_normalized.replace(0, np.nan))
        )
        
        return trend.fillna(0)
    
    def calculate_drs(
        self,
        momentum: pd.Series,
        rel_vol: pd.Series,
        vix: float = 20.0,
        sector_beta: pd.Series = None,
    ) -> pd.Series:
        """
        计算防御轮动分数 (Defensive Rotation Score)
        
        根据市场波动率动态调整动量和低波因子的权重
        
        Args:
            momentum: 动量分数
            rel_vol: 相对波动率
            vix: 当前VIX水平
            sector_beta: 行业贝塔（可选）
            
        Returns:
            防御轮动分数
        """
        # 动态权重 (VIX > 25 时增加低波权重)
        w_mom = 1 / (1 + np.exp(-0.5 * (vix - 25)))
        w_vol = 1 - w_mom
        
        # 归一化
        mom_z = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
        vol_z = -(rel_vol - rel_vol.mean()) / (rel_vol.std() + 1e-8)  # 低波反转
        
        return (w_mom * mom_z + w_vol * vol_z).fillna(0)
    
    # =====================================================================
    # 综合信号计算
    # =====================================================================
    
    def calculate_combined_signal(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        vix: float = 20.0,
    ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        计算综合信号分数
        
        Returns:
            综合分数和各分项信号
        """
        # 计算各信号
        ramom = self.calculate_ramom(
            close, 
            self.config.momentum_lookback, 
            self.config.vol_window
        )
        
        dttf = self.calculate_dttf(
            close, high, low,
            self.config.short_ma,
            self.config.long_ma,
            self.config.atr_window,
            self.config.atr_threshold,
        )
        
        # 相对波动率
        rel_vol = close.pct_change().rolling(self.config.vol_window).std()
        rel_vol = rel_vol / rel_vol.mean()
        
        drs = self.calculate_drs(ramom, rel_vol, vix)
        
        # 综合得分
        combined = 0.4 * ramom + 0.3 * dttf + 0.3 * drs
        
        return combined, {
            'ramom': ramom,
            'dttf': dttf,
            'drs': drs,
        }
    
    # =====================================================================
    # 组合构建
    # =====================================================================
    
    def calculate_portfolio_weights(
        self,
        signals: Dict[str, pd.Series],
        current_positions: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """
        计算目标组合权重
        
        Args:
            signals: 各ETF的信号分数
            current_positions: 当前持仓权重
            
        Returns:
            目标权重字典
        """
        if current_positions is None:
            current_positions = {}
        
        # 1. 筛选正分数的标的
        eligible = {
            symbol: score for symbol, score in signals.items()
            if score > self.config.score_threshold
        }
        
        if not eligible:
            # 无合格标的，全部现金
            return {'SHY': 1.0}
        
        # 2. 按波动率倒数加权
        # 使用最近20天收益率标准差作为波动率代理
        weights = {}
        for symbol in eligible:
            weights[symbol] = 1.0 / (eligible[symbol] + 1e-8)
        
        # 归一化
        total = sum(weights.values())
        for symbol in weights:
            weights[symbol] /= total
        
        # 3. 应用约束
        # 单标最大权重
        for symbol in weights:
            weights[symbol] = min(weights[symbol], self.config.max_weight)
        
        # 归一化
        total = sum(weights.values())
        for symbol in weights:
            weights[symbol] /= total
        
        # 4. 确保防御类最小权重
        defensive_weight = sum(
            weights.get(s, 0) for s in self.etf_pool['defensive']
            if s in weights
        )
        if defensive_weight < self.config.defensive_min:
            # 增加防御类权重
            deficit = self.config.defensive_min - defensive_weight
            for s in self.etf_pool['defensive']:
                if s in weights:
                    weights[s] += deficit / len(self.etf_pool['defensive'])
            # 重新归一化
            total = sum(weights.values())
            for symbol in weights:
                weights[symbol] /= total
        
        # 5. 换手率限制
        if current_positions:
            turnover = sum(
                abs(weights.get(s, 0) - current_positions.get(s, 0))
                for s in set(list(weights.keys()) + list(current_positions.keys()))
            ) / 2
            
            if turnover > self.config.monthly_turnover_limit:
                # 渐进调整
                scale = self.config.monthly_turnover_limit / turnover
                for symbol in weights:
                    weights[symbol] *= scale
                
                # 剩余仓位给现金
                allocated = sum(weights.values())
                weights['SHY'] = 1.0 - allocated
        
        return weights
    
    # =====================================================================
    # Freqtrade 协议方法
    # =====================================================================
    
    def populate_indicators(
        self, 
        dataframe: pd.DataFrame, 
        metadata: Dict = None
    ) -> pd.DataFrame:
        """计算技术指标"""
        symbol = metadata.get('symbol', 'unknown') if metadata else 'unknown'
        
        close = dataframe['close']
        high = dataframe['high']
        low = dataframe['low']
        
        # 计算信号
        combined, signals = self.calculate_combined_signal(close, high, low)
        
        # 添加到 dataframe
        dataframe['ramom'] = signals['ramom']
        dataframe['dttf'] = signals['dttf']
        dataframe['drs'] = signals['drs']
        dataframe['combined_score'] = combined
        
        self.logger.debug(f"{symbol}: RAMOM={signals['ramom'].iloc[-1]:.4f}, "
                         f"DTTF={signals['dttf'].iloc[-1]:.4f}, "
                         f"Combined={combined.iloc[-1]:.4f}")
        
        return dataframe
    
    def populate_entry_trend(
        self, 
        dataframe: pd.DataFrame, 
        metadata: Dict = None
    ) -> pd.DataFrame:
        """生成进场信号"""
        dataframe['enter_long'] = 0
        dataframe['enter_tag'] = ""
        
        # 每月末检查是否需要调仓
        # 这里简化：只要综合分数 > 0 就生成信号
        entry_condition = dataframe['combined_score'] > self.config.score_threshold
        
        dataframe.loc[entry_condition, 'enter_long'] = 1
        dataframe.loc[entry_condition, 'enter_tag'] = "robust_rotation_signal"
        
        return dataframe
    
    def populate_exit_trend(
        self, 
        dataframe: pd.DataFrame, 
        metadata: Dict = None
    ) -> pd.DataFrame:
        """生成离场信号"""
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ""
        
        # 综合分数 < 0 时离场
        exit_condition = dataframe['combined_score'] < self.config.score_threshold
        
        dataframe.loc[exit_condition, 'exit_long'] = 1
        dataframe.loc[exit_condition, 'exit_tag'] = "rotation_exit"
        
        return dataframe
    
    # =====================================================================
    # 生命周期回调
    # =====================================================================
    
    async def bot_start(self, **kwargs) -> None:
        """策略启动"""
        self.logger.info(f"🚀 启动稳健ETF轮动策略")
        self.logger.info(f"   ETF池: {self.etf_pool}")
        self.logger.info(f"   动量回溯: {self.config.momentum_lookback}天")
        self.logger.info(f"   趋势滤波: {self.config.short_ma}/{self.config.long_ma} MA")
    
    async def botShutdown(self, **kwargs) -> None:
        """策略停止"""
        self.logger.info(f"🛑 稳健ETF轮动策略已停止")

"""
策略模板系统

提供常用策略模式的可复用模板，用户只需继承并配置参数。
"""

from abc import ABC
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base.indicators import (
    sma, ema, wma, dema, tema, rsi, macd, bollinger_bands, stochastic,
)
from src.utils.logger import get_logger

logger = get_logger("strategy.templates")


# ============ 策略配置类（内联定义避免循环导入） ============

class StrategyConfig:
    """策略配置"""
    def __init__(
        self,
        strategy_name: str = "",
        strategy_id: str = "",
        timeframe: str = "1d",
        stoploss: float = -0.10,
        trailing_stop: bool = False,
        trailing_stop_positive: float = 0.0,
        trailing_stop_positive_offset: float = 0.0,
        use_exit_signal: bool = True,
        **kwargs
    ):
        self.strategy_name = strategy_name
        self.strategy_id = strategy_id
        self.timeframe = timeframe
        self.stoploss = stoploss
        self.trailing_stop = trailing_stop
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.use_exit_signal = use_exit_signal
        for k, v in kwargs.items():
            setattr(self, k, v)


# ============ 趋势跟踪模板 ============

class TrendFollowingStrategy(ABC):
    """
    趋势跟踪策略模板
    
    使用双均线系统，当短期均线上穿长期均线时买入。
    """
    
    config: StrategyConfig = StrategyConfig(strategy_name="Trend Following Strategy", stoploss=-0.05)
    fast_period: int = 10
    slow_period: int = 20
    ma_type: str = "sma"
    entry_threshold: float = 0.0
    
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        ma_map = {'sma': sma, 'ema': ema, 'wma': wma, 'dema': dema, 'tema': tema}
        ma_func = ma_map.get(self.ma_type, sma)
        
        dataframe['fast_ma'] = ma_func(dataframe['close'], self.fast_period)
        dataframe['slow_ma'] = ma_func(dataframe['close'], self.slow_period)
        dataframe['ma_diff'] = (dataframe['fast_ma'] - dataframe['slow_ma']) / dataframe['slow_ma']
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        if not dataframe.empty:
            prev_diff = dataframe['ma_diff'].shift(1)
            curr_diff = dataframe['ma_diff']
            dataframe.loc[(prev_diff <= self.entry_threshold) & (curr_diff > self.entry_threshold), 'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        return dataframe


# ============ RSI 策略模板 ============

class RSIStrategy(ABC):
    """RSI 策略模板"""
    
    config: StrategyConfig = StrategyConfig(strategy_name="RSI Strategy", stoploss=-0.05)
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_exit: float = 50.0
    use_exit_signal: bool = True
    
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['rsi'] = rsi(dataframe['close'], self.rsi_period)
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        if not dataframe.empty:
            dataframe.loc[dataframe['rsi'] < self.rsi_oversold, 'enter_long'] = 1
            dataframe.loc[dataframe['rsi'] > self.rsi_overbought, 'enter_short'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        if self.use_exit_signal and not dataframe.empty:
            dataframe.loc[(dataframe['rsi'] > self.rsi_exit) & (dataframe['rsi'].shift(1) <= self.rsi_exit), 'exit_long'] = 1
            dataframe.loc[(dataframe['rsi'] < self.rsi_exit) & (dataframe['rsi'].shift(1) >= self.rsi_exit), 'exit_short'] = 1
        return dataframe


# ============ MACD 策略模板 ============

class MACDStrategy(ABC):
    """MACD 策略模板"""
    
    config: StrategyConfig = StrategyConfig(strategy_name="MACD Strategy", stoploss=-0.05)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_strict: bool = False
    
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        result = macd(dataframe['close'], self.macd_fast, self.macd_slow, self.macd_signal)
        dataframe['macd'] = result.macd
        dataframe['macd_signal'] = result.signal
        dataframe['macd_hist'] = result.hist
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        if not dataframe.empty:
            prev_hist = dataframe['macd_hist'].shift(1)
            curr_hist = dataframe['macd_hist']
            dataframe.loc[(prev_hist <= 0) & (curr_hist > 0), 'enter_long'] = 1
            dataframe.loc[(prev_hist >= 0) & (curr_hist < 0), 'enter_short'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe


# ============ 布林带策略模板 ============

class BollingerBandStrategy(ABC):
    """布林带策略模板"""
    
    config: StrategyConfig = StrategyConfig(strategy_name="Bollinger Band Strategy", stoploss=-0.08)
    bb_period: int = 20
    bb_std: float = 2.0
    use_exit_signal: bool = True
    
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        bb = bollinger_bands(dataframe['close'], self.bb_period, self.bb_std)
        dataframe['bb_upper'] = bb.upper
        dataframe['bb_middle'] = bb.middle
        dataframe['bb_lower'] = bb.lower
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        if not dataframe.empty:
            dataframe.loc[(dataframe['close'] > dataframe['bb_upper']), 'enter_long'] = 1
            dataframe.loc[(dataframe['close'] < dataframe['bb_lower']), 'enter_short'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe


# ============ 随机指标策略模板 ============

class StochasticStrategy(ABC):
    """随机指标策略模板"""
    
    config: StrategyConfig = StrategyConfig(strategy_name="Stochastic Strategy", stoploss=-0.05)
    k_period: int = 14
    d_period: int = 3
    overbought: float = 80.0
    oversold: float = 20.0
    use_exit_signal: bool = True
    
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        stoch = stochastic(dataframe['high'], dataframe['low'], dataframe['close'], self.k_period, self.d_period)
        dataframe['stoch_k'] = stoch.k
        dataframe['stoch_d'] = stoch.d
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        if not dataframe.empty:
            dataframe.loc[(dataframe['stoch_k'] < self.oversold) & (dataframe['stoch_d'] < self.oversold), 'enter_long'] = 1
            dataframe.loc[(dataframe['stoch_k'] > self.overbought) & (dataframe['stoch_d'] > self.overbought), 'enter_short'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe


# ============ 动量策略模板 ============

class MomentumStrategy(ABC):
    """动量策略模板"""
    
    config: StrategyConfig = StrategyConfig(strategy_name="Momentum Strategy", stoploss=-0.10)
    lookback_period: int = 20
    momentum_type: str = "returns"
    
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['returns'] = dataframe['close'].pct_change(self.lookback_period)
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        return dataframe


# ============ 均值回归策略模板 ============

class MeanReversionStrategy(ABC):
    """均值回归策略模板"""
    
    config: StrategyConfig = StrategyConfig(strategy_name="Mean Reversion Strategy", stoploss=-0.08)
    band_std: float = 2.0
    lookback: int = 20
    
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['middle'] = sma(dataframe['close'], self.lookback)
        dataframe['std'] = dataframe['close'].rolling(self.lookback).std()
        dataframe['upper'] = dataframe['middle'] + self.band_std * dataframe['std']
        dataframe['lower'] = dataframe['middle'] - self.band_std * dataframe['std']
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        if not dataframe.empty:
            dataframe.loc[dataframe['close'] < dataframe['lower'], 'enter_long'] = 1
            dataframe.loc[dataframe['close'] > dataframe['upper'], 'enter_short'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe


# ============ 多时间框架策略模板 ============

class MultiTimeframeStrategy(ABC):
    """多时间框架策略模板"""
    
    config: StrategyConfig = StrategyConfig(strategy_name="Multi-Timeframe Strategy", stoploss=-0.05)
    entry_timeframe: str = "15m"
    exit_timeframe: str = "1h"
    
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    def get_informative_pairs(self) -> List[tuple]:
        return [(self.config.strategy_id, self.entry_timeframe), (self.config.strategy_id, self.exit_timeframe)]
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        timeframe = metadata.get('timeframe', self.entry_timeframe)
        if timeframe == self.exit_timeframe:
            dataframe['trend_ma'] = sma(dataframe['close'], 20)
        else:
            dataframe['entry_rsi'] = rsi(dataframe['close'], 14)
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe


# ============ 策略模板工厂 ============

class StrategyTemplateFactory:
    """策略模板工厂"""
    
    _templates: Dict[str, type] = {
        'trend_following': TrendFollowingStrategy,
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'rsi': RSIStrategy,
        'macd': MACDStrategy,
        'bollinger': BollingerBandStrategy,
        'stochastic': StochasticStrategy,
        'multi_timeframe': MultiTimeframeStrategy,
    }
    
    @classmethod
    def get_template(cls, name: str) -> Optional[type]:
        return cls._templates.get(name.lower())
    
    @classmethod
    def list_templates(cls) -> List[str]:
        return list(cls._templates.keys())
    
    @classmethod
    def create(cls, name: str, strategy_id: Optional[str] = None, **kwargs) -> Any:
        """
        创建策略实例
        
        Args:
            name: 模板名称（如 'rsi', 'trend_following'）
            strategy_id: 策略ID（可选）
            **kwargs: 其他策略参数
            
        Returns:
            策略实例
        """
        template_cls = cls.get_template(name)
        if template_cls is None:
            raise ValueError(f"Unknown template: {name}. Available: {cls.list_templates()}")
        
        # 分离配置参数和策略参数
        config_kwargs = {}
        strategy_kwargs = {}
        
        config_params = ['strategy_name', 'timeframe', 'stoploss', 'trailing_stop', 
                         'trailing_stop_positive', 'trailing_stop_positive_offset', 'use_exit_signal']
        
        for k, v in kwargs.items():
            if k in config_params:
                config_kwargs[k] = v
            else:
                strategy_kwargs[k] = v
        
        # 创建配置
        config = StrategyConfig(
            strategy_id=strategy_id or f"{name}_strategy",
            **config_kwargs
        )
        
        # 创建策略类
        class DynamicStrategy(template_cls):
            pass
        
        # 应用策略参数
        for k, v in strategy_kwargs.items():
            setattr(DynamicStrategy, k, v)
        
        DynamicStrategy.config = config
        if strategy_id:
            DynamicStrategy.__name__ = f"{strategy_id.title().replace('_', '')}Strategy"
        
        return DynamicStrategy()


# ============ 导出 ============

__all__ = [
    'TrendFollowingStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'BollingerBandStrategy',
    'StochasticStrategy',
    'MultiTimeframeStrategy',
    'StrategyTemplateFactory',
    'StrategyConfig',
]

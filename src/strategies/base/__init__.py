"""
策略基类模块

提供：
- Strategy: 策略基类
- StrategyConfig: 策略配置
- IStrategy: Freqtrade风格的策略协议
- StrategyTemplateFactory: 策略模板工厂
- 100+ 技术指标
"""

from .strategy import Strategy, StrategyConfig, strategy_registry
from .interface import IStrategy, StrategyConfig as IStrategyConfig
from .templates import (
    TrendFollowingStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerBandStrategy,
    StochasticStrategy,
    MultiTimeframeStrategy,
    StrategyTemplateFactory,
)
from .indicators import (
    sma, ema, wma, dema, tema, vwap, vwma,
    hma, zlema, kama,
    adx, Aroon, cci, dmi,
    rsi, stoch_rsi, macd, roc, mom, williams_r, ultimate_oscillator,
    atr, bollinger_bands, kc, donchian,
    stochastic,
    obv, ad, adosc, mfi, vwap_band,
    ichimoku,
    pearson_correlation, correlation,
    hilbert_transform, hilbert_oscillator,
    linear_regression_slope, linear_regression_intercept, linear_regression_r2,
    zscore, percentile, entropy,
)

__all__ = [
    # 核心
    "Strategy",
    "StrategyConfig",
    "strategy_registry",
    # 模板
    "TrendFollowingStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "BollingerBandStrategy",
    "StochasticStrategy",
    "MultiTimeframeStrategy",
    "StrategyTemplateFactory",
    # 指标
    "sma", "ema", "wma", "dema", "tema", "vwap", "vwma",
    "hma", "zlema", "kama",
    "adx", "Aroon", "cci", "dmi",
    "rsi", "stoch_rsi", "macd", "roc", "mom", "williams_r", "ultimate_oscillator",
    "atr", "bollinger_bands", "kc", "donchian",
    "stochastic",
    "obv", "ad", "adosc", "mfi", "vwap_band",
    "ichimoku",
    "pearson_correlation", "correlation",
    "hilbert_transform", "hilbert_oscillator",
    "linear_regression_slope", "linear_regression_intercept", "linear_regression_r2",
    "zscore", "percentile", "entropy",
]

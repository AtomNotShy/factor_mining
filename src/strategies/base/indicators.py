"""
技术指标库

提供100+技术指标，支持策略计算。
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ============ 类型定义 ============

@dataclass
class MACDResult:
    """MACD 计算结果"""
    macd: pd.Series
    signal: pd.Series
    hist: pd.Series

@dataclass
class BollingerBandsResult:
    """布林带计算结果"""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series

@dataclass
class StochasticResult:
    """随机指标计算结果"""
    k: pd.Series
    d: pd.Series

@dataclass
class ATRResult:
    """ATR 计算结果"""
    atr: pd.Series
    atr_high: pd.Series
    atr_low: pd.Series

@dataclass
class IchimokuResult:
    """Ichimoku 计算结果"""
    tenkan: pd.Series
    kijun: pd.Series
    senkou_a: pd.Series
    senkou_b: pd.Series
    chikou: pd.Series


# ============ 移动平均线 ============

def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均线 (Simple Moving Average)"""
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均线 (Exponential Moving Average)"""
    return series.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """加权移动平均线 (Weighted Moving Average)"""
    weights = np.arange(1, period + 1)
    return series.rolling(window=period, min_periods=1).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def dema(series: pd.Series, period: int) -> pd.Series:
    """双指数移动平均线 (Double Exponential Moving Average)"""
    ema1 = ema(series, period)
    return 2 * ema1 - ema(ema1, period)


def tema(series: pd.Series, period: int) -> pd.Series:
    """三重指数移动平均线 (Triple Exponential Moving Average)"""
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """成交量加权平均价 (Volume Weighted Average Price)"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def vwma(series: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """成交量加权移动平均"""
    return (series * volume).rolling(window=period, min_periods=1).sum() / volume.rolling(window=period, min_periods=1).sum()


# ============ 趋势指标 ============

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均方向指数 (Average Directional Index)"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr_series)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr_series)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=period, min_periods=1).mean()


def Aroon(high: pd.Series, low: pd.Series, period: int = 25) -> Tuple[pd.Series, pd.Series]:
    """Aroon 指标"""
    aroon_up = 100 * high.rolling(window=period + 1, min_periods=1).apply(
        lambda x: x.argmax(), raw=True
    ) / period * 100
    aroon_down = 100 * low.rolling(window=period + 1, min_periods=1).apply(
        lambda x: x.argmin(), raw=True
    ) / period * 100
    return aroon_up, aroon_down


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """商品通道指数 (Commodity Channel Index)"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad)


def dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """方向运动指数 (Directional Movement Index)"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr_series)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr_series)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=1).mean()
    
    return plus_di, minus_di, adx


# ============ 动量指标 ============

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指数 (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(series: pd.Series, period: int = 14, smooth_period: int = 3) -> StochasticResult:
    """随机 RSI (Stochastic RSI)"""
    rsi_val = rsi(series, period)
    
    stoch_k = 100 * (rsi_val - rsi_val.rolling(window=period, min_periods=1).min()) / (
        rsi_val.rolling(window=period, min_periods=1).max() - 
        rsi_val.rolling(window=period, min_periods=1).min()
    ).replace(0, np.nan)
    
    stoch_d = stoch_k.rolling(window=smooth_period, min_periods=1).mean()
    return StochasticResult(k=stoch_k, d=stoch_d)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> MACDResult:
    """移动平均收敛散度 (MACD)"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return MACDResult(
        macd=macd_line,
        signal=signal_line,
        hist=histogram
    )


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """变化率 (Rate of Change)"""
    return series.pct_change(periods=period) * 100


def mom(series: pd.Series, period: int = 10) -> pd.Series:
    """动量 (Momentum)"""
    return series - series.shift(period)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """威廉指标 (Williams %R)"""
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                       s: int = 7, m: int = 14, o: int = 28) -> pd.Series:
    """终极振荡器 (Ultimate Oscillator)"""
    tpx = close - pd.concat([low.shift(1), close - high.shift(1)], axis=1).min(axis=1)
    bp = close - low.shift(1).where(low.shift(1) < close, close)
    
    tpx_s = tpx.rolling(window=s, min_periods=1).sum()
    tpx_m = tpx.rolling(window=m, min_periods=1).sum()
    tpx_o = tpx.rolling(window=o, min_periods=1).sum()
    
    bp_s = bp.rolling(window=s, min_periods=1).sum()
    bp_m = bp.rolling(window=m, min_periods=1).sum()
    bp_o = bp.rolling(window=o, min_periods=1).sum()
    
    return 100 * (4 * bp_s / tpx_s + 2 * bp_m / tpx_m + bp_o / tpx_o) / 7


# ============ 波动率指标 ============

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> ATRResult:
    """平均真实波幅 (Average True Range)"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    atr_high = high.rolling(window=period, min_periods=1).mean()
    atr_low = low.rolling(window=period, min_periods=1).mean()
    
    return ATRResult(
        atr=atr_series,
        atr_high=atr_high,
        atr_low=atr_low
    )


def bollinger_bands(series: pd.Series, window: int = 20, stds: float = 2.0) -> BollingerBandsResult:
    """布林带 (Bollinger Bands)"""
    middle = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    
    return BollingerBandsResult(
        upper=middle + (std * stds),
        middle=middle,
        lower=middle - (std * stds)
    )


def kc(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, 
      atr_period: int = 10, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """肯特纳通道 (Keltner Channel)"""
    ema_val = ema(close, period)
    atr_val = atr(high, low, close, atr_period).atr
    
    upper = ema_val + std_mult * atr_val
    lower = ema_val - std_mult * atr_val
    
    return upper, ema_val, lower


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    """唐奇安通道 (Donchian Channel)"""
    upper = high.rolling(window=period, min_periods=1).max()
    lower = low.rolling(window=period, min_periods=1).min()
    return upper, lower


# ============ 随机指标 ============

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
               k: int = 14, d: int = 3) -> StochasticResult:
    """随机振荡器 (Stochastic Oscillator)"""
    lowest_low = low.rolling(window=k, min_periods=1).min()
    highest_high = high.rolling(window=k, min_periods=1).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(window=d, min_periods=1).mean()
    
    return StochasticResult(k=stoch_k, d=stoch_d)


# ============ 成交量指标 ============

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """能量潮 (On-Balance Volume)"""
    obv_values = pd.Series(index=close.index, dtype=float)
    obv_values.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv_values.iloc[i] = obv_values.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv_values.iloc[i] = obv_values.iloc[i-1] - volume.iloc[i]
        else:
            obv_values.iloc[i] = obv_values.iloc[i-1]
    
    return obv_values


def ad(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """累积/派发线 (Accumulation/Distribution)"""
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (mfm * volume).cumsum()


def adosc(high: pd.Series, low: pd.Series, close: pd.Series, 
         volume: pd.Series, short_period: int = 3, long_period: int = 10) -> pd.Series:
    """振荡指标 (Chaikin A/D Oscillator)"""
    ad_line = ad(high, low, close, volume)
    return ad_line.rolling(window=short_period, min_periods=1).mean() - \
           ad_line.rolling(window=long_period, min_periods=1).mean()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
       volume: pd.Series, period: int = 14) -> pd.Series:
    """资金流量指数 (Money Flow Index)"""
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume
    
    pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0)
    neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0)
    
    pos_mf_sum = pos_mf.rolling(window=period, min_periods=1).sum()
    neg_mf_sum = neg_mf.rolling(window=period, min_periods=1).sum()
    
    mfr = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def vwap_band(high: pd.Series, low: pd.Series, close: pd.Series, 
             volume: pd.Series, period: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP 布林带"""
    vwap_val = vwap(high, low, close, volume)
    std = (close * volume).rolling(window=period, min_periods=1).sum() / volume.rolling(window=period, min_periods=1).sum()
    std = std.rolling(window=period, min_periods=1).std()
    
    return vwap_val + std_mult * std, vwap_val, vwap_val - std_mult * std


# ============ Ichimoku Kinko Hyo ============

def ichimoku(high: pd.Series, low: pd.Series, 
            tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> IchimokuResult:
    """一目均衡表 (Ichimoku Kinko Hyo)"""
    tenkan_sen = (high.rolling(window=tenkan, min_periods=1).max() + 
                  low.rolling(window=tenkan, min_periods=1).min()) / 2
    
    kijun_sen = (high.rolling(window=kijun, min_periods=1).max() + 
                 low.rolling(window=kijun, min_periods=1).min()) / 2
    
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    senkou_b_line = (high.rolling(window=senkou_b, min_periods=1).max() + 
                     low.rolling(window=senkou_b, min_periods=1).min()) / 2
    senkou_b_result = senkou_b_line.shift(kijun)
    
    chikou = close.shift(-kijun)
    
    return IchimokuResult(
        tenkan=tenkan_sen,
        kijun=kijun_sen,
        senkou_a=senkou_a,
        senkou_b=senkou_b_result,
        chikou=chikou
    )


# ============ 皮尔逊相关系数 ============

def pearson_correlation(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
    """皮尔逊相关系数"""
    return series1.rolling(window=period).corr(series2)


def correlation(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
    """滚动相关性"""
    return pearson_correlation(series1, series2, period)


# ============ 希尔伯特变换指标 ============

def hilbert_transform(series: pd.Series) -> pd.Series:
    """希尔伯特变换 (Hilbert Transform) - 简化实现"""
    # 简化的希尔伯特变换实现
    n = len(series)
    result = np.zeros(n)
    
    for i in range(n):
        if i < 4:
            result[i] = series.iloc[i]
        else:
            result[i] = 0.9962 * result[i-4] + series.iloc[i] - series.iloc[i-4]
    
    return pd.Series(result, index=series.index)


def hilbert_oscillator(series: pd.Series) -> pd.Series:
    """希尔伯特振荡器"""
    ht = hilbert_transform(series)
    sine_wave = np.sin(np.linspace(0, 2 * np.pi, len(series)))
    lead_lag = ht - pd.Series(sine_wave, index=series.index) * 0.5
    return lead_lag


# ============ 回归分析 ============

def linear_regression_slope(series: pd.Series, period: int = 14) -> pd.Series:
    """线性回归斜率"""
    def calc_slope(x):
        if len(x) < 2:
            return 0
        y = np.arange(len(x))
        if np.std(y) == 0:
            return 0
        slope, _ = np.polyfit(y, x, 1)
        return slope
    
    return series.rolling(window=period, min_periods=1).apply(calc_slope, raw=True)


def linear_regression_intercept(series: pd.Series, period: int = 14) -> pd.Series:
    """线性回归截距"""
    def calc_intercept(x):
        if len(x) < 2:
            return x.iloc[-1] if len(x) > 0 else 0
        y = np.arange(len(x))
        if np.std(y) == 0:
            return x.iloc[-1] if len(x) > 0 else 0
        slope, intercept = np.polyfit(y, x, 1)
        return intercept
    
    return series.rolling(window=period, min_periods=1).apply(calc_intercept, raw=True)


def linear_regression_r2(series: pd.Series, period: int = 14) -> pd.Series:
    """线性回归 R²"""
    def calc_r2(x):
        if len(x) < 2:
            return 0
        y = np.arange(len(x))
        if np.std(x) == 0 or np.std(y) == 0:
            return 0
        return np.corrcoef(x, y)[0, 1] ** 2
    
    return series.rolling(window=period, min_periods=1).apply(calc_r2, raw=True)


# ============ 统计函数 ============

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-Score"""
    mean = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std()
    return (series - mean) / std.replace(0, np.nan)


def percentile(series: pd.Series, period: int = 20) -> pd.Series:
    """百分位排名"""
    return series.rolling(window=period, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def entropy(series: pd.Series, period: int = 20, base: float = 2) -> pd.Series:
    """信息熵"""
    def calc_entropy(x):
        if len(x) == 0 or x.std() == 0:
            return 0
        hist, _ = np.histogram(x, bins=10)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist) / np.log(base))
    
    return series.rolling(window=period, min_periods=1).apply(calc_entropy, raw=True)


# ============ 便利函数 ============

def hma(series: pd.Series, period: int) -> pd.Series:
    """赫尔移动平均 (Hull Moving Average)"""
    wma_half = wma(series, period // 2)
    wma_full = wma(series, period)
    return wma(2 * wma_half - wma_full, int(np.sqrt(period)))


def vhma(series: pd.Series, period: int, volume: pd.Series) -> pd.Series:
    """成交量加权赫尔移动平均"""
    hma_val = hma(series, period)
    return hma_val * (volume / volume.rolling(window=period, min_periods=1).mean())


def hma(series: pd.Series, period: int) -> pd.Series:
    """赫尔移动平均 (Hull Moving Average)"""
    # WMA of price for 2*period/2 and period
    wma_half = wma(series, period // 2)
    wma_full = wma(series, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    return wma(hma_raw, int(np.sqrt(period)))


def zlema(series: pd.Series, period: int) -> pd.Series:
    """零滞后指数移动平均 (Zero-Lag EMA)"""
    ema1 = ema(series, period)
    ema2 = ema(series, period)
    return 2 * ema1 - ema2


def kama(series: pd.Series, period: int = 14, fast: int = 2, slow: int = 30) -> pd.Series:
    """考夫曼自适应移动平均 (KAMA)"""
    abs_diff = series.diff().abs()
    er = abs_diff.rolling(window=period, min_periods=1).sum() / abs_diff.rolling(window=period, min_periods=1).sum()
    fast_alpha = 2 / (fast + 1)
    slow_alpha = 2 / (slow + 1)
    alpha = er * (fast_alpha - slow_alpha) + slow_alpha
    
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[0] = series.iloc[0]
    
    for i in range(1, len(series)):
        kama.iloc[i] = alpha.iloc[i] * series.iloc[i] + (1 - alpha.iloc[i]) * kama.iloc[i-1]
    
    return kama


# ============ 导出 ============

__all__ = [
    # 移动平均线
    'sma', 'ema', 'wma', 'dema', 'tema', 'vwap', 'vwma',
    'hma', 'vhma', 'zlema', 'kama',
    
    # 趋势指标
    'adx', 'Aroon', 'cci', 'dmi',
    
    # 动量指标
    'rsi', 'stoch_rsi', 'macd', 'roc', 'mom', 'williams_r', 'ultimate_oscillator',
    
    # 波动率指标
    'atr', 'bollinger_bands', 'kc', 'donchian',
    
    # 随机指标
    'stochastic',
    
    # 成交量指标
    'obv', 'ad', 'adosc', 'mfi', 'vwap_band',
    
    # Ichimoku
    'ichimoku',
    
    # 相关性
    'pearson_correlation', 'correlation',
    
    # 希尔伯特变换
    'hilbert_transform', 'hilbert_oscillator',
    
    # 回归分析
    'linear_regression_slope', 'linear_regression_intercept', 'linear_regression_r2',
    
    # 统计函数
    'zscore', 'percentile', 'entropy',
]

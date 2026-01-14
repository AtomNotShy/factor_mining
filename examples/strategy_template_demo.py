"""
示例策略脚本

演示如何使用新的策略模板系统。
"""

from datetime import date
import pandas as pd
import numpy as np

from src.strategies.base import (
    RSIStrategy,
    TrendFollowingStrategy,
    MACDStrategy,
    StrategyTemplateFactory,
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
)


# ============ 示例 1: 继承策略模板 ============

class SimpleRSIStrategy(RSIStrategy):
    """简单 RSI 策略 - 当 RSI 低于 30 时买入，高于 70 时卖出"""
    
    strategy_name = "Simple RSI Strategy"
    timeframe = "1d"
    
    # RSI 参数
    rsi_period = 14
    rsi_overbought = 70.0
    rsi_oversold = 30.0
    
    # 止损设置
    stoploss = -0.05


class DualMAStrategy(TrendFollowingStrategy):
    """双均线策略 - 短期均线上穿长期均线时买入"""
    
    strategy_name = "Dual MA Crossover"
    timeframe = "1d"
    
    # 均线参数
    fast_period = 10
    slow_period = 20
    ma_type = "sma"
    
    # 进场阈值（均线金叉强度）
    entry_threshold = 0.0


# ============ 示例 2: 使用策略工厂 ============

def create_custom_strategy():
    """使用工厂创建自定义策略"""
    
    # 创建 RSI 策略
    rsi_strategy = StrategyTemplateFactory.create(
        'rsi',
        strategy_id='custom_rsi',
        rsi_period=10,
        rsi_overbought=65,
        rsi_oversold=35,
        strategy_name="Custom RSI"
    )
    
    # 创建均线策略
    ma_strategy = StrategyTemplateFactory.create(
        'trend_following',
        strategy_id='custom_ma',
        fast_period=5,
        slow_period=15,
        strategy_name="Custom MA"
    )
    
    return rsi_strategy, ma_strategy


# ============ 示例 3: 手动计算指标 ============

def calculate_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    """演示如何手动计算技术指标"""
    
    # 移动平均线
    dataframe['sma_20'] = sma(dataframe['close'], 20)
    dataframe['ema_12'] = ema(dataframe['close'], 12)
    dataframe['ema_26'] = ema(dataframe['close'], 26)
    
    # RSI
    dataframe['rsi_14'] = rsi(dataframe['close'], 14)
    
    # MACD
    macd_result = macd(dataframe['close'], 12, 26, 9)
    dataframe['macd'] = macd_result.macd
    dataframe['macd_signal'] = macd_result.signal
    dataframe['macd_hist'] = macd_result.hist
    
    # 布林带
    bb = bollinger_bands(dataframe['close'], 20, 2)
    dataframe['bb_upper'] = bb.upper
    dataframe['bb_middle'] = bb.middle
    dataframe['bb_lower'] = bb.lower
    
    # ATR
    dataframe['atr_14'] = atr(dataframe['high'], dataframe['low'], dataframe['close'], 14)
    
    return dataframe


# ============ 示例 4: 生成交易信号 ============

def generate_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """基于指标生成交易信号"""
    
    dataframe['enter_long'] = 0
    dataframe['exit_long'] = 0
    
    # RSI 信号
    dataframe.loc[dataframe['rsi_14'] < 30, 'enter_long'] = 1
    dataframe.loc[dataframe['rsi_14'] > 70, 'exit_long'] = 1
    
    # MACD 金叉/死叉
    dataframe['macd_prev'] = dataframe['macd'].shift(1)
    dataframe['signal_prev'] = dataframe['macd_signal'].shift(1)
    
    # 金叉
    macd_cross_up = (
        (dataframe['macd'] > dataframe['macd_signal']) & 
        (dataframe['macd_prev'] <= dataframe['signal_prev'])
    )
    
    # 死叉
    macd_cross_down = (
        (dataframe['macd'] < dataframe['macd_signal']) & 
        (dataframe['macd_prev'] >= dataframe['signal_prev'])
    )
    
    dataframe.loc[macd_cross_up, 'enter_long'] = 1
    dataframe.loc[macd_cross_down, 'exit_long'] = 1
    
    return dataframe


# ============ 主测试函数 ============

def main():
    """运行示例"""
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1d')
    
    # 生成随机价格数据
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    dataframe = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + np.abs(np.random.randn(100)),
        'low': prices - np.abs(np.random.randn(100)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
    }, index=dates)
    
    print("=" * 60)
    print("示例 1: 继承策略模板")
    print("=" * 60)
    
    rsi_strategy = SimpleRSIStrategy()
    print(f"策略 ID: {rsi_strategy.strategy_id}")
    print(f"策略名称: {rsi_strategy.config.strategy_name}")
    print(f"时间框架: {rsi_strategy.config.timeframe}")
    print(f"RSI 周期: {rsi_strategy.rsi_period}")
    print(f"RSI 超买: {rsi_strategy.rsi_overbought}")
    print(f"RSI 超卖: {rsi_strategy.rsi_oversold}")
    
    ma_strategy = DualMAStrategy()
    print(f"\n均线策略 ID: {ma_strategy.strategy_id}")
    print(f"快线周期: {ma_strategy.fast_period}")
    print(f"慢线周期: {ma_strategy.slow_period}")
    
    print("\n" + "=" * 60)
    print("示例 2: 策略工厂")
    print("=" * 60)
    
    rsi_s, ma_s = create_custom_strategy()
    print(f"工厂 RSI 策略: {rsi_s.strategy_id}")
    print(f"工厂 MA 策略: {ma_s.strategy_id}")
    
    print("\n" + "=" * 60)
    print("示例 3 & 4: 指标计算和信号生成")
    print("=" * 60)
    
    # 计算指标
    dataframe = calculate_indicators(dataframe)
    
    # 生成信号
    dataframe = generate_signals(dataframe)
    
    # 显示结果
    print("\n最后 10 根 K 线:")
    print(dataframe[['close', 'rsi_14', 'macd', 'macd_signal', 'enter_long', 'exit_long']].tail(10))
    
    # 统计信号
    print(f"\n买入信号数量: {dataframe['enter_long'].sum()}")
    print(f"卖出信号数量: {dataframe['exit_long'].sum()}")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

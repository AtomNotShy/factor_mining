#!/usr/bin/env python3
"""
测试动量计算
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 创建测试数据：上升趋势
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
np.random.seed(42)

# 创建上升趋势
trend = np.linspace(100, 150, len(dates))
noise = np.random.normal(0, 2, len(dates))
prices = trend + noise

df = pd.DataFrame({
    'close': prices,
    'open': prices * 0.99,
    'high': prices * 1.01,
    'low': prices * 0.98,
    'volume': np.random.randint(1000000, 10000000, len(dates))
}, index=dates)

print("测试数据:")
print(f"数据点数量: {len(df)}")
print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
print(f"总收益率: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")

# 手动计算简单动量（20天收益率）
lookback = 20
if len(df) > lookback:
    simple_return = (df['close'].iloc[-1] / df['close'].iloc[-lookback] - 1) * 100
    annualized = simple_return * (252 / lookback)
    print(f"\n简单动量计算 ({lookback}天):")
    print(f"  简单收益率: {simple_return:.2f}%")
    print(f"  年化收益率: {annualized:.2f}%")

# 测试加权线性回归
from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy

strategy = ETFMomentumJoinQuantStrategy(lookback_days=20, r2_threshold=0.3)

# 测试动量计算
prices_series = pd.Series(df['close'].values, index=df.index)
result = strategy._calculate_weighted_momentum(prices_series)

print("\n加权线性回归动量计算:")
print(f"  动量(年化收益率): {result['momentum'] * 100:.2f}%")
print(f"  R²: {result['r2']:.4f}")
print(f"  年化波动率: {result['annual_volatility'] * 100:.2f}%")
print(f"  是否有效: {result['valid']}")

# 测试动量评分
score = strategy._calculate_momentum_score(result)
print(f"  动量评分: {score:.4f}")

# 测试整个数据框
df_test = df.copy()
df_test['symbol'] = 'TEST'
df_with_indicators = strategy.populate_indicators(df_test, {"symbol": "TEST"})

print("\n数据框指标计算:")
if 'momentum_score' in df_with_indicators.columns:
    last_score = df_with_indicators['momentum_score'].iloc[-1]
    last_r2 = df_with_indicators['momentum_r2'].iloc[-1]
    print(f"  最后动量评分: {last_score:.4f}")
    print(f"  最后R²: {last_r2:.4f}")
    
    # 检查有多少天的动量评分为正
    positive_days = (df_with_indicators['momentum_score'] > 0).sum()
    print(f"  动量评分为正的天数: {positive_days}/{len(df_with_indicators)}")
    
    # 检查进场信号
    df_with_entry = strategy.populate_entry_trend(df_with_indicators, {"symbol": "TEST"})
    enter_signals = df_with_entry[df_with_entry['enter_long'] == 1]
    print(f"  进场信号数量: {len(enter_signals)}")
    
    if not enter_signals.empty:
        print(f"  第一个进场信号: {enter_signals.index[0]}")
        print(f"  最后一个进场信号: {enter_signals.index[-1]}")

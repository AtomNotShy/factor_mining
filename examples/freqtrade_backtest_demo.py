"""
Freqtrade 策略回测示例

演示使用新的 Freqtrade 风格策略进行回测：
1. Dual MA Crossover Strategy
2. Mean Reversion Strategy  
3. ETF Momentum JoinQuant Strategy

用法:
    python examples/freqtrade_backtest_demo.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import date, timedelta
from loguru import logger

from src.strategies.user_strategies.dual_ma import DualMAStrategy, create_dual_ma_strategy
from src.strategies.user_strategies.mean_reversion import MeanReversionStrategy, create_mean_reversion_strategy
from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine, UnifiedConfig, FeatureFlag
from src.evaluation.backtesting.config import TradeConfig, TimeConfig


def generate_mock_data(
    symbol: str,
    start_date: date,
    end_date: date,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0005
) -> pd.DataFrame:
    """生成模拟市场数据"""
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(all_dates)
    returns = np.random.normal(drift, volatility, n)
    prices = initial_price * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, n))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, n))),
        'close': prices,
        'volume': np.random.uniform(1_000_000, 10_000_000, n).astype(int)
    }, index=all_dates)

    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data


async def run_dual_ma_backtest():
    """运行 Dual MA 策略回测"""
    print("\n" + "=" * 60)
    print("Dual MA Crossover Strategy Backtest")
    print("=" * 60)

    # 创建策略（使用默认参数）
    strategy = DualMAStrategy()
    print(f"策略: {strategy.strategy_name}")
    print(f"Fast Period: {strategy.fast_period}")
    print(f"Slow Period: {strategy.slow_period}")
    print(f"MA Type: {strategy.ma_type}")
    print(f"Stoploss: {strategy.stoploss}")
    print(f"Minimal ROI: {strategy.minimal_roi}")
    print()

    # 或使用工厂创建自定义策略
    # custom_strategy = create_dual_ma_strategy(
    #     fast_period=5, slow_period=20, ma_type='ema',
    #     strategy_id='fast_dual_ma'
    # )

    # 生成模拟数据
    end_date = date.today()
    start_date = end_date - timedelta(days=90)
    symbols = ['SPY', 'QQQ']

    market_data = {}
    for symbol in symbols:
        if symbol == 'SPY':
            price, vol, drift = 450.0, 0.015, 0.0005
        else:
            price, vol, drift = 380.0, 0.025, 0.0008

        market_data[symbol] = generate_mock_data(symbol, start_date, end_date, price, vol, drift)
        print(f"{symbol}: {len(market_data[symbol])} days, "
              f"price ${market_data[symbol]['close'].iloc[-1]:.2f}")

    # 配置
    config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=100000.0,
            commission_rate=0.001,
        ),
        time=TimeConfig(signal_timeframe="1d"),
        features=FeatureFlag.ALL,
    )
    print(f"\n配置: 初始资金 ${config.trade.initial_capital:,.0f}, 手续费 {config.trade.commission_rate*100:.2f}%")

    # 创建引擎
    engine = UnifiedBacktestEngine(config=config)

    # 运行回测
    print("\n运行回测...")
    results = await engine.run(
        strategies=[strategy],
        universe=symbols,
        start=start_date,
        end=end_date,
    )

    # 显示结果
    result_dict = results.to_dict() if results else {}
    print(f"\n📊 回测结果:")
    print(f"   总收益: {result_dict.get('total_return_pct', 0):.2f}%")
    print(f"   最大回撤: {result_dict.get('max_drawdown_pct', 0):.2f}%")
    print(f"   交易次数: {result_dict.get('total_trades', 0)}")
    print(f"   胜率: {result_dict.get('win_rate_pct', 0):.1f}%")
    print(f"   夏普比率: {result_dict.get('sharpe_ratio', 0):.2f}")

    return results


async def run_mean_reversion_backtest():
    """运行均值回归策略回测"""
    print("\n" + "=" * 60)
    print("Mean Reversion Strategy Backtest")
    print("=" * 60)

    # 创建策略
    strategy = MeanReversionStrategy()
    print(f"策略: {strategy.strategy_name}")
    print(f"Lookback: {strategy.lookback}")
    print(f"Entry Z-Score: {strategy.entry_z}")
    print(f"Exit Z-Score: {strategy.exit_z}")
    print(f"Stoploss: {strategy.stoploss}")
    print()

    # 生成均值回归数据 (OU 过程)
    end_date = date.today()
    start_date = end_date - timedelta(days=90)
    
    np.random.seed(42)
    n = len(pd.date_range(start=start_date, end=end_date, freq='B'))
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    theta = 0.1  # 回复速度
    mu = 100.0   # 长期均值
    sigma = 2.0  # 波动率
    prices = np.zeros(n)
    prices[0] = mu

    for t in range(1, n):
        prices[t] = prices[t-1] + theta * (mu - prices[t-1]) + np.random.normal(0, sigma)

    market_data = {
        'SPY': pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, n))),
            'close': prices,
            'volume': np.random.uniform(1_000_000, 10_000_000, n).astype(int)
        }, index=dates)
    }

    print(f"SPY: {len(market_data['SPY'])} days, "
          f"price ${market_data['SPY']['close'].iloc[-1]:.2f}")

    # 配置
    config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=100000.0,
            commission_rate=0.001,
        ),
        time=TimeConfig(signal_timeframe="1d"),
        features=FeatureFlag.ALL,
    )
    print(f"\n配置: 初始资金 ${config.trade.initial_capital:,.0f}")

    # 创建引擎
    engine = UnifiedBacktestEngine(config=config)

    # 运行回测
    print("\n运行回测...")
    results = await engine.run(
        strategies=[strategy],
        universe=['SPY'],
        start=start_date,
        end=end_date,
    )

    # 显示结果
    result_dict = results.to_dict() if results else {}
    print(f"\n📊 回测结果:")
    print(f"   总收益: {result_dict.get('total_return_pct', 0):.2f}%")
    print(f"   最大回撤: {result_dict.get('max_drawdown_pct', 0):.2f}%")
    print(f"   交易次数: {result_dict.get('total_trades', 0)}")
    print(f"   胜率: {result_dict.get('win_rate_pct', 0):.1f}%")

    return results


async def demonstrate_strategy_usage():
    """演示策略的各种用法"""
    print("\n" + "=" * 60)
    print("Strategy Usage Demonstration")
    print("=" * 60)

    # 1. Dual MA Strategy
    print("\n1. Dual MA Strategy:")
    print("   默认参数:")
    dual_ma = DualMAStrategy()
    print(f"   - fast_period: {dual_ma.fast_period}")
    print(f"   - slow_period: {dual_ma.slow_period}")
    print(f"   - ma_type: {dual_ma.ma_type}")

    print("\n   自定义参数 (工厂模式):")
    custom_dual_ma = create_dual_ma_strategy(
        fast_period=5,
        slow_period=20,
        ma_type='ema',
        strategy_id='fast_ema_crossover'
    )
    print(f"   - strategy_id: {custom_dual_ma.strategy_id}")
    print(f"   - fast_period: {custom_dual_ma.fast_period}")
    print(f"   - slow_period: {custom_dual_ma.slow_period}")
    print(f"   - ma_type: {custom_dual_ma.ma_type}")

    # 2. Mean Reversion Strategy
    print("\n2. Mean Reversion Strategy:")
    print("   默认参数:")
    mean_rev = MeanReversionStrategy()
    print(f"   - lookback: {mean_rev.lookback}")
    print(f"   - entry_z: {mean_rev.entry_z}")
    print(f"   - exit_z: {mean_rev.exit_z}")

    print("\n   自定义参数:")
    custom_mean_rev = create_mean_reversion_strategy(
        lookback=40,
        entry_z=-1.5,
        exit_z=-0.5,
        strategy_id='fast_mean_reversion'
    )
    print(f"   - lookback: {custom_mean_rev.lookback}")
    print(f"   - entry_z: {custom_mean_rev.entry_z}")
    print(f"   - exit_z: {custom_mean_rev.exit_z}")

    # 3. ETF Momentum Strategy
    print("\n3. ETF Momentum JoinQuant Strategy:")
    print("   默认参数:")
    etf_mom = ETFMomentumJoinQuantStrategy()
    print(f"   - etf_pool: {etf_mom.etf_pool[:5]}...")
    print(f"   - lookback_days: {etf_mom.lookback_days}")
    print(f"   - r2_threshold: {etf_mom.r2_threshold}")
    print(f"   - target_positions: {etf_mom.target_positions}")

    # 4. 测试信号生成
    print("\n4. Signal Generation Test:")
    import pandas as pd
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 105 + np.cumsum(np.random.randn(100) * 0.5),
        'low': 95 + np.cumsum(np.random.randn(100) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1_000_000, 10_000_000, 100)
    }, index=dates)

    # Dual MA 信号
    df = dual_ma.populate_indicators(data.copy(), {'pair': 'TEST'})
    entry = dual_ma.populate_entry_trend(df.copy(), {'pair': 'TEST'})
    exit_df = dual_ma.populate_exit_trend(df.copy(), {'pair': 'TEST'})
    print(f"   Dual MA - Entry signals: {entry['enter_long'].sum()}")
    print(f"   Dual MA - Exit signals: {exit_df['exit_long'].sum()}")

    # Mean Reversion 信号
    df2 = mean_rev.populate_indicators(data.copy(), {'pair': 'TEST'})
    entry2 = mean_rev.populate_entry_trend(df2.copy(), {'pair': 'TEST'})
    exit2 = mean_rev.populate_exit_trend(df2.copy(), {'pair': 'TEST'})
    print(f"   Mean Reversion - Entry signals: {entry2['enter_long'].sum()}")
    print(f"   Mean Reversion - Exit signals: {exit2['exit_long'].sum()}")


async def main():
    """主函数"""
    # 移除默认 logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))

    print("\n" + "=" * 60)
    print("Freqtrade Strategy Backtest Demo")
    print("=" * 60)
    print("\n此演示展示如何使用新的 Freqtrade 风格策略")

    # 演示策略用法
    await demonstrate_strategy_usage()

    # 运行 Dual MA 回测
    await run_dual_ma_backtest()

    # 运行 Mean Reversion 回测
    await run_mean_reversion_backtest()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    print("\n💡 使用 CLI 运行回测:")
    print("   python src/main.py backtest --list-strategies")
    print("   python src/main.py backtest --strategy dual_ma --symbols SPY --days 90")
    print("   python src/main.py backtest --strategy mean_reversion --symbols SPY --days 90")
    print()


if __name__ == "__main__":
    asyncio.run(main())

"""
使用示例
演示如何使用新系统进行回测
"""

from datetime import date, datetime
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.core.types import MarketData
from src.evaluation.backtesting.engine import BacktestEngine
from src.strategies.vwap.vwap_pullback_v2 import VWAPPullbackStrategyV2, VWAPPullbackParams
from src.evaluation.walk_forward import WalkForwardAnalyzer, Gate
from src.evaluation.metrics.performance import PerformanceAnalyzer
from src.data.collectors.polygon import PolygonCollector
from src.data.storage.parquet_store import ParquetDataFrameStore
from src.config.settings import get_settings


async def example_backtest():
    """回测示例"""
    print("=" * 60)
    print("回测示例")
    print("=" * 60)
    
    # 1. 创建运行上下文
    calendar = TradingCalendar()
    ctx = RunContext.create(
        env=Environment.RESEARCH,
        trading_calendar=calendar,
    )
    print(f"运行上下文创建成功: env={ctx.env.value}, code_version={ctx.code_version}")
    
    # 2. 创建策略
    params = VWAPPullbackParams()
    strategy = VWAPPullbackStrategyV2(params)
    print(f"策略创建成功: {strategy.strategy_id}")
    
    # 3. 加载数据（示例：从Polygon获取）
    collector = PolygonCollector()
    await collector.connect()
    
    try:
        bars = await collector.get_ohlcv(
            symbol="SPY",
            timeframe="1d",
            since=datetime(2024, 1, 1),
            data_version=ctx.data_version,
        )
        print(f"数据加载成功: {len(bars)} 条记录")
        
        if bars.empty:
            print("警告: 没有可用数据")
            return
        
        # 4. 创建市场数据
        md = MarketData(bars=bars)
        
        # 5. 运行回测
        engine = BacktestEngine(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
        )
        
        result = engine.run(
            strategies=[strategy],
            universe=["SPY"],
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            ctx=ctx,
            bars=bars,
        )
        
        print(f"\n回测完成:")
        print(f"  Run ID: {result.get('run_id')}")
        print(f"  最终资产: ${result.get('final_equity', 0):,.2f}")
        print(f"  总收益: {result.get('total_return', 0):.2%}")
        print(f"  信号数: {len(result.get('signals', []))}")
        print(f"  订单数: {len(result.get('orders', []))}")
        print(f"  成交数: {len(result.get('fills', []))}")
        
        # 6. 计算性能指标
        portfolio_daily = result.get('portfolio_daily')
        if portfolio_daily is not None and not portfolio_daily.empty:
            returns = portfolio_daily['daily_return']
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.comprehensive_analysis(returns)
            
            print(f"\n性能指标:")
            print(f"  CAGR: {metrics.get('cagr', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            
            # 7. Gate准入检查
            gate = Gate()
            passed, failures = gate.pass_gate(metrics)
            
            print(f"\nGate准入:")
            print(f"  通过: {passed}")
            if failures:
                print(f"  失败原因: {failures}")
        
    finally:
        await collector.disconnect()


async def example_walk_forward():
    """Walk-Forward分析示例"""
    print("=" * 60)
    print("Walk-Forward分析示例")
    print("=" * 60)
    
    # 创建策略和上下文
    params = VWAPPullbackParams()
    strategy = VWAPPullbackStrategyV2(params)
    
    calendar = TradingCalendar()
    ctx = RunContext.create(
        env=Environment.RESEARCH,
        trading_calendar=calendar,
    )
    
    # 运行WFA
    analyzer = WalkForwardAnalyzer()
    wfa_result = analyzer.run(
        strategies=[strategy],
        universe=["SPY"],
        start=date(2023, 1, 1),
        end=date(2024, 12, 31),
        train_window_days=252,
        test_window_days=63,
        step_days=21,
        ctx=ctx,
    )
    
    print(f"WFA完成:")
    print(f"  总窗口数: {wfa_result.get('total_windows', 0)}")
    print(f"  通过率: {wfa_result.get('pass_rate', 0):.2%}")


if __name__ == "__main__":
    import asyncio
    
    # 运行回测示例
    asyncio.run(example_backtest())
    
    # 运行WFA示例（需要数据）
    # asyncio.run(example_walk_forward())

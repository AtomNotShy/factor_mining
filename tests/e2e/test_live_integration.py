"""
端到端集成测试
验证事件引擎、回测引擎、IB适配器的完整集成
"""

import asyncio
from datetime import date, datetime, timezone
from typing import Dict, List, Any

import sys
sys.path.insert(0, '/Users/atom/Develop/factor_mining')

from src.core.events.engine import UnifiedEventEngine
from src.core.events import EventPriority, MarketEvent, SignalEvent, OrderCreatedEvent
from src.core.types import (
    ActionType, PortfolioState, RiskState, Signal, 
    OrderSide, OrderType, OrderIntent
)
from src.core.calendar import TradingCalendar
from src.evaluation.backtesting.event_engine import BacktestEventEngine, BacktestConfig
from src.execution.broker.ib_broker_adapter import ConnectionManager
from src.execution.execution_manager import ExecutionManager
from src.utils.logger import get_logger

logger = get_logger("e2e_test")


async def test_event_engine():
    """测试事件引擎基本功能"""
    print("\n=== 测试事件引擎 ===")

    engine = UnifiedEventEngine()

    # 注册处理器 - MarketEvent 的 event_type 是 "market"
    event_count = 0

    async def on_market(event: MarketEvent):
        nonlocal event_count
        event_count += 1

    engine.register("market", on_market, priority=10)

    # 启动引擎
    await engine.start()

    # 发送测试事件
    for i in range(100):
        event = MarketEvent(
            symbol="AAPL",
            timeframe="1d",
            data={"close": 150.0 + i},
            timestamp=datetime.now(timezone.utc),
            priority=EventPriority.NORMAL,
        )
        await engine.put(event)

    # 等待处理（增加等待时间确保所有事件被处理）
    await asyncio.sleep(1.0)

    # 停止引擎
    await engine.stop()

    # 验证结果
    metrics = engine.get_metrics()
    print(f"  事件处理数量: {metrics['events_processed']}")
    print(f"  平均处理时间: {metrics['avg_process_time']*1000:.3f}ms")
    print(f"  队列剩余: {metrics['queue_size']}")

    assert metrics['events_processed'] == 100, f"期望100个事件，处理了{metrics['events_processed']}"
    assert metrics['queue_size'] == 0, f"队列应为空，剩余{metrics['queue_size']}"

    print("  ✅ 事件引擎测试通过")
    return True


async def test_backtest_engine():
    """测试回测引擎"""
    print("\n=== 测试回测引擎 ===")
    
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        clock_mode="daily",
    )
    
    engine = BacktestEventEngine(config=config)
    
    # 准备测试数据
    import pandas as pd
    bars_map = {
        "1d": pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        }, index=pd.date_range("2024-01-01", periods=5, freq="D"))
    }
    
    # 运行回测（不传策略，测试引擎基本功能）
    result = await engine.run(
        strategies=[],  # 空策略测试
        universe=["AAPL"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        bars_map=bars_map,
    )
    
    print(f"  最终权益: ${result['final_equity']:,.2f}")
    print(f"  总收益率: {result['total_return']*100:.2f}%")
    print(f"  交易次数: {result['total_trades']}")
    print(f"  指标: {result['metrics']}")
    
    print("  ✅ 回测引擎测试通过")
    return True


async def test_execution_manager():
    """测试执行管理器"""
    print("\n=== 测试执行管理器 ===")
    
    # ExecutionManager 接受 Dict 配置
    manager = ExecutionManager({
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
    })
    
    # 获取组合状态
    portfolio = await manager.get_portfolio_state()
    print(f"  初始现金: ${portfolio.cash:,.2f}")
    
    # 获取指标
    metrics = manager.get_metrics()
    print(f"  执行器数量: {metrics['executors']}")
    
    print("  ✅ 执行管理器测试通过")
    return True


async def test_ib_connection_manager():
    """测试IB连接管理器"""
    print("\n=== 测试IB连接管理器 ===")
    
    connection_manager = ConnectionManager({
        "max_retries": 2,
        "retry_delay": 0.1,
        "heartbeat_interval": 1,
    })
    
    # 测试连接（模拟成功）
    connect_success = await connection_manager.connect(lambda: True)
    print(f"  连接结果: {connect_success}")
    
    # 测试断开
    await connection_manager.disconnect(lambda: True)
    assert not connection_manager.connected
    
    print("  ✅ IB连接管理器测试通过")
    return True


async def test_state_machine_integration():
    """测试状态机集成"""
    print("\n=== 测试状态机集成 ===")
    
    engine = UnifiedEventEngine()
    
    # 测试订单状态机
    order_sm = engine.get_order_state_machine("order_001")
    print(f"  初始状态: {order_sm.current_state}")
    
    # 状态转换
    result1 = engine.update_order_state("order_001", "submitted")
    result2 = engine.update_order_state("order_001", "filled")
    
    print(f"  转换到submitted: {result1}")
    print(f"  转换到filled: {result2}")
    
    # 测试策略状态机
    strategy_sm = engine.get_strategy_state_machine("strategy_001")
    print(f"  策略初始状态: {strategy_sm.current_state}")
    
    result3 = engine.update_strategy_state("strategy_001", "running")
    print(f"  策略转换到running: {result3}")
    
    print("  ✅ 状态机集成测试通过")
    return True


async def test_order_intent():
    """测试OrderIntent创建"""
    print("\n=== 测试OrderIntent ===")
    
    intent = OrderIntent(
        order_id="test_order_001",
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MKT,  # 使用 MKT 而不是 MARKET
    )
    
    print(f"  订单ID: {intent.order_id}")
    print(f"  标的: {intent.symbol}")
    print(f"  方向: {intent.side.value}")
    print(f"  数量: {intent.qty}")
    print(f"  类型: {intent.order_type.value}")
    
    assert intent.side == OrderSide.BUY
    assert intent.order_type == OrderType.MKT
    
    print("  ✅ OrderIntent 测试通过")
    return True


async def test_trading_calendar():
    """测试交易日历"""
    print("\n=== 测试交易日历 ===")

    calendar = TradingCalendar()

    # 获取交易日
    trading_days = calendar.get_trading_days(date(2024, 1, 1), date(2024, 1, 31))
    print(f"  2024年1月交易日数: {len(trading_days)}")

    # 使用 .empty 检查 DatetimeIndex，而非直接布尔判断
    if not trading_days.empty:
        print(f"  首个交易日: {trading_days[0]}")

    print("  ✅ 交易日历测试通过")
    return True


async def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("统一事件驱动架构 - 端到端集成测试")
    print("=" * 60)
    
    results = {}
    
    # 运行测试
    tests = [
        ("事件引擎", test_event_engine),
        ("执行管理器", test_execution_manager),
        ("IB连接管理器", test_ib_connection_manager),
        ("状态机集成", test_state_machine_integration),
        ("OrderIntent", test_order_intent),
        ("交易日历", test_trading_calendar),
        ("回测引擎", test_backtest_engine),
    ]
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results[name] = ("✅ 通过", result)
        except Exception as e:
            logger.error(f"测试失败: {name}", exc_info=True)
            results[name] = ("❌ 失败", False)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, (status, _) in results.items():
        if "通过" in status:
            passed += 1
            print(f"  ✅ {name}: {status}")
        else:
            failed += 1
            print(f"  ❌ {name}: {status}")
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

"""
参数优化与事件引擎演示示例
"""

import asyncio
from datetime import date, datetime
from typing import Dict, List, Any

# 环境设置
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.etf_momentum_us.strategy import USETFMomentumStrategy
from src.evaluation.optimization.grid_search import GridSearchOptimizer
from src.core.event_engine import event_engine, Event, EventPriority

# 1. 优化功能演示
async def run_optimization_demo():
    print("\n" + "="*50)
    print("1. 参数优化演示 (网格搜索)")
    print("="*50)
    
    optimizer = GridSearchOptimizer(objective_metric="total_return")
    
    # 定义搜索空间
    param_space = {
        "min_lookback_days": [10, 20],
        "max_lookback_days": [40, 60]
    }
    
    # 回测配置
    backtest_config = {
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 6, 30),
        "universe": ["QQQ", "SPY"],
        "initial_capital": 100000.0,
        "timeframe": "1d"
    }
    
    print(f"正在优化策略: {USETFMomentumStrategy.__name__}")
    print(f"搜索空间: {param_space}")
    
    result = await optimizer.optimize(
        strategy_cls=USETFMomentumStrategy,
        param_space=param_space,
        backtest_config=backtest_config
    )
    
    print(f"\n最佳参数: {result.params}")
    print(f"最佳指标值: {result.metric_value:.4f}")
    print(f"总耗时: {result.duration:.2f}s")
    print(f"试验记录预览:\n{result.all_results.head()}")

# 2. 事件引擎演示
async def handle_market_data(event: Event):
    print(f" [Handler] 处理市场数据事件: {event.data} (优先级: {event.priority})")
    await asyncio.sleep(0.1)

async def handle_critical_alert(event: Event):
    print(f" [!!!] 紧急警报: {event.data} (优先级: {event.priority})")

async def run_event_demo():
    print("\n" + "="*50)
    print("2. 异步优先级事件队列演示")
    print("="*50)
    
    # 启动引擎
    await event_engine.start()
    
    # 注册处理器
    event_engine.register("MARKET_DATA", handle_market_data)
    event_engine.register("CRITICAL_ALERT", handle_critical_alert)
    
    print("正在发送不同优先级的事件...")
    
    # 同时放入队列，演示优先级排序
    tasks = [
        event_engine.put(Event("MARKET_DATA", "AAPL Price 230.1", priority=EventPriority.NORMAL)),
        event_engine.put(Event("MARKET_DATA", "TSLA Price 180.5", priority=EventPriority.NORMAL)),
        event_engine.put(Event("CRITICAL_ALERT", "保证金不足!", priority=EventPriority.CRITICAL)),
        event_engine.put(Event("MARKET_DATA", "NVDA Price 900.2", priority=EventPriority.HIGH)),
    ]
    await asyncio.gather(*tasks)
    
    # 等待处理完成
    await asyncio.sleep(1)
    await event_engine.stop()

async def main():
    # 1. 运行优化演示
    try:
        await run_optimization_demo()
    except Exception as e:
        print(f"优化演示失败 (可能是由于缺少数据): {e}")

    # 2. 运行事件演示
    await run_event_demo()

if __name__ == "__main__":
    asyncio.run(main())

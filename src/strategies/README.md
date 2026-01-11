# 策略系统使用指南

## 概述

策略系统使用统一的 v2 接口：`generate_signals()` 生成信号，`size_positions()` 输出订单意图。

## 开源示例策略

位于 [`src/strategies/example/`](example/__init__.py)：

| 策略ID | 名称 | 说明 |
|--------|------|------|
| `simple_momentum` | 简单动量 | 收益率排名，买入最强的 |
| `simple_ma` | 简单均线 | MA 交叉，金叉做多 |

```python
from src.strategies.example import SimpleMomentumStrategy, SimpleMAStrategy

# 动量策略
s1 = SimpleMomentumStrategy()

# 均线策略
s2 = SimpleMAStrategy()
```

## 快速开始

```python
import asyncio
from datetime import date
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.evaluation.backtesting.engine import BacktestEngine
from src.strategies.example import SimpleMomentumStrategy

async def run_backtest():
    strategy = SimpleMomentumStrategy()
    engine = BacktestEngine(initial_capital=100000)

    ctx = RunContext(
        env=Environment.BACKTEST,
        run_id="demo",
        trading_calendar=TradingCalendar(),
    )

    await engine.run(
        strategies=[strategy],
        universe=["SPY", "QQQ", "TLT"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        ctx=ctx,
    )

asyncio.run(run_backtest())
```

## 创建自定义策略

```python
from typing import List
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.core.types import Signal, OrderIntent, MarketData, PortfolioState, RiskState, ActionType, OrderSide, OrderType
from src.core.context import RunContext

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__(StrategyConfig(
            strategy_id="my_strategy",
            timeframe="1d",
            params={}
        ))

    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        return []  # 实现信号生成逻辑

    def size_positions(self, signals, portfolio, risk, ctx) -> List[OrderIntent]:
        return []  # 实现仓位计算逻辑
```

策略会自动注册，无需手动调用。

##

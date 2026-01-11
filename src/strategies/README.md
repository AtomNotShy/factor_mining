# 策略系统使用指南 (v2)

## 概述

策略系统使用统一的 v2 接口：`generate_signals()` 生成信号，`size_positions()` 输出订单意图。

## 目录结构

```
src/strategies/
├── __init__.py                # 策略模块入口/注册
├── base/
│   ├── __init__.py
│   └── strategy.py            # Strategy/StrategyConfig/strategy_registry
├── etf_momentum_us/           # ETF 动量策略
│   └── strategy.py
└── vwap/
    └── vwap_pullback_v2.py    # VWAP 回踩策略 (v2)
```

## 快速开始

```python
import asyncio
from datetime import date

from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.evaluation.backtesting.engine import BacktestEngine
from src.strategies.vwap.vwap_pullback_v2 import VWAPPullbackStrategyV2, VWAPPullbackParams


async def run_backtest():
    strategy = VWAPPullbackStrategyV2(VWAPPullbackParams())
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0005, slippage_rate=0.0002)

    ctx = RunContext(
        env=Environment.BACKTEST,
        run_id="manual_run",
        trading_calendar=TradingCalendar(),
    )

    result = await engine.run(
        strategies=[strategy],
        universe=["SPY"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        ctx=ctx,
        auto_download=True,
    )
    print(result.keys())


asyncio.run(run_backtest())
```

## 创建新策略

### 1) 定义参数

```python
from dataclasses import dataclass


@dataclass
class MyStrategyParams:
    lookback: int = 20
    position_cash_frac: float = 0.3
```

### 2) 实现策略类

```python
from typing import List
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.core.types import Signal, OrderIntent, MarketData, PortfolioState, RiskState, ActionType, OrderSide, OrderType
from src.core.context import RunContext


class MyStrategy(Strategy):
    def __init__(self, params: MyStrategyParams | None = None):
        if params is None:
            params = MyStrategyParams()
        config = StrategyConfig(
            strategy_id="my_strategy",
            timeframe="1d",
            params=params.__dict__,
        )
        super().__init__(config)
        self.params = params

    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        signals: List[Signal] = []
        # 根据 md.bars/ctx 生成信号
        return signals

    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        intents: List[OrderIntent] = []
        # 根据信号与组合状态生成订单意图
        return intents


# 默认会自动注册策略（继承 Strategy 即可）
```

## 最佳实践

1. **参数集中**: 将可调参数集中在 `StrategyConfig.params`（可用 dataclass 转 dict）
2. **数据校验**: 在 `generate_signals` 内验证 bar 数据是否足够
3. **清晰日志**: 使用 `get_logger("strategy.xxx")` 输出关键决策日志
4. **最小副作用**: 仅在 `size_positions` 返回意图，不直接修改组合状态

## 示例

- `src/strategies/etf_momentum_us/strategy.py` - ETF 动量轮动策略

## 回测

 python3 backtest_cli.py --strategy us_etf_momentum --symbols SPY,QQQ,IWM --start 2023-01-01 --end 2024-12-31

##



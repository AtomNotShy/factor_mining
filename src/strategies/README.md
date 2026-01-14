# 策略系统使用指南

## 目录结构

```
src/strategies/
├── base/               # 框架核心（用户不直接编辑）
│   ├── strategy.py         # Strategy 基类
│   ├── parameters.py       # 参数定义
│   ├── vectorized_strategy.py  # 向量化策略基类
│   └── informative.py      # 多时间框架支持
│
├── components/         # 可复用组件
│   ├── scorers.py      # 评分器（MomentumScorer, RSIScorer 等）
│   └── filters.py      # 过滤器（RangeFilter, DrawdownFilter 等）
│
├── examples/           # 教学示例（供学习参考）
│   ├── simple_momentum.py  # 简单动量策略
│   └── simple_ma.py        # 简单均线策略
│
└── user_strategies/    # 用户策略（用户编写处）
    ├── __init__.py          # 自动注册所有策略
    ├── etf_momentum_us.py   # ETF动量轮动策略
    └── your_strategy.py     # 你的新策略
```

## 策略分类

### 框架代码 (`base/`, `components/`)
**用户不应修改**，包含：
- [`Strategy`](base/strategy.py) - 所有策略的基类
- 参数系统和自动注册机制
- 多时间框架支持

### 教学示例 (`examples/`)
**供学习参考**，包含简单策略实现：
- `SimpleMomentumStrategy` - 简单动量排名
- `SimpleMAStrategy` - 均线交叉

### 用户策略 (`user_strategies/`)
**用户编写策略的目录**，框架会自动加载此目录下的策略。

## 导入示例

```python
# 导入用户策略
from src.strategies.user_strategies import USETFMomentumStrategy

# 导入示例策略
from src.strategies.example import SimpleMomentumStrategy, SimpleMAStrategy

# 导入框架基类
from src.strategies.base.strategy import Strategy
from src.strategies.base.parameters import IntParameter, DecimalParameter
```

## 快速开始

```python
import asyncio
from datetime import date
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine, UnifiedConfig, TradeConfig
from src.strategies.user_strategies import USETFMomentumStrategy

async def run_backtest():
    strategy = USETFMomentumStrategy()
    config = UnifiedConfig(
        trade=TradeConfig(initial_capital=100000),
    )
    engine = UnifiedBacktestEngine(config=config)

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

将新策略放在 `user_strategies/` 目录下：

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

# 使用指南

本文档介绍如何使用新的量化系统进行策略回测和评估。

## 目录

1. [快速开始](#快速开始)
2. [核心概念](#核心概念)
3. [策略开发](#策略开发)
4. [回测流程](#回测流程)
5. [评估与Gate](#评估与gate)
6. [Paper Trading](#paper-trading)
7. [API使用](#api使用)

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（.env文件）
POLYGON_API_KEY=your_api_key
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=factor_mining
```

### 2. 初始化数据库

```python
from src.data.storage.db_store import DatabaseStore

# 自动创建表结构
db_store = DatabaseStore()
```

### 3. 运行简单回测

```python
from datetime import date
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.evaluation.backtesting.engine_v2 import BacktestEngine
from src.strategies.vwap.vwap_pullback_v2 import VWAPPullbackStrategyV2, VWAPPullbackParams

# 创建运行上下文
calendar = TradingCalendar()
ctx = RunContext.create(env=Environment.RESEARCH, trading_calendar=calendar)

# 创建策略
params = VWAPPullbackParams()
strategy = VWAPPullbackStrategyV2(params)

# 运行回测
engine = BacktestEngine(initial_capital=100000.0)
result = engine.run(
    strategies=[strategy],
    universe=["SPY"],
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    ctx=ctx,
)

print(f"总收益: {result['total_return']:.2%}")
```

## 核心概念

### 运行上下文 (RunContext)

`RunContext` 包含运行时的所有上下文信息：

- `env`: 运行环境（research/paper/live）
- `code_version`: 代码版本（git commit SHA）
- `data_version`: 数据版本
- `config_hash`: 配置hash（用于可重放）
- `trading_calendar`: 交易日历

### 领域模型

- **Signal**: 策略生成的交易信号
- **OrderIntent**: 订单意图（策略层输出）
- **Fill**: 成交记录
- **MarketData**: 市场数据容器
- **PortfolioState**: 组合状态

### 版本化

所有数据产物都带有版本信息：

- `data_version`: 数据拉取批次标识
- `code_version`: 代码版本
- `config_hash`: 配置hash

这确保了结果的可重放性。

## 策略开发

### 实现策略接口

```python
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.core.types import Signal, OrderIntent, MarketData, PortfolioState, RiskState
from src.core.context import RunContext

class MyStrategy(Strategy):
    def __init__(self):
        config = StrategyConfig(
            strategy_id="my_strategy",
            timeframe="1d",
        )
        super().__init__(config)
    
    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        """生成信号"""
        signals = []
        # 实现信号生成逻辑
        return signals
    
    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        """计算目标仓位"""
        order_intents = []
        # 实现仓位计算逻辑
        return order_intents
```

## 回测流程

### 1. 数据准备

```python
from src.data.collectors.polygon import PolygonCollector

collector = PolygonCollector()
await collector.connect()

bars = await collector.get_ohlcv(
    symbol="SPY",
    timeframe="1d",
    since=datetime(2024, 1, 1),
)
```

### 2. 运行回测

```python
from src.evaluation.backtesting.engine_v2 import BacktestEngine

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
```

### 3. 查看结果

回测结果包含：

- `signals`: 所有信号
- `orders`: 所有订单
- `fills`: 所有成交
- `portfolio_daily`: 每日组合状态
- `run_id`: 运行ID（用于查询）

结果自动保存到Parquet和数据库。

## 评估与Gate

### 性能指标计算

```python
from src.evaluation.metrics.performance import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
returns = portfolio_daily['daily_return']
metrics = analyzer.comprehensive_analysis(returns)

print(f"CAGR: {metrics['cagr']:.2%}")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Max DD: {metrics['max_drawdown']:.2%}")
```

### Walk-Forward分析

```python
from src.evaluation.walk_forward import WalkForwardAnalyzer

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
```

### Gate准入检查

```python
from src.evaluation.walk_forward import Gate

gate = Gate()
passed, failures = gate.pass_gate(metrics, robustness)

if passed:
    print("策略通过Gate准入")
else:
    print(f"未通过: {failures}")
```

## Paper Trading

### 使用Paper Broker

```python
from src.execution.paper import PaperBroker

broker = PaperBroker(initial_cash=100000.0)

# 提交订单
order_ids = broker.place_orders(order_intents, ctx)

# 轮询成交
fills = broker.poll_fills(ctx, current_prices={"SPY": 450.0})

# 获取持仓
portfolio = broker.get_positions(ctx)
```

### 监控告警

```python
from src.monitoring.alerts import AlertManager

alert_manager = AlertManager(
    daily_loss_limit=0.05,
    consecutive_loss_limit=5,
)

# 检查告警
alert = alert_manager.check_daily_loss(portfolio, initial_equity, timestamp)
if alert:
    print(f"告警: {alert.message}")
```

## API使用

### 运行回测

```bash
curl -X POST http://localhost:8000/api/v1/strategy-backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "vwap_pullback",
    "symbol": "SPY",
    "timeframe": "1d",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000
  }'
```

### 查询回测运行

```bash
# 列出所有运行
curl http://localhost:8000/api/v1/strategy-backtest/runs

# 获取运行详情
curl http://localhost:8000/api/v1/strategy-backtest/runs/{run_id}
```

## 最佳实践

1. **版本化**: 始终使用版本化的数据和配置
2. **可重放**: 确保相同输入产生相同输出
3. **成本模型**: 回测中始终启用成本模型
4. **Gate检查**: 生产前必须通过Gate准入
5. **监控**: Paper Trading时启用监控告警

## 更多示例

参见 `examples/usage_example.py` 获取完整示例代码。

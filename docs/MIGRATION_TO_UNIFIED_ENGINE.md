# UnifiedBacktestEngine 迁移指南

## 概述

`UnifiedBacktestEngine` 是新的统一回测引擎，整合了以下功能：
- 原 `BacktestEngine` 的核心功能
- `FreqtradeBacktestEngine` 的 Freqtrade 协议支持
- `BacktestEventEngine` 的事件驱动模式

## 为什么要迁移？

1. **单一入口**：不再需要选择使用哪个引擎
2. **插件式特性**：按需启用功能，避免冗余
3. **统一配置**：所有配置通过 `UnifiedConfig` 管理
4. **更好的维护**：单一代码库，降低维护成本

## 快速迁移

### 方式1：使用默认配置（推荐）

**旧代码：**
```python
from src.evaluation.backtesting import BacktestEngine, EngineConfig

config = EngineConfig(
    commission_rate=0.001,
    clock_mode="daily",
)
engine = BacktestEngine(config=config)
```

**新代码：**
```python
from src.evaluation.backtesting import UnifiedBacktestEngine, UnifiedConfig

# 使用默认配置
engine = UnifiedBacktestEngine()

# 或自定义配置
config = UnifiedConfig(
    trade=TradeConfig(commission_rate=0.001),
    time=TimeConfig(clock_mode="daily"),
)
engine = UnifiedBacktestEngine(config=config)
```

### 方式2：Freqtrade 风格

**旧代码：**
```python
from src.evaluation.backtesting import FreqtradeBacktestEngine, FreqtradeBacktestConfig

config = FreqtradeBacktestConfig(
    dry_run_wallet=100000,
    timeframe="1h",
    stoploss=-0.05,
    trailing_stop=True,
)
engine = FreqtradeBacktestEngine(config)
```

**新代码：**
```python
from src.evaluation.backtesting import (
    UnifiedBacktestEngine, 
    UnifiedConfig,
    FeatureFlag,
    StoplossConfig,
)

config = UnifiedConfig(
    trade=TradeConfig(initial_capital=100000),
    time=TimeConfig(signal_timeframe="1h"),
    stoploss=StoplossConfig(stoploss=-0.05, trailing_stop=True),
    features=FeatureFlag.STOPLOSS_MANAGER | FeatureFlag.VECTORIZED,
)
engine = UnifiedBacktestEngine(config=config)
```

### 方式3：启用所有特性

```python
from src.evaluation.backtesting import UnifiedBacktestEngine, UnifiedConfig, FeatureFlag

config = UnifiedConfig(
    features=FeatureFlag.ALL,  # 启用所有特性
)
engine = UnifiedBacktestEngine(config=config)
```

## 配置映射

| 旧配置 | 新配置 | 说明 |
|--------|--------|------|
| `EngineConfig.commission_rate` | `UnifiedConfig.trade.commission_rate` | 手续费率 |
| `EngineConfig.slippage_rate` | `UnifiedConfig.trade.slippage_rate` | 滑点率 |
| `EngineConfig.clock_mode` | `UnifiedConfig.time.clock_mode` | 时钟模式 |
| `EngineConfig.max_position_size` | `UnifiedConfig.trade.max_position_size` | 最大仓位 |
| `FreqtradeBacktestConfig.dry_run_wallet` | `UnifiedConfig.trade.initial_capital` | 初始资金 |
| `FreqtradeBacktestConfig.stoploss` | `UnifiedConfig.stoploss.stoploss` | 止损比例 |
| `FreqtradeBacktestConfig.timeframe` | `UnifiedConfig.time.signal_timeframe` | 时间框架 |

## 特性开关

```python
from src.evaluation.backtesting import FeatureFlag

# 只启用向量化预计算
config = UnifiedConfig(features=FeatureFlag.VECTORIZED)

# 启用向量化 + 止损管理
config = UnifiedConfig(
    features=FeatureFlag.VECTORIZED | FeatureFlag.STOPLOSS_MANAGER
)

# 启用所有特性
config = UnifiedConfig(features=FeatureFlag.ALL)
```

### 可用特性

| 特性 | 说明 |
|------|------|
| `VECTORIZED` | 向量化预计算（Freqtrade 协议） |
| `FREQTRADE_PROTOCOL` | Freqtrade 协议支持 |
| `STOPLOSS_MANAGER` | 止损/ROI 管理 |
| `TRAILING_STOP` | 追踪止损 |
| `PROTECTIONS` | 保护机制（冷却期等） |
| `EVENT_DRIVEN` | 事件驱动模式 |
| `MULTI_TIMEFRAME` | 多时间框架支持 |
| `DETAIL_TIMEFRAME` | Detail 时间框架 |
| `SAVE_RESULTS` | 保存回测结果 |

## 完整示例

```python
import asyncio
from datetime import date
from src.evaluation.backtesting import (
    UnifiedBacktestEngine,
    UnifiedConfig,
    FeatureFlag,
    TradeConfig,
    TimeConfig,
    StoplossConfig,
    ProtectionConfig,
)
from src.strategies.example import SimpleMomentumStrategy


async def main():
    # 创建配置
    config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=100000,
            commission_rate=0.001,
            max_position_size=0.2,
        ),
        time=TimeConfig(
            signal_timeframe="1d",
            clock_mode="daily",
            warmup_days=260,
        ),
        stoploss=StoplossConfig(
            stoploss=-0.05,
            trailing_stop=True,
            minimal_roi={0: 0.05, "60": 0.03},
        ),
        protection=ProtectionConfig(
            enabled=True,
            max_drawdown=0.15,
        ),
        features=FeatureFlag.ALL,
    )
    
    # 创建引擎
    engine = UnifiedBacktestEngine(config=config)
    
    # 运行回测
    result = await engine.run(
        strategies=[SimpleMomentumStrategy()],
        universe=["AAPL", "MSFT", "GOOGL"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
    )
    
    # 输出结果
    print(f"总收益率: {result.total_return_pct:.2f}%")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"交易次数: {result.total_trades}")
    print(f"胜率: {result.win_rate:.2f}%")
    

if __name__ == "__main__":
    asyncio.run(main())
```

## 回测结果

```python
# 访问结果
result = await engine.run(...)

# 基本指标
print(result.run_id)           # 回测 ID
print(result.strategy_name)    # 策略名称
print(result.total_return_pct) # 总收益率 (%)
print(result.annualized_return)# 年化收益率
print(result.max_drawdown_pct) # 最大回撤 (%)
print(result.sharpe_ratio)     # 夏普比率
print(result.win_rate)         # 胜率 (%)

# 转换为字典 (API 返回格式)
result_dict = result.to_dict()

# 详细数据
print(result.signals)      # 信号列表
print(result.orders)       # 订单列表
print(result.fills)        # 成交列表
print(result.portfolio_daily)  # 每日净值
```

## 兼容性

旧 API 仍然可用：

```python
# 旧 API - 仍然支持
from src.evaluation.backtesting import BacktestEngine, EngineConfig

config = EngineConfig(...)
engine = BacktestEngine(config=config)
```

## 常见问题

### Q: 我应该使用哪个引擎？

**A:** 新项目推荐使用 `UnifiedBacktestEngine`，旧项目可以继续使用 `BacktestEngine`。

### Q: 如何迁移现有的回测代码？

**A:**
1. 将 `EngineConfig` 替换为 `UnifiedConfig`
2. 将 `FreqtradeBacktestConfig` 替换为 `UnifiedConfig` + `StoplossConfig`
3. 根据需要启用/禁用特性

### Q: 性能有影响吗？

**A:** 不会。禁用的特性不会产生额外开销。

### Q: 为什么不直接删除旧引擎？

**A:** 为了给用户足够的迁移时间。计划在下一个主要版本中删除旧引擎。

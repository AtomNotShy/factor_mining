# 聚宽ETF轮动策略转换说明

## 概述

已将聚宽（JoinQuant）上的ETF轮动策略成功转换为针对美股ETF的策略，并集成到当前因子挖掘系统中。

## 原始策略来源

1. **主策略**：克隆自聚宽文章《【ETF轮动策略】年化163%，回撤7%》
   - 作者：zfs1
   - 链接：https://www.joinquant.com/post/62821

2. **改进版本**：参考《【思路分享】动量ETF轮动之基于历史波动率动态调整历史回溯期》
   - 作者：0xtao
   - 链接：https://www.joinquant.com/post/60824

## 策略核心逻辑

### 1. 动量评分计算
- **公式**：`score = 年化收益率 × R²`
- **计算方法**：加权线性回归（近期权重更大）
- **年化收益率**：基于回归斜率计算
- **R²**：拟合优度，衡量趋势的稳定性

### 2. 动态回溯期调整
- **基础参数**：最小20天，最大60天，默认25天
- **调整逻辑**：基于ATR波动率动态调整
  - 波动率高 → 缩短lookback（快速反应）
  - 波动率低 → 延长lookback（捕捉趋势）

### 3. 风险过滤
实现聚宽策略的3种风险过滤条件：
1. **短期大幅下跌**：3天内有一天跌超5%
2. **连续下跌1**：3天连跌且总跌幅超4%
3. **连续下跌2**：4天连跌且总跌幅超4%

### 4. 交易逻辑
- **调仓频率**：每天调仓
- **持仓数量**：持有排名最高的1只ETF
- **资金分配**：等权重配置

## 转换成果

### 1. 策略落地位置
- **位置**：`src/strategies/etf_momentum_us/strategy.py`
- **类名**：`USETFMomentumStrategy`
- **说明**：已将 JoinQuant 逻辑合并为 v2 策略实现

### 2. 美股ETF池
默认包含以下美股ETF：
```python
DEFAULT_US_ETF_POOL = [
    # 美股大盘
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "IWM",    # Russell 2000
    "VTI",    # Total Stock Market
    "VOO",    # S&P 500 (Vanguard)
    "DIA",    # Dow Jones
    # 债券
    "TLT",    # 20+ Year Treasury
    "IEF",    # 7-10 Year Treasury
    "SHY",    # 1-3 Year Treasury
    # 商品
    "GLD",    # Gold
    "SLV",    # Silver
    "USO",    # Oil
    "UNG",    # Natural Gas
    # 其他
    "VNQ",    # REITs
]
```

### 3. 测试文件
1. **单元测试**：`test_joinquant_strategy.py`
   - 测试策略初始化
   - 测试风险过滤器
   - 测试动量计算
   - 测试回溯期调整
   - 测试策略函数

2. **回测示例**：`examples/joinquant_etf_backtest.py`
   - 完整的回测流程
   - 数据加载和策略执行

## 使用方法

### 1. 导入策略
```python
from src.strategies.etf_momentum_us.strategy import USETFMomentumStrategy
```

### 2. 创建策略实例
```python
strategy = USETFMomentumStrategy()
strategy.config.params.update(
    {
        "etf_pool": ["SPY", "QQQ", "IWM", "TLT", "GLD"],
        "target_positions": 1,
        "auto_adjust_lookback": False,
        "default_lookback_days": 25,
    }
)
```

### 3. 运行回测
```python
from datetime import date
from src.core.calendar import TradingCalendar
from src.core.context import RunContext, Environment
from src.evaluation.backtesting.engine import BacktestEngine

engine = BacktestEngine(
    initial_capital=100000.0,
    commission_rate=0.0002,
    slippage_rate=0.001,
)
ctx = RunContext.create(env=Environment.RESEARCH, trading_calendar=TradingCalendar())

results = await engine.run(
    strategies=[strategy],
    universe=strategy.config.params["etf_pool"],
    start=date(2023, 1, 1),
    end=date(2024, 12, 31),
    ctx=ctx,
    auto_download=True,
)
```

## 与现有策略的差异

### 聚宽策略 vs 现有ETF动量策略

| 特性 | 聚宽策略 | 现有策略 |
|------|----------|----------|
| **评分公式** | `年化收益 × R²` | `(年化收益/波动率) × R²` |
| **风险过滤** | 3种严格条件 | 1种条件 |
| **ETF溢价率** | 支持（可选） | 不支持 |
| **调仓频率** | 每天 | 可配置 |
| **数据需求** | 收盘价、高、低 | 收盘价、高、低 |

## 注意事项

1. **数据需求**：策略需要日频的OHLC数据
2. **ETF溢价率**：美股ETF溢价率通常较低，默认关闭此功能
3. **风险规避**：当没有ETF通过筛选时，可配置风险规避标的（默认：SHY）
4. **参数调优**：可根据市场环境调整风险过滤阈值和评分范围

## 验证结果

所有单元测试已通过：
- ✅ 信号生成测试
- ✅ 订单意图测试

## 后续建议

1. **实盘测试**：在模拟环境中进行更长时间的回测
2. **参数优化**：对风险阈值、评分范围等参数进行优化
3. **数据源扩展**：考虑添加更多类型的ETF（行业ETF、国际ETF等）
4. **功能增强**：添加止损、仓位管理等功能

## 文件清单

1. `src/strategies/etf_momentum_us/strategy.py` - 策略实现
2. `test_joinquant_strategy.py` - 单元测试（v2）
3. `examples/joinquant_etf_backtest.py` - 回测示例（v2）
4. `聚宽ETF轮动策略转换说明.md` - 本文档

---
**完成时间**：2026年1月10日  
**版本**：1.0.0  
**状态**：已通过所有单元测试，可集成使用

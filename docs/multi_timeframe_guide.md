# 多时间框架策略开发指南

本指南说明如何在factor_mining系统中使用多个时间框架开发策略。

---

## 概述

多时间框架策略是指在不同的时间周期上分析市场数据，例如：
- 主时间框架：1小时
- 辅助时间框架：4小时（用于判断趋势）、15分钟（用于寻找入场点）
- 目标：利用不同时间框架的信息组合，提高交易决策质量

---

## 快速开始

```python
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.strategies.base.informative import informative
import pandas_ta as ta
import pandas as pd
import numpy as np

class MyMultiTfStrategy(Strategy):
    def __init__(self):
        super().__init__(StrategyConfig(
            strategy_id="my_multi_tf",
            timeframe="1h",  # 主时间框架
            params={}
        ))
        
        # 声明需要的informative时间框架
        self.informative_timeframes = ['4h', '15m', '1d']
    
    @informative('4h')
    def populate_indicators_4h(self, dataframe, metadata):
        # 4小时数据：判断大趋势
        dataframe['ema_20'] = ta.ema(dataframe['close'], length=20)
        dataframe['ema_50'] = ta.ema(dataframe['close'], length=50)
        dataframe['trend'] = np.where(
            dataframe['ema_20'] > dataframe['ema_50'], 1, -1
        )
        return dataframe
    
    @informative('15m')
    def populate_indicators_15m(self, dataframe, metadata):
        # 15分钟数据：寻找入场信号
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        dataframe['bb_upper'] = ta.bbands(dataframe['close'], length=20)['BBU_20_2.0']
        dataframe['bb_lower'] = ta.bbands(dataframe['close'], length=20)['BBL_20_2.0']
        return dataframe
    
    @informative('1d')
    def populate_indicators_1d(self, dataframe, metadata):
        # 1天数据：长期趋势
        dataframe['ema_200'] = ta.ema(dataframe['close'], length=200)
        return dataframe
    
    def populate_indicators(self, dataframe, metadata):
        # 主时间框架（1h）指标
        dataframe['atr'] = ta.atr(
            dataframe['high'], dataframe['low'], dataframe['close'], length=14
        )
        
        # 访问informative数据（已自动对齐到主时间框架）
        dataframe['trend_4h'] = self.get_informative_pair('4h', 'trend')
        dataframe['rsi_15m'] = self.get_informative_pair('15m', 'rsi')
        dataframe['ema_1d'] = self.get_informative_pair('1d', 'ema_200')
        
        return dataframe
    
    def populate_entry_trend(self, dataframe, metadata):
        # 多时间框架进场信号
        # 条件1: 4小时趋势向上
        # 条件2: 15分钟RSI超卖
        # 条件3: 1天EMA向上（长期趋势）
        
        condition_trend = dataframe['trend_4h'] > 0
        condition_rsi = dataframe['rsi_15m'] < 30
        condition_ema = dataframe['ema_1d'] > dataframe['ema_1d'].shift(1)
        
        dataframe['enter_long'] = np.where(
            condition_trend & condition_rsi & condition_ema,
            1, 0
        )
        
        return dataframe
    
    def generate_signals(self, md, ctx):
        # 使用向量化结果生成信号（由Engine处理）
        return []
    
    def size_positions(self, signals, portfolio, risk, ctx):
        # 仓位管理
        orders = []
        
        for signal in signals:
            if signal.action == ActionType.LONG:
                target_value = portfolio.equity / len(signals)
                qty = target_value / md.latest_price(signal.symbol)
                
                orders.append(OrderIntent(
                    ts_utc=signal.ts_utc,
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    qty=qty,
                    order_type=OrderType.LMT,
                    limit_price=md.latest_price(signal.symbol),
                    strategy_id=self.strategy_id,
                ))
        
        return orders
```

---

## 核心功能

### 1. @informative装饰器

使用装饰器标记辅助时间框架的方法：

```python
@informative('4h')
def populate_indicators_4h(self, dataframe, metadata):
    dataframe['ema'] = ta.ema(dataframe['close'], length=20)
    return dataframe
```

**装饰器参数：**
- `timeframe`: 辅助时间框架字符串（如'1h', '4h', '1d', '5m', '15m'等）
- `asset`: 可选，跨资产时间框架（如'BTC/USDT'）
- `ffill`: 可选，是否前向填充（默认True）

**工作机制：**
1. 装饰器标记方法为informative方法
2. 策略类通过`get_informative_methods()`获取所有informative方法
3. BacktestEngine自动调用informative方法，传入对应时间框架的数据
4. 计算结果自动合并到主时间框架DataFrame

### 2. 访问informative数据

在主时间框架的方法中访问informative数据：

```python
def populate_indicators(self, dataframe, metadata):
    # 方法1：通过self.get_informative_pair()
    trend_4h = self.get_informative_pair('4h', 'trend')
    
    # 方法2：列名格式为 "{column}_{timeframe}"
    dataframe['rsi_1h'] = dataframe['rsi_1h']  # 直接从合并后的DataFrame访问
    
    return dataframe
```

**列名格式：**
- `{indicator}_{timeframe}` - 例如：`trend_4h`, `rsi_1h`, `ema_4h`

### 3. 数据合并

**自动合并（由BacktestEngine处理）：**
- 辅助时间框架的指标自动合并到主时间框架
- 列名自动添加时间框架后缀
- 自动前向填充使低频数据在每根bar上可用

**前视偏差保护：**
- 自动检测informative数据是否包含未来数据
- 如果检测到，会发出警告

---

## 配置回测引擎

在`EngineConfig`中指定需要的时间框架：

```python
from src.evaluation.backtesting.config import EngineConfig

config = EngineConfig(
    signal_timeframe="1h",        # 主时间框架
    informative_timeframes=["4h", "15m", "1d"],  # 辅助时间框架
    clock_mode="daily",              # 回测模式
    commission_rate=0.001,
    slippage_rate=0.0005,
)
```

**参数说明：**
- `signal_timeframe`: 主时间框架，用于信号生成和订单执行
- `informative_timeframes`: 辅助时间框架列表，用于指标计算
- 引擎会自动加载所有指定的时间框架数据

---

## 策略示例

### 示例1：多时间框架动量策略

参见`src/strategies/example/multi_tf_momentum.py`

**特点：**
- 4小时时间框架判断大趋势
- 15分钟时间框架寻找入场信号
- 1小时主时间框架执行交易

**信号逻辑：**
```python
# 长期趋势向上
trend_4h > 0

# 中期超卖（15m RSI < 30）
rsi_15m < 30

# 长期上升（1日 EMA > 前一日）
ema_1d > ema_1d.shift(1)
```

### 示例2：日内交易策略

```python
class IntradayStrategy(Strategy):
    def __init__(self):
        super().__init__(StrategyConfig(
            strategy_id="intraday",
            timeframe="5m",
            params={}
        ))
        
        self.informative_timeframes = ['15m', '1h']
    
    @informative('15m')
    def populate_indicators_15m(self, dataframe, metadata):
        # 超短周期指标
        dataframe['vwap'] = ta.vwap(dataframe)
        return dataframe
    
    @informative('1h')
    def populate_indicators_1h(self, dataframe, metadata):
        # 确认趋势
        dataframe['ema'] = ta.ema(dataframe['close'], length=200)
        return dataframe
    
    def populate_indicators(self, dataframe, metadata):
        # 5分钟主时间框架
        dataframe['volume_sma'] = ta.sma(dataframe['volume'], length=20)
        
        # 访问informative数据
        dataframe['vwap_15m'] = self.get_informative_pair('15m', 'vwap')
        dataframe['trend_1h'] = self.get_informative_pair('1h', 'ema')
        
        # 进场条件：趋势向上 + 15分钟VWAP偏离 + 成交量萎缩
        dataframe['enter_long'] = np.where(
            (dataframe['trend_1h'] > 0) &
            (abs(dataframe['vwap_15m'] - dataframe['vwap']) > dataframe['vwap'] * 0.005) &
            (dataframe['volume'] < dataframe['volume_sma']),
            1, 0
        )
        
        return dataframe
```

---

## 常见使用模式

### 模式1：多时间框架趋势确认

**场景：** 使用多个时间框架确认同一趋势

```python
@informative('4h')
def populate_indicators_4h(self, dataframe, metadata):
    dataframe['ema_20'] = ta.ema(dataframe['close'], length=20)
    dataframe['ema_50'] = ta.ema(dataframe['close'], length=50)
    dataframe['trend'] = np.where(
        dataframe['ema_20'] > dataframe['ema_50'], 1, -1
    )
    return dataframe

@informative('1d')
def populate_indicators_1d(self, dataframe, metadata):
    dataframe['ema'] = ta.ema(dataframe['close'], length=200)
    return dataframe

def populate_indicators(self, dataframe, metadata):
    # 只有当4小时和1天趋势都向上时才进场
    dataframe['trend_4h'] = self.get_informative_pair('4h', 'trend')
    dataframe['trend_1d'] = self.get_informative_pair('1d', 'ema')
    
    dataframe['enter_long'] = np.where(
        (dataframe['trend_4h'] > 0) &
        (dataframe['trend_1d'] > 0),
        1, 0
    )
    
    return dataframe
```

### 模式2：多时间框架组合信号

**场景：** 使用不同时间框架的指标组合生成信号

```python
def populate_indicators(self, dataframe, metadata):
    # 访问informative数据
    rsi_15m = self.get_informative_pair('15m', 'rsi')
    macd_4h = self.get_informative_pair('4h', 'macd')
    ema_1d = self.get_informative_pair('1d', 'ema')
    
    # 组合信号条件
    # 条件1: 短期RSI超卖
    # 条件2: 中期MACD金叉（4h）
    # 条件3: 长期趋势向上（1d）
    
    dataframe['enter_long'] = np.where(
        (rsi_15m < 30) &
        (macd_4h > macd_4h.shift(1)) &
        (ema_1d > ema_1d.shift(1)),
        1, 0
    )
    
    return dataframe
```

### 模式3：时间框架过滤

**场景：** 根据不同时间框架的条件过滤

```python
def populate_entry_trend(self, dataframe, metadata):
    # 1小时主时间框架
    atr_1h = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
    
    # 访问informative数据
    volatility_4h = self.get_informative_pair('4h', 'atr')  # 4小时波动率
    volatility_1d = self.get_informative_pair('1d', 'atr')   # 1天波动率
    
    # 进场条件：短期低波动 + 长期低波动 + ATR过滤
    dataframe['enter_long'] = np.where(
        (dataframe['atr'] < dataframe['atr'].rolling(20).mean()) &  # 当前ATR低于20日均值
        (volatility_1d < volatility_1d.rolling(20).mean()) &  # 长期波动率低
        (dataframe['atr'] < dataframe['atr'] * 0.8),  # ATR小于当日ATR的80%
        1, 0
    )
    
    return dataframe
```

---

## 工具函数

### resample_to_interval

时间框架重采样：

```python
from src.strategies.base.informative import resample_to_interval

# 重采样到1小时
df_1h = resample_to_interval(df_5m, 60)
```

### merge_informative_pair

合并informative数据到主时间框架：

```python
from src.strategies.base.informative import merge_informative_pair

# 自动合并、列重命名、前向填充
merged = merge_informative_pair(df_5m, df_1h, '5m', '1h', ffill=True)
```

---

## 测试

运行单元测试：

```bash
python3 tests/test_informative.py -v
```

测试覆盖：
- 时间框架转换
- 时间框架重采样
- @informative装饰器
- informative数据合并
- 批量informative合并

---

## 最佳实践

1. **时间框架选择**
   - 遵循原则：高频用于信号生成，低频用于趋势判断
   - 常见组合：5m主 + 15m/1h/4h辅助

2. **前视偏差避免**
   - 所有informative数据自动对齐，防止未来数据泄露
   - 不在informative方法中使用主时间框架的值

3. **性能优化**
   - informative方法使用向量化计算
   - 避免重复计算
   - 使用缓存

4. **错误处理**
   - 检查informative数据是否存在
   - 处理缺失值（NaN）
   - 验证时间框架支持

5. **调试**
   - 使用logger输出informative数据加载信息
   - 打印合并后的列名
   - 检查时间框架对齐

---

## API支持

策略接口已自动集成多时间框架支持，无需额外配置：

```python
# 策略类会自动处理informative方法
# BacktestEngine会自动加载informative_timeframes指定的数据
# 数据自动合并并传递给策略的populate_indicators方法
```

---

## 与Freqtrade对比

| 特性 | Freqtrade | Factor Mining |
|------|----------|----------------|
| @informative | ✅ 支持 | ✅ 支持 |
| 时间框架格式 | '1h', '15m' | '1h', '15m' |
| 数据合并 | merge_informative_pair | merge_informative_pair |
| 前向填充 | 自动 | 自动 |
| 前视检测 | 内置 | 内置 |
| 策略接口 | IStrategy | Strategy |
| 配置方式 | config.json | EngineConfig |

---

## 故障排除

**问题：informative数据全是NaN**
- 检查informative方法是否被正确调用
- 验证时间框架数据是否正确加载
- 检查列名是否正确（应为`{column}_{timeframe}`）

**问题：时间框架不匹配**
- 验证EngineConfig.informative_timeframes配置
- 检查策略的informative_timeframes声明
- 检查数据加载日志

**问题：合并失败**
- 检查主时间框架和informative时间框架的日期范围
- 验证pandas DataFrame结构
- 查看错误日志

---

## 相关资源

- 策略基类：`src/strategies/base/strategy.py`
- Informative工具：`src/strategies/base/informative.py`
- 回测引擎：`src/evaluation/backtesting/engine.py`
- 配置类：`src/evaluation/backtesting/config.py`
- 示例策略：`src/strategies/example/multi_tf_momentum.py`
- 单元测试：`tests/test_informative.py`

---

## 更新日志

- **2025-01-13**: 初始版本
  - 实现核心@informative装饰器
  - 实现merge_informative_pair函数
  - 扩展Strategy基类
  - 扩展EngineConfig
  - 修改BacktestEngine集成
  - 创建多时间框架策略示例
  - 编写完整单元测试
  - 添加使用文档

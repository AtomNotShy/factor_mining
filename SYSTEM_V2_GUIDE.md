# Factor Mining Framework V2 (Freqtrade-Style) 使用指南

本系统是对原框架的重大升级，采用了类似 **Freqtrade** 的“向量化预计算 + 事件驱动执行”混合架构，旨在提供高性能、可配置、且易于实盘扩展的美股交易环境。

---

## 1. 环境准备

### 推荐环境
- **Python**: 3.10 + (推荐 3.12)
- **OS**: macOS / Linux

### 安装依赖
由于系统引入了 `pydantic`, `pandas-ta`, `optuna`, `plotly` 和 `sqlalchemy`，请确保安装最新依赖：

```bash
# 进入虚拟环境后运行
pip install pandas pandas-ta pydantic pyyaml plotly jinja2 optuna sqlalchemy
```

### 解决 `ModuleNotFoundError`
运行命令时，Python 需要知道根目录在哪。请始终在项目根目录下执行，并设置 `PYTHONPATH`：

```bash
export PYTHONPATH=$PYTHONPATH:.
```

---

## 2. 核心配置文件 (`config.yaml`)

不再需要修改 Python 脚本来调整参数。所有配置都统一在 YAML 文件中：

```yaml
trading:
  stake_amount: 100000    # 初始资金
  max_open_trades: 5      # 最大同时持仓
  timeframe: "1d"         # 周期

broker:
  name: "simulated"       # 模拟成交
  commission: 0.001       # 佣金
  slippage: 0.0005        # 滑点

data:
  datadir: "./data"
  startup_candle_count: 200 # 自动多加载 200 根线用于计算指标

strategy:
  name: "VectorizedRSISignal" # 策略类名
  params:
    rsi_period: 14        # 策略自定义参数
```

---

## 3. 运行回测 (Backtest)

通过统一的 CLI 入口运行回测：

```bash
python src/main.py backtest -c config.example.yaml
```

### 输出产物：
1. **控制台摘要**：最终净值、总收益率、夏普比率等。
2. **可视化报告**：在 `./reports/` 目录下生成交互式 HTML 报告（基于 Plotly）。
3. **数据库记录**：所有成交记录自动保存到 `./data/trades.db` (SQLite)。

---

## 4. 自动参数调参 (Hyperopt)

系统集成贝叶斯优化，自动寻找最优参数：

```bash
# 运行 50 次试验，目标是最大化夏普比率
python src/main.py hyperopt -c config.example.yaml -n 50 --metric sharpe_ratio
```

**如何定义寻优范围？**
在策略类中实现 `hyperopt_space` 方法：
```python
def hyperopt_space(self):
    return {
        "rsi_period": ("int", 7, 30),
        "rsi_lower": ("int", 20, 40)
    }
```

---

## 5. 策略开发指南

### 向量化计算
新策略必须继承 `Strategy` 基类，并实现向量化方法，这能提高 10-100 倍的速度：

```python
class MyStrategy(Strategy):
    def populate_indicators(self, dataframe, metadata):
        # 使用 self.ta 直接调用 pandas-ta 指标
        dataframe['rsi'] = self.ta.rsi(dataframe['close'], length=14)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        # 向量化生成买入信号
        dataframe['enter_long'] = (dataframe['rsi'] < 30).astype(int)
        return dataframe
```

### 自动数据预热
如果你需要计算 MA200，只需在配置中设置 `startup_candle_count: 200`，系统会自动多加载历史数据，确保你回测的第一天就有准确的指标。

---

## 6. 常见问题排查

- **Q: 提示找不到 `src` 模块？**
  - A: 运行 `export PYTHONPATH=.`。
- **Q: 数据库在哪里看？**
  - A: 使用任何 SQLite 浏览器打开 `./data/trades.db`。
- **Q: 怎么查看图形化报告？**
  - A: 直接用浏览器打开 `./reports/` 下的 HTML 文件，它是完全交互式的。

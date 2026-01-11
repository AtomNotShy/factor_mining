# 策略回测 CLI 使用指南（v2）

本项目提供 `backtest_cli.py` 用于在命令行运行策略回测，并输出专业的 Performance Summary/Trades 报告。

---

## 1. 前置条件

- Python 3（建议使用虚拟环境）
- 已安装依赖：`pip install -r requirements.txt`

可选（建议）：
- 已配置 IB 数据源（用于缺数据时自动补齐）

---

## 2. 快速开始

### 2.1 查看帮助

```bash
python3 backtest_cli.py --help
```

### 2.2 列出可用策略

```bash
python3 backtest_cli.py --list-strategies
```

### 2.3 回测一个策略（最常用）

```bash
python3 backtest_cli.py \
  --strategy us_etf_momentum \
  --symbols SPY,QQQ,IWM \
  --start 2023-01-01 \
  --end 2024-12-31
```

---

## 3. 参数说明

### 3.1 核心参数

- `--strategy`：策略名（如 `us_etf_momentum`）
- `--symbols`：回测标的列表，逗号分隔（如 `SPY,QQQ,IWM`）
- `--start` / `--end`：回测区间（`YYYY-MM-DD`）
- `--days`：回测天数（仅在未指定 `--start` 时生效）

### 3.2 资金与成本

- `--initial-capital`：初始资金（默认 100000）
- `--commission`：手续费率（默认由 CLI 参数决定）
- `--slippage`：滑点率（默认由 CLI 参数决定）

### 3.3 基准（用于 α/β、超额收益、对比曲线）

- `--benchmark`：基准标的（优先级最高）
- 若不传 `--benchmark`，将使用策略参数 `benchmark_symbol`
- 若策略也未设置，则回退为 `--symbols` 的第一个标的

### 3.4 数据自动补齐

- `--auto-download`：自动补齐缺失数据（默认开启）
- `--no-auto-download`：关闭自动补齐（完全使用本地数据）

### 3.5 策略参数传递（两种方式）

**方式 A：JSON**

```bash
python3 backtest_cli.py \
  --strategy us_etf_momentum \
  --symbols SPY,QQQ,IWM \
  --start 2023-01-01 --end 2024-12-31 \
  --params '{"target_positions":2,"rebalance_frequency":"weekly"}'
```

**方式 B：多次 `key=value`（推荐，易读）**

```bash
python3 backtest_cli.py \
  --strategy us_etf_momentum \
  --symbols SPY,QQQ,IWM \
  --start 2023-01-01 --end 2024-12-31 \
  --param target_positions=2 \
  --param rebalance_frequency=weekly
```

> 说明：`--param key=value` 支持自动类型转换（整数/浮点/布尔/列表/JSON）。

---

## 4. 示例

### 4.1 美股 ETF 动量轮动（多标的）

```bash
python3 backtest_cli.py \
  --strategy us_etf_momentum \
  --symbols SPY,QQQ,IWM,VTI,VOO,DIA,TLT \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --benchmark SPY \
  --param target_positions=1 \
  --param rebalance_frequency=weekly
```

### 4.2 美股小市值低频示例（默认小盘 ETF 池）

```bash
python3 backtest_cli.py \
  --strategy us_smallcap_lowfreq \
  --symbols IWM,IJR,VB,VBR,SCHA,IWN,IWO \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --param target_positions=3 \
  --param rebalance_frequency=monthly
```

> 注意：该策略默认参数使用 `small_cap_pool`，CLI 的 `--symbols` 会同时用于“数据下载/回测 universe”与该池的覆盖。

---

## 5. 输出内容（你会看到什么）

CLI 会输出多张表格（示例）：
- `BACKTEST SUMMARY`：策略、标的、区间、最终净值、信号/订单/成交数量
- `PERFORMANCE SUMMARY`：CAGR、年化收益/波动、夏普/索提诺、最大回撤、α/β 等
- `TRADE SUMMARY`：交易次数、胜率、利润因子、期望值等
- `BACKTESTING REPORT`：按标的统计
- `ENTER TAG STATS / EXIT REASON STATS / MIXED TAG STATS`：按入场标签/退出原因聚合统计

---

## 6. 常见问题

### Q1：为什么回测提示缺数据并尝试下载？

如果本地缓存不完整，且 `--auto-download` 开启，框架会尝试从数据源补齐并落盘缓存。

### Q2：策略里用的是“池”，为什么我还要传 `--symbols`？

回测引擎的数据加载是以 `universe` 为准；你不传 `--symbols` 时，会尝试从策略参数里解析（如 `etf_pool` / `small_cap_pool`）。
为了可控与可复现，建议显式传入 `--symbols`。


# 数据下载指南

本文档介绍如何下载和管理市场数据。

## 目录

- [支持的數據源](#支持的數據源)
- [CLI 下載方式](#cli-下載方式)
- [Python API](#python-api)
- [自動下載配置](#自動下載配置)
- [數據存儲結構](#數據存儲結構)
- [常見問題](#常見問題)

---

## 支持的数据源

| 数据源 | 优先级 | 说明 | 配置要求 |
|--------|--------|------|----------|
| **Alpaca** | ⭐⭐⭐⭐⭐ | 默认首选，免费、稳定 | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |
| **Polygon** | ⭐⭐⭐ | 备用选择 | `POLYGON_API_KEY` |
| **Interactive Brokers** | ⭐⭐ | 需要运行 TWS/Gateway | IB Gateway/TWS 运行中 |

### 数据源优先级（自动模式）

```
auto 模式: Alpaca → Polygon → IB
alpaca 模式: Alpaca (失败则回退到 Polygon/IB)
```

---

## CLI 下载方式

### 基本命令

```bash
# 下载单个标的（默认使用 Alpaca）
python src/main.py download --symbols AAPL

# 下载多个标的
python src/main.py download --symbols "AAPL,MSFT,GOOGL"

# 指定天数
python src/main.py download --symbols "QQQ,SPY" --days 365

# 指定时间周期
python src/main.py download --symbols "AAPL" --timeframe 1d

# 指定数据源
python src/main.py download --symbols "AAPL" --provider alpaca

python src/main.py download --symbols "DIA, EEM, VWO, VEA, USO, BND" --timeframe 1d --days 700 --provider alpaca 
```

### CLI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbols` | 必填 | 标的代码，逗号分隔 |
| `--days` | 365 | 回溯下载天数 |
| `--timeframe` | 1d | 时间周期 (1d, 1h, 1m, etc.) |
| `--provider` | auto | 数据源 (auto/ib/polygon/alpaca) |

---

## Python API

### 方式 1：使用 Downloader 类

```python
import asyncio
from datetime import datetime
from src.data.downloader import Downloader

async def download_data():
    downloader = Downloader()
    
    # 方式 A: 使用天数
    await downloader.download_symbols(
        symbols=["AAPL", "MSFT", "GOOGL"],
        days=365,
        timeframe="1d",
        provider="alpaca"  # 或 "auto"
    )
    
    # 方式 B: 指定日期范围
    await downloader.download_symbols(
        symbols=["QQQ", "SPY"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        timeframe="1d",
        provider="alpaca"
    )

asyncio.run(download_data())
```

### 方式 2：使用 data_manager

```python
import asyncio
from datetime import datetime
from src.data.manager import data_manager

async def get_data_with_download():
    # 自动下载缺失数据
    df = await data_manager.get_ohlcv(
        symbol="AAPL",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        timeframe="1d",
        auto_download=True,        # 开启自动下载
        source_preference="alpaca"  # 优先使用 Alpaca
    )
    
    print(f"获取到 {len(df)} 条数据")
    return df

asyncio.run(get_data_with_download())
```

### 方式 3：在回测中自动下载

```python
import asyncio
from datetime import date
from src.evaluation.backtesting.engine import UnifiedBacktestEngine
from src.strategies.example.simple_etf_momentum import SimpleETFMomentumStrategy

async def run_backtest_with_auto_download():
    engine = UnifiedBacktestEngine()
    
    # auto_download=True 时，数据缺失会自动下载
    result = await engine.run(
        strategies=[SimpleETFMomentumStrategy()],
        universe=["QQQ", "SPY", "TLT", "GLD", "IWM"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        auto_download=True,  # 默认开启
    )
    
    return result

asyncio.run(run_backtest_with_auto_download())
```

---

## 自动下载配置

### 环境变量配置

```bash
# Alpaca（推荐，默认使用）
export ALPACA_API_KEY=your_api_key_id
export ALPACA_SECRET_KEY=your_secret_key
# 可选：指定数据源
export ALPACA_FEED=iex  # iex 或 sip

# Polygon（备用）
export POLYGON_API_KEY=your_polygon_api_key

# Interactive Brokers（需要运行 TWS/Gateway）
export IB_HOST=127.0.0.1
export IB_PORT=7497  # 7497=模拟账户，7496=实盘
export IB_CLIENT_ID=2
```

### 配置文件（config.yaml）

```yaml
data:
  datadir: "./data"

alpaca:
  api_key: "${ALPACA_API_KEY}"
  secret_key: "${ALPACA_SECRET_KEY}"
  base_url: "https://data.alpaca.markets"
  feed: "iex"

polygon:
  api_key: "${POLYGON_API_KEY}"
  adjusted: true

ib:
  host: "127.0.0.1"
  port: 7497
  collector_client_id: 2
```

---

## 数据存储结构

### 统一存储路径（推荐）

所有数据下载后会自动保存到统一路径：

```
data/
└── bars/
    ├── AAPL/
    │   ├── 1d/
    │   │   └── data.parquet
    │   └── 1m/
    │       └── data.parquet
    ├── QQQ/
    │   ├── 1d/
    │   │   └── data.parquet
    │   └── 1m/
    │       └── data.parquet
    └── ...
```

### 旧缓存路径（兼容）

数据也可能保存在以下旧路径：

| 数据源 | 路径 |
|--------|------|
| IB | `data/ib/ohlcv/{timeframe}/{symbol}.parquet` |
| Polygon | `data/polygon/ohlcv/{adjusted|raw}/utc/{timeframe}/{symbol}.parquet` |
| Alpaca | `data/alpaca/ohlcv/{feed}/utc/{timeframe}/{symbol}.parquet` |

### 读取优先级

系统会按以下顺序查找数据：

1. `data/bars/{symbol}/{timeframe}/data.parquet` ← 统一路径（最高优先级）
2. `data/ib/ohlcv/{timeframe}/{symbol}.parquet`
3. `data/daily/{symbol}_{timeframe}.parquet`
4. `data/polygon/ohlcv/{adjusted|raw}/utc/{timeframe}/{symbol}.parquet`

---

## 常见问题

### Q1: 下载失败，提示 "API Key 未配置"

**解决**：检查环境变量是否正确设置

```bash
# 检查 Alpaca 配置
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY

# 如果为空，设置后再下载
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

### Q2: 数据下载后显示 "数据为空"

**可能原因**：
1. 数据未成功写入 `data/bars/` 目录
2. 数据格式不正确

**解决**：

```bash
# 检查数据文件是否存在
ls -la data/bars/AAPL/1d/

# 检查文件内容
python3 -c "
import pandas as pd
df = pd.read_parquet('data/bars/AAPL/1d/data.parquet')
print(f'数据行数: {len(df)}')
print(df.head())
"
```

### Q3: 如何更新已存在的数据？

数据下载是**增量更新**的，不会重复下载已有数据：

```bash
# 假设已经下载了 2024 年数据
python src/main.py download --symbols AAPL --days 30

# 只会下载最近 30 天的新数据（增量更新）
```

### Q4: 如何强制全量重新下载？

在代码中设置 `use_cache=False`：

```python
from src.data.collectors.ib_history import IBHistoryCollector

collector = IBHistoryCollector()
await collector.connect()

# 强制全量下载（不使用缓存）
df = await collector.get_ohlcv_async(
    symbol="AAPL",
    timeframe="1d",
    since=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    use_cache=False  # 关键参数
)
```

### Q5: Alpaca 和 Polygon 有什么区别？

| 特性 | Alpaca | Polygon |
|------|--------|---------|
| 价格 | 免费 | 需要付费计划 |
| 数据延迟 | 实时 (IEX) / 15min (SIP) | 实时 |
| 历史数据 | 有限 | 丰富 |
| 适合场景 | 快速测试、日常使用 | 专业交易 |

### Q6: 如何选择使用哪个数据源？

```python
# 推荐：使用 Alpaca（免费、稳定）
source_preference="alpaca"

# 如果 Alpaca 不可用，回退到 Polygon
source_preference="polygon"

# 需要 IB 数据（需要运行 TWS/Gateway）
source_preference="ib"

# 自动选择（优先级：Alpaca → Polygon → IB）
source_preference="auto"
```

---

## 故障排除

### 检查日志

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 运行下载
python src/main.py download --symbols AAPL --days 30
```

### 验证数据源配置

```python
python3 -c "
from src.data.manager import data_manager

# 检查各数据源配置
print('Alpaca:', bool(data_manager.settings.alpaca.api_key))
print('Polygon:', bool(data_manager.settings.polygon.api_key))
print('IB Host:', data_manager.settings.ib.host)

# 测试采集器初始化
alpaca = data_manager._get_alpaca_collector()
polygon = data_manager._get_polygon_collector()
print('Alpaca Collector:', alpaca is not None)
print('Polygon Collector:', polygon is not None)
"
```

### 手动测试 API 连接

```python
# 测试 Alpaca
python3 -c "
import asyncio
from src.data.collectors.alpaca import AlpacaCollector

async def test():
    collector = AlpacaCollector()
    connected = await collector.connect()
    print('Alpaca 连接:', connected)
    
    df = await collector.get_ohlcv('AAPL', '1d', limit=10)
    print('数据行数:', len(df))

asyncio.run(test())
"
```

---

## 相关文档

- [AGENTS.md](../AGENTS.md) - 开发指南和构建命令
- [策略系统使用指南](策略系统使用指南.md) - 策略回测配置
- [回测系统使用说明](回测系统使用说明.md) - 回测引擎详解

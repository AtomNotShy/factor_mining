# Factor Mining System

一个因子挖掘与回测系统，支持Python FastAPI后端 + React/TypeScript前端。为ETF/股票策略开发提供数据驱动的决策支持，可支持多数据来源：IB, Polygon.io, Biance。

## 项目结构

```
factor_mining/
├── src/                    # Python后端 (FastAPI)
│   ├── api/               # 路由模块 (7个模块)
│   ├── config/            # Pydantic配置 (嵌套env_prefix)
│   ├── core/              # 核心域类型 (Signal, Order, PortfolioState)
│   ├── data/              # 数据采集器 (IB, Polygon, CCXT) + 存储
│   ├── evaluation/        # 双引擎回测 + 评估指标
│   ├── execution/         # 券商实现
│   ├── factors/           # 40+ 技术因子
│   ├── strategies/        # 策略实现 (v2)
│   └── utils/             # Loguru日志
├── frontend/              # React/TypeScript + Vite
│   └── src/
│       ├── components/    # 图表 (Recharts + TradingView), 组件
│       ├── pages/         # 页面 (Dashboard, Backtest, History, Monitoring, Settings)
│       ├── services/      # Axios API服务
│       └── stores/        # Zustand状态管理
├── examples/              # 13个示例脚本
├── tests/                 # 测试目录
├── data/                  # 本地Parquet缓存, IB OHLCV数据
└── docs/                  # 详细文档
```

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动服务

**方式一：直接运行**

```bash
# 使用run.py启动
python3 run.py

# 或使用uvicorn启动
python3 -m uvicorn src.api.main:app --reload --port 8000
```

**方式二：Docker Compose（推荐）**

```bash
# 启动完整系统（包括数据库、缓存等）
docker-compose up -d

# 查看日志
docker-compose logs -f factor-mining

# 停止服务
docker-compose down
```

### 访问系统

- API文档: http://localhost:8000/docs
- Web界面: http://localhost:3000
- 健康检查: http://localhost:8000/health

## 功能特性

### 📊 数据采集
- ✅ 多数据源支持 (Interactive Brokers, Polygon.io, CCXT)
- ✅ 加密货币数据 (Binance, OKX等)
- ✅ 美股/ETF数据 (Polygon，本地Parquet缓存)
- ✅ 实时市场数据获取
- ✅ 历史数据回填
- ✅ 数据质量检查

### 🧮 因子计算
- ✅ 40+ 技术因子库
  - 动量因子: 价格动量、RSI动量、MACD动量等
  - 波动率因子: 历史波动率、ATR、GARCH波动率等
  - 反转因子: 短期反转、RSI反转、布林带反转等
- ✅ 自定义因子开发框架
- ✅ 因子注册表系统

### 📈 因子评估
- ✅ IC分析 (信息系数)
- ✅ 回测引擎 (v2)
- ✅ 分层回测分析
- ✅ 多空组合构建
- ✅ 性能指标计算
- ✅ 因子排名系统
- ✅ 步进向前分析

### 🎯 策略系统
- ✅ ETF动量策略
- ✅ 简单移动平均策略
- ✅ 简单动量策略
- ✅ 策略自动注册
- ✅ 策略回测CLI工具
- ✅ 批量因子测试

### 📡 执行与监控
- ✅ Interactive Brokers TWS集成
- ✅ 模拟交易模式
- ✅ 实时监控预警
- ✅ 任务管理

### 🌐 前端界面
- ✅ 仪表盘
- ✅ 回测页面
- ✅ 历史记录
- ✅ 监控面板
- ✅ 设置页面
- ✅ TradingView图表集成
- ✅ 回撤图、权益曲线、月度收益热力图

## 使用示例

### 1. 运行示例脚本

```bash
# 简单测试
python3 examples/simple_test.py

# API客户端测试（需要先启动服务）
python3 examples/api_client_demo.py

# 分析SPY因子IC表现
python3 examples/spy_factor_ic_4m.py

# 下载日线数据
python3 examples/download_daily_data.py
```

### 2. 使用回测CLI

```bash
# 运行回测
python3 backtest_cli.py --strategy etf_momentum_us --symbol SPY --start 2023-01-01

# 批量测试
python3 batch_factor_test.py

# 优化分析
python3 batch_sharpe_optimization.py
```

### 3. API使用示例

```python
import aiohttp
import asyncio

async def get_factor_data():
    async with aiohttp.ClientSession() as session:
        # 获取因子列表
        async with session.get("http://localhost:8000/api/v1/factors/list") as resp:
            factors = await resp.json()
            print(f"可用因子: {factors['count']} 个")
        
        # 计算动量因子
        params = {"symbol": "SPY", "timeframe": "1d", "limit": 100}
        async with session.post(
            "http://localhost:8000/api/v1/factors/calculate/momentum_20",
            json=params
        ) as resp:
            result = await resp.json()
            print(f"动量因子计算结果")

asyncio.run(get_factor_data())
```

### 4. 获取市场数据

```bash
# 获取美股数据
curl -X POST "http://localhost:8000/api/v1/data/polygon/ohlcv" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "AAPL", "timeframe": "1m", "limit": 500}'

# 获取加密货币数据
curl -X POST "http://localhost:8000/api/v1/data/ohlcv" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "BTC/USDT", "timeframe": "1h", "limit": 50}'
```

## 开发指南

### 添加新因子

1. 在 `src/factors/technical/` 中创建新的因子文件
2. 继承 `TechnicalFactor` 基类
3. 实现 `calculate` 方法
4. 在模块末尾注册因子

```python
from src.factors.base.factor import TechnicalFactor, FactorMetadata, factor_registry

class MyCustomFactor(TechnicalFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="my_custom_factor",
            description="我的自定义因子",
            category="technical",
            sub_category="custom",
            calculation_window=20,
            update_frequency="1d",
            data_requirements=["close"],
        )
        super().__init__(metadata)
    
    def calculate(self, data, **kwargs):
        # 实现因子计算逻辑
        return data['close'].pct_change()

# 注册因子
factor_registry.register(MyCustomFactor())
```

### 添加新策略

1. 在 `src/strategies/` 或 `src/strategies/example/` 中创建策略文件
2. 继承 `Strategy` 基类
3. 实现 `generate_signals` 方法
4. 在 `src/strategies/__init__.py` 中导入以自动注册

```python
from src.strategies.base.strategy import Strategy, Signal, SignalAction

class MyStrategy(Strategy):
    name = "my_strategy"
    description = "我的策略"
    
    def generate_signals(self, data, portfolio_state=None):
        # 实现信号生成逻辑
        return Signal(
            ts_utc=data.index[-1],
            symbol=self.symbol,
            action=SignalAction.BUY,
            strength=1.0
        )
```

### 配置环境变量

系统支持通过环境变量进行配置，参考 `.env.example`：

```bash
# Interactive Brokers配置
export IB_HOST=127.0.0.1
export IB_PORT=7497
export IB_CLIENT_ID=1

# Polygon API配置
export POLYGON_API_KEY=your_api_key

# 数据库配置
export DB_HOST=localhost
export DB_PORT=5432

# API服务配置
export API_HOST=0.0.0.0
export API_PORT=8000
```

## 项目结构说明

| 目录 | 说明 |
|------|------|
| `src/api/` | FastAPI RESTful接口 |
| `src/data/` | 数据采集和处理模块 |
| `src/factors/` | 因子计算模块 |
| `src/evaluation/` | 回测和评估模块 |
| `src/strategies/` | 策略实现模块 |
| `src/execution/` | 券商集成模块 |
| `src/core/` | 核心类型定义 |
| `src/monitoring/` | 监控和预警模块 |
| `frontend/` | React前端应用 |
| `examples/` | 使用示例脚本 |
| `docs/` | 详细文档 |

## 相关文档

- [开发指南](AGENTS.md) - 详细的开发规范和代码地图
- [策略回测CLI使用指南](docs/策略回测CLI使用指南.md)
- [策略回测Web界面使用指南](docs/策略回测Web界面使用指南.md)
- [策略系统使用指南](docs/策略系统使用指南.md)
- [批量因子测试使用指南](docs/批量因子测试使用指南.md)
- [前端开发指南](frontend/AGENTS.md)
- [前端快速开始](frontend/QUICK_START.md)

## 许可证

MIT License

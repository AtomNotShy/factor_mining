# Factor Mining System

> 基于 Interactive Brokers 的美股 ETF 量化策略回测与实盘交易系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React + TypeScript](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)

## 📋 目录

- [项目概述](#项目概述)
- [✨ 主要特性](#-主要特性)
- [🛠 技术栈](#-技术栈)
- [🚀 快速开始](#-快速开始)
- [📁 项目结构](#-项目结构)
- [📖 文档](#-文档)
- [📊 使用示例](#-使用示例)
- [🧪 测试](#-测试)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)

## 项目概述

Factor Mining System 是一个专业的量化交易因子挖掘与策略回测平台，支持：

- **多数据源**：Interactive Brokers、Polygon、CCXT 等
- **丰富因子库**：40+ 技术指标因子
- **灵活策略系统**：Freqtrade 风格策略模板，支持 v2/v3 双模式
- **统一事件驱动架构**：回测与实盘共用核心引擎
- **完整回测评估**：双回测引擎 + 全面性能指标
- **现代化 Web 界面**：React + TypeScript + Vite

## ✨ 主要特性

### 核心功能

| 功能 | 描述 |
|------|------|
| **多时间框架** | 支持多时间框架分析 (1m, 5m, 1h, 1d, 1w) |
| **因子研究** | 40+ 技术指标，支持自定义因子扩展 |
| **策略模板** | RSI、MACD、布林带、随机指标、动量等 8 种模板 |
| **事件驱动** | 统一事件流架构，确保回测确定性 |
| **风险管理** | 完整风险检查、日亏损限制、最大回撤保护 |
| **佣金模型** | 灵活的佣金、滑点、隔夜利息模型 |

### 回测引擎

- **UnifiedBacktestEngine**：统一事件驱动回测引擎
- **FreqtradeBacktestEngine**：Freqtrade 风格回测引擎
- 支持资金曲线、回撤、收益分布、月度热力图等可视化

### 实盘交易

- **Interactive Brokers** 集成
- **模拟交易**模式
- **Freqtrade 风格**交易协议

## 🛠 技术栈

### 后端

| 技术 | 用途 |
|------|------|
| **Python 3.10+** | 主语言 |
| **FastAPI** | Web 框架 |
| **Pydantic** | 数据验证 |
| **Pandas/NumPy** | 数据处理 |
| **ib-insync** | IB 接口 |
| **loguru** | 日志 |

### 前端

| 技术 | 用途 |
|------|------|
| **React 18** | UI 框架 |
| **TypeScript** | 类型安全 |
| **Vite** | 构建工具 |
| **Tailwind CSS** | 样式框架 |
| **Recharts** | 图表可视化 |
| **Zustand** | 状态管理 |

### 工具链

- **Black** - 代码格式化 (line-length: 100)
- **Flake8** - 代码检查
- **MyPy** - 类型检查
- **Pytest** - 单元测试
- **Docker** - 容器化

## 🚀 快速开始

### 前置条件

- Python 3.10+
- Node.js 18+
- pnpm (可选)

### 1. 克隆项目

```bash
git clone <repository-url>
cd factor_mining
```

### 2. 安装后端依赖

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -e ".[dev]"  # 安装开发依赖
```

### 3. 安装前端依赖

```bash
cd frontend
pnpm install
```

### 4. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置
# 主要配置项：
# - IB__HOST: IB Gateway/TWS 主机地址
# - IB__PORT: IB 端口
# - IB__CLIENT_ID: 客户端 ID
```

### 5. 启动服务

**后端服务：**

```bash
# 开发模式 (自动重载)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 或使用 Python 模块
python3 -m src.api.main
```

**前端服务：**

```bash
cd frontend
pnpm dev
```

访问 http://localhost:3000 查看前端界面，API 服务在 http://localhost:8000。

### 6. 运行示例

```bash
# 策略模板演示
python3 examples/strategy_template_demo.py

# Freqtrade 回测演示
python3 examples/freqtrade_backtest_demo.py
```

## 📁 项目结构

```
factor_mining/
├── src/                          # Python 后端
│   ├── api/                      # FastAPI 路由
│   │   └── routers/              # API 端点 (7个模块)
│   ├── config/                   # Pydantic 配置
│   ├── core/                     # 核心类型和状态
│   │   ├── types.py              # Signal, Order, PortfolioState
│   │   ├── calendar.py           # 交易日历
│   │   ├── context.py            # 运行上下文
│   │   ├── risk_manager.py       # 风险管理
│   │   └── state_machine.py      # 状态机
│   ├── data/                     # 数据层
│   │   ├── collectors/           # 数据采集器 (IB, Polygon, CCXT)
│   │   ├── storage/              # 数据存储 (Parquet, SQLite)
│   │   └── adapter/              # 数据适配器
│   ├── evaluation/               # 评估层
│   │   ├── backtesting/          # 回测引擎
│   │   │   ├── unified_engine.py # 统一回测引擎
│   │   │   ├── freqtrade_engine.py
│   │   │   └── report.py         # 报告生成
│   │   └── metrics/              # 性能指标
│   ├── execution/                # 执行层
│   │   ├── order_engine.py       # 订单引擎
│   │   └── providers/            # 执行提供者 (IB, 模拟)
│   ├── factors/                  # 因子库
│   │   └── technical/            # 技术因子 (40+)
│   ├── strategies/               # 策略系统
│   │   ├── base/                 # 策略基类
│   │   │   ├── strategy.py       # Strategy 基类
│   │   │   ├── freqtrade_interface.py
│   │   │   └── templates.py      # 策略模板
│   │   ├── example/              # 示例策略
│   │   └── user_strategies/      # 用户策略
│   └── utils/                    # 工具函数
│       └── logger.py             # 日志
│
├── frontend/                     # React 前端
│   └── src/
│       ├── components/           # 组件
│       │   ├── charts/           # 图表组件
│       │   └── tradingview/      # 交易视图
│       ├── pages/                # 页面
│       │   ├── Dashboard.tsx     # 仪表盘
│       │   ├── Backtest.tsx      # 回测页面
│       │   ├── History.tsx       # 历史记录
│       │   ├── Monitoring.tsx    # 监控
│       │   └── Settings.tsx      # 设置
│       ├── services/             # API 服务
│       ├── stores/               # 状态管理
│       └── i18n/                 # 国际化
│
├── examples/                     # 示例脚本
├── tests/                        # 测试用例
├── docs/                         # 文档
├── data/                         # 数据缓存
├── reports/                      # 回测报告
├── logs/                         # 日志文件
│
├── pyproject.toml               # Python 项目配置
├── requirements.txt             # 依赖列表
├── frontend/package.json        # 前端依赖
├── docker-compose.yml           # Docker 配置
└── README.md                    # 本文件
```

## 📖 文档

| 文档 | 描述 |
|------|------|
| [AGENTS.md](AGENTS.md) | 开发指南、代码规范、构建命令 |
| [docs/策略系统使用指南.md](docs/策略系统使用指南.md) | 策略系统详细用法 |
| [docs/策略回测CLI使用指南.md](docs/策略回测CLI使用指南.md) | CLI 回测指南 |
| [docs/策略回测Web界面使用指南.md](docs/策略回测Web界面使用指南.md) | Web 界面使用指南 |
| [docs/回测系统使用说明.md](docs/回测系统使用说明.md) | 回测系统说明 |
| [docs/批量因子测试使用指南.md](docs/批量因子测试使用指南.md) | 因子测试指南 |
| [docs/新增策略开发指南.md](docs/新增策略开发指南.md) | 新策略开发指南 |
| [docs/unified_event_driven_architecture.md](docs/unified_event_driven_architecture.md) | 事件驱动架构设计 |
| [frontend/README.md](frontend/README.md) | 前端开发指南 |

## 📊 使用示例

### 策略模板使用

```python
from src.strategies.base.templates import RSIStrategy

class MyRSIStrategy(RSIStrategy):
    strategy_name = "My RSI Strategy"
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
```

### 统一回测引擎

```python
import asyncio
from datetime import date
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine, UnifiedConfig, TradeConfig
from src.strategies.user_strategies import USETFMomentumStrategy

async def run_backtest():
    strategy = USETFMomentumStrategy()
    config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005,
        ),
    )
    engine = UnifiedBacktestEngine(config=config)

    await engine.run(
        strategies=[strategy],
        universe=["SPY", "QQQ", "TLT"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
    )

asyncio.run(run_backtest())
```

### Freqtrade 风格策略

```python
import pandas as pd
from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin

class MyFreqtradeStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    strategy_name = "My Freqtrade Strategy"
    strategy_id = "my_freqtrade"
    timeframe = "1d"
    stoploss = -0.10
    trailing_stop = False
    minimal_roi = {0: 0.02, "60": 0.01}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 计算指标
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = False
        dataframe['enter_tag'] = ''
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = False
        dataframe['exit_tag'] = ''
        return dataframe
```

## 🧪 测试

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行特定测试

```bash
pytest tests/test_informative.py                # 单个文件
pytest tests/test_informative.py::TestClass     # 单个类
pytest tests/test_informative.py::TestClass::test_method  # 单个测试
```

### 带覆盖率测试

```bash
pytest --cov=src --cov-report=term-missing
```

### 代码质量检查

```bash
black src/ && flake8 src/ && mypy src/
```

## 🤝 贡献指南

### 代码规范

- **Python**: 遵循 [AGENTS.md](AGENTS.md) 中的代码风格指南
- **Imports**: Standard library → Third-party → Local
- **Line length**: 100 字符
- **类型注解**: 所有函数和变量必须标注类型
- **错误处理**: 禁止 bare `except:`，捕获具体异常
- **日志**: 使用 `logger` 而非 `print()`

### 提交规范

- feat: 新功能
- fix: Bug 修复
- docs: 文档更新
- style: 代码格式调整
- refactor: 重构
- test: 测试相关
- chore: 构建或辅助工具变动

### 开发流程

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [Freqtrade](https://www.freqtrade.io/) - 策略系统和回测引擎的设计参考
- [pandas](https://pandas.pydata.org/) - 数据处理
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [React](https://react.dev/) - 前端框架

# Factor Mining System

一个系统化的因子挖掘系统，为freqtrade量化交易策略提供数据驱动的决策支持。

## 项目结构

```
factor/
├── src/
│   ├── data/                   # 数据管理模块
│   │   ├── collectors/         # 数据采集器
│   │   ├── processors/         # 数据处理器
│   │   └── storage/           # 数据存储
│   ├── factors/               # 因子计算模块
│   │   ├── base/              # 基础因子类
│   │   ├── technical/         # 技术因子
│   │   ├── fundamental/       # 基本面因子
│   │   └── alternative/       # 另类因子
│   ├── evaluation/            # 因子评估模块
│   │   ├── metrics/           # 评估指标
│   │   ├── backtesting/       # 回测引擎
│   │   └── analysis/          # 分析工具
│   ├── strategy/              # 策略生成模块
│   │   ├── generators/        # 策略生成器
│   │   └── freqtrade/         # freqtrade集成
│   ├── api/                   # API接口
│   │   ├── routers/           # 路由模块
│   │   └── schemas/           # 数据模型
│   ├── monitoring/            # 监控模块
│   │   ├── alerts/            # 预警系统
│   │   └── reports/           # 报告生成
│   ├── config/                # 配置管理
│   └── utils/                 # 工具函数
├── frontend/                  # 前端界面
├── tests/                     # 测试代码
├── docs/                      # 文档
├── scripts/                   # 脚本工具
├── docker/                    # Docker配置
└── requirements.txt           # Python依赖
```

## 快速开始

### 方式一：直接运行

#### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

可选（机器学习相关）：

```bash
pip install -r requirements-ml.txt
```

macOS/arm64 若安装 `lightgbm` 失败，建议优先用 Docker 运行；或改用 conda-forge 安装 `lightgbm`。

#### 2. 启动服务

```bash
# 直接启动（使用默认配置）
python3 run.py

# 或者使用uvicorn启动
python3 -m uvicorn src.api.main:app --reload --port 8000
```

#### 3. 访问系统

- API文档: http://localhost:8000/docs
- 系统健康检查: http://localhost:8000/health

### 方式二：Docker Compose（推荐）

```bash
# 启动完整系统（包括数据库、缓存等）
docker-compose up -d

# 查看日志
docker-compose logs -f factor-mining

# 停止服务
docker-compose down
```

启动后可访问：
- 因子挖掘API: http://localhost:8000/docs
- Grafana监控: http://localhost:3000 (admin/admin)

### 方式三：单独Docker

```bash
# 构建镜像
docker build -t factor-mining .

# 运行容器
docker run -p 8000:8000 factor-mining
```

## 功能特性

### 📊 数据采集
- ✅ 多交易所数据采集 (Binance, OKX)
- ✅ 美股/ETF 数据采集 (Polygon，支持本地Parquet缓存)
- ✅ 实时市场数据获取
- ✅ 历史数据回填
- ✅ 数据质量检查

### 🧮 因子计算
- ✅ 40+ 技术因子库
  - 动量类因子 (10+): 价格动量、RSI动量、MACD动量等
  - 波动率因子 (10+): 历史波动率、ATR、GARCH波动率等
  - 反转类因子 (10+): 短期反转、RSI反转、布林带反转等
- ✅ 自定义因子开发框架
- ✅ 因子批量计算API

### 📈 因子评估
- ✅ IC分析 (信息系数)
- ✅ 因子回测引擎
- ✅ 分层回测分析
- ✅ 多空组合构建
- ✅ 性能指标计算
- ✅ 因子排名系统

### 🎯 策略生成
- ✅ freqtrade策略生成
- ✅ 实时监控预警
- ✅ 可视化分析界面

## 使用示例

### 1. 本地测试

```bash
# 运行基础功能测试
python3 examples/simple_test.py

# 运行API客户端测试（需要先启动服务）
python3 examples/api_client_demo.py

# 分析 SPY 近4个月因子 IC 表现（需要配置 POLYGON_API_KEY）
python3 examples/spy_factor_ic_4m.py
```

### 2. API使用示例

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
        params = {"symbol": "BTC/USDT", "timeframe": "1h", "limit": 100}
        async with session.post(
            "http://localhost:8000/api/v1/factors/calculate/momentum_20",
            params=params
        ) as resp:
            result = await resp.json()
            print(f"动量因子计算结果: {result['statistics']}")

asyncio.run(get_factor_data())
```

### 3. 获取市场数据

```bash
# 使用curl获取BTC/USDT数据
curl -X POST "http://localhost:8000/api/v1/data/ohlcv" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "BTC/USDT", "timeframe": "1h", "limit": 50}'
```

### 4. 获取美股/ETF数据（Polygon，本地缓存）

先设置环境变量 `POLYGON_API_KEY`，然后：

```bash
curl -X POST "http://localhost:8000/api/v1/data/polygon/ohlcv" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "AAPL", "timeframe": "1m", "limit": 500}'
```

数据会缓存到 `data/polygon/ohlcv/`（Docker Compose 已挂载 `./data:/app/data`，可持久化）。

## 项目结构说明

- `src/data/` - 数据采集和处理模块
- `src/factors/` - 因子计算模块，包含各种因子算法
- `src/api/` - RESTful API接口
- `src/config/` - 配置管理
- `src/utils/` - 工具函数
- `examples/` - 使用示例和测试脚本
- `docker/` - Docker相关配置

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

### 配置环境变量

系统支持通过环境变量进行配置：

```bash
# 数据库配置
export DB_HOST=
export DB_PORT=
export DB_USERNAME=
export DB_PASSWORD=

# 交易所API配置
export EXCHANGE_BINANCE_API_KEY=
export EXCHANGE_BINANCE_SECRET=

# API服务配置
export API_HOST=0.0.0.0
export API_PORT=8000
export API_DEBUG=false
```

## 许可证

MIT License # factor_mining

# 策略回测Web界面使用指南

## 概述

已创建完整的策略回测Web界面，可以：
- 📊 可视化回测结果
- 📈 在价格图表上标记买入/卖出点
- 📉 展示投资组合价值曲线
- 📊 显示详细的回测统计信息

## 快速开始

### 1. 启动API服务

```bash
# 从项目根目录启动
python -m src.api.main

# 或使用uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 访问Web界面

打开浏览器访问：
```
http://localhost:8000/api/v1/strategy-backtest/viewer
```

### 3. 运行回测

1. **选择策略**：从下拉菜单中选择要回测的策略（如 `vwap_pullback`）
2. **配置参数**：
   - Symbol: 交易标的（如 `SPY`）
   - Timeframe: 时间周期（1m, 5m, 15m, 1h, 1d）
   - **Date Range Mode**: 选择日期模式
     - **Use Days**: 使用天数（从今天往前推N天）
     - **Use Date Range**: 使用日期范围（指定开始和结束日期）
   - Days: 回测天数（仅在"Use Days"模式下使用）
   - Start Date / End Date: 开始和结束日期（仅在"Use Date Range"模式下使用）
   - Initial Capital: 初始资金
   - Commission Rate: 手续费率
   - Slippage Rate: 滑点率
3. **点击 "Run Backtest"** 开始回测

## 功能说明

### 1. 价格图表（带买卖信号）

- **绿色向上三角形**：买入点
- **红色向下三角形**：卖出点
- **蓝色线**：价格走势

图表支持：
- 鼠标悬停查看详细信息
- 缩放和平移
- 时间轴联动

### 2. 投资组合价值曲线

显示回测期间投资组合价值的变化：
- 绿色线表示组合价值
- 可以查看每个时间点的具体价值

### 3. 统计信息卡片

显示关键指标：
- **Final Value**: 最终组合价值
- **Total Return**: 总收益率
- **Sharpe Ratio**: 夏普比率
- **Max Drawdown**: 最大回撤
- **Total Trades**: 总交易次数
- **Win Rate**: 胜率

## API端点

### 1. 获取可用策略列表

```bash
GET /api/v1/strategy-backtest/strategies
```

响应示例：
```json
{
  "strategies": [
    {
      "name": "vwap_pullback",
      "description": "VWAP回踩策略",
      "category": "vwap",
      "version": "1.0.0"
    }
  ],
  "count": 1
}
```

### 2. 运行策略回测

**方式一：使用天数**
```bash
POST /api/v1/strategy-backtest/run
Content-Type: application/json

{
  "strategy_name": "vwap_pullback",
  "symbol": "SPY",
  "timeframe": "1m",
  "days": 10,
  "initial_capital": 100000.0,
  "commission_rate": 0.0005,
  "slippage_rate": 0.0002
}
```

**方式二：使用日期范围**
```bash
POST /api/v1/strategy-backtest/run
Content-Type: application/json

{
  "strategy_name": "vwap_pullback",
  "symbol": "SPY",
  "timeframe": "1m",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "initial_capital": 100000.0,
  "commission_rate": 0.0005,
  "slippage_rate": 0.0002
}
```

**注意**：
- `days` 和 `start_date`/`end_date` 二选一
- 如果同时提供，`days` 优先
- 日期格式：`YYYY-MM-DD`（如 `2024-01-01`）

响应包含：
- 价格数据（OHLCV）
- 交易记录（买入/卖出点）
- 回测结果（统计信息、组合价值等）

### 3. Web界面

```bash
GET /api/v1/strategy-backtest/viewer
```

返回HTML页面，包含完整的交互式界面。

## 使用示例

### Python客户端示例

**使用天数**
```python
import requests

# 运行回测（使用天数）
response = requests.post(
    "http://localhost:8000/api/v1/strategy-backtest/run",
    json={
        "strategy_name": "vwap_pullback",
        "symbol": "SPY",
        "timeframe": "1m",
        "days": 10,
        "initial_capital": 100000.0,
        "commission_rate": 0.0005,
        "slippage_rate": 0.0002
    }
)
```

**使用日期范围**
```python
import requests

# 运行回测（使用日期范围）
response = requests.post(
    "http://localhost:8000/api/v1/strategy-backtest/run",
    json={
        "strategy_name": "vwap_pullback",
        "symbol": "SPY",
        "timeframe": "1m",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": 100000.0,
        "commission_rate": 0.0005,
        "slippage_rate": 0.0002
    }
)
```

data = response.json()

# 查看结果
print(f"Final Value: ${data['results']['final_value']:,.2f}")
print(f"Total Return: {data['results']['total_return']:.2%}")
print(f"Total Trades: {data['results']['trade_stats']['total_trades']}")

# 查看交易记录
for trade in data['trades']:
    print(f"{trade['order_type']}: {trade['size']} @ ${trade['price']:.2f} at {trade['timestamp']}")
```

### JavaScript/前端示例

```javascript
// 运行回测
async function runBacktest() {
    const response = await fetch('/api/v1/strategy-backtest/run', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            strategy_name: 'vwap_pullback',
            symbol: 'SPY',
            timeframe: '1m',
            days: 10,
            initial_capital: 100000.0,
            commission_rate: 0.0005,
            slippage_rate: 0.0002
        })
    });
    
    const data = await response.json();
    
    // 绘制图表
    plotPriceChart(data.price_data, data.trades);
    plotPortfolioChart(data.results.portfolio_value);
}

// 绘制价格图表（带买卖点）
function plotPriceChart(priceData, trades) {
    const buyTrades = trades.filter(t => t.order_type === 'buy');
    const sellTrades = trades.filter(t => t.order_type === 'sell');
    
    // 使用Plotly绘制
    // ... 图表代码
}
```

## 图表说明

### 买入/卖出标记

- **买入点（绿色▲）**：策略发出买入信号的位置
  - X轴：交易时间
  - Y轴：买入价格
  
- **卖出点（红色▼）**：策略发出卖出信号的位置
  - X轴：交易时间
  - Y轴：卖出价格

### 交互功能

- **悬停**：鼠标悬停在数据点上查看详细信息
- **缩放**：双击图表区域放大，拖拽选择区域缩放
- **平移**：点击并拖拽图表移动视图
- **重置**：双击图表重置视图

## 注意事项

1. **API Key配置**：确保已配置 `POLYGON_API_KEY` 环境变量（用于获取美股数据）
2. **数据获取**：首次运行可能需要一些时间获取数据
3. **数据缓存**：数据会自动缓存到本地，后续运行会更快
4. **时间范围**：回测天数越多，数据获取和处理时间越长

## 故障排除

### 问题：无法获取数据

**解决方案**：
1. 检查 `POLYGON_API_KEY` 是否配置
2. 检查网络连接
3. 检查API Key权限

### 问题：策略不存在

**解决方案**：
1. 检查策略名称是否正确
2. 访问 `/api/v1/strategy-backtest/strategies` 查看可用策略列表

### 问题：图表不显示

**解决方案**：
1. 检查浏览器控制台是否有错误
2. 确保Plotly.js已加载
3. 检查数据格式是否正确

## 扩展功能

### 添加新策略

1. 在 `src/strategies/` 下创建新策略
2. 策略会自动注册到策略注册表
3. Web界面会自动显示新策略

### 自定义图表

可以修改 `strategy_backtest.py` 中的图表代码来自定义：
- 图表样式
- 颜色方案
- 标记形状
- 布局配置

## 相关文档

- `策略系统使用指南.md` - 策略系统详细说明
- `回测系统使用说明.md` - 回测引擎详细文档
- `src/strategies/README.md` - 策略开发文档

# Backtest Analyzer Skill

## 描述
分析策略回测结果，识别性能瓶颈，并提供具体的优化建议

## 分析维度

### 1. 收益分析
- 总收益率、年化收益率
- 相对基准的超额收益
- 月度/季度收益分布

### 2. 风险分析
- 最大回撤及持续时间
- 回撤原因分析
- 波动率、夏普比率、索提诺比率

### 3. 交易分析
- 交易次数、胜率、盈亏比
- 每笔交易平均收益
- 连续盈利/亏损次数
- 交易成本占比

### 4. 时间分析
- 持仓时间分布
- 信号产生频率
- 市场环境适配度

## 可用工具

### analyze_backtest_results
分析回测结果，提供改进建议

**参数：**
```json
{
  "type": "object",
  "properties": {
    "results": {
      "type": "object",
      "description": "回测结果字典，包含 performance_stats, trade_stats, trades 等"
    },
    "strategy_name": {
      "type": "string",
      "description": "策略名称（可选，用于上下文）"
    },
    "focus_area": {
      "type": "string",
      "description": "重点分析领域：收益/风险/交易/综合（默认综合）",
      "enum": ["收益", "风险", "交易", "综合"]
    }
  },
  "required": ["results"]
}
```

**返回：**
```json
{
  "summary": "整体评估摘要",
  "scores": {
    "overall": 85,
    "returns": 80,
    "risk": 75,
    "trading": 90
  },
  "strengths": [
    "胜率达到65%，表现良好"
  ],
  "weaknesses": [
    "最大回撤超过15%，需要改进"
  ],
  "recommendations": [
    {
      "priority": "高",
      "area": "风险控制",
      "suggestion": "建议增加止损机制，将最大回撤控制在10%以内",
      "expected_impact": "减少50%的最大回撤"
    },
    {
      "priority": "中",
      "area": "参数优化",
      "suggestion": "建议将持仓周期从5天延长到10天，减少交易频率",
      "expected_impact": "提升夏普比率0.2"
    }
  ],
  "detailed_analysis": {
    "returns": { ... },
    "risk": { ... },
    "trading": { ... }
  }
}
```

### compare_strategies
比较多个回测结果，识别最佳策略

**参数：**
```json
{
  "type": "object",
  "properties": {
    "results_list": {
      "type": "array",
      "description": "多个回测结果数组"
    },
    "strategy_names": {
      "type": "array",
      "description": "对应的策略名称列表"
    },
    "comparison_metric": {
      "type": "string",
      "description": "比较指标：sharpe_ratio/total_return/max_drawdown",
      "enum": ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
    }
  },
  "required": ["results_list"]
}
```

### generate_optimization_report
生成详细的策略优化报告

**参数：**
```json
{
  "type": "object",
  "properties": {
    "results": {
      "type": "object",
      "description": "回测结果"
    },
    "strategy_params": {
      "type": "object",
      "description": "当前策略参数（可选）"
    },
    "market_conditions": {
      "type": "string",
      "description": "市场环境假设：牛市/熊市/震荡市",
      "enum": ["牛市", "熊市", "震荡市", "综合"]
    }
  },
  "required": ["results"]
}
```

## 分析规则

### 评分标准

| 指标 | 优秀 | 良好 | 一般 | 需改进 |
|------|------|------|------|--------|
| 夏普比率 | >2.0 | 1.5-2.0 | 1.0-1.5 | <1.0 |
| 最大回撤 | <5% | 5-10% | 10-20% | >20% |
| 胜率 | >70% | 55-70% | 45-55% | <45% |
| 盈亏比 | >2.0 | 1.5-2.0 | 1.0-1.5 | <1.0 |

### 改进建议优先级

**高优先级：**
- 最大回撤 > 15%
- 夏普比率 < 1.0
- 胜率 < 45%

**中优先级：**
- 10% < 最大回撤 ≤ 15%
- 1.0 ≤ 夏普比率 < 1.5
- 交易成本占比 > 3%

**低优先级：**
- 月度收益波动较大
- 持仓时间分布不均匀
- 信号频率过高/过低

## 使用示例

```json
{
  "results": {
    "total_return": 0.15,
    "annual_return": 0.18,
    "sharpe_ratio": 1.3,
    "max_drawdown": 0.12,
    "win_rate": 0.58,
    "profit_loss_ratio": 1.8,
    "total_trades": 45,
    "trades": [...]
  }
}
```

## 输出格式

分析结果以结构化 JSON 格式返回，包含：
- `overall_score`: 整体评分 (0-100)
- `summary`: 简要总结
- `detailed_analysis`: 详细分析
- `recommendations`: 改进建议列表
- `action_items`: 具体可操作项

"""
参数优化基类
定义优化器的通用接口和数据结构
"""

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

@dataclass
class OptimizationResult:
    """优化结果"""
    params: Dict[str, Any]      # 最佳参数组合
    metric_value: float         # 目标指标值 (如夏普比率)
    all_results: pd.DataFrame   # 所有试验的历史记录
    best_backtest_result: Dict[str, Any] # 最佳参数对应的回测详情
    duration: float             # 优化耗时

class OptimizerBase(abc.ABC):
    """优化器基类"""
    
    def __init__(self, objective_metric: str = "sharpe_ratio"):
        """
        Args:
            objective_metric: 优化目标指标，如 sharpe_ratio, total_return, max_drawdown
        """
        self.objective_metric = objective_metric

    @abc.abstractmethod
    async def optimize(
        self,
        strategy_cls: Any,
        param_space: Dict[str, List[Any]],
        backtest_config: Dict[str, Any],
    ) -> OptimizationResult:
        """运行优化"""
        pass

"""
因子评估模块
提供因子有效性评估、IC分析、回测等功能
"""

from .metrics.ic_analysis import ICAnalyzer
from .metrics.performance import PerformanceAnalyzer
from .backtesting.engine import BacktestEngine

__all__ = [
    "ICAnalyzer",
    "PerformanceAnalyzer", 
    "BacktestEngine"
] 
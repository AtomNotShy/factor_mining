"""
因子计算模块
提供各种因子的计算功能
"""

from .base.factor import BaseFactor, TechnicalFactor, FundamentalFactor, AlternativeFactor, factor_registry

# 导入所有因子类型，触发自动注册
from . import technical

__all__ = [
    "BaseFactor",
    "TechnicalFactor", 
    "FundamentalFactor",
    "AlternativeFactor",
    "factor_registry",
    "technical",
]

"""
简单演示策略示例
- simple_momentum: 简单动量策略
- simple_ma: 简单均线策略

用于学习策略框架的使用方法。
"""

from .simple_momentum import SimpleMomentumStrategy
from .simple_ma import SimpleMAStrategy

__all__ = ["SimpleMomentumStrategy", "SimpleMAStrategy"]

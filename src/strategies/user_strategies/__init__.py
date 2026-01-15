"""
用户策略目录
用户编写的策略放在此目录下，框架会自动加载。

目录结构：
user_strategies/
├── __init__.py                        # 自动注册所有策略
├── etf_momentum_joinquant.py          # 聚宽ETF动量轮动策略
├── etf_momentum_rotation.py           # ETF动量轮动策略（完整参数化版）
├── etf_momentum_rotation_simple.py    # ETF动量轮动策略（简化版）
├── simple_etf_momentum.py             # 简单ETF动量策略
└── your_strategy.py                   # 你的新策略
"""



# 导入注册表
from src.strategies.simple_strategy import StrategyRegistry
from src.strategies.user_strategies.us_etf_joinquant_rotation import USETFJoinQuantRotationStrategy

# 注册策略


__all__ = [
    "USETFJoinQuantRotationStrategy",
]

"""
用户策略目录
框架会自动发现和加载此目录下所有策略，无需手动注册。

目录结构：
user_strategies/
├── __init__.py                        # 自动注册所有策略
├── us_etf_joinquant_rotation.py       # 聚宽ETF动量轮动策略
├── your_strategy.py                   # 你的新策略（只需写在这里即可自动被识别）

使用方式：
1. 在 user_strategies/ 目录下创建新的策略文件
2. 继承 Strategy 或 FreqtradeStrategy
3. 定义 strategy_id 类属性
4. 无需在任何地方写注册代码，框架自动发现并注册
"""

from __future__ import annotations

import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.base.strategy import Strategy
    from src.strategies.base.freqtrade_interface import FreqtradeStrategy

# 自动导入所有策略模块，触发自动注册
_this_dir = Path(__file__).parent
for module_info in pkgutil.iter_modules([str(_this_dir)], prefix="user_strategies."):
    module_name = module_info.name
    # 跳过 __init__ 和已导入的模块
    if module_name.endswith(".__init__"):
        continue
    # 动态导入模块，触发 __init_subclass__ 自动注册
    try:
        __import__(f"src.strategies.{module_name}", fromlist=[""])
    except ImportError as exc:
        # 静默忽略导入错误（可能是依赖问题）
        pass

__all__ = []

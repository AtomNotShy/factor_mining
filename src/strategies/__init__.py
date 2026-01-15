"""
策略模块
提供统一的策略接口和策略管理功能

自动发现机制：
- 框架会自动扫描 strategies 目录下的所有模块
- 继承 Strategy 或 FreqtradeStrategy 的类会自动注册
- 无需手动注册或导入

目录结构：
strategies/
├── base/              # 策略基类（Strategy, FreqtradeStrategy）
├── user_strategies/   # 用户策略（自动加载）
└── example/           # 示例策略（自动加载）
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Iterable

from src.utils.logger import get_logger
from .base.strategy import Strategy, StrategyConfig, strategy_registry

if TYPE_CHECKING:
    from src.strategies.base.freqtrade_interface import FreqtradeStrategy

logger = get_logger("strategies")


def _iter_strategy_modules() -> Iterable[str]:
    """遍历策略模块，排除 base 目录"""
    package_name = __name__
    for module_info in pkgutil.walk_packages(__path__, prefix=f"{package_name}."):
        name = module_info.name
        # 跳过 base 目录
        if ".base." in name or name.endswith(".base"):
            continue
        # 跳过 __init__
        if name.endswith(".__init__"):
            continue
        yield name


def auto_discover() -> None:
    """自动导入策略模块，触发策略注册。

    注意：由于使用了 __init_subclass__ 自动注册机制，
    此函数主要用于确保所有策略模块被加载。
    """
    for module_name in _iter_strategy_modules():
        try:
            importlib.import_module(module_name)
            logger.debug(f"加载策略模块: {module_name}")
        except Exception as exc:
            logger.warning(f"跳过策略模块 {module_name}: {exc}")


# 自动发现并加载所有策略
auto_discover()

__all__ = [
    "Strategy",
    "StrategyConfig",
    "strategy_registry",
    "auto_discover",
]

"""
策略模块
提供统一的策略接口和策略管理功能
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable

from src.utils.logger import get_logger
from .base.strategy import Strategy, StrategyConfig, strategy_registry

logger = get_logger("strategies")


def _iter_strategy_modules() -> Iterable[str]:
    package_name = __name__
    for module_info in pkgutil.walk_packages(__path__, prefix=f"{package_name}."):
        name = module_info.name
        if ".base." in name or name.endswith(".base"):
            continue
        if name.endswith(".__init__"):
            continue
        yield name


def auto_discover() -> None:
    """自动导入策略模块，触发策略注册."""
    for module_name in _iter_strategy_modules():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            logger.warning(f"跳过策略模块 {module_name}: {exc}")


# 自动发现并加载所有策略实现
auto_discover()

__all__ = [
    "Strategy",
    "StrategyConfig",
    "strategy_registry",
    "auto_discover",
]

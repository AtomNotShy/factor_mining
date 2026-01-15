"""
CLI 策略管理模块
处理策略查找、注册、列表等功能
"""

import re
from typing import Any, Dict, Optional, Type

from src.strategies.base.strategy import strategy_registry
from src.strategies.base.freqtrade_interface import FreqtradeStrategy

# Freqtrade 策略注册表
_freqtrade_strategy_registry: Dict[str, Type[FreqtradeStrategy]] = {}


def register_freqtrade_strategies() -> None:
    """注册 Freqtrade 风格策略"""
    global _freqtrade_strategy_registry

    try:
        from src.strategies.user_strategies.dual_ma import DualMAStrategy
        _freqtrade_strategy_registry['dual_ma'] = DualMAStrategy
    except ImportError:
        pass
    
    try:
        from src.strategies.user_strategies.mean_reversion import MeanReversionStrategy
        _freqtrade_strategy_registry['mean_reversion'] = MeanReversionStrategy
    except ImportError:
        pass
    
    try:
        from src.strategies.user_strategies.us_etf_joinquant_rotation import USETFJoinQuantRotationStrategy
        _freqtrade_strategy_registry['us_etf_joinquant_rotation'] = USETFJoinQuantRotationStrategy
        _freqtrade_strategy_registry['us_etf_rotation'] = USETFJoinQuantRotationStrategy
    except ImportError:
        pass


def get_strategy_class(strategy_name: str) -> Optional[Type]:
    """获取策略类 - 支持多种名称格式"""
    if not strategy_name:
        return None

    # 生成候选名称
    candidates = [
        strategy_name,
        strategy_name.lower(),
        strategy_name.upper(),
        strategy_name.replace("_", "").lower(),
    ]

    # 驼峰转下划线
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', strategy_name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    candidates.append(s2)
    candidates = list(dict.fromkeys(candidates))

    # 1. 优先尝试 Freqtrade 策略
    for name in candidates:
        if name in _freqtrade_strategy_registry:
            return _freqtrade_strategy_registry[name]

    # 2. 然后尝试 v2 策略
    for name in candidates:
        strategy_class = strategy_registry.get_strategy(name)
        if strategy_class:
            return strategy_class

    return None


def get_strategy_instance(strategy_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """获取策略实例"""
    strategy_class = get_strategy_class(strategy_name)
    if not strategy_class:
        return None

    # Freqtrade 策略需要实例化
    if strategy_name in _freqtrade_strategy_registry:
        strategy = strategy_class()
        if params:
            for key, value in params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
        return strategy
    else:
        # v2 策略
        if params:
            strategy_class.set_params(params)
        return strategy_class


def list_strategies() -> None:
    """列出所有可用策略"""
    strategies = strategy_registry.list_strategies()
    freqtrade_strategies = list(_freqtrade_strategy_registry.keys())
    all_strategies = list(set(strategies + freqtrade_strategies))

    if not all_strategies:
        print("未发现策略")
        return

    print("=" * 50)
    print("可用策略列表")
    print("=" * 50)

    # Freqtrade 策略
    for name in sorted(freqtrade_strategies):
        strategy_class = _freqtrade_strategy_registry.get(name)
        if strategy_class:
            desc = getattr(strategy_class, '__doc__', '') or getattr(strategy_class, 'strategy_name', name)
            desc = desc.strip().split('\n')[0] if isinstance(desc, str) else name
            print(f"  • {name} [Freqtrade]: {desc}")

    # v2 策略
    for name in sorted(strategies):
        if name not in freqtrade_strategies:
            strategy_class = strategy_registry.get_strategy(name)
            if strategy_class:
                desc = strategy_class.__doc__ or "无描述"
                desc = desc.strip().split('\n')[0] if desc else "无描述"
                print(f"  • {name}: {desc}")

    print("=" * 50)


def get_default_universe(strategy_class: Type) -> list:
    """获取策略默认标的池"""
    if hasattr(strategy_class, 'etf_pool'):
        return list(strategy_class.etf_pool)
    elif hasattr(strategy_class, 'small_cap_pool'):
        return list(strategy_class.small_cap_pool)
    elif hasattr(strategy_class, 'universe'):
        return list(strategy_class.universe)
    return []


# 初始化时注册 Freqtrade 策略
register_freqtrade_strategies()

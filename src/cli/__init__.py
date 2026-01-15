"""
CLI 模块

提供命令行接口功能，包括：
- 参数解析 (parser)
- 配置管理 (config)
- 策略管理 (strategies)
- 回测执行 (backtest)
- 数据下载 (download)
"""

from .parser import build_parser
from .strategies import list_strategies, get_strategy_class, get_strategy_instance
from .backtest import run_backtest
from .download import run_download

__all__ = [
    "build_parser",
    "list_strategies",
    "get_strategy_class",
    "get_strategy_instance",
    "run_backtest",
    "run_download",
]

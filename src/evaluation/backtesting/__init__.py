"""
回测模块

统一回测框架（单一入口）：
- 引擎：UnifiedBacktestEngine
- 配置：UnifiedConfig
- 成本模型、止损管理、多时间框架、报告、可视化

Usage:
    from src.evaluation.backtesting import (
        UnifiedBacktestEngine,
        UnifiedConfig,
        FeatureFlag,
    )
"""

from .unified_engine import UnifiedBacktestEngine

from .config import (
    UnifiedConfig,
    FeatureFlag,
    TradeConfig,
    TimeConfig,
    FillConfig,
    StoplossConfig,
    ProtectionConfig,
    OutputConfig,
    BacktestResult,
)

from .cost_model import (
    USStockCostModel,
    CostModel,
    Side,
    TradeInfo,
)

from .stoploss_manager import (
    StoplossManager,
    ExitReason,
)

from .multi_timeframe import (
    MultiTimeframeDataLoader,
    MultiTimeframeAnalyzer,
    TimeframeConfig,
    InformativePair,
    create_multi_timeframe_config,
)

from .report import (
    EnhancedBacktestReport,
    EnhancedReportGenerator,
    TradeDetail,
    SignalAnalysis,
)

from .visualizer import (
    EnhancedBacktestVisualizer,
)

__all__ = [
    "UnifiedBacktestEngine",
    "UnifiedConfig",
    "FeatureFlag",
    "TradeConfig",
    "TimeConfig",
    "FillConfig",
    "StoplossConfig",
    "ProtectionConfig",
    "OutputConfig",
    "BacktestResult",
    "USStockCostModel",
    "CostModel",
    "Side",
    "TradeInfo",
    "StoplossManager",
    "ExitReason",
    "MultiTimeframeDataLoader",
    "MultiTimeframeAnalyzer",
    "TimeframeConfig",
    "InformativePair",
    "create_multi_timeframe_config",
    "EnhancedBacktestReport",
    "EnhancedReportGenerator",
    "TradeDetail",
    "SignalAnalysis",
    "EnhancedBacktestVisualizer",
]

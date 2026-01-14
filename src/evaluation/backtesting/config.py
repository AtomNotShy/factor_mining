"""
回测引擎配置

统一配置模块：
1. UnifiedConfig - 统一配置（新标准）
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from enum import Flag, auto


class FeatureFlag(Flag):
    """引擎特性开关"""
    NONE = 0
    VECTORIZED = auto()
    FREQTRADE_PROTOCOL = auto()
    STOPLOSS_MANAGER = auto()
    TRAILING_STOP = auto()
    PROTECTIONS = auto()
    EVENT_DRIVEN = auto()
    MULTI_TIMEFRAME = auto()
    DETAIL_TIMEFRAME = auto()
    SAVE_RESULTS = auto()
    ALL = (
        VECTORIZED | FREQTRADE_PROTOCOL | STOPLOSS_MANAGER | 
        TRAILING_STOP | PROTECTIONS | EVENT_DRIVEN | 
        MULTI_TIMEFRAME | DETAIL_TIMEFRAME | SAVE_RESULTS
    )


@dataclass
class TradeConfig:
    """交易配置"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_size: float = 0.2
    max_positions: int = 10
    stake_amount: Optional[float] = None


@dataclass
class TimeConfig:
    """时间框架配置"""
    signal_timeframe: str = "1d"
    execution_timeframe: str = "1d"
    detail_timeframe: Optional[str] = None
    informative_timeframes: List[str] = field(default_factory=list)
    execution_time: Optional[str] = None
    clock_mode: Literal["daily", "bar", "hybrid"] = "daily"
    warmup_days: int = 260


@dataclass
class FillConfig:
    """成交配置"""
    fill_price: Literal["open", "close", "high", "low"] = "close"
    missing_bar_handling: Literal["previous", "next", "fail"] = "previous"


@dataclass
class StoplossConfig:
    """止损配置"""
    stoploss: float = -0.10
    trailing_stop: bool = False
    trailing_stop_positive: float = 0.0
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False
    minimal_roi: Dict[int, float] = field(default_factory=lambda: {0: float('inf')})


@dataclass
class ProtectionConfig:
    """保护机制配置"""
    enabled: bool = True
    cooldown_protection: bool = True
    cooldown_duration: int = 60
    max_drawdown_protection: bool = True
    max_drawdown: float = 0.15
    max_open_trades_protection: bool = True
    max_open_trades: int = 3


@dataclass
class OutputConfig:
    """输出配置"""
    save_results: bool = True
    export_format: Literal["parquet", "json", "csv"] = "parquet"
    export_dir: str = "data/backtest_results"
    backtest_breakdown: List[str] = field(default_factory=lambda: ["day", "week", "month"])


@dataclass
class UnifiedConfig:
    """
    统一回测配置
    
    Usage:
        config = UnifiedConfig(
            trade=TradeConfig(initial_capital=500000),
            time=TimeConfig(signal_timeframe="1h", clock_mode="hybrid"),
            stoploss=StoplossConfig(stoploss=-0.08, minimal_roi={0: 0.05, 60: 0.03}),
            features=FeatureFlag.ALL,
        )
    """
    trade: TradeConfig = field(default_factory=TradeConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    fill: FillConfig = field(default_factory=FillConfig)
    stoploss: StoplossConfig = field(default_factory=StoplossConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    features: FeatureFlag = FeatureFlag.ALL
    universe: List[str] = field(default_factory=list)
    timerange: Optional[str] = None
    
    def validate(self) -> None:
        assert self.trade.initial_capital > 0
        assert self.trade.commission_rate >= 0
        assert -1.0 <= self.stoploss.stoploss <= 0
        assert self.time.warmup_days >= 0


@dataclass
class BacktestResult:
    """回测结果"""
    run_id: str = ""
    strategy_name: str = ""
    timeframe: str = "1d"
    timerange: str = ""
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    signals: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_daily: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "timerange": self.timerange,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return_pct": round(self.annualized_return * 100, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "win_rate_pct": round(self.win_rate * 100, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "profit_factor": round(self.profit_factor, 4),
            "signals": self.signals,
            "orders": self.orders,
            "fills": self.fills,
            "portfolio_daily": self.portfolio_daily,
        }

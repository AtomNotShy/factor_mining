"""
核心类型定义
按照设计文档定义统一的领域模型
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
from enum import Enum


class ActionType(str, Enum):
    """信号动作类型"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """订单类型"""
    MKT = "MKT"  # 市价单
    LMT = "LMT"  # 限价单
    STP = "STP"  # 止损单


class OrderStatus(str, Enum):
    """订单状态"""
    NEW = "NEW"
    SENT = "SENT"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TimeInForce(str, Enum):
    """订单有效期"""
    DAY = "DAY"  # 当日有效
    GTC = "GTC"  # 撤销前有效


@dataclass
class Signal:
    """策略信号"""
    ts_utc: datetime
    symbol: str
    strategy_id: str
    action: ActionType  # LONG/SHORT/FLAT
    strength: float  # 信号强度 (0-1 或任意尺度)
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    ttl_bars: Optional[int] = None  # 信号有效期（bar数）
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderIntent:
    """订单意图（策略层输出）"""
    order_id: str = ""
    ts_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    qty: float = 0.0
    order_type: OrderType = OrderType.MKT
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """成交记录"""
    fill_id: str
    order_id: str
    ts_fill_utc: datetime
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float  # 手续费
    slippage_est: Optional[float] = None  # 相对参考价的滑点估计
    liquidity_flag: Optional[str] = None  # maker/taker
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketData:
    """市场数据容器"""
    bars: pd.DataFrame  # indexed by ts_utc, columns: open/high/low/close/volume...
    bars_map: Optional[Dict[str, pd.DataFrame]] = None  # 多周期数据映射: timeframe -> DataFrame
    bars_all: Optional[pd.DataFrame] = None  # 完整历史数据（包含预热期）
    features: Optional[pd.DataFrame] = None  # aligned (ts_utc, symbol)
    actions: Optional[pd.DataFrame] = None  # corporate actions if needed

    def __post_init__(self):
        """验证数据格式"""
        if self.bars.empty:
            raise ValueError("bars 不能为空")
        if not isinstance(self.bars.index, pd.DatetimeIndex):
            raise ValueError("bars 必须使用 DatetimeIndex")

    def get_bars(self, timeframe: str) -> pd.DataFrame:
        """获取指定时间框架的数据"""
        if self.bars_map and timeframe in self.bars_map:
            return self.bars_map[timeframe]
        return self.bars


@dataclass
class PortfolioState:
    """组合状态"""
    cash: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)
    avg_price: Dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    daily_loss: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_position(self, symbol: str) -> float:
        """获取持仓数量"""
        return self.positions.get(symbol, 0.0)
    
    def get_avg_price(self, symbol: str) -> float:
        """获取平均成本价"""
        return self.avg_price.get(symbol, 0.0)


@dataclass
class RiskState:
    """风险状态"""
    daily_loss_limit: Optional[float] = None  # 日亏损限制
    max_position_size: Optional[float] = None  # 单标的最大仓位
    max_positions: Optional[int] = None  # 最大持仓数
    blacklist: List[str] = field(default_factory=list)  # 黑名单
    max_drawdown_limit: Optional[float] = None  # 最大回撤限制
    metadata: Dict[str, Any] = field(default_factory=dict)

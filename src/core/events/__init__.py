"""
事件类型定义
提供统一的事件类型供回测和实盘系统使用
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class EventPriority(Enum):
    """事件优先级（数值越小优先级越高）"""
    CRITICAL = 0  # 系统关键事件（错误、连接中断）
    HIGH = 10     # 高优先级（成交、订单状态变更）
    NORMAL = 20   # 普通优先级（市场数据、信号）
    LOW = 30      # 低优先级（日志、监控）


@dataclass
class Event:
    """事件基类"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = field(default="base")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: EventPriority = field(default=EventPriority.NORMAL)
    source: str = field(default="system")
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """确保时间戳是UTC时区"""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def __lt__(self, other: "Event") -> bool:
        """比较方法（用于优先级队列）"""
        if not isinstance(other, Event):
            return NotImplemented
        # 优先级高的排前面（值小的优先）
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        # 同优先级时，时间早的排前面
        return self.timestamp < other.timestamp

    def __le__(self, other: "Event") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Event") -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return other < self

    def __ge__(self, other: "Event") -> bool:
        return self == other or other < self

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "data": self.data,
        }


# ============ 市场事件 ============

@dataclass
class MarketEvent(Event):
    """市场数据事件基类"""
    event_type: str = field(default="market", init=False)

    symbol: str = ""
    timeframe: str = "1d"
    data_type: str = "bar"  # bar, tick, quote, trade

    @property
    def open(self) -> Optional[float]:
        return self.data.get("open")

    @property
    def high(self) -> Optional[float]:
        return self.data.get("high")

    @property
    def low(self) -> Optional[float]:
        return self.data.get("low")

    @property
    def close(self) -> Optional[float]:
        return self.data.get("close")

    @property
    def volume(self) -> Optional[float]:
        return self.data.get("volume")


@dataclass
class BarEvent(MarketEvent):
    """K线事件"""
    event_type: str = field(default="bar", init=False)
    data_type: str = field(default="bar", init=False)


@dataclass
class TickEvent(MarketEvent):
    """Tick事件"""
    event_type: str = field(default="tick", init=False)
    data_type: str = field(default="tick", init=False)
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None


@dataclass
class QuoteEvent(MarketEvent):
    """报价事件"""
    event_type: str = field(default="quote", init=False)
    data_type: str = field(default="quote", init=False)
    bid: Optional[float] = None
    ask: Optional[float] = None


# ============ 策略事件 ============

@dataclass
class StrategyEvent(Event):
    """策略事件基类"""
    event_type: str = field(default="strategy", init=False)
    strategy_id: str = ""


@dataclass
class SignalEvent(StrategyEvent):
    """策略信号事件"""
    event_type: str = field(default="signal", init=False)

    signal_type: str = ""  # entry, exit, modify
    action: str = ""       # buy, sell, hold, flat
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    symbol: str = ""
    quantity: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    valid_until: Optional[datetime] = None


@dataclass
class StrategyStatusEvent(StrategyEvent):
    """策略状态事件"""
    event_type: str = field(default="strategy_status", init=False)

    status: str = ""  # running, paused, stopped, error
    message: str = ""
    performance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyErrorEvent(StrategyEvent):
    """策略错误事件"""
    event_type: str = field(default="strategy_error", init=False)
    priority: EventPriority = field(default=EventPriority.CRITICAL, init=False)

    error_type: str = ""
    error_message: str = ""
    stack_trace: Optional[str] = None
    recoverable: bool = True


# ============ 订单事件 ============

@dataclass
class OrderEvent(Event):
    """订单事件基类"""
    event_type: str = field(default="order", init=False)

    order_id: str = ""
    symbol: str = ""
    order_type: str = ""  # market, limit, stop
    side: str = ""        # buy, sell
    quantity: float = 0.0
    strategy_id: str = ""


@dataclass
class OrderCreatedEvent(OrderEvent):
    """订单创建事件"""
    event_type: str = field(default="order_created", init=False)

    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    reason: Optional[str] = None


@dataclass
class OrderSubmittedEvent(OrderEvent):
    """订单提交事件"""
    event_type: str = field(default="order_submitted", init=False)
    priority: EventPriority = field(default=EventPriority.HIGH, init=False)

    broker_order_id: Optional[str] = None
    submit_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderFilledEvent(OrderEvent):
    """订单成交事件"""
    event_type: str = field(default="order_filled", init=False)
    priority: EventPriority = field(default=EventPriority.HIGH, init=False)

    fill_id: str = ""
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    fill_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    commission: float = 0.0
    slippage: float = 0.0
    liquidity: Optional[str] = None  # maker, taker


@dataclass
class OrderPartiallyFilledEvent(OrderFilledEvent):
    """订单部分成交事件"""
    event_type: str = field(default="order_partially_filled", init=False)

    remaining_quantity: float = 0.0


@dataclass
class OrderCancelledEvent(OrderEvent):
    """订单取消事件"""
    event_type: str = field(default="order_cancelled", init=False)

    cancel_reason: str = ""
    cancelled_quantity: float = 0.0
    cancel_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderRejectedEvent(OrderEvent):
    """订单拒绝事件"""
    event_type: str = field(default="order_rejected", init=False)
    priority: EventPriority = field(default=EventPriority.CRITICAL, init=False)

    reject_reason: str = ""
    reject_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============ 系统事件 ============

@dataclass
class SystemEvent(Event):
    """系统事件基类"""
    event_type: str = field(default="system", init=False)


@dataclass
class TimerEvent(SystemEvent):
    """定时器事件"""
    event_type: str = field(default="timer", init=False)

    timer_id: str = ""
    interval: Optional[float] = None  # 秒
    scheduled_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HeartbeatEvent(SystemEvent):
    """心跳事件"""
    event_type: str = field(default="heartbeat", init=False)

    component: str = ""
    status: str = "healthy"
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorEvent(SystemEvent):
    """系统错误事件"""
    event_type: str = field(default="error", init=False)
    priority: EventPriority = field(default=EventPriority.CRITICAL, init=False)

    component: str = ""
    error_type: str = ""
    error_message: str = ""
    stack_trace: Optional[str] = None
    fatal: bool = False


@dataclass
class ConfigurationEvent(SystemEvent):
    """配置变更事件"""
    event_type: str = field(default="configuration", init=False)

    config_type: str = ""
    changes: Dict[str, Any] = field(default_factory=dict)
    effective_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============ 风险事件 ============

@dataclass
class RiskEvent(Event):
    """风险事件基类"""
    event_type: str = field(default="risk", init=False)
    priority: EventPriority = field(default=EventPriority.HIGH, init=False)


@dataclass
class RiskLimitEvent(RiskEvent):
    """风险限额事件"""
    event_type: str = field(default="risk_limit", init=False)

    limit_type: str = ""  # position, exposure, loss, etc.
    current_value: float = 0.0
    limit_value: float = 0.0
    threshold: float = 0.8  # 阈值比例


@dataclass
class RiskViolationEvent(RiskEvent):
    """风险违规事件"""
    event_type: str = field(default="risk_violation", init=False)

    violation_type: str = ""
    severity: str = ""  # warning, error, critical
    action_taken: str = ""  # reject, close, alert
    details: Dict[str, Any] = field(default_factory=dict)


# ============ 工厂函数 ============

def create_bar_event(
    symbol: str,
    timeframe: str,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float,
    timestamp: Optional[datetime] = None,
) -> BarEvent:
    """创建K线事件"""
    return BarEvent(
        symbol=symbol,
        timeframe=timeframe,
        data={
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        },
        timestamp=timestamp or datetime.now(timezone.utc),
    )


def create_signal_event(
    strategy_id: str,
    signal_type: str,
    action: str,
    symbol: str,
    strength: float = 1.0,
    quantity: Optional[float] = None,
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SignalEvent:
    """创建信号事件"""
    return SignalEvent(
        strategy_id=strategy_id,
        signal_type=signal_type,
        action=action,
        symbol=symbol,
        strength=strength,
        quantity=quantity,
        limit_price=limit_price,
        stop_price=stop_price,
        metadata=metadata or {},
    )


def create_order_created_event(
    order_id: str,
    symbol: str,
    order_type: str,
    side: str,
    quantity: float,
    strategy_id: str,
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    time_in_force: str = "GTC",
    reason: Optional[str] = None,
) -> OrderCreatedEvent:
    """创建订单创建事件"""
    return OrderCreatedEvent(
        order_id=order_id,
        symbol=symbol,
        order_type=order_type,
        side=side,
        quantity=quantity,
        strategy_id=strategy_id,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force,
        reason=reason,
    )

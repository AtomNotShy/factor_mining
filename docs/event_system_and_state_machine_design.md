# 事件系统与状态机设计

## 1. 事件系统设计

### 1.1 事件基类

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import uuid

class EventPriority(Enum):
    """事件优先级"""
    CRITICAL = 0    # 系统关键事件（错误、连接中断）
    HIGH = 10       # 高优先级（成交、订单状态变更）
    NORMAL = 20     # 普通优先级（市场数据、信号）
    LOW = 30        # 低优先级（日志、监控）

@dataclass
class Event:
    """事件基类"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = field(default="base")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = field(default=EventPriority.NORMAL)
    source: str = field(default="system")
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 确保时间戳是UTC时区
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
```

### 1.2 市场事件

```python
@dataclass
class MarketEvent(Event):
    """市场数据事件"""
    event_type: str = field(default="market")
    
    symbol: str = field(default="")
    timeframe: str = field(default="1d")
    data_type: str = field(default="bar")  # bar, tick, quote, trade
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 常用数据字段
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
    event_type: str = field(default="bar")
    data_type: str = field(default="bar")

@dataclass
class TickEvent(MarketEvent):
    """Tick事件"""
    event_type: str = field(default="tick")
    data_type: str = field(default="tick")

@dataclass
class QuoteEvent(MarketEvent):
    """报价事件"""
    event_type: str = field(default="quote")
    data_type: str = field(default="quote")
    bid: Optional[float] = field(default=None)
    ask: Optional[float] = field(default=None)
    bid_size: Optional[float] = field(default=None)
    ask_size: Optional[float] = field(default=None)
```

### 1.3 策略事件

```python
@dataclass
class StrategyEvent(Event):
    """策略事件基类"""
    event_type: str = field(default="strategy")
    strategy_id: str = field(default="")

@dataclass
class SignalEvent(StrategyEvent):
    """策略信号事件"""
    event_type: str = field(default="signal")
    
    signal_type: str = field(default="")  # entry, exit, modify
    action: str = field(default="")       # buy, sell, hold
    strength: float = field(default=1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 信号相关字段
    symbol: str = field(default="")
    quantity: Optional[float] = field(default=None)
    limit_price: Optional[float] = field(default=None)
    stop_price: Optional[float] = field(default=None)
    valid_until: Optional[datetime] = field(default=None)

@dataclass
class StrategyStatusEvent(StrategyEvent):
    """策略状态事件"""
    event_type: str = field(default="strategy_status")
    
    status: str = field(default="")  # running, paused, stopped, error
    message: str = field(default="")
    performance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyErrorEvent(StrategyEvent):
    """策略错误事件"""
    event_type: str = field(default="strategy_error")
    priority: EventPriority = field(default=EventPriority.CRITICAL)
    
    error_type: str = field(default="")
    error_message: str = field(default="")
    stack_trace: Optional[str] = field(default=None)
    recoverable: bool = field(default=True)
```

### 1.4 订单事件

```python
@dataclass
class OrderEvent(Event):
    """订单事件基类"""
    event_type: str = field(default="order")
    
    order_id: str = field(default="")
    symbol: str = field(default="")
    order_type: str = field(default="")  # market, limit, stop
    side: str = field(default="")        # buy, sell
    quantity: float = field(default=0.0)
    strategy_id: str = field(default="")

@dataclass
class OrderCreatedEvent(OrderEvent):
    """订单创建事件"""
    event_type: str = field(default="order_created")
    
    limit_price: Optional[float] = field(default=None)
    stop_price: Optional[float] = field(default=None)
    time_in_force: str = field(default="GTC")  # GTC, IOC, FOK

@dataclass
class OrderSubmittedEvent(OrderEvent):
    """订单提交事件"""
    event_type: str = field(default="order_submitted")
    priority: EventPriority = field(default=EventPriority.HIGH)
    
    broker_order_id: Optional[str] = field(default=None)
    submit_time: datetime = field(default_factory=datetime.utcnow)

@dataclass
class OrderFilledEvent(OrderEvent):
    """订单成交事件"""
    event_type: str = field(default="order_filled")
    priority: EventPriority = field(default=EventPriority.HIGH)
    
    fill_id: str = field(default="")
    fill_price: float = field(default=0.0)
    fill_quantity: float = field(default=0.0)
    fill_time: datetime = field(default_factory=datetime.utcnow)
    commission: float = field(default=0.0)
    slippage: float = field(default=0.0)
    liquidity: Optional[str] = field(default=None)  # maker, taker

@dataclass
class OrderPartiallyFilledEvent(OrderFilledEvent):
    """订单部分成交事件"""
    event_type: str = field(default="order_partially_filled")
    
    remaining_quantity: float = field(default=0.0)

@dataclass
class OrderCancelledEvent(OrderEvent):
    """订单取消事件"""
    event_type: str = field(default="order_cancelled")
    
    cancel_reason: str = field(default="")
    cancelled_quantity: float = field(default=0.0)
    cancel_time: datetime = field(default_factory=datetime.utcnow)

@dataclass
class OrderRejectedEvent(OrderEvent):
    """订单拒绝事件"""
    event_type: str = field(default="order_rejected")
    priority: EventPriority = field(default=EventPriority.CRITICAL)
    
    reject_reason: str = field(default="")
    reject_time: datetime = field(default_factory=datetime.utcnow)
```

### 1.5 系统事件

```python
@dataclass
class SystemEvent(Event):
    """系统事件基类"""
    event_type: str = field(default="system")

@dataclass
class TimerEvent(SystemEvent):
    """定时器事件"""
    event_type: str = field(default="timer")
    
    timer_id: str = field(default="")
    interval: Optional[float] = field(default=None)  # 秒
    scheduled_time: datetime = field(default_factory=datetime.utcnow)

@dataclass
class HeartbeatEvent(SystemEvent):
    """心跳事件"""
    event_type: str = field(default="heartbeat")
    
    component: str = field(default="")
    status: str = field(default="healthy")
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorEvent(SystemEvent):
    """系统错误事件"""
    event_type: str = field(default="error")
    priority: EventPriority = field(default=EventPriority.CRITICAL)
    
    component: str = field(default="")
    error_type: str = field(default="")
    error_message: str = field(default="")
    stack_trace: Optional[str] = field(default=None)
    fatal: bool = field(default=False)

@dataclass
class ConfigurationEvent(SystemEvent):
    """配置变更事件"""
    event_type: str = field(default="configuration")
    
    config_type: str = field(default="")
    changes: Dict[str, Any] = field(default_factory=dict)
    effective_time: datetime = field(default_factory=datetime.utcnow)
```

### 1.6 风险事件

```python
@dataclass
class RiskEvent(Event):
    """风险事件基类"""
    event_type: str = field(default="risk")
    priority: EventPriority = field(default=EventPriority.HIGH)

@dataclass
class RiskLimitEvent(RiskEvent):
    """风险限额事件"""
    event_type: str = field(default="risk_limit")
    
    limit_type: str = field(default="")  # position, exposure, loss, etc.
    current_value: float = field(default=0.0)
    limit_value: float = field(default=0.0)
    threshold: float = field(default=0.8)  # 阈值比例

@dataclass
class RiskViolationEvent(RiskEvent):
    """风险违规事件"""
    event_type: str = field(default="risk_violation")
    priority: EventPriority = field(default=EventPriority.CRITICAL)
    
    violation_type: str = field(default="")
    severity: str = field(default="")  # warning, error, critical
    action_taken: str = field(default="")  # reject, close, alert
    details: Dict[str, Any] = field(default_factory=dict)
```

## 2. 状态机设计

### 2.1 订单状态机

```python
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class OrderState(Enum):
    """订单状态枚举"""
    CREATED = "created"           # 订单已创建
    VALIDATING = "validating"     # 验证中
    VALIDATED = "validated"       # 验证通过
    REJECTED = "rejected"         # 验证拒绝
    PENDING_SUBMIT = "pending_submit"  # 等待提交
    SUBMITTING = "submitting"     # 提交中
    SUBMITTED = "submitted"       # 已提交到券商
    PENDING_CANCEL = "pending_cancel"  # 等待取消
    CANCELLING = "cancelling"     # 取消中
    CANCELLED = "cancelled"       # 已取消
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"             # 完全成交
    EXPIRED = "expired"           # 已过期
    ERROR = "error"               # 错误状态

class OrderStateMachine:
    """订单状态机"""
    
    # 状态转移规则
    TRANSITIONS: Dict[OrderState, List[OrderState]] = {
        OrderState.CREATED: [
            OrderState.VALIDATING,
            OrderState.REJECTED,
        ],
        OrderState.VALIDATING: [
            OrderState.VALIDATED,
            OrderState.REJECTED,
        ],
        OrderState.VALIDATED: [
            OrderState.PENDING_SUBMIT,
            OrderState.REJECTED,
        ],
        OrderState.PENDING_SUBMIT: [
            OrderState.SUBMITTING,
            OrderState.CANCELLED,
        ],
        OrderState.SUBMITTING: [
            OrderState.SUBMITTED,
            OrderState.REJECTED,
            OrderState.ERROR,
        ],
        OrderState.SUBMITTED: [
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.EXPIRED,
            OrderState.ERROR,
        ],
        OrderState.PARTIALLY_FILLED: [
            OrderState.FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.EXPIRED,
            OrderState.ERROR,
        ],
        OrderState.PENDING_CANCEL: [
            OrderState.CANCELLING,
            OrderState.FILLED,  # 可能在取消过程中成交
        ],
        OrderState.CANCELLING: [
            OrderState.CANCELLED,
            OrderState.FILLED,  # 可能在取消过程中成交
            OrderState.ERROR,
        ],
    }
    
    # 终态
    TERMINAL_STATES = {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.ERROR,
    }
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.current_state = OrderState.CREATED
        self.state_history: List[Tuple[datetime, OrderState, Optional[str]]] = []
        self._record_state_change(self.current_state, "initial")
    
    def _record_state_change(self, new_state: OrderState, reason: Optional[str] = None):
        """记录状态变更"""
        self.state_history.append((datetime.utcnow(), new_state, reason))
    
    def can_transition(self, target_state: OrderState) -> bool:
        """检查是否可以转移到目标状态"""
        if self.current_state in self.TERMINAL_STATES:
            return False
        
        allowed_transitions = self.TRANSITIONS.get(self.current_state, [])
        return target_state in allowed_transitions
    
    def transition(self, target_state: OrderState, reason: Optional[str] = None) -> bool:
        """执行状态转移"""
        if not self.can_transition(target_state):
            return False
        
        old_state = self.current_state
        self.current_state = target_state
        self._record_state_change(target_state, reason)
        
        # 触发状态变更事件
        self._on_state_changed(old_state, target_state, reason)
        return True
    
    def _on_state_changed(self, old_state: OrderState, new_state: OrderState, reason: Optional[str]):
        """状态变更回调"""
        # 这里可以触发事件或执行其他逻辑
        pass
    
    def get_state_history(self) -> List[Dict]:
        """获取状态历史"""
        return [
            {
                "timestamp": ts,
                "state": state.value,
                "reason": reason
            }
            for ts, state, reason in self.state_history
        ]
    
    def is_terminal(self) -> bool:
        """检查是否处于终态"""
        return self.current_state in self.TERMINAL_STATES
    
    def get_time_in_state(self) -> float:
        """获取在当前状态的停留时间（秒）"""
        if not self.state_history:
            return 0.0
        
        last_change_time = self.state_history[-1][0]
        return (datetime.utcnow() - last_change_time).total_seconds()
```

### 2.2 策略状态机

```python
class StrategyState(Enum):
    """策略状态枚举"""
    INITIALIZING = "initializing"     # 初始化中
    READY = "ready"                   # 准备就绪
    RUNNING = "running"               # 运行中
    PAUSED = "paused"                 # 已暂停
    STOPPING = "stopping"             # 停止中
    STOPPED = "stopped"               # 已停止
    ERROR = "error"                   # 错误状态
    DISABLED = "disabled"             # 已禁用

class StrategyStateMachine:
    """策略状态机"""
    
    TRANSITIONS: Dict[StrategyState, List[StrategyState]] = {
        StrategyState.INITIALIZING: [
            StrategyState.READY,
            StrategyState.ERROR,
            StrategyState.DISABLED,
        ],
        StrategyState.READY: [
            StrategyState.RUNNING,
            StrategyState.DISABLED,
        ],
        StrategyState.RUNNING: [
            StrategyState.PAUSED,
            StrategyState.STOPPING,
            StrategyState.ERROR,
        ],
        StrategyState.PAUSED: [
            StrategyState.RUNNING,
            StrategyState.STOPPING,
            StrategyState.ERROR,
        ],
        StrategyState.STOPPING: [
            StrategyState.STOPPED,
            StrategyState.ERROR,
        ],
        StrategyState.STOPPED: [
            StrategyState.READY,
            StrategyState.DISABLED,
        ],
        StrategyState.ERROR: [
            StrategyState.READY,
            StrategyState.DISABLED,
        ],
        StrategyState.DISABLED: [
            StrategyState.READY,
        ],
    }
    
    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        self.current_state = StrategyState.INITIALIZING
        self.state_history: List[Tuple[datetime, StrategyState, Optional[str]]] = []
        self.error_count = 0
        self.last_error: Optional[Exception] = None
    
    def transition(self, target_state: StrategyState, reason: Optional[str] = None) -> bool:
        """执行状态转移"""
        allowed_transitions = self.TRANSITIONS.get(self.current_state, [])
        if target_state not in allowed_transitions:
            return False
        
        # 记录状态变更
        self.state_history.append((datetime.utcnow(), target_state, reason
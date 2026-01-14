"""
状态机实现
提供统一的状态机基类用于订单和策略状态管理
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid


class StateType(Enum):
    """状态类型枚举"""
    CREATED = "created"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PENDING_SUBMIT = "pending_submit"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    PENDING_CANCEL = "pending_cancel"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    EXPIRED = "expired"
    ERROR = "error"


class StrategyState(Enum):
    """策略状态枚举"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class StateTransition:
    """状态转移记录"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    from_state: str = ""
    to_state: str = ""
    reason: Optional[str] = None
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
            "event_id": self.event_id,
        }


@dataclass
class BaseStateMachine(ABC):
    """状态机基类"""

    entity_id: str = ""
    current_state: str = ""
    state_history: List[StateTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def terminal_states(self) -> set:
        """终态集合"""
        pass

    @property
    @abstractmethod
    def transitions(self) -> Dict[str, List[str]]:
        """状态转移规则"""
        pass

    def can_transition(self, target_state: str) -> bool:
        """检查是否可以转移到目标状态"""
        if self.current_state in self.terminal_states:
            return False

        allowed_transitions = self.transitions.get(self.current_state, [])
        return target_state in allowed_transitions

    def transition(
        self,
        target_state: str,
        reason: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> bool:
        """执行状态转移"""
        if not self.can_transition(target_state):
            return False

        old_state = self.current_state
        self.current_state = target_state

        # 记录状态变更
        transition_record = StateTransition(
            timestamp=datetime.now(timezone.utc),
            from_state=old_state,
            to_state=target_state,
            reason=reason,
            event_id=event_id,
        )
        self.state_history.append(transition_record)

        # 触发回调
        self._on_state_changed(old_state, target_state, reason)

        return True

    def _on_state_changed(
        self,
        old_state: str,
        new_state: str,
        reason: Optional[str],
    ) -> None:
        """状态变更回调（可重写）"""
        pass

    def get_state_history(self) -> List[Dict[str, Any]]:
        """获取状态历史"""
        return [t.to_dict() for t in self.state_history]

    def is_terminal(self) -> bool:
        """检查是否处于终态"""
        return self.current_state in self.terminal_states

    def get_time_in_state(self) -> float:
        """获取在当前状态的停留时间（秒）"""
        if not self.state_history:
            return 0.0

        last_change_time = self.state_history[-1].timestamp
        return (datetime.now(timezone.utc) - last_change_time).total_seconds()

    def reset(self, initial_state: str) -> None:
        """重置状态机"""
        self.current_state = initial_state
        self.state_history = []
        self.metadata = {}


# ============ 订单状态机 ============

class OrderStateMachine(BaseStateMachine):
    """订单状态机"""

    # 终态
    TERMINAL_STATES: set = {
        StateType.FILLED.value,
        StateType.CANCELLED.value,
        StateType.REJECTED.value,
        StateType.EXPIRED.value,
        StateType.ERROR.value,
    }

    # 状态转移规则
    TRANSITIONS: Dict[str, List[str]] = {
        StateType.CREATED.value: [
            StateType.VALIDATING.value,
            StateType.REJECTED.value,
        ],
        StateType.VALIDATING.value: [
            StateType.VALIDATED.value,
            StateType.REJECTED.value,
        ],
        StateType.VALIDATED.value: [
            StateType.PENDING_SUBMIT.value,
            StateType.REJECTED.value,
        ],
        StateType.PENDING_SUBMIT.value: [
            StateType.SUBMITTING.value,
            StateType.CANCELLED.value,
        ],
        StateType.SUBMITTING.value: [
            StateType.SUBMITTED.value,
            StateType.REJECTED.value,
            StateType.ERROR.value,
        ],
        StateType.SUBMITTED.value: [
            StateType.PARTIALLY_FILLED.value,
            StateType.FILLED.value,
            StateType.PENDING_CANCEL.value,
            StateType.EXPIRED.value,
            StateType.ERROR.value,
        ],
        StateType.PARTIALLY_FILLED.value: [
            StateType.FILLED.value,
            StateType.PENDING_CANCEL.value,
            StateType.EXPIRED.value,
            StateType.ERROR.value,
        ],
        StateType.PENDING_CANCEL.value: [
            StateType.CANCELLING.value,
            StateType.FILLED.value,  # 可能在取消过程中成交
        ],
        StateType.CANCELLING.value: [
            StateType.CANCELLED.value,
            StateType.FILLED.value,  # 可能在取消过程中成交
            StateType.ERROR.value,
        ],
    }

    def __init__(self, order_id: str, initial_state: str = StateType.CREATED.value):
        self.entity_id = order_id
        self.current_state = initial_state
        self.state_history = []
        self.metadata = {
            "order_id": order_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # 记录初始状态
        self.state_history.append(
            StateTransition(
                timestamp=datetime.now(timezone.utc),
                from_state="",
                to_state=initial_state,
                reason="initial",
            )
        )

    @property
    def terminal_states(self) -> set:
        """终态集合"""
        return self.TERMINAL_STATES

    @property
    def transitions(self) -> Dict[str, List[str]]:
        """状态转移规则"""
        return self.TRANSITIONS

    def validate(self, reason: Optional[str] = None) -> bool:
        """验证订单"""
        return self.transition(StateType.VALIDATING.value, reason)

    def validate_pass(self, reason: Optional[str] = None) -> bool:
        """验证通过"""
        return self.transition(StateType.VALIDATED.value, reason)

    def reject(self, reason: str) -> bool:
        """拒绝订单"""
        return self.transition(StateType.REJECTED.value, reason)

    def submit(self, reason: Optional[str] = None) -> bool:
        """提交订单"""
        return self.transition(StateType.SUBMITTING.value, reason)

    def submitted(self, broker_order_id: Optional[str] = None) -> bool:
        """订单已提交"""
        if broker_order_id:
            self.metadata["broker_order_id"] = broker_order_id
        return self.transition(StateType.SUBMITTED.value, "submitted_to_broker")

    def partial_fill(self, fill_id: str, filled_qty: float) -> bool:
        """部分成交"""
        self.metadata["last_fill_id"] = fill_id
        self.metadata["filled_qty"] = filled_qty
        return self.transition(StateType.PARTIALLY_FILLED.value, "partial_fill")

    def fill(self, fill_id: str, filled_qty: float, price: float) -> bool:
        """完全成交"""
        self.metadata["last_fill_id"] = fill_id
        self.metadata["filled_qty"] = filled_qty
        self.metadata["fill_price"] = price
        self.metadata["filled_at"] = datetime.now(timezone.utc).isoformat()
        return self.transition(StateType.FILLED.value, "filled")

    def request_cancel(self, reason: str = "user_request") -> bool:
        """请求取消"""
        return self.transition(StateType.PENDING_CANCEL.value, reason)

    def cancel(self, reason: str = "cancelled") -> bool:
        """取消订单"""
        return self.transition(StateType.CANCELLING.value, reason)

    def cancelled(self, reason: str = "cancelled") -> bool:
        """订单已取消"""
        self.metadata["cancelled_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["cancel_reason"] = reason
        return self.transition(StateType.CANCELLED.value, reason)

    def expire(self, reason: str = "expired") -> bool:
        """订单过期"""
        return self.transition(StateType.EXPIRED.value, reason)

    def error(self, error_message: str) -> bool:
        """订单错误"""
        self.metadata["error"] = error_message
        return self.transition(StateType.ERROR.value, error_message)

    def get_filled_quantity(self) -> float:
        """获取已成交数量"""
        return self.metadata.get("filled_qty", 0.0)

    def get_fill_price(self) -> Optional[float]:
        """获取成交价格"""
        return self.metadata.get("fill_price")

    def get_status_dict(self) -> Dict[str, Any]:
        """获取状态信息字典"""
        return {
            "order_id": self.entity_id,
            "state": self.current_state,
            "is_terminal": self.is_terminal(),
            "history": self.get_state_history(),
            "metadata": self.metadata,
            "time_in_state": self.get_time_in_state(),
        }


# ============ 策略状态机 ============

class StrategyStateMachine(BaseStateMachine):
    """策略状态机"""

    # 状态转移规则
    TRANSITIONS: Dict[str, List[str]] = {
        StrategyState.INITIALIZING.value: [
            StrategyState.READY.value,
            StrategyState.ERROR.value,
            StrategyState.DISABLED.value,
        ],
        StrategyState.READY.value: [
            StrategyState.RUNNING.value,
            StrategyState.DISABLED.value,
        ],
        StrategyState.RUNNING.value: [
            StrategyState.PAUSED.value,
            StrategyState.STOPPING.value,
            StrategyState.ERROR.value,
        ],
        StrategyState.PAUSED.value: [
            StrategyState.RUNNING.value,
            StrategyState.STOPPING.value,
            StrategyState.ERROR.value,
        ],
        StrategyState.STOPPING.value: [
            StrategyState.STOPPED.value,
            StrategyState.ERROR.value,
        ],
        StrategyState.STOPPED.value: [
            StrategyState.READY.value,
            StrategyState.DISABLED.value,
        ],
        StrategyState.ERROR.value: [
            StrategyState.READY.value,
            StrategyState.DISABLED.value,
        ],
        StrategyState.DISABLED.value: [
            StrategyState.READY.value,
        ],
    }

    # 终态
    TERMINAL_STATES: set = {
        StrategyState.STOPPED.value,
        StrategyState.DISABLED.value,
    }

    def __init__(self, strategy_id: str):
        self.entity_id = strategy_id
        self.current_state = StrategyState.INITIALIZING.value
        self.state_history = []
        self.metadata = {
            "strategy_id": strategy_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error_count": 0,
            "last_error": None,
        }
        self.error_count = 0

        # 记录初始状态
        self.state_history.append(
            StateTransition(
                timestamp=datetime.now(timezone.utc),
                from_state="",
                to_state=StrategyState.INITIALIZING.value,
                reason="initial",
            )
        )

    @property
    def terminal_states(self) -> set:
        return self.TERMINAL_STATES

    @property
    def transitions(self) -> Dict[str, List[str]]:
        return self.TRANSITIONS

    def initialize_complete(self, reason: Optional[str] = None) -> bool:
        """初始化完成"""
        return self.transition(StrategyState.READY.value, reason or "init_complete")

    def start(self, reason: Optional[str] = None) -> bool:
        """启动策略"""
        return self.transition(StrategyState.RUNNING.value, reason or "start")

    def pause(self, reason: Optional[str] = None) -> bool:
        """暂停策略"""
        return self.transition(StrategyState.PAUSED.value, reason or "pause")

    def resume(self, reason: Optional[str] = None) -> bool:
        """恢复策略"""
        return self.transition(StrategyState.RUNNING.value, reason or "resume")

    def stop(self, reason: Optional[str] = None) -> bool:
        """停止策略"""
        return self.transition(StrategyState.STOPPING.value, reason or "stop")

    def stopped(self, reason: Optional[str] = None) -> bool:
        """策略已停止"""
        self.metadata["stopped_at"] = datetime.now(timezone.utc).isoformat()
        return self.transition(StrategyState.STOPPED.value, reason or "stopped")

    def error(self, error_message: str, recoverable: bool = True) -> bool:
        """策略错误"""
        self.error_count += 1
        self.metadata["error_count"] = self.error_count
        self.metadata["last_error"] = {
            "message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": recoverable,
        }
        return self.transition(StrategyState.ERROR.value, error_message)

    def disable(self, reason: str = "disabled") -> bool:
        """禁用策略"""
        self.metadata["disabled_at"] = datetime.now(timezone.utc).isoformat()
        return self.transition(StrategyState.DISABLED.value, reason)

    def enable(self, reason: Optional[str] = None) -> bool:
        """启用策略"""
        return self.transition(StrategyState.READY.value, reason or "enabled")

    def record_signal(self):
        """记录信号生成"""
        if "signal_count" not in self.metadata:
            self.metadata["signal_count"] = 0
        self.metadata["signal_count"] += 1
        self.metadata["last_activity"] = datetime.now(timezone.utc).isoformat()

    def record_order(self):
        """记录订单创建"""
        if "order_count" not in self.metadata:
            self.metadata["order_count"] = 0
        self.metadata["order_count"] += 1
        self.metadata["last_activity"] = datetime.now(timezone.utc).isoformat()

    def get_status_dict(self) -> Dict[str, Any]:
        """获取状态信息字典"""
        return {
            "strategy_id": self.entity_id,
            "state": self.current_state,
            "is_terminal": self.is_terminal(),
            "history": self.get_state_history(),
            "metadata": self.metadata,
            "error_count": self.error_count,
            "time_in_state": self.get_time_in_state(),
        }


# ============ 辅助函数 ============

def create_order_state_machine(order_id: str) -> OrderStateMachine:
    """创建订单状态机"""
    return OrderStateMachine(order_id)


def create_strategy_state_machine(strategy_id: str) -> StrategyStateMachine:
    """创建策略状态机"""
    return StrategyStateMachine(strategy_id)


def state_to_enum(state_str: str) -> Optional[StateType]:
    """将状态字符串转换为枚举"""
    try:
        return StateType(state_str)
    except ValueError:
        return None


def strategy_state_to_enum(state_str: str) -> Optional[StrategyState]:
    """将策略状态字符串转换为枚举"""
    try:
        return StrategyState(state_str)
    except ValueError:
        return None

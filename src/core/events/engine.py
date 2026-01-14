"""
增强的事件引擎
支持优先级队列、事件历史记录和状态机集成
"""

import asyncio
import enum
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
import heapq
import uuid

from src.utils.logger import get_logger
from src.core.events import Event, EventPriority, MarketEvent
from src.core.state_machine import OrderStateMachine, StrategyStateMachine

logger = logging.getLogger("event_engine")


class EventHandler:
    """事件处理器"""

    def __init__(
        self,
        handler: Callable,
        priority: int = 0,
        once: bool = False,
    ):
        self.handler = handler
        self.priority = priority
        self.once = once
        self.call_count = 0
        # 缓存is_coroutine检查结果，避免重复调用
        self._is_coroutine = asyncio.iscoroutinefunction(handler)

    async def call(self, event: Event) -> None:
        """调用处理器"""
        self.call_count += 1
        try:
            if self._is_coroutine:
                await self.handler(event)
            else:
                self.handler(event)
        except Exception as e:
            logger.error(f"事件处理器执行失败: {self.handler.__name__}, 错误: {e}")

    def __lt__(self, other: "EventHandler") -> bool:
        """比较方法（用于优先级队列）"""
        return self.priority < other.priority


@dataclass
class EventSubscription:
    """事件订阅"""
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    handler: Optional[Callable] = None
    priority: int = 0
    once: bool = False
    active: bool = True


class UnifiedEventEngine:
    """统一事件引擎"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._queue: List = []  # 优先级队列，使用List存储可比较对象
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        self._active = False
        self._worker_task: Optional[asyncio.Task] = None
        self._condition = asyncio.Condition()
        self._max_history_size = self.config.get("max_history_size", 10000)
        self._event_history: deque = deque(maxlen=self._max_history_size)
        self._order_state_machines: Dict[str, OrderStateMachine] = {}
        self._strategy_state_machines: Dict[str, StrategyStateMachine] = {}
        self._metrics = EventEngineMetrics()

    def register(
        self,
        event_type: str,
        handler: Callable,
        priority: int = 0,
    ) -> str:
        """注册事件处理器"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        event_handler = EventHandler(handler=handler, priority=priority)
        self._handlers[event_type].append(event_handler)
        # 按优先级排序（高优先级在前）
        self._handlers[event_type].sort(key=lambda h: -h.priority)

        logger.debug(f"已注册事件处理器: {event_type} (优先级: {priority})")
        return str(uuid.uuid4())

    def unregister(self, event_type: str, handler: Callable) -> bool:
        """取消注册事件处理器"""
        if event_type in self._handlers:
            original_len = len(self._handlers[event_type])
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h.handler != handler
            ]
            return len(self._handlers[event_type]) < original_len
        return False

    def subscribe(
        self,
        event_type: str,
        handler: Callable,
        priority: int = 0,
        once: bool = False,
    ) -> str:
        """订阅事件（返回订阅ID）"""
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []

        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            priority=priority,
            once=once,
        )
        self._subscriptions[event_type].append(subscription)

        # 同时注册为处理器
        self.register(event_type, handler, priority)

        return subscription.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        for event_type, subs in self._subscriptions.items():
            for sub in subs:
                if sub.subscription_id == subscription_id:
                    sub.active = False
                    if sub.handler is not None:
                        self.unregister(event_type, sub.handler)
                    return True
        return False

    async def put(self, event: Event) -> None:
        """将事件送入队列"""
        async with self._condition:
            # 使用负优先级使高优先级事件先处理
            heapq.heappush(self._queue, (-event.priority.value, event))
            self._condition.notify_all()
            self._metrics.events_queued += 1
            logger.debug(f"事件入队: {event.event_type} (P={event.priority.value})")

    async def start(self) -> None:
        """启动事件引擎"""
        if self._active:
            return
        self._active = True
        self._worker_task = asyncio.create_task(self._run())
        logger.info("事件引擎已启动")

    async def stop(self) -> None:
        """停止事件引擎"""
        self._active = False
        async with self._condition:
            self._condition.notify_all()
        if self._worker_task:
            await self._worker_task
        logger.info("事件引擎已停止")

    async def _run(self) -> None:
        """核心处理循环"""
        while self._active:
            event = None
            async with self._condition:
                while not self._queue and self._active:
                    await self._condition.wait()

                if not self._active:
                    break

                if self._queue:
                    _, event = heapq.heappop(self._queue)

            if event:
                start_time = asyncio.get_running_loop().time()
                await self._process_event(event)
                process_time = asyncio.get_running_loop().time() - start_time
                # 使用增量平均计算更高效
                self._metrics.avg_process_time = (
                    self._metrics.avg_process_time * 0.9 + process_time * 0.1
                )

    async def _process_event(self, event: Event) -> None:
        """处理单个事件"""
        # 记录事件历史
        self._add_to_history(event)

        handlers = self._handlers.get(event.event_type)
        if not handlers:
            return

        tasks = []
        for handler in handlers:
            try:
                if handler._is_coroutine:
                    tasks.append(handler.call(event))
                else:
                    # 同步处理器在单独线程中执行
                    loop = asyncio.get_running_loop()
                    tasks.append(loop.run_in_executor(None, handler.call, event))
            except Exception as e:
                logger.error(f"事件 {event.event_type} 处理出错: {e}")
                self._metrics.errors += 1

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # 更新指标
        self._metrics.events_processed += 1

    def _add_to_history(self, event: Event) -> None:
        """添加事件到历史记录（deque自动管理大小）"""
        self._event_history.append(event)

    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """获取事件历史"""
        if event_type:
            filtered = [e for e in self._event_history if e.event_type == event_type]
        else:
            filtered = list(self._event_history)
        return filtered[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """获取引擎指标"""
        return {
            "events_queued": self._metrics.events_queued,
            "events_processed": self._metrics.events_processed,
            "errors": self._metrics.errors,
            "avg_process_time": self._metrics.avg_process_time,
            "queue_size": len(self._queue),
            "active_handlers": sum(len(h) for h in self._handlers.values()),
            "active_subscriptions": sum(
                1 for subs in self._subscriptions.values() for s in subs if s.active
            ),
        }

    # ============ 订单状态机集成 ============

    def get_order_state_machine(self, order_id: str) -> OrderStateMachine:
        """获取订单状态机"""
        if order_id not in self._order_state_machines:
            self._order_state_machines[order_id] = OrderStateMachine(order_id)
        return self._order_state_machines[order_id]

    def update_order_state(
        self,
        order_id: str,
        new_state: str,
        reason: Optional[str] = None,
    ) -> bool:
        """更新订单状态"""
        state_machine = self.get_order_state_machine(order_id)
        return state_machine.transition(new_state, reason)

    # ============ 策略状态机集成 ============

    def get_strategy_state_machine(self, strategy_id: str) -> StrategyStateMachine:
        """获取策略状态机"""
        if strategy_id not in self._strategy_state_machines:
            self._strategy_state_machines[strategy_id] = StrategyStateMachine(
                strategy_id
            )
        return self._strategy_state_machines[strategy_id]

    def update_strategy_state(
        self,
        strategy_id: str,
        new_state: str,
        reason: Optional[str] = None,
    ) -> bool:
        """更新策略状态"""
        state_machine = self.get_strategy_state_machine(strategy_id)
        return state_machine.transition(new_state, reason)

    # ============ 便捷方法 ============

    async def publish_market_event(
        self,
        symbol: str,
        data: Dict[str, Any],
        event_type: str = "bar",
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """发布市场事件"""
        event = MarketEvent(
            symbol=symbol,
            data=data,
            priority=priority,
            source="market_adapter",
        )
        # 设置event_type（通过覆盖默认值）
        event.event_type = event_type
        await self.put(event)

    async def publish_signal_event(
        self,
        strategy_id: str,
        signal_type: str,
        action: str,
        symbol: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """发布信号事件"""
        from src.core.events import SignalEvent

        event = SignalEvent(
            strategy_id=strategy_id,
            signal_type=signal_type,
            action=action,
            symbol=symbol,
            strength=strength,
            metadata=metadata or {},
            priority=EventPriority.HIGH,
            source="strategy",
        )
        await self.put(event)

    async def publish_order_event(
        self,
        event_type: str,
        order_id: str,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        strategy_id: str,
        **kwargs,
    ) -> None:
        """发布订单事件"""
        from src.core.events import (
            OrderCreatedEvent,
            OrderSubmittedEvent,
            OrderFilledEvent,
            OrderCancelledEvent,
            OrderRejectedEvent,
        )

        event_map = {
            "order_created": OrderCreatedEvent,
            "order_submitted": OrderSubmittedEvent,
            "order_filled": OrderFilledEvent,
            "order_cancelled": OrderCancelledEvent,
            "order_rejected": OrderRejectedEvent,
        }

        event_class = event_map.get(event_type)
        if not event_class:
            logger.warning(f"未知的事件类型: {event_type}")
            return

        event = event_class(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            strategy_id=strategy_id,
            **kwargs,
        )
        await self.put(event)


@dataclass
class EventEngineMetrics:
    """事件引擎指标"""
    events_queued: int = 0
    events_processed: int = 0
    errors: int = 0
    avg_process_time: float = 0.0


# 全局事件引擎单例
event_engine = UnifiedEventEngine()

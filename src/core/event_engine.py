"""
异步事件引擎
支持优先级队列和异步事件处理
"""

import asyncio
import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import heapq
from src.utils.logger import get_logger

logger = get_logger("event_engine")

class EventPriority(enum.IntEnum):
    """事件优先级（数值越小优先级越高）"""
    CRITICAL = 0
    HIGH = 10
    NORMAL = 20
    LOW = 30

@dataclass(order=True)
class Event:
    """事件对象"""
    priority: int
    timestamp: datetime = field(compare=False)
    event_type: str = field(compare=False)
    data: Any = field(default=None, compare=False)
    
    def __init__(self, event_type: str, data: Any = None, priority: EventPriority = EventPriority.NORMAL):
        self.event_type = event_type
        self.data = data
        self.priority = int(priority)
        self.timestamp = datetime.now()

class EventEngine:
    """优先级事件引擎"""
    
    def __init__(self):
        self._queue = []  # 优先级队列 (heapq)
        self._handlers: Dict[str, List[Callable]] = {}
        self._active = False
        self._worker_task: Optional[asyncio.Task] = None
        self._condition = asyncio.Condition()

    def register(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"已注册事件处理器: {event_type}")

    def unregister(self, event_type: str, handler: Callable):
        """取消注册事件处理器"""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"已取消注册事件处理器: {event_type}")

    async def put(self, event: Event):
        """将事件送入队列"""
        async with self._condition:
            heapq.heappush(self._queue, event)
            self._condition.notify_all()
            logger.trace(f"事件入队: {event.event_type} (P={event.priority})")

    async def start(self):
        """启动事件引擎"""
        if self._active:
            return
        self._active = True
        self._worker_task = asyncio.create_task(self._run())
        logger.info("事件引擎已启动")

    async def stop(self):
        """停止事件引擎"""
        self._active = False
        async with self._condition:
            self._condition.notify_all()
        if self._worker_task:
            await self._worker_task
        logger.info("事件引擎已停止")

    async def _run(self):
        """核心处理循环"""
        while self._active:
            event = None
            async with self._condition:
                while not self._queue and self._active:
                    await self._condition.wait()
                
                if not self._active:
                    break
                    
                if self._queue:
                    event = heapq.heappop(self._queue)
            
            if event:
                await self._process_event(event)

    async def _process_event(self, event: Event):
        """处理单个事件"""
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            return

        tasks = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"事件 {event.event_type} 处理出错: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# 全局单例
event_engine = EventEngine()

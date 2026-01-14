"""
IB实时数据适配器
用于实盘交易的Interactive Brokers数据连接
"""

from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timezone, timedelta
import asyncio
import logging
import pandas as pd

from src.data.adapter import LiveDataAdapter, DataMetadata, DataLoadError
from src.utils.logger import get_logger

logger = logging.getLogger("ib_adapter")


class IBDataStream:
    """IB数据流管理器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ib = None
        self.connected = False
        self.subscriptions: Dict[str, Dict] = {}
        self.reconnect_delay = self.config.get("reconnect_delay", 5)
        self.heartbeat_interval = self.config.get("heartbeat_interval", 30)

    async def connect(self) -> bool:
        """连接到IB"""
        try:
            # 这里应该使用 ib_insync 库
            # 由于是代码框架实现，使用伪代码
            logger.info("连接到Interactive Brokers...")
            # self.ib = IB()
            # await self.ib.connectAsync(...)
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"IB连接失败: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """断开连接"""
        if self.ib:
            self.ib.disconnect()
        self.connected = False

    async def subscribe_market_data(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """订阅市场数据"""
        subscription_id = f"{symbol}_{id(callback)}"

        self.subscriptions[subscription_id] = {
            "symbol": symbol,
            "callback": callback,
            "active": True,
        }

        # 启动数据订阅
        logger.info(f"订阅市场数据: {symbol}")

        return subscription_id

    async def unsubscribe_market_data(self, subscription_id: str) -> bool:
        """取消订阅市场数据"""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            return True
        return False

    async def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新数据"""
        # 返回模拟的最新数据
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last": 150.0,
            "bid": 149.9,
            "ask": 150.1,
            "volume": 1000000,
        }

    async def get_available_symbols(self) -> List[str]:
        """获取可用标的列表"""
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]


class IBLiveAdapter(LiveDataAdapter):
    """IB实时数据适配器"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        config["cache_ttl"] = config.get("cache_ttl", 1)  # 实时数据缓存1秒
        super().__init__(IBDataStream(config), config)

        self.data_stream: IBDataStream = self.data_stream
        self.connection_monitor_task = None

    async def connect(self) -> bool:
        """建立连接"""
        return await self.data_stream.connect()

    async def disconnect(self):
        """断开连接"""
        await self.data_stream.disconnect()
        if self.connection_monitor_task:
            self.connection_monitor_task.cancel()

    async def _ensure_connection(self):
        """确保连接有效"""
        if not self.data_stream.connected:
            await self.connect()

    async def _start_connection_monitor(self):
        """启动连接监控"""
        self.connection_monitor_task = asyncio.create_task(self._monitor_connection())

    async def _monitor_connection(self):
        """监控连接状态"""
        while True:
            try:
                await asyncio.sleep(self.data_stream.heartbeat_interval)
                if not self.data_stream.connected:
                    logger.warning("IB连接断开，尝试重连...")
                    await self.connect()
            except asyncio.CancelledError:
                break

    async def subscribe_real_time(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """订阅实时数据"""
        await self._ensure_connection()

        # 启动连接监控（首次订阅时）
        if self.connection_monitor_task is None:
            await self._start_connection_monitor()

        return await self.data_stream.subscribe_market_data(symbol, callback)

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        await self._ensure_connection()

        try:
            data = await self.data_stream.get_latest_data(symbol)
            if data and "last" in data:
                return float(data["last"])
        except Exception as e:
            logger.error(f"获取当前价格失败: {symbol}: {e}")

        return None

    async def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取历史数据（IB实时适配器不支持历史数据，返回空DataFrame）"""
        logger.warning(f"IBLiveAdapter 不支持历史数据获取: {symbol} {timeframe} {start} - {end}")
        return pd.DataFrame()

    async def unsubscribe_real_time(self, subscription_id: str) -> bool:
        """取消订阅实时数据"""
        return await self.data_stream.unsubscribe_market_data(subscription_id)

    async def get_available_symbols(self) -> List[str]:
        """获取可用标的列表"""
        await self._ensure_connection()
        return await self.data_stream.get_available_symbols()

    async def get_data_metadata(
        self, symbol: str, timeframe: str
    ) -> DataMetadata:
        """获取数据元数据"""
        return DataMetadata(
            symbol=symbol,
            timeframe=timeframe,
            last_updated=datetime.now(timezone.utc),
            source="ib_live",
        )


class IBEventAdapter:
    """IB事件适配器 - 将IB事件转换为系统事件"""

    def __init__(self, event_engine):
        self.event_engine = event_engine
        self.logger = get_logger("ib_event_adapter")
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.handlers[event_type] = handler

    async def on_connected(self):
        """连接事件"""
        from src.core.events import SystemEvent

        event = SystemEvent(
            data={"status": "connected", "timestamp": datetime.now(timezone.utc).isoformat(), "event_type": "connection"},
            source="ib_adapter",
        )
        await self.event_engine.put(event)

    async def on_disconnected(self):
        """断开连接事件"""
        from src.core.events import SystemEvent

        event = SystemEvent(
            data={"status": "disconnected", "timestamp": datetime.now(timezone.utc).isoformat(), "event_type": "connection"},
            source="ib_adapter",
        )
        await self.event_engine.put(event)

    async def on_order_status(self, order_id: str, status: str, filled: float):
        """订单状态事件"""
        from src.core.events import OrderEvent

        event = OrderEvent(
            order_id=order_id,
            data={"status": status, "filled": filled},
            source="ib_adapter",
        )
        await self.event_engine.put(event)

    async def on_execution(self, order_id: str, trade):
        """成交事件"""
        from src.core.events import OrderFilledEvent

        event = OrderFilledEvent(
            order_id=order_id,
            symbol=trade.contract.symbol,
            order_type="market",
            side=trade.order.action,
            quantity=trade.fills[0].execution.shares,
            fill_id=trade.fills[0].execution.execId,
            fill_price=trade.fills[0].execution.price,
            fill_quantity=trade.fills[0].execution.shares,
            fill_time=trade.fills[0].execution.time,
            commission=trade.fills[0].commissionReport.commission if trade.fills[0].commissionReport else 0,
            source="ib_adapter",
        )
        await self.event_engine.put(event)

"""
执行器抽象接口
抽象 ExecutionProvider：回测用模拟撮合，实盘用 IB 执行；统一发 Order 事件
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, AsyncIterator
import asyncio
import uuid

from src.core.types import OrderIntent, Fill, PortfolioState, OrderSide, OrderType
from src.core.events import (
    OrderCreatedEvent, OrderSubmittedEvent, OrderFilledEvent,
    OrderCancelledEvent, OrderRejectedEvent,
)
from src.core.events.engine import UnifiedEventEngine
from src.utils.logger import get_logger

logger = get_logger("execution_provider")


@dataclass
class ExecutionProviderConfig:
    """执行器配置"""
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    fill_price_type: str = "close"  # "close", "open", "high", "low"
    missing_bar_handling: str = "previous"  # "previous", "next", "fail"
    initial_capital: float = 100000.0


class ExecutionProvider(ABC):
    """
    执行器抽象基类

    实现方式：
    - SimulatedExecutionProvider：模拟撮合（回测用）
    - IBExecutionProvider：Interactive Brokers（实盘用）
    - CCXTExecutionProvider：加密货币交易所（实盘用）
    """

    # 同步兼容方法的默认属性（子类可覆盖）
    _orders: Dict[str, OrderIntent] = {}
    _fills: List[Fill] = []
    _current_prices: Dict[str, float] = {}
    config: ExecutionProviderConfig = field(default_factory=ExecutionProviderConfig)

    @abstractmethod
    async def initialize(self) -> None:
        """初始化执行器"""
        pass

    @abstractmethod
    async def submit_order(self, order: OrderIntent) -> Dict[str, Any]:
        """提交订单"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """取消订单"""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        pass

    @abstractmethod
    async def poll_fills(self) -> List[Fill]:
        """轮询成交"""
        pass

    @abstractmethod
    def get_portfolio_state(self) -> PortfolioState:
        """获取组合状态"""
        pass

    @abstractmethod
    def reset(self, capital: Optional[float] = None) -> None:
        """重置执行器"""
        pass

    # ============ 同步兼容方法（用于向后兼容） ============

    def place_order(self, intent: OrderIntent, current_ts: Optional[datetime] = None) -> str:
        """同步订单提交（兼容 SimulatedBroker 接口）- 同步执行

        Args:
            intent: 订单意图
            current_ts: 当前时间戳（回测时用于设置成交时间），如果不提供则使用系统时间
        """
        order_id = intent.order_id or str(uuid.uuid4())
        intent.order_id = order_id

        # 使用传入的时间戳或当前系统时间
        fill_ts = current_ts if current_ts is not None else datetime.now(timezone.utc)

        try:
            # 1. 创建订单
            self._orders[order_id] = intent

            # 2. 计算成交价格（同步方式）
            fill_price = self._calculate_fill_price_sync(intent)

            # 3. 计算手续费和滑点（同步方式）
            commission = self._calculate_commission(intent, fill_price)
            slippage = self._calculate_slippage_sync(intent, fill_price)

            # 4. 创建成交记录（使用 current_ts 作为成交时间）
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order_id,
                ts_fill_utc=fill_ts,
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                price=fill_price,
                fee=commission,
                slippage_est=slippage,
            )

            self._fills.append(fill)

            # 5. 更新当前价格（用于后续计算最终权益）
            self._current_prices[intent.symbol] = fill_price

            # 6. 更新组合状态 - 调用子类的方法
            self._update_portfolio_sync(fill)

            logger.info(
                f"模拟成交: {order_id} {intent.symbol} {intent.side.value if hasattr(intent.side, 'value') else intent.side} "
                f"{intent.qty} @ {fill_price:.4f} (ts={fill_ts.strftime('%Y-%m-%d')}, commission={commission:.2f}, slippage={slippage:.4f})"
            )

            return order_id
        except Exception as e:
            logger.error(f"订单执行失败 {order_id}: {e}")
            return order_id

    def _calculate_fill_price_sync(self, order: OrderIntent) -> float:
        """同步计算成交价格"""
        if order.limit_price:
            return float(order.limit_price)
        
        # 从当前价格数据获取
        if order.symbol in self._current_prices:
            return float(self._current_prices[order.symbol])
        
        # 没有价格数据时抛出异常，避免使用无意义的默认值
        raise ValueError(
            f"无法计算成交价格: 标的 {order.symbol} 没有可用价格数据。 "
            f"请确保已正确设置市场数据。"
        )
    
    def _calculate_slippage_sync(self, order: OrderIntent, fill_price: float) -> float:
        """同步计算滑点（默认实现）"""
        return fill_price * self.config.slippage_rate

    def _calculate_commission(self, order: OrderIntent, fill_price: float) -> float:
        """计算手续费（默认实现）"""
        return order.qty * fill_price * self.config.commission_rate

    def _update_portfolio(self, fill: Fill) -> None:
        """更新组合状态（默认实现 - 空操作）"""
        pass
    
    def _update_portfolio_sync(self, fill: Fill) -> None:
        """同步更新组合状态（默认实现 - 空操作）"""
        pass

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderIntent]:
        """获取未成交订单（兼容接口）"""
        # 简化实现：返回空列表
        return []

    def update_account_state(self, current_ts: datetime, current_prices: Dict[str, Any]) -> None:
        """更新账户状态（兼容接口）"""
        # 更新当前价格
        self._current_prices.update(current_prices)

    def on_tick(self, ts: datetime, bars: Dict[str, Any]) -> List[Fill]:
        """处理 tick 数据（兼容 SimulatedBroker 接口）- 返回已成交的订单"""
        # 返回累积的 fills，但不清空，因为 engine 会在 _context.all_fills 中保存副本
        fills = self._fills.copy()
        # 不清空 self._fills，保留用于 get_fills() 等方法
        return fills


class SimulatedExecutionProvider(ExecutionProvider):
    """
    模拟撮合执行器（用于回测）
    
    特点：
    - 从 DataFeed 获取真实市场价格
    - 支持滑点模拟
    - 自动发布订单事件到事件引擎
    """
    
    def __init__(
        self,
        config: Optional[ExecutionProviderConfig] = None,
        data_feed: Optional[Any] = None,
        event_engine: Optional[UnifiedEventEngine] = None,
    ):
        self.config = config or ExecutionProviderConfig()
        self._data_feed = data_feed
        self._event_engine = event_engine
        self._portfolio = PortfolioState(
            cash=self.config.initial_capital,
            positions={},
            avg_price={},
            equity=self.config.initial_capital,
        )
        self._fills: List[Fill] = []
        self._orders: Dict[str, OrderIntent] = {}
        self._current_prices: Dict[str, float] = {}  # 用于同步计算成交价格
        self._closed = False
        
        logger.info(
            f"SimulatedExecutionProvider 初始化完成: "
            f"commission={self.config.commission_rate}, "
            f"slippage={self.config.slippage_rate}, "
            f"fill_price={self.config.fill_price_type}"
        )
    
    def set_data_feed(self, data_feed: Any) -> None:
        """设置数据源（用于获取实时价格）"""
        self._data_feed = data_feed
    
    def set_event_engine(self, event_engine: UnifiedEventEngine) -> None:
        """设置事件引擎（用于发布订单事件）"""
        self._event_engine = event_engine
    
    async def initialize(self) -> None:
        """初始化执行器"""
        logger.info("SimulatedExecutionProvider 已初始化")
    
    async def submit_order(self, order: OrderIntent) -> Dict[str, Any]:
        """提交订单"""
        order_id = order.order_id or str(uuid.uuid4())
        order.order_id = order_id
        
        # 1. 创建订单
        self._orders[order_id] = order
        
        # 2. 发布订单创建事件
        if self._event_engine:
            await self._event_engine.publish_order_event(
                event_type="order_created",
                order_id=order_id,
                symbol=order.symbol,
                order_type=order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                side=order.side.value if hasattr(order.side, 'value') else str(order.side),
                quantity=order.qty,
                strategy_id=order.strategy_id,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )
        
        # 3. 计算成交价格
        fill_price = await self._calculate_fill_price(order)
        
        # 4. 计算手续费和滑点
        commission = self._calculate_commission(order, fill_price)
        slippage = await self._calculate_slippage(order, fill_price)
        
        # 5. 创建成交记录
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            ts_fill_utc=datetime.now(timezone.utc),
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=commission,
            slippage_est=slippage,
        )
        
        self._fills.append(fill)
        
        # 6. 更新组合状态
        self._update_portfolio(fill)
        
        # 7. 发布订单成交事件
        if self._event_engine:
            await self._event_engine.publish_order_event(
                event_type="order_filled",
                order_id=order_id,
                symbol=order.symbol,
                order_type=order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                side=order.side.value if hasattr(order.side, 'value') else str(order.side),
                quantity=order.qty,
                strategy_id=order.strategy_id,
                fill_id=fill.fill_id,
                fill_price=fill_price,
                fill_quantity=order.qty,
                fill_time=fill.ts_fill_utc,
                commission=commission,
                slippage=slippage,
            )
        
        logger.info(
            f"模拟成交: {order_id} {order.symbol} {order.side.value if hasattr(order.side, 'value') else order.side} "
            f"{order.qty} @ {fill_price:.4f} (commission={commission:.2f}, slippage={slippage:.4f})"
        )
        
        return {
            "success": True,
            "order_id": order_id,
            "fill_id": fill.fill_id,
            "fill_price": fill_price,
            "fill_quantity": order.qty,
            "commission": commission,
            "slippage": slippage,
        }
    
    async def _calculate_fill_price(self, order: OrderIntent) -> float:
        """计算成交价格（从市场数据获取真实价格）"""
        if self._data_feed is None:
            logger.warning("没有设置 data_feed，使用默认价格 100.0")
            return 100.0
        
        # 获取当前K线数据
        bar = await self._data_feed.get_current_bar(order.symbol)
        
        if not bar:
            logger.warning(f"没有找到 {order.symbol} 的市场数据，使用默认价格 100.0")
            return 100.0
        
        # 根据配置获取价格类型
        price_type = self.config.fill_price_type
        base_price = bar.get(price_type, bar.get('close', 100.0))
        
        # 考虑滑点
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + self.config.slippage_rate)
        else:
            fill_price = base_price * (1 - self.config.slippage_rate)
        
        return fill_price
    
    def _calculate_commission(self, order: OrderIntent, fill_price: float) -> float:
        """计算手续费"""
        return order.qty * fill_price * self.config.commission_rate
    
    async def _calculate_slippage(self, order: OrderIntent, fill_price: float) -> float:
        """计算滑点（相对于参考价）"""
        if self._data_feed is None:
            return 0.0

        # 获取参考价
        bar = await self._data_feed.get_current_bar(order.symbol)
        if not bar:
            return 0.0

        reference_price = bar.get('close', fill_price)
        if reference_price == 0:
            return 0.0

        return abs(fill_price - reference_price) / reference_price
    
    def _update_portfolio(self, fill: Fill) -> None:
        """更新组合状态"""
        symbol = fill.symbol
        
        if fill.side == OrderSide.BUY:
            # 买入：更新持仓和平均成本
            current_qty = self._portfolio.positions.get(symbol, 0.0)
            current_avg = self._portfolio.avg_price.get(symbol, fill.price)
            
            new_qty = current_qty + fill.qty
            if new_qty != 0:
                new_avg = (current_qty * current_avg + fill.qty * fill.price) / new_qty
            else:
                new_avg = fill.price
            
            self._portfolio.positions[symbol] = new_qty
            self._portfolio.avg_price[symbol] = new_avg
            self._portfolio.cash -= fill.qty * fill.price + fill.fee
            
        else:
            # 卖出：更新持仓
            current_qty = self._portfolio.positions.get(symbol, 0.0)
            
            new_qty = current_qty - fill.qty
            if abs(new_qty) < 1e-8:
                new_qty = 0.0
                self._portfolio.avg_price[symbol] = 0.0
            
            self._portfolio.positions[symbol] = new_qty
            self._portfolio.cash += fill.qty * fill.price - fill.fee
        
        # 更新权益
        self._update_equity()
    
    def _update_portfolio_sync(self, fill: Fill) -> None:
        """同步更新组合状态（与 _update_portfolio 相同）"""
        self._update_portfolio(fill)
    
    def _update_equity(self) -> None:
        """更新权益"""
        positions_value = sum(
            abs(qty) * self._portfolio.avg_price.get(symbol, 0.0)
            for symbol, qty in self._portfolio.positions.items()
            if abs(qty) > 1e-8
        )
        self._portfolio.equity = self._portfolio.cash + positions_value
    
    async def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """取消订单"""
        if order_id not in self._orders:
            return False
        
        # 移除订单
        del self._orders[order_id]
        
        # 发布订单取消事件
        if self._event_engine:
            await self._event_engine.publish_order_event(
                event_type="order_cancelled",
                order_id=order_id,
                symbol="",
                order_type="",
                side="",
                quantity=0,
                strategy_id="",
                reason=reason,
            )
        
        logger.info(f"订单取消: {order_id} (reason: {reason})")
        return True
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        return {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, 'value') else order.side,
            "quantity": order.qty,
            "filled_quantity": order.qty,  # 模拟立即成交
            "status": "FILLED",
            "filled_price": self._fills[-1].price if self._fills else None,
        }
    
    async def poll_fills(self) -> List[Fill]:
        """轮询成交"""
        return self._fills.copy()
    
    def get_portfolio_state(self) -> PortfolioState:
        """获取组合状态"""
        self._update_equity()
        return self._portfolio
    
    def get_fills(self) -> List[Fill]:
        """获取所有成交（兼容方法）"""
        return self._fills.copy()
    
    def reset(self, capital: Optional[float] = None) -> None:
        """重置执行器"""
        if capital:
            self.config.initial_capital = capital
        
        self._portfolio = PortfolioState(
            cash=self.config.initial_capital,
            positions={},
            avg_price={},
            equity=self.config.initial_capital,  # 初始权益 = 初始资金
        )
        self._fills = []
        self._orders = {}
        self._current_prices = {}
        logger.info("SimulatedExecutionProvider 已重置")
    
    async def close(self) -> None:
        """关闭执行器"""
        self._closed = True
        self._fills.clear()
        self._orders.clear()
        logger.info("SimulatedExecutionProvider 已关闭")


class ExecutionProviderFactory:
    """执行器工厂"""
    
    @classmethod
    def create_simulated_provider(
        cls,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        fill_price_type: str = "close",
        initial_capital: float = 100000.0,
        data_feed: Optional[Any] = None,
        event_engine: Optional[UnifiedEventEngine] = None,
    ) -> SimulatedExecutionProvider:
        """创建模拟撮合执行器"""
        config = ExecutionProviderConfig(
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            fill_price_type=fill_price_type,
            initial_capital=initial_capital,
        )
        provider = SimulatedExecutionProvider(
            config=config,
            data_feed=data_feed,
            event_engine=event_engine,
        )
        return provider
    
    @classmethod
    def create_live_provider(cls, source: str = "ib", **kwargs) -> ExecutionProvider:
        """创建实盘执行器（需要外部实现）"""
        if source == "ib":
            from src.execution.providers.ib import IBExecutionProvider
            return IBExecutionProvider(**kwargs)
        else:
            raise ValueError(f"未知的执行器: {source}. 当前仅支持 'ib'")

"""
IB执行器 - Interactive Brokers实盘交易执行

功能：
- 连接到IB Gateway/TWS
- 提交市价单、限价单、止损单
- 轮询订单状态和成交
- 获取账户组合状态
- 发布订单事件到事件引擎

使用方式：
```python
from src.execution.providers import ExecutionProviderFactory

provider = ExecutionProviderFactory.create_live_provider(
    source="ib",
    host="127.0.0.1",
    port=7497,
    client_id=1,
    account_id="DU1234567",
)
await provider.initialize()
```

注意：需要IB Gateway/TWS运行并配置API访问
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ib_insync import IB, Contract, Stock, Order as IBOrder, MarketOrder, LimitOrder, StopOrder, StopLimitOrder, Trade

from src.config.settings import get_settings
from src.core.events.engine import UnifiedEventEngine
from src.core.types import (
    Fill, OrderIntent, OrderSide, OrderStatus, OrderType,
    PortfolioState, TimeInForce,
)
from src.execution.providers.base import ExecutionProvider, ExecutionProviderConfig
from src.utils.logger import get_logger

logger = get_logger("ib_execution_provider")


@dataclass
class IBExecutionProviderConfig:
    """IB执行器配置"""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account_id: str = ""  # IB账户ID
    paper_trading: bool = False  # 模拟交易模式
    commission_rate: float = 0.001  # 手续费率 (IB通常为0.005 USD/share, 最低1 USD)
    slippage_rate: float = 0.0005  # 滑点率
    fill_price_type: str = "close"  # 成交价格类型
    reconnect_delay: int = 5  # 重连延迟(秒)
    heartbeat_interval: int = 30  # 心跳间隔(秒)
    initial_capital: float = 100000.0


class IBExecutionProvider(ExecutionProvider):
    """
    Interactive Brokers实盘交易执行器
    
    特点：
    - 使用ib_insync库连接IB Gateway/TWS
    - 支持市价单、限价单、止损单
    - 自动处理订单状态更新
    - 发布事件到事件引擎
    """
    
    def __init__(
        self,
        config: Optional[IBExecutionProviderConfig] = None,
        event_engine: Optional[UnifiedEventEngine] = None,
    ):
        self.config = config or IBExecutionProviderConfig()
        self._event_engine = event_engine
        self._ib: Optional[IB] = None
        self._connected = False
        self._account_id: Optional[str] = None
        self._portfolio = PortfolioState(
            cash=self.config.initial_capital,
            positions={},
            avg_price={},
            equity=self.config.initial_capital,
        )
        self._orders: Dict[str, OrderIntent] = {}
        self._fills: List[Fill] = []
        self._trades: Dict[str, Trade] = {}  # IB Trade对象映射
        self._closed = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        
        # IB合约缓存
        self._contracts: Dict[str, Contract] = {}
        
        logger.info(
            f"IBExecutionProvider 初始化: "
            f"host={self.config.host}:{self.config.port}, "
            f"client_id={self.config.client_id}, "
            f"account={self.config.account_id or 'default'}, "
            f"paper={self.config.paper_trading}"
        )
    
    def set_event_engine(self, event_engine: UnifiedEventEngine) -> None:
        """设置事件引擎"""
        self._event_engine = event_engine
    
    def _create_contract(self, symbol: str) -> Contract:
        """创建IB合约"""
        if symbol in self._contracts:
            return self._contracts[symbol]
        
        # 支持股票和ETF
        contract = Stock(symbol, "SMART", "USD")
        self._contracts[symbol] = contract
        return contract
    
    def _to_ib_order(
        self,
        intent: OrderIntent,
    ) -> IBOrder:
        """将OrderIntent转换为IB Order"""
        action = "BUY" if intent.side == OrderSide.BUY else "SELL"
        
        # IB使用整股交易
        quantity = int(intent.qty)
        
        # 设置TIF
        tif_map = {
            TimeInForce.DAY: "DAY",
            TimeInForce.GTC: "GTC",
        }
        time_in_force = tif_map.get(TimeInForce.DAY, "DAY")
        
        # 根据订单类型创建不同的IB订单
        order: IBOrder
        if intent.order_type == OrderType.MKT:
            order = MarketOrder(action, quantity)
        elif intent.order_type == OrderType.LMT:
            if intent.limit_price is None:
                raise ValueError(f"限价单需要指定limit_price: {intent}")
            order = LimitOrder(action, quantity, intent.limit_price)
        elif intent.order_type == OrderType.STP:
            if intent.limit_price is not None:
                # 止损限价单
                if intent.stop_price is None:
                    raise ValueError(f"止损限价单需要指定stop_price: {intent}")
                order = StopLimitOrder(action, quantity, intent.stop_price, intent.limit_price)
            else:
                # 止损单
                if intent.stop_price is None:
                    raise ValueError(f"止损单需要指定stop_price: {intent}")
                order = StopOrder(action, quantity, intent.stop_price)
        else:
            # 默认使用市价单
            order = MarketOrder(action, quantity)
        
        # 设置TIF
        order.tif = time_in_force
        
        # 设置订单ID（用于跟踪）
        order_id = intent.order_id or str(uuid.uuid4())
        order.orderId = order_id
        order.account = self.config.account_id
        order.orderRef = order_id
        
        # 模拟交易模式：添加前缀
        if self.config.paper_trading:
            order.orderId = f"paper_{order_id}"
        
        return order
    
    async def _ensure_connected(self) -> bool:
        """确保已连接"""
        # 模拟交易模式不需要真实连接
        if self.config.paper_trading:
            self._connected = True
            return True
        
        if self._connected and self._ib is not None:
            return True
        
        return await self.initialize()
    
    async def initialize(self) -> None:
        """初始化执行器（连接到IB）"""
        try:
            # 检查是否已有运行的事件循环
            try:
                running_loop = asyncio.get_running_loop()
                # 如果在运行中的事件循环内，创建新线程执行IB连接
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            self._ib = IB()
            
            def connect_sync():
                if self._ib is None:
                    return False
                return self._ib.connect(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id,
                    timeout=10,
                    readonly=False,
                )
            
            if loop.is_running():
                # 在运行中的事件循环内，使用run_until_complete
                await asyncio.sleep(0)  # 允许其他任务运行
                connected = await asyncio.get_event_loop().run_in_executor(None, connect_sync)
            else:
                connected = await loop.run_in_executor(None, connect_sync)
            
            if not connected:
                raise RuntimeError("IB连接失败")
            
            self._connected = True
            self._account_id = self.config.account_id or self._ib.account
            
            logger.info(
                f"已连接到IB: {self.config.host}:{self.config.port} "
                f"(client_id={self.config.client_id}, account={self._account_id})"
            )
            
            # 启动连接监控
            self._monitor_task = asyncio.create_task(self._monitor_connection())
            
            # 同步账户信息
            await self._sync_portfolio()
            
        except Exception as e:
            logger.error(f"IB初始化失败: {e}")
            self._connected = False
            raise
    
    async def _monitor_connection(self) -> None:
        """监控连接状态"""
        while not self._closed:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self._ib is None or not self._ib.isConnected():
                    logger.warning("IB连接断开，尝试重连...")
                    await self._reconnect()
                else:
                    # 刷新账户信息
                    try:
                        self._ib.reqAccountUpdates(True, self._account_id)
                    except Exception as e:
                        logger.warning(f"账户更新失败: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"连接监控错误: {e}")
    
    async def _reconnect(self) -> None:
        """重新连接"""
        try:
            if self._ib is not None and self._ib.isConnected():
                self._ib.disconnect()
        except Exception:
            pass
        
        self._connected = False
        
        # 延迟重连
        await asyncio.sleep(self.config.reconnect_delay)
        
        try:
            await self.initialize()
        except Exception as e:
            logger.error(f"IB重连失败: {e}")
    
    async def _sync_portfolio(self) -> None:
        """同步账户信息"""
        if self._ib is None or not self._connected:
            return
        
        try:
            # 获取账户摘要
            account_summary = self._ib.accountSummary(self._account_id)
            
            # 解析现金和净值
            cash = 0.0
            net_liquidation = 0.0
            for item in account_summary:
                if item.tag == "CashBalance":
                    cash = float(item.value)
                elif item.tag == "NetLiquidation":
                    net_liquidation = float(item.value)
            
            # 更新组合状态
            self._portfolio.cash = cash
            self._portfolio.equity = net_liquidation
            
            # 获取持仓
            positions = self._ib.positions()
            
            for position in positions:
                if position.account != self._account_id:
                    continue
                
                symbol = position.contract.symbol
                quantity = position.position
                
                if quantity != 0:
                    self._portfolio.positions[symbol] = quantity
                    # IB不直接提供平均成本，使用当前市价
                    # 实际应用中应该从成交记录计算
                    self._portfolio.avg_price[symbol] = 0.0
            
            logger.info(
                f"账户同步完成: cash=${self._portfolio.cash:.2f}, "
                f"equity=${self._portfolio.equity:.2f}, "
                f"positions={len(self._portfolio.positions)}"
            )
            
        except Exception as e:
            logger.error(f"账户同步失败: {e}")
    
    async def submit_order(self, order: OrderIntent) -> Dict[str, Any]:
        """提交订单到IB"""
        try:
            await self._ensure_connected()
            
            order_id = order.order_id or str(uuid.uuid4())
            order.order_id = order_id
            
            # 创建合约和订单
            contract = self._create_contract(order.symbol)
            ib_order = self._to_ib_order(order)
            
            # 保存订单意图
            self._orders[order_id] = order
            
            # 发布订单创建事件
            if self._event_engine:
                await self._event_engine.publish_order_event(
                    event_type="order_created",
                    order_id=order_id,
                    symbol=order.symbol,
                    order_type=order.order_type.value,
                    side=order.side.value,
                    quantity=order.qty,
                    strategy_id=order.strategy_id,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                )
            
            if self.config.paper_trading:
                # 模拟交易：直接返回成功
                fill = await self._simulate_fill(order, order_id)
                self._fills.append(fill)
                self._update_portfolio(fill)
                
                result = {
                    "success": True,
                    "order_id": order_id,
                    "fill_id": fill.fill_id,
                    "fill_price": fill.price,
                    "fill_quantity": fill.qty,
                    "commission": fill.fee,
                    "slippage": fill.slippage_est,
                }
                
                # 发布成交事件
                if self._event_engine:
                    await self._event_engine.publish_order_event(
                        event_type="order_filled",
                        order_id=order_id,
                        symbol=order.symbol,
                        order_type=order.order_type.value,
                        side=order.side.value,
                        quantity=order.qty,
                        strategy_id=order.strategy_id,
                        fill_id=fill.fill_id,
                        fill_price=fill.price,
                        fill_quantity=fill.qty,
                        fill_time=fill.ts_fill_utc,
                        commission=fill.fee,
                        slippage=fill.slippage_est,
                    )
                
                logger.info(f"模拟成交: {order_id} {order.symbol} {order.side.value} {order.qty} @ {fill.price:.4f}")
                return result
            
            # 实盘交易
            trade = self._ib.placeOrder(contract, ib_order)
            self._trades[order_id] = trade
            
            logger.info(
                f"订单已提交: {order_id} {order.symbol} {order.side.value} "
                f"{order.qty} @ {order.order_type.value}"
            )
            
            # 等待订单状态更新
            status = await self._wait_for_status(order_id)
            
            if status.get("filled", False):
                # 获取成交
                fill = await self._get_fill_from_trade(order_id, trade)
                if fill:
                    self._fills.append(fill)
                    self._update_portfolio(fill)
                    
                    result = {
                        "success": True,
                        "order_id": order_id,
                        "fill_id": fill.fill_id,
                        "fill_price": fill.price,
                        "fill_quantity": fill.qty,
                        "commission": fill.fee,
                        "slippage": fill.slippage_est,
                    }
                    
                    # 发布成交事件
                    if self._event_engine:
                        await self._event_engine.publish_order_event(
                            event_type="order_filled",
                            order_id=order_id,
                            symbol=order.symbol,
                            order_type=order.order_type.value,
                            side=order.side.value,
                            quantity=order.qty,
                            strategy_id=order.strategy_id,
                            fill_id=fill.fill_id,
                            fill_price=fill.price,
                            fill_quantity=fill.qty,
                            fill_time=fill.ts_fill_utc,
                            commission=fill.fee,
                            slippage=fill.slippage_est,
                        )
                    
                    logger.info(
                        f"订单成交: {order_id} {order.symbol} {order.side.value} "
                        f"{fill.qty} @ {fill.price:.4f}"
                    )
                    return result
            
            return {
                "success": True,
                "order_id": order_id,
                "status": status.get("status", "SUBMITTED"),
            }
            
        except Exception as e:
            logger.error(f"订单提交失败: {e}")
            
            # 发布拒绝事件
            if self._event_engine and order.order_id:
                await self._event_engine.publish_order_event(
                    event_type="order_rejected",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    order_type=order.order_type.value,
                    side=order.side.value,
                    quantity=order.qty,
                    strategy_id=order.strategy_id,
                    reason=str(e),
                )
            
            return {
                "success": False,
                "order_id": order.order_id if order.order_id else "",
                "error": str(e),
            }
    
    async def _simulate_fill(self, order: OrderIntent, order_id: str) -> Fill:
        """模拟成交（用于模拟交易模式）"""
        # 获取当前价格
        base_price = 100.0
        if self._ib and self._connected:
            try:
                ticker = self._ib.reqMktData(self._create_contract(order.symbol))
                if ticker.last:
                    base_price = ticker.last
            except Exception:
                pass
        
        # 计算成交价格
        if order.order_type == OrderType.LMT and order.limit_price:
            fill_price = float(order.limit_price)
        else:
            fill_price = base_price
        
        # 考虑滑点
        if order.side == OrderSide.BUY:
            fill_price *= (1 + self.config.slippage_rate)
        else:
            fill_price *= (1 - self.config.slippage_rate)
        
        # 计算手续费
        commission = order.qty * fill_price * self.config.commission_rate
        
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            ts_fill_utc=datetime.now(timezone.utc),
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=commission,
            slippage_est=self.config.slippage_rate,
        )
        
        return fill
    
    async def _wait_for_status(
        self,
        order_id: str,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """等待订单状态更新"""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                return {"status": "TIMEOUT", "filled": False}
            
            trade = self._trades.get(order_id)
            if trade is None:
                await asyncio.sleep(0.1)
                continue
            
            order_status = trade.orderStatus
            
            if order_status.status == "Filled":
                return {"status": "FILLED", "filled": True}
            elif order_status.status == "Cancelled":
                return {"status": "CANCELLED", "filled": False}
            elif order_status.status == "Submitted":
                return {"status": "SUBMITTED", "filled": False}
            elif order_status.status == "ApiPending":
                return {"status": "PENDING", "filled": False}
            elif order_status.status == "Inactive":
                return {"status": "INACTIVE", "filled": False}
            elif order_status.status == "Rejected":
                return {"status": "REJECTED", "filled": False}
            
            await asyncio.sleep(0.1)
    
    async def _get_fill_from_trade(self, order_id: str, trade: Trade) -> Optional[Fill]:
        """从IB Trade获取成交记录"""
        if not trade.fills:
            return None
        
        fill_data = trade.fills[0]  # 只取第一笔成交
        execution = fill_data.execution
        
        # 计算手续费
        commission = 0.0
        if fill_data.commissionReport:
            commission = fill_data.commissionReport.commission
        
        fill = Fill(
            fill_id=execution.execId,
            order_id=order_id,
            ts_fill_utc=execution.time,
            symbol=trade.contract.symbol,
            side=OrderSide.BUY if execution.side == "BUY" else OrderSide.SELL,
            qty=execution.shares,
            price=execution.price,
            fee=commission,
            slippage_est=None,
        )
        
        return fill
    
    async def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """取消订单"""
        try:
            await self._ensure_connected()
            
            # 模拟交易模式：检查订单是否在_orders中
            if self.config.paper_trading:
                if order_id in self._orders:
                    del self._orders[order_id]
                    logger.info(f"模拟订单取消: {order_id}")
                    return True
                logger.warning(f"订单不存在: {order_id}")
                return False
            
            # 实盘交易模式
            trade = self._trades.get(order_id)
            if trade is None:
                logger.warning(f"订单不存在: {order_id}")
                return False
            
            # 取消IB订单
            self._ib.cancelOrder(trade.order)
            
            # 删除订单
            if order_id in self._orders:
                del self._orders[order_id]
            
            # 发布取消事件
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
            
            logger.info(f"订单已取消: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"订单取消失败: {order_id} - {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        try:
            await self._ensure_connected()
            
            trade = self._trades.get(order_id)
            if trade is None:
                # 检查是否是未提交的订单
                order = self._orders.get(order_id)
                if order is None:
                    return None
                
                return {
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.qty,
                    "filled_quantity": 0,
                    "status": "CREATED",
                }
            
            order_status = trade.orderStatus
            
            return {
                "order_id": order_id,
                "symbol": trade.contract.symbol,
                "side": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "filled_quantity": order_status.filled,
                "avg_fill_price": order_status.avgFillPrice,
                "status": order_status.status,
                "commission": order_status.commission,
            }
            
        except Exception as e:
            logger.error(f"获取订单状态失败: {order_id} - {e}")
            return None
    
    async def poll_fills(self) -> List[Fill]:
        """轮询成交"""
        # 过滤出已成交的订单
        filled = [f for f in self._fills if f.order_id not in self._trades]
        # 移除已处理的成交
        self._fills = [f for f in self._fills if f.order_id in self._trades]
        return filled
    
    async def get_portfolio_state(self) -> PortfolioState:
        """获取组合状态"""
        await self._ensure_connected()
        
        # 同步账户信息
        await self._sync_portfolio()
        
        return self._portfolio
    
    def get_fills(self) -> List[Fill]:
        """获取所有成交"""
        return self._fills.copy()
    
    def reset(self, capital: Optional[float] = None) -> None:
        """重置执行器"""
        if capital:
            self.config.initial_capital = capital
        
        self._portfolio = PortfolioState(
            cash=self.config.initial_capital,
            positions={},
            avg_price={},
            equity=self.config.initial_capital,
        )
        self._fills = []
        self._orders = {}
        self._trades = {}
        logger.info("IBExecutionProvider 已重置")
    
    async def close(self) -> None:
        """关闭执行器"""
        self._closed = True
        
        # 停止监控任务
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # 断开IB连接
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
                logger.info("已断开IB连接")
            except Exception as e:
                logger.warning(f"断开IB连接时出错: {e}")
        
        self._connected = False
        self._ib = None
        self._fills.clear()
        self._orders.clear()
        self._trades.clear()
        logger.info("IBExecutionProvider 已关闭")
    
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
    
    def _update_equity(self) -> None:
        """更新权益"""
        positions_value = sum(
            abs(qty) * self._portfolio.avg_price.get(symbol, 0.0)
            for symbol, qty in self._portfolio.positions.items()
            if abs(qty) > 1e-8
        )
        self._portfolio.equity = self._portfolio.cash + positions_value


class IBProviderFactory:
    """IB执行器工厂"""
    
    @classmethod
    def create(
        cls,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account_id: str = "",
        paper_trading: bool = False,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        event_engine: Optional[UnifiedEventEngine] = None,
    ) -> IBExecutionProvider:
        """创建IB执行器"""
        config = IBExecutionProviderConfig(
            host=host,
            port=port,
            client_id=client_id,
            account_id=account_id,
            paper_trading=paper_trading,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )
        
        return IBExecutionProvider(
            config=config,
            event_engine=event_engine,
        )

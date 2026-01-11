"""
Interactive Brokers Broker
使用 ib_insync 连接 TWS/IB Gateway 进行模拟或实盘交易
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid
import asyncio

from ib_insync import IB, MarketOrder, LimitOrder, StopOrder, Stock, Contract, OrderStatus as IBOrderStatus

from src.core.types import OrderIntent, Fill, PortfolioState, OrderSide, OrderStatus, OrderType
from src.core.context import RunContext
from src.config.settings import get_settings
from src.utils.logger import get_logger


class IBBroker:
    """Interactive Brokers Broker"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        account: Optional[str] = None,
        timeout: Optional[float] = None,
        readonly: Optional[bool] = None,
    ):
        """
        初始化 IBBroker
        
        Args:
            host: TWS/IB Gateway 主机地址
            port: 端口（7497=模拟账户，7496=实盘账户）
            client_id: 客户端ID
            account: 账户ID（如果为None，使用默认账户）
            timeout: 连接超时时间（秒）
            readonly: 是否只读模式
        """
        settings = get_settings()
        ib_settings = settings.ib
        
        self.host = host or ib_settings.host
        self.port = port or ib_settings.port
        self.client_id = client_id or ib_settings.broker_client_id
        self.account = account or ib_settings.account
        self.timeout = timeout or ib_settings.timeout
        self.readonly = readonly or ib_settings.readonly
        
        self.ib = IB()
        self.logger = get_logger("ib_broker")
        
        # 内部状态
        self._connected = False
        self._order_id_map: Dict[str, int] = {}  # 内部订单ID -> IB订单ID
        self._ib_order_id_map: Dict[int, str] = {}  # IB订单ID -> 内部订单ID
        self._fills_cache: Dict[str, Fill] = {}  # order_id -> Fill
        
    async def connect(self) -> None:
        """连接到 TWS/IB Gateway"""
        if self._connected:
            self.logger.warning("已经连接到 IB，跳过重复连接")
            return
        
        try:
            self.logger.info(f"正在连接到 IB: {self.host}:{self.port} (client_id={self.client_id})")
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=self.readonly,
            )
            self._connected = True
            
            # 获取账户信息
            if self.account is None:
                accounts = self.ib.accountValues()
                if accounts:
                    # 使用第一个账户
                    self.account = accounts[0].account
                    self.logger.info(f"使用默认账户: {self.account}")
                else:
                    self.logger.warning("未找到账户信息")
            else:
                self.logger.info(f"使用指定账户: {self.account}")
            
            self.logger.info("成功连接到 IB")
            
        except Exception as e:
            self.logger.error(f"连接 IB 失败: {e}")
            self._connected = False
            raise
    
    async def disconnect(self) -> None:
        """断开连接"""
        if self._connected:
            try:
                self.ib.disconnect()
                self._connected = False
                self.logger.info("已断开 IB 连接")
            except Exception as e:
                self.logger.error(f"断开连接时出错: {e}")
    
    def _ensure_connected(self) -> None:
        """确保已连接"""
        if not self._connected:
            raise RuntimeError("IBBroker 未连接，请先调用 connect()")
    
    def _symbol_to_contract(self, symbol: str) -> Contract:
        """
        将交易符号转换为 IB Contract
        
        Args:
            symbol: 交易符号（如 "AAPL", "SPY"）
            
        Returns:
            IB Contract 对象
        """
        # 假设是美股，使用 Stock
        # 如果需要支持其他市场，可以扩展此方法
        contract = Stock(symbol, 'SMART', 'USD')
        return contract
    
    def _order_intent_to_ib_order(self, order: OrderIntent) -> Tuple[Contract, object]:
        """
        将 OrderIntent 转换为 IB Order
        
        Args:
            order: 订单意图
            
        Returns:
            (Contract, IB Order) 元组
        """
        contract = self._symbol_to_contract(order.symbol)
        
        # 根据订单类型创建 IB Order
        if order.order_type == OrderType.MKT:
            ib_order = MarketOrder(
                action='BUY' if order.side == OrderSide.BUY else 'SELL',
                totalQuantity=int(order.qty),
            )
        elif order.order_type == OrderType.LMT:
            if order.limit_price is None:
                raise ValueError(f"限价单必须提供 limit_price")
            ib_order = LimitOrder(
                action='BUY' if order.side == OrderSide.BUY else 'SELL',
                totalQuantity=int(order.qty),
                lmtPrice=order.limit_price,
            )
        elif order.order_type == OrderType.STP:
            if order.stop_price is None:
                raise ValueError(f"止损单必须提供 stop_price")
            ib_order = StopOrder(
                action='BUY' if order.side == OrderSide.BUY else 'SELL',
                totalQuantity=int(order.qty),
                auxPrice=order.stop_price,
            )
        else:
            raise ValueError(f"不支持的订单类型: {order.order_type}")
        
        # 设置账户
        if self.account:
            ib_order.account = self.account
        
        return contract, ib_order
    
    async def place_orders(
        self,
        orders: List[OrderIntent],
        ctx: RunContext,
    ) -> List[str]:
        """
        提交订单
        
        Args:
            orders: 订单意图列表
            ctx: 运行上下文
            
        Returns:
            订单ID列表
        """
        self._ensure_connected()
        
        order_ids = []
        
        for order in orders:
            try:
                normalized_qty = int(order.qty)
                if normalized_qty <= 0:
                    self.logger.info(
                        f"跳过订单（股数<1）: {order.symbol} {order.side.value} {order.qty}"
                    )
                    continue
                if normalized_qty != order.qty:
                    order.qty = float(normalized_qty)

                # 生成内部订单ID
                internal_order_id = str(uuid.uuid4())
                
                # 转换为 IB Order
                contract, ib_order = self._order_intent_to_ib_order(order)
                
                # 确保合约信息完整
                contract = await self.ib.qualifyContractsAsync(contract)
                if not contract:
                    self.logger.error(f"无法解析合约: {order.symbol}")
                    continue
                
                # 提交订单
                trade = self.ib.placeOrder(contract[0], ib_order)
                
                # 记录订单ID映射
                ib_order_id = trade.order.orderId
                self._order_id_map[internal_order_id] = ib_order_id
                self._ib_order_id_map[ib_order_id] = internal_order_id
                
                order_ids.append(internal_order_id)
                self.logger.info(
                    f"提交订单: {internal_order_id} -> IB订单ID={ib_order_id}, "
                    f"{order.symbol} {order.side.value} {order.qty}"
                )
                
            except Exception as e:
                self.logger.error(f"提交订单失败: {order.symbol}, 错误: {e}")
                # 继续处理其他订单
        
        return order_ids
    
    async def poll_fills(
        self,
        ctx: RunContext,
    ) -> List[Fill]:
        """
        轮询成交
        
        Args:
            ctx: 运行上下文
            
        Returns:
            成交列表
        """
        self._ensure_connected()
        
        new_fills = []
        
        # 获取所有活跃交易
        trades = self.ib.trades()
        
        for trade in trades:
            ib_order_id = trade.order.orderId
            
            # 检查是否是我们管理的订单
            if ib_order_id not in self._ib_order_id_map:
                continue
            
            internal_order_id = self._ib_order_id_map[ib_order_id]
            
            # 检查订单状态
            if trade.orderStatus.status == IBOrderStatus.Filled:
                # 检查是否已经处理过这个成交
                if internal_order_id in self._fills_cache:
                    continue
                
                # 获取成交信息
                fills = trade.fills
                if not fills:
                    continue
                
                # 创建 Fill 对象（使用第一个成交，如果有多个成交会合并）
                total_qty = sum(f.execution.shares for f in fills)
                avg_price = sum(f.execution.price * f.execution.shares for f in fills) / total_qty if total_qty > 0 else 0
                total_fee = sum(f.commissionReport.commission if f.commissionReport else 0.0 for f in fills)
                
                # 确定订单方向
                order_side = OrderSide.BUY if trade.order.action == 'BUY' else OrderSide.SELL
                
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=internal_order_id,
                    ts_fill_utc=datetime.utcnow(),  # IB 的成交时间需要从 execution 中获取
                    symbol=trade.contract.symbol,
                    side=order_side,
                    qty=float(total_qty),
                    price=float(avg_price),
                    fee=float(total_fee),
                    slippage_est=None,  # IB 不直接提供滑点估计
                    liquidity_flag=None,
                    metadata={
                        'ib_order_id': ib_order_id,
                        'ib_execution_id': fills[0].execution.execId if fills else None,
                    },
                )
                
                new_fills.append(fill)
                self._fills_cache[internal_order_id] = fill
                
                self.logger.info(
                    f"订单成交: {internal_order_id}, {fill.symbol} {fill.side.value} "
                    f"{fill.qty} @ {fill.price}, 手续费={fill.fee}"
                )
        
        return new_fills
    
    async def cancel(
        self,
        order_id: str,
        ctx: RunContext,
    ) -> None:
        """
        撤销订单
        
        Args:
            order_id: 订单ID（内部订单ID）
            ctx: 运行上下文
        """
        self._ensure_connected()
        
        if order_id not in self._order_id_map:
            self.logger.warning(f"订单不存在: {order_id}")
            return
        
        ib_order_id = self._order_id_map[order_id]
        
        # 查找对应的交易
        trades = self.ib.trades()
        target_trade = None
        for trade in trades:
            if trade.order.orderId == ib_order_id:
                target_trade = trade
                break
        
        if target_trade is None:
            self.logger.warning(f"找不到对应的 IB 交易: {ib_order_id}")
            return
        
        try:
            self.ib.cancelOrder(target_trade.order)
            self.logger.info(f"撤销订单: {order_id} (IB订单ID={ib_order_id})")
        except Exception as e:
            self.logger.error(f"撤销订单失败: {order_id}, 错误: {e}")
            raise
    
    async def get_positions(self, ctx: RunContext) -> PortfolioState:
        """
        获取持仓
        
        Args:
            ctx: 运行上下文
            
        Returns:
            组合状态
        """
        self._ensure_connected()
        
        # 获取持仓
        positions = self.ib.positions()
        
        # 转换为 PortfolioState
        positions_dict: Dict[str, float] = {}
        avg_price_dict: Dict[str, float] = {}
        
        for pos in positions:
            if self.account and pos.account != self.account:
                continue
            
            symbol = pos.contract.symbol
            qty = float(pos.position)
            avg_cost = float(pos.avgCost) if pos.avgCost else 0.0
            
            positions_dict[symbol] = qty
            avg_price_dict[symbol] = avg_cost
        
        # 获取账户价值
        cash = await self.get_cash(ctx)
        
        # 计算总资产（简化：现金 + 持仓成本）
        equity = cash + sum(
            qty * avg_price_dict.get(symbol, 0.0)
            for symbol, qty in positions_dict.items()
        )
        
        return PortfolioState(
            cash=cash,
            positions=positions_dict,
            avg_price=avg_price_dict,
            equity=equity,
            daily_loss=0.0,  # IB 不直接提供日亏损，需要计算
        )
    
    async def get_cash(self, ctx: RunContext) -> float:
        """
        获取现金余额
        
        Args:
            ctx: 运行上下文
            
        Returns:
            现金余额
        """
        self._ensure_connected()
        
        # 获取账户摘要
        account_values = self.ib.accountValues()
        
        cash = 0.0
        for av in account_values:
            if self.account and av.account != self.account:
                continue
            
            # 查找可用现金
            if av.tag == 'AvailableFunds' or av.tag == 'CashBalance':
                try:
                    cash = float(av.value)
                    break
                except (ValueError, TypeError):
                    continue
        
        return cash

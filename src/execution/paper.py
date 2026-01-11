"""
Paper Trading Broker
模拟执行，复用回测的成本模型和撮合逻辑
"""

from typing import List, Dict, Optional
from datetime import datetime
import uuid
import pandas as pd

from src.core.types import OrderIntent, Fill, PortfolioState, OrderSide, OrderStatus
from src.core.context import RunContext, Environment
from src.evaluation.backtesting.engine import CostModel
from src.utils.logger import get_logger
from src.data.storage.parquet_store import ParquetDataFrameStore
from src.config.settings import get_settings


class PaperBroker:
    """Paper Trading Broker"""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        store: Optional[ParquetDataFrameStore] = None,
    ):
        self.initial_cash = initial_cash
        self.cost_model = CostModel(commission_rate, slippage_rate)
        settings = get_settings()
        self.store = store or ParquetDataFrameStore(settings.storage.data_dir)
        self.logger = get_logger("paper_broker")
        
        # 内部状态
        self.portfolio = PortfolioState(
            cash=initial_cash,
            positions={},
            avg_price={},
            equity=initial_cash,
            daily_loss=0.0,
        )
        self.pending_orders: Dict[str, OrderIntent] = {}
        self.fills: List[Fill] = []
        self.orders: Dict[str, Dict] = {}  # order_id -> order info
    
    async def place_orders(
        self,
        orders: List[OrderIntent],
        ctx: RunContext,
    ) -> List[str]:
        """提交订单"""
        order_ids = []
        
        for order in orders:
            normalized_qty = int(order.qty)
            if normalized_qty <= 0:
                self.logger.info(
                    f"跳过订单（股数<1）: {order.symbol} {order.side.value} {order.qty}"
                )
                continue
            if normalized_qty != order.qty:
                order.qty = float(normalized_qty)

            order_id = str(uuid.uuid4())
            self.pending_orders[order_id] = order
            self.orders[order_id] = {
                'order': order,
                'status': OrderStatus.NEW,
                'created_at': ctx.now_utc,
            }
            order_ids.append(order_id)
            self.logger.info(f"提交订单: {order_id}, {order.symbol} {order.side.value} {order.qty}")
        
        return order_ids
    
    async def poll_fills(
        self,
        ctx: RunContext,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[Fill]:
        """
        轮询成交（模拟撮合）
        
        Args:
            ctx: 运行上下文
            current_prices: 当前价格字典 {symbol: price}（如果为None，从数据源获取）
            
        Returns:
            成交列表
        """
        new_fills = []
        
        for order_id, order in list(self.pending_orders.items()):
            # 获取当前价格
            if current_prices is None:
                price = self._get_current_price(order.symbol, ctx)
            else:
                price = current_prices.get(order.symbol)
            
            if price is None:
                continue
            
            # 模拟成交（市价单立即成交）
            if order.order_type.value == "MKT":
                fill = self._create_fill(order, order_id, price, ctx.now_utc)
                new_fills.append(fill)
                self.fills.append(fill)
                
                # 更新订单状态
                self.orders[order_id]['status'] = OrderStatus.FILLED
                self.orders[order_id]['filled_at'] = ctx.now_utc
                del self.pending_orders[order_id]
                
                # 更新组合状态
                self._update_portfolio(fill, price)
        
        return new_fills
    
    async def cancel(
        self,
        order_id: str,
        ctx: RunContext,
    ) -> None:
        """撤销订单"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self.orders[order_id]['status'] = OrderStatus.CANCELLED
            self.logger.info(f"撤销订单: {order_id}")
    
    async def get_positions(self, ctx: RunContext) -> PortfolioState:
        """获取持仓"""
        # 更新equity
        self.portfolio.equity = self._calculate_equity()
        return self.portfolio
    
    async def get_cash(self, ctx: RunContext) -> float:
        """获取现金"""
        return self.portfolio.cash
    
    def _get_current_price(self, symbol: str, ctx: RunContext) -> Optional[float]:
        """获取当前价格（从数据源）"""
        # 简化：从store读取最新价格
        try:
            bars = self.store.read_dataset(
                dataset="bars",
                partition={"symbol": symbol, "timeframe": "1d"},
                data_version=ctx.data_version,
            )
            if not bars.empty:
                return bars.iloc[-1]['close']
        except Exception:
            pass
        return None
    
    def _create_fill(
        self,
        order: OrderIntent,
        order_id: str,
        fill_price: float,
        fill_time: datetime,
    ) -> Fill:
        """创建成交记录"""
        fill_id = str(uuid.uuid4())
        
        fee = self.cost_model.estimate_fee(order, fill_price)
        slippage = self.cost_model.estimate_slippage(order, fill_price)
        
        return Fill(
            fill_id=fill_id,
            order_id=order_id,
            ts_fill_utc=fill_time,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=fee,
            slippage_est=slippage,
        )
    
    def _update_portfolio(self, fill: Fill, fill_price: float):
        """更新组合状态"""
        current_qty = self.portfolio.positions.get(fill.symbol, 0.0)
        current_avg = self.portfolio.avg_price.get(fill.symbol, fill_price)
        
        if fill.side == OrderSide.BUY:
            new_qty = current_qty + fill.qty
            if new_qty != 0:
                new_avg = (current_qty * current_avg + fill.qty * fill_price) / new_qty
            else:
                new_avg = fill_price
        else:  # SELL
            new_qty = current_qty - fill.qty
            if abs(new_qty) < 1e-8:
                new_qty = 0.0
                new_avg = 0.0
            else:
                new_avg = current_avg
        
        self.portfolio.positions[fill.symbol] = new_qty
        self.portfolio.avg_price[fill.symbol] = new_avg
        
        # 更新现金
        if fill.side == OrderSide.BUY:
            cost = fill.qty * fill_price + fill.fee
            self.portfolio.cash -= cost
        else:
            proceeds = fill.qty * fill_price - fill.fee
            self.portfolio.cash += proceeds
    
    def _calculate_equity(self) -> float:
        """计算总资产净值"""
        # 简化：需要当前价格，这里返回cash + 持仓成本
        positions_value = sum(
            qty * avg_price
            for qty, avg_price in zip(
                self.portfolio.positions.values(),
                self.portfolio.avg_price.values(),
            )
        )
        return self.portfolio.cash + positions_value

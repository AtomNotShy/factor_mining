"""
订单引擎模块
提供流畅的 API 简化订单创建过程

设计目标：
1. 减少重复代码（ts_utc, strategy_id, order_type 等）
2. 提供流畅接口支持链式调用
3. 支持批量订单操作
4. 保持向后兼容性
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
import uuid

from src.core.types import (
    OrderIntent, OrderSide, OrderType, 
    PortfolioState, Signal, ActionType
)
from src.core.context import RunContext
from src.utils.logger import get_logger


class OrderBuilder:
    """
    订单构建器 - 流畅接口模式
    
    使用示例：
        order = OrderBuilder(strategy_id="my_strategy", now_utc=ctx.now_utc)\
            .buy("AAPL", 100)\
            .with_limit(150.0)\
            .with_metadata(reason="breakout")\
            .execute()
    """
    
    def __init__(self, strategy_id: str, now_utc: datetime):
        self.strategy_id = strategy_id
        self.now_utc = now_utc
        
        # 订单属性
        self.symbol: str = ""
        self.side: OrderSide = OrderSide.BUY
        self.qty: float = 0.0
        self.order_type: OrderType = OrderType.MKT
        self.limit_price: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
    
    def buy(self, symbol: str, qty: float) -> "OrderBuilder":
        """设置买入订单"""
        self.symbol = symbol
        self.side = OrderSide.BUY
        self.qty = qty
        return self
    
    def sell(self, symbol: str, qty: float) -> "OrderBuilder":
        """设置卖出订单"""
        self.symbol = symbol
        self.side = OrderSide.SELL
        self.qty = qty
        return self
    
    def close(self, symbol: str, qty: Optional[float] = None) -> "OrderBuilder":
        """设置平仓订单（自动决定方向）"""
        self.symbol = symbol
        self.side = OrderSide.SELL  # 默认卖出平仓
        self.qty = qty if qty is not None else 0  # 0 表示全平
        self.metadata["reason"] = "close"
        return self
    
    def with_limit(self, price: float) -> "OrderBuilder":
        """设置限价单"""
        self.order_type = OrderType.LMT
        self.limit_price = price
        return self
    
    def with_stop(self, price: float) -> "OrderBuilder":
        """设置止损单"""
        self.order_type = OrderType.STP
        self.stop_price = price
        return self
    
    def with_market(self) -> "OrderBuilder":
        """设置市价单（默认）"""
        self.order_type = OrderType.MKT
        return self
    
    def with_metadata(self, **kwargs) -> "OrderBuilder":
        """添加元数据"""
        self.metadata.update(kwargs)
        return self
    
    def with_reason(self, reason: str) -> "OrderBuilder":
        """设置订单原因"""
        self.metadata["reason"] = reason
        return self
    
    def execute(self) -> OrderIntent:
        """构建并返回 OrderIntent"""
        return OrderIntent(
            order_id=str(uuid.uuid4()),
            ts_utc=self.now_utc,
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            order_type=self.order_type,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            strategy_id=self.strategy_id,
            metadata=self.metadata.copy(),
        )


class OrderEngine:
    """
    订单引擎 - 简化订单创建
    
    主要功能：
    1. 快速创建常见订单类型
    2. 批量操作
    3. 与策略上下文集成
    """
    
    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        self._now_utc: datetime = datetime.now(timezone.utc)
        self.logger = get_logger(f"order_engine.{strategy_id}")
    
    def update_context(self, now_utc: Optional[datetime] = None):
        """更新运行时上下文"""
        if now_utc:
            self._now_utc = now_utc
    
    def builder(self) -> OrderBuilder:
        """获取订单构建器"""
        return OrderBuilder(self.strategy_id, self._now_utc)
    
    # ===== 快速创建方法 =====
    
    def buy(self, symbol: str, qty: float, **metadata) -> OrderIntent:
        """快速创建买入订单"""
        return OrderIntent(
            order_id=str(uuid.uuid4()),
            ts_utc=self._now_utc,
            symbol=symbol,
            side=OrderSide.BUY,
            qty=qty,
            order_type=OrderType.MKT,
            strategy_id=self.strategy_id,
            metadata=metadata,
        )
    
    def sell(self, symbol: str, qty: float, **metadata) -> OrderIntent:
        """快速创建卖出订单"""
        return OrderIntent(
            order_id=str(uuid.uuid4()),
            ts_utc=self._now_utc,
            symbol=symbol,
            side=OrderSide.SELL,
            qty=qty,
            order_type=OrderType.MKT,
            strategy_id=self.strategy_id,
            metadata=metadata,
        )
    
    def close_position(self, symbol: str, qty: Optional[float] = None, **metadata) -> OrderIntent:
        """平仓订单"""
        # 使用 setdefault 确保不会覆盖传入的 reason
        metadata.setdefault("reason", "close")
        return OrderIntent(
            order_id=str(uuid.uuid4()),
            ts_utc=self._now_utc,
            symbol=symbol,
            side=OrderSide.SELL,  # 默认卖出平仓
            qty=qty if qty is not None else 0,  # 0 表示全平
            order_type=OrderType.MKT,
            strategy_id=self.strategy_id,
            metadata=metadata.copy(),  # 使用 copy 避免修改原始 dict
        )
    
    def limit_buy(self, symbol: str, qty: float, limit_price: float, **metadata) -> OrderIntent:
        """限价买入"""
        return OrderIntent(
            order_id=str(uuid.uuid4()),
            ts_utc=self._now_utc,
            symbol=symbol,
            side=OrderSide.BUY,
            qty=qty,
            order_type=OrderType.LMT,
            limit_price=limit_price,
            strategy_id=self.strategy_id,
            metadata=metadata,
        )
    
    def limit_sell(self, symbol: str, qty: float, limit_price: float, **metadata) -> OrderIntent:
        """限价卖出"""
        return OrderIntent(
            order_id=str(uuid.uuid4()),
            ts_utc=self._now_utc,
            symbol=symbol,
            side=OrderSide.SELL,
            qty=qty,
            order_type=OrderType.LMT,
            limit_price=limit_price,
            strategy_id=self.strategy_id,
            metadata=metadata,
        )
    
    # ===== 批量操作 =====
    
    def close_all_positions(self, portfolio: PortfolioState, **metadata) -> List[OrderIntent]:
        """平掉所有持仓"""
        orders = []
        for symbol, qty in portfolio.positions.items():
            if abs(qty) > 1e-8:  # 有持仓
                # 合并元数据
                order_metadata = {"reason": "close_all"}
                order_metadata.update(metadata)
                orders.append(
                    self.close_position(symbol, abs(qty), **order_metadata)
                )
        return orders
    
    def close_symbols(self, symbols: List[str], **metadata) -> List[OrderIntent]:
        """平掉指定标的"""
        return [
            self.close_position(symbol, **metadata)
            for symbol in symbols
        ]
    
    def from_signals(self, signals: List[Signal], portfolio: PortfolioState, 
                     cash_per_signal: Optional[float] = None, **metadata) -> List[OrderIntent]:
        """
        从信号列表生成订单
        
        Args:
            signals: 信号列表
            portfolio: 当前组合状态
            cash_per_signal: 每个信号的分配资金（如果为None，则平均分配可用资金）
            metadata: 附加元数据
        """
        orders = []
        
        # 筛选 LONG 信号
        long_signals = [s for s in signals if s.action == ActionType.LONG]
        if not long_signals:
            return orders
        
        # 计算可用资金
        available_cash = portfolio.cash
        if cash_per_signal is None:
            cash_per_signal = available_cash / len(long_signals)
        
        for signal in long_signals:
            price = signal.metadata.get("price", 0)
            if price <= 0:
                continue
            
            qty = cash_per_signal / price
            if qty * price < 100:  # 最小交易金额检查
                continue
            
            orders.append(
                self.buy(
                    signal.symbol,
                    qty,
                    reason="signal_entry",
                    strength=signal.strength,
                    **metadata,
                )
            )
        
        return orders
    
    def rebalance_to_target(self, target_weights: Dict[str, float], 
                           portfolio: PortfolioState, total_value: float,
                           **metadata) -> List[OrderIntent]:
        """
        再平衡到目标权重
        
        Args:
            target_weights: 目标权重 {symbol: weight}
            portfolio: 当前组合状态
            total_value: 总资产价值
            metadata: 附加元数据
        """
        orders = []
        
        for symbol, target_weight in target_weights.items():
            current_qty = portfolio.positions.get(symbol, 0.0)
            current_price = portfolio.avg_price.get(symbol, 0.0)
            
            if current_price <= 0:
                continue
            
            target_value = total_value * target_weight
            target_qty = target_value / current_price
            
            qty_diff = target_qty - current_qty
            
            if abs(qty_diff) < 1e-6:
                continue
            
            if qty_diff > 0:
                orders.append(
                    self.buy(symbol, qty_diff, reason="rebalance_buy", **metadata)
                )
            else:
                orders.append(
                    self.sell(symbol, abs(qty_diff), reason="rebalance_sell", **metadata)
                )
        
        return orders


# ===== 工厂函数 =====

def create_order(side: Union[str, OrderSide], symbol: str, qty: float,
                 strategy_id: str, now_utc: datetime,
                 order_type: Union[str, OrderType] = OrderType.MKT,
                 limit_price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 **metadata) -> OrderIntent:
    """
    工厂函数 - 最简单的订单创建
    
    Args:
        side: "BUY" 或 "SELL" 或 OrderSide 枚举
        symbol: 标的代码
        qty: 数量
        strategy_id: 策略ID
        now_utc: 当前时间
        order_type: 订单类型
        limit_price: 限价（仅限价单有效）
        stop_price: 止损价（仅止损单有效）
        **metadata: 元数据
    """
    # 转换 side
    if isinstance(side, str):
        side = OrderSide(side.upper())
    
    # 转换 order_type
    if isinstance(order_type, str):
        order_type = OrderType(order_type.upper())
    
    return OrderIntent(
        order_id=str(uuid.uuid4()),
        ts_utc=now_utc,
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        strategy_id=strategy_id,
        metadata=metadata,
    )


def buy(symbol: str, qty: float, strategy_id: str, now_utc: datetime, **metadata) -> OrderIntent:
    """快速创建买入订单（工厂函数）"""
    return create_order("BUY", symbol, qty, strategy_id, now_utc, **metadata)


def sell(symbol: str, qty: float, strategy_id: str, now_utc: datetime, **metadata) -> OrderIntent:
    """快速创建卖出订单（工厂函数）"""
    return create_order("SELL", symbol, qty, strategy_id, now_utc, **metadata)


def limit_buy(symbol: str, qty: float, limit_price: float, 
              strategy_id: str, now_utc: datetime, **metadata) -> OrderIntent:
    """快速创建限价买入订单"""
    return create_order(
        "BUY", symbol, qty, strategy_id, now_utc,
        order_type="LMT", limit_price=limit_price, **metadata
    )


def limit_sell(symbol: str, qty: float, limit_price: float,
               strategy_id: str, now_utc: datetime, **metadata) -> OrderIntent:
    """快速创建限价卖出订单"""
    return create_order(
        "SELL", symbol, qty, strategy_id, now_utc,
        order_type="LMT", limit_price=limit_price, **metadata
    )


# ===== 策略集成辅助函数 =====

def create_engine_for_strategy(strategy_id: str, ctx: Optional[RunContext] = None) -> OrderEngine:
    """
    为策略创建订单引擎
    
    Args:
        strategy_id: 策略ID
        ctx: 运行上下文（可选，用于设置时间）
    """
    engine = OrderEngine(strategy_id)
    if ctx:
        engine.update_context(ctx.now_utc)
    return engine
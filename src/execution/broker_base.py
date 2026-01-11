"""
执行Broker基类
定义统一的执行接口
"""

from abc import ABC, abstractmethod
from typing import List, Protocol, runtime_checkable

from src.core.types import OrderIntent, Fill, PortfolioState
from src.core.context import RunContext


@runtime_checkable
class ExecutionBroker(Protocol):
    """执行Broker协议（使用Protocol定义接口，支持异步方法）"""
    
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
        ...
    
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
        ...
    
    async def cancel(
        self,
        order_id: str,
        ctx: RunContext,
    ) -> None:
        """
        撤销订单
        
        Args:
            order_id: 订单ID
            ctx: 运行上下文
        """
        ...
    
    async def get_positions(
        self,
        ctx: RunContext,
    ) -> PortfolioState:
        """
        获取持仓
        
        Args:
            ctx: 运行上下文
            
        Returns:
            组合状态
        """
        ...
    
    async def get_cash(
        self,
        ctx: RunContext,
    ) -> float:
        """
        获取现金
        
        Args:
            ctx: 运行上下文
            
        Returns:
            现金余额
        """
        ...

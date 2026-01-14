"""
策略生命周期混入类

为现有 Strategy 类添加 Freqtrade 风格的回调支持
"""

from typing import Dict, Optional
from datetime import datetime
import pandas as pd

from src.core.types import Fill, OrderSide
from src.utils.logger import get_logger


logger = get_logger("strategy.lifecycle")


class FreqtradeLifecycleMixin:
    """
    Freqtrade 风格策略生命周期混入类
    
    提供所有 Freqtrade 策略回调的默认实现
    可以被其他 Strategy 类继承
    """
    
    async def bot_start(self, **kwargs) -> None:
        """机器人启动时调用"""
        self.logger.info(f"Bot started: {self.strategy_name}")
    
    async def bot_loop_start(self, **kwargs) -> None:
        """每轮开始时调用"""
        pass
    
    def custom_stoploss(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> float:
        """自定义止损"""
        pass
    
    def custom_sell(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> Optional[str]:
        """自定义卖出"""
        pass
    
    def custom_buy(
        self,
        pair: str,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> Optional[str]:
        """自定义买入"""
        pass
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """确认订单进入"""
        pass
    
    def confirm_trade_exit(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """确认订单退出"""
        pass
    
    def adjust_trade_position(
        self,
        trade: Dict,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        current_stake_amount: float,
        **kwargs
    ) -> Optional[float]:
        """调整仓位"""
        pass
    
    def order_filled(
        self,
        pair: str,
        order: Fill,
        current_time: datetime,
        **kwargs
    ) -> None:
        """订单成交后"""
        self.logger.info(
            f"Order filled: {pair} {order.side.value} {order.qty} @ {order.price}"
        )
    
    def botShutdown(self, **kwargs) -> None:
        """机器人关闭"""
        self.logger.info(f"Bot shutdown: {self.strategy_name}")

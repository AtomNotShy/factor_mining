"""
止损/ROI 管理器

Freqtrade 风格的止损和收益管理
"""

from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from src.utils.logger import get_logger


logger = get_logger("stoploss_manager")


@dataclass
class ExitReason:
    """离场原因"""
    reason: str
    price: Optional[float] = None
    timestamp: Optional[datetime] = None


class StoplossManager:
    """
    止损/ROI 管理器
    
    Freqtrade 规则：
    1. Stoploss: 优先级最高，保护本金优先
    2. ROI: 次优先级
    3. Trailing Stoploss: 最后优先级
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,
    ):
        self.commission_rate = commission_rate
        self.logger = get_logger("stoploss_manager")
    
    def check_exit(
        self,
        symbol: str,
        current_price: float,
        avg_price: float,
        entry_time: datetime,
        current_time: datetime,
        stoploss_price: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        roi_table: Optional[Dict[int, float]] = None,
    ) -> Optional[ExitReason]:
        """检查是否应该离场"""
        if abs(avg_price) < 1e-8:
            return None
        
        current_pnl = (current_price - avg_price) / avg_price
        
        # 止损检查（优先级最高）
        if stoploss_price is not None:
            actual_loss = (avg_price - stoploss_price) / avg_price
            loss_with_fee = actual_loss + (2 * self.commission_rate)
            
            if current_pnl <= -abs(loss_with_fee):
                return ExitReason(
                    reason="stoploss",
                    price=stoploss_price,
                    timestamp=current_time,
                )
        
        # ROI 检查（次优先级）
        if roi_table is not None:
            duration_minutes = int((current_time - entry_time).total_seconds() / 60)

            # 找到最接近的 ROI 目标（Duration >= ROI threshold）
            # 使用 reverse=True 从大到小查找，按整数键排序
            sorted_roi = sorted(roi_table.items(), key=lambda x: int(x[0]), reverse=True)
            target_minutes = None
            target_profit = 0.0
            for minutes, profit in sorted_roi:
                if duration_minutes >= minutes:
                    target_minutes = minutes
                    target_profit = profit
                    break
            
            if target_minutes is None:
                return None
            
            target_profit_pct = target_profit
            target_price = avg_price * (1.0 + target_profit_pct)
            
            if current_pnl >= target_profit_pct:
                return ExitReason(
                    reason="roi",
                    price=target_price,
                    timestamp=current_time,
                )
        
        # 追踪止损检查（最后优先级）
        if trailing_stop is not None and stoploss_price is not None:
            if trailing_stop > stoploss_price:
                distance_to_stop = current_price - trailing_stop
                if distance_to_stop <= 0:
                    return ExitReason(
                        reason="trailing_stop",
                        price=trailing_stop,
                        timestamp=current_time,
                    )
        
        return None
    
    def calculate_stoploss_price(
        self,
        avg_price: float,
        stoploss_pct: float,
        is_short: bool = False,
    ) -> float:
        """计算止损价格"""
        if is_short:
            return avg_price * (1.0 + stoploss_pct)
        return avg_price * (1.0 - stoploss_pct)

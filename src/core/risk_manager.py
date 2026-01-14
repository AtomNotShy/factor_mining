"""
独立风控管理器
实现类似 Freqtrade 的止损、移动止损和 ROI 止盈逻辑。
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import pandas as pd
from src.utils.logger import get_logger

class RiskManager:
    """
    风控管理器 (RiskManager)
    负责在回测或实盘过程中，根据配置的参数对持仓进行实时风控检查。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化风控管理器
        
        Args:
            config: 包含 risk 字段的配置字典
        """
        self.logger = get_logger("risk_manager")
        self.risk_config = config.get("risk", {})
        
        # 核心参数
        self.stoploss = self.risk_config.get("stoploss")
        self.trailing_stop = self.risk_config.get("trailing_stop", False)
        self.trailing_stop_positive = self.risk_config.get("trailing_stop_positive")
        self.trailing_stop_positive_offset = self.risk_config.get("trailing_stop_positive_offset", 0.0)
        self.roi_table = self.risk_config.get("roi_table", {})
        
        # 内部状态：记录每笔持仓自进入以来的最高收益率（用于移动止损）
        self._high_water_marks: Dict[str, float] = {} # symbol -> max_profit_pct
        
    def check_exit(
        self, 
        symbol: str, 
        current_price: float, 
        avg_price: float, 
        entry_time: datetime, 
        current_time: datetime
    ) -> Optional[str]:
        """
        检查指定标的是否触发风控离场逻辑。
        
        Args:
            symbol: 标的代码
            current_price: 当前市场价格
            avg_price: 持仓平均成本
            entry_time: 入场时间
            current_time: 当前时间
            
        Returns:
            如果触发风控，返回离场原因字符串；否则返回 None。
        """
        if avg_price <= 0:
            return None
            
        profit_pct = (current_price - avg_price) / avg_price
        
        # 1. 更新最高水位线 (High Water Mark)
        if symbol not in self._high_water_marks:
            self._high_water_marks[symbol] = profit_pct
        else:
            if profit_pct > self._high_water_marks[symbol]:
                self._high_water_marks[symbol] = profit_pct
        
        hwm = self._high_water_marks[symbol]

        # 2. 静态止损 (Static Stop Loss)
        # 如果未开启移动止损，或者开启了但还没达到触发移动的阈值
        if self.stoploss is not None:
            # 基础止损检查
            if profit_pct <= self.stoploss:
                return f"stop_loss ({profit_pct:.2%})"

        # 3. 移动止损 (Trailing Stop)
        if self.trailing_stop and self.stoploss is not None:
            # 如果配置了 trailing_stop_positive，则需要达到特定利润才启用更紧的止损
            actual_stoploss = self.stoploss
            is_trailing_active = True
            
            if self.trailing_stop_positive is not None:
                if hwm >= self.trailing_stop_positive_offset:
                    actual_stoploss = self.trailing_stop_positive
                else:
                    # 尚未达到正向移动止损的激活阈值
                    is_trailing_active = False
            
            if is_trailing_active:
                # 触发点 = 最高水位 - 止损深度
                # 例子: hwm=0.10, actual_stoploss=-0.02 (2%回撤) -> trigger=0.08
                # 注意：我们这里统一假设 stoploss 是负数或代表回撤深度
                # 如果 actual_stoploss 是 -0.02, 则 trigger_pct = hwm - 0.02
                trigger_pct = hwm + actual_stoploss 
                if profit_pct < trigger_pct:
                    return f"trailing_stop_loss (hwm: {hwm:.2%}, trigger: {trigger_pct:.2%}, curr: {profit_pct:.2%})"

        # 4. ROI 表止盈 (Take Profit Table)
        if self.roi_table:
            # 计算持仓时间（分钟）
            duration_min = (current_time - entry_time).total_seconds() / 60
            
            # 找到对应时间点的最低利润要求
            # roi_table 格式 e.g.: {"0": 0.2, "30": 0.1, "60": 0.05}
            suitable_threshold = None
            # 转换 key 为 int 并排序
            sorted_keys = sorted([int(k) for k in self.roi_table.keys()], reverse=True)
            for time_key in sorted_keys:
                if duration_min >= time_key:
                    suitable_threshold = self.roi_table[str(time_key)]
                    break
            
            if suitable_threshold is not None and profit_pct >= suitable_threshold:
                return f"roi_hit (duration: {duration_min:.0f}min, profit: {profit_pct:.2%}, target: {suitable_threshold:.2%})"

        return None

    def on_exit(self, symbol: str):
        """标的离场后的清理工作"""
        if symbol in self._high_water_marks:
            del self._high_water_marks[symbol]
            self.logger.debug(f"已重置 {symbol} 的风控状态线")

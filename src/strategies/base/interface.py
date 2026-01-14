"""
IStrategy 接口定义

Freqtrade 风格的策略协议，定义所有策略必须实现的方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol, Type, TypeVar
import pandas as pd
import numpy as np

from src.core.types import (
    Signal, OrderIntent, Fill, PortfolioState, RiskState,
    MarketData, ActionType, OrderSide, OrderType,
)
from src.core.context import RunContext
from src.utils.logger import get_logger

logger = get_logger("strategy.interface")


# ============ 策略配置 ============

@dataclass
class SellType:
    """卖出原因类型"""
    ROI = "roi"           # 达到目标收益
    STOP_LOSS = "stop_loss"  # 止损
    TRAILING_STOP = "trailing_stop"  # 追踪止损
    TIMEOUT = "timeout"   # 超时
    CUSTOM = "custom"     # 自定义


@dataclass 
class BuyType:
    """买入原因类型"""
    LONG = "long"         # 正常做多
    SHORT = "short"       # 做空
    REBuy = "rebuy"       # 补仓


@dataclass
class StrategyConfig:
    """
    策略配置（类属性，自动从策略类读取）
    
    使用方式：
        class MyStrategy(IStrategy):
            strategy_name = "My Strategy"
            timeframe = "1d"
            stoploss = -0.05
            trailing_stop = False
            trailing_stop_positive = 0.02
            trailing_stop_positive_offset = 0.03
            startup_candle_count = 20
            order_types = {
                "entry": "limit",
                "exit": "limit", 
                "stoploss": "market",
                "stoploss_on_exchange": False,
            }
            position_adjustment_enable = False
            use_exit_signal = True
            exit_profit_only = False
            ignore_roi_if_entry_signal = False
    """
    # 策略标识
    strategy_name: str = ""
    strategy_id: str = ""  # 简短的策略ID，用于文件命名
    
    # 时间框架
    timeframe: str = "1d"
    startup_candle_count: int = 0  # 策略启动所需的最小K线数量
    
    # 止损配置
    stoploss: float = -0.10  # 止损比例（如 -0.05 表示亏损5%止损）
    trailing_stop: bool = False
    trailing_stop_positive: float = 0.0
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False
    trailing_stop_dry_run: bool = False  # 仅测试模式，不真正下单
    
    # 订单类型
    order_types: Dict[str, str] = field(default_factory=dict)  # {"entry": "limit", "exit": "limit", "stoploss": "market"}
    
    # 仓位配置
    position_size: float = 1.0  # 仓位比例 (0.0 - 1.0)
    stake_amount: Optional[float] = None  # 固定金额，如果为None则使用资金的比例
    default_stake_amount: float = 1000.0  # 默认每笔交易金额
    ask_last_balance: float = 0.0  # 保留现金比例
    
    # 止盈配置
    minimal_roi: Dict[str, float] = field(default_factory=dict)  # {"0": 0.05, "60": 0.03, "240": 0.02}
    
    # 信号配置
    use_exit_signal: bool = True
    exit_profit_only: bool = False
    ignore_roi_if_entry_signal: bool = False
    
    # 仓位调整
    position_adjustment_enable: bool = False
    
    # 交易时间段
    can_short: bool = False  # 是否支持做空
    use_book_value: bool = False
    
    # 保护配置
    ignore_protections: bool = False
    protections: List[Dict] = field(default_factory=list)
    
    # 白名单/黑名单
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)
    
    # 循环模式
    process_only_new_candles: bool = False
    keep_interested_candles: bool = False
    
    # 期货/杠杆配置
    futures_mode: bool = False
    leverage: float = 1.0
    
    class Config:
        arbitrary_types_allowed = True


# ============ 生命周期钩子 ============

class IStrategyLifecycle(Protocol):
    """
    策略生命周期钩子协议
    
    策略可以实现这些方法来响应不同的生命周期事件。
    """
    
    def bot_start(self, **kwargs) -> None:
        """机器人启动时调用"""
        pass
    
    def bot_loop_start(self, **kwargs) -> None:
        """每个循环开始时调用"""
        pass
    
    def custom_stoploss(self, pair: str, current_profit: float, 
                       current_rate: float, **kwargs) -> float:
        """
        自定义止损逻辑
        
        Args:
            pair: 交易对
            current_profit: 当前盈亏比例
            current_rate: 当前价格
            
        Returns:
            止损价格或比例
        """
        return self.config.stoploss
    
    def custom_sell(self, pair: str, current_profit: float, 
                   current_rate: float, current_time: datetime, **kwargs) -> Optional[str]:
        """
        自定义卖出逻辑
        
        Args:
            pair: 交易对
            current_profit: 当前盈亏比例
            current_rate: 当前价格
            current_time: 当前时间
            
        Returns:
            卖出原因或None（使用默认逻辑）
        """
        return None
    
    def custom_buy(self, pair: str, current_rate: float, 
                  current_time: datetime, **kwargs) -> Optional[str]:
        """
        自定义买入逻辑
        
        Args:
            pair: 交易对
            current_rate: 当前价格
            current_time: 当前时间
            
        Returns:
            买入原因或None（使用默认逻辑）
        """
        return None
    
    def confirm_trade_entry(self, pair: str, order_type: str, 
                           amount: float, rate: float, time_in_force: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        确认订单进入（可用于二次确认）
        
        Returns:
            True: 确认下单
            False: 取消订单
        """
        return True
    
    def confirm_trade_exit(self, pair: str, order_type: str, 
                          amount: float, rate: float, time_in_force: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        确认订单退出（可用于二次确认）
        
        Returns:
            True: 确认下单
            False: 取消订单
        """
        return True
    
    def adjust_trade_position(self, trade: Dict, current_time: datetime,
                             current_rate: float, current_profit: float,
                             min_stake: float, max_stake: float,
                             current_stake_amount: float, **kwargs) -> Optional[float]:
        """
        调整仓位（加仓/减仓）
        
        Args:
            trade: 交易信息字典
            current_time: 当前时间
            current_rate: 当前价格
            current_profit: 当前盈亏
            min_stake: 最小仓位
            max_stake: 最大仓位
            current_stake_amount: 当前仓位金额
            
        Returns:
            调整金额（正数加仓，负数减仓）或None（不调整）
        """
        return None
    
    def botShutdown(self, **kwargs) -> None:
        """机器人关闭时调用"""
        pass
    
    def analyze_strategy(self, *args, **kwargs) -> Dict[str, Any]:
        """
        策略分析（用于 hyperopt）
        
        Returns:
            优化结果字典
        """
        return {}


# ============ 核心策略接口 ============

class IStrategy(ABC):
    """
    策略核心接口（类似 freqtrade 的 IStrategy）
    
    所有策略必须继承此类并实现必要的方法。
    
    使用方式：
        class MyStrategy(IStrategy):
            strategy_name = "My Strategy"
            timeframe = "1d"
            
            def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
                # 计算指标
                dataframe['rsi'] = ta.rsi(dataframe['close'])
                return dataframe
            
            def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
                # 生成买入信号
                dataframe['enter_long'] = dataframe['rsi'] < 30
                return dataframe
            
            def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
                # 生成卖出信号
                dataframe['exit_long'] = dataframe['rsi'] > 70
                return dataframe
    """
    
    # 策略配置（子类应覆盖这些类属性）
    config: StrategyConfig = StrategyConfig()
    
    # 保留旧版接口兼容性
    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.__class__.__name__.lower()
    
    @property
    def strategy_name(self) -> str:
        return self.config.strategy_name or self.__class__.__name__
    
    @property
    def timeframe(self) -> str:
        return self.config.timeframe
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """初始化策略"""
        if config is not None:
            self.config = config
        
        self.logger = get_logger(f"strategy.{self.strategy_id}")
        self._prepared = False
        
    # ============ 核心方法（必须实现） ============
    
    @abstractmethod
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            dataframe: K线数据
            metadata: 元数据（包含 symbol, timeframe 等）
            
        Returns:
            添加了指标的 DataFrame
        """
        pass
    
    @abstractmethod
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        生成买入信号
        
        在 dataframe 中添加 'enter_long', 'enter_short' 等列
        enter_long=1 表示买入信号
        
        Args:
            dataframe: K线数据（包含指标）
            metadata: 元数据
            
        Returns:
            添加了买入信号的 DataFrame
        """
        pass
    
    @abstractmethod
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        生成卖出信号
        
        在 dataframe 中添加 'exit_long', 'exit_short' 等列
        exit_long=1 表示卖出信号
        
        Args:
            dataframe: K线数据（包含指标）
            metadata: 元数据
            
        Returns:
            添加了卖出信号的 DataFrame
        """
        pass
    
    # ============ 可选方法（提供默认实现） ============
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "timeframe": self.config.timeframe,
            "stoploss": self.config.stoploss,
            "trailing_stop": self.config.trailing_stop,
            "minimal_roi": self.config.minimal_roi,
            "order_types": self.config.order_types,
        }
    
    def version(self) -> int:
        """策略版本号"""
        return 1
    
    def check_buy_timeout(self, pair: str, current_time: datetime, 
                         current_rate: float, **kwargs) -> bool:
        """检查买入是否超时（返回 True 表示取消订单）"""
        return False
    
    def check_sell_timeout(self, pair: str, current_time: datetime,
                          current_rate: float, **kwargs) -> bool:
        """检查卖出是否超时（返回 True 表示取消订单）"""
        return False
    
    def adjust_entry_price(self, pair: str, current_time: datetime,
                          current_rate: float, proposed_rate: float,
                          current_order: Dict, **kwargs) -> float:
        """
        调整入场价格
        
        Args:
            pair: 交易对
            current_time: 当前时间
            current_rate: 当前价格
            proposed_rate: 提议的价格
            current_order: 当前订单信息
            
        Returns:
            调整后的价格
        """
        return proposed_rate
    
    def fee(self, pair: str, order_type: str, is_maker: bool, 
           amount: float, price: float) -> float:
        """
        计算手续费
        
        Args:
            pair: 交易对
            order_type: 订单类型
            is_maker: 是否是 maker
            amount: 数量
            price: 价格
            
        Returns:
            手续费金额
        """
        return 0.0
    
    def should_roi_on_trade(self, trade: Dict, current_time: datetime,
                           current_profit: float, min_roi: Dict,
                           max_roi: Dict, last_candle: pd.Series,
                           previous_candle: pd.Series, **kwargs) -> bool:
        """
        检查是否应该根据 ROI 退出交易
        """
        return True
    
    def should_exit(self, pair: str, current_time: datetime,
                   current_rate: float, current_profit: float,
                   dataframe: pd.DataFrame, **kwargs) -> bool:
        """
        决定是否退出交易
        
        Args:
            pair: 交易对
            current_time: 当前时间
            current_rate: 当前价格
            current_profit: 当前盈亏
            dataframe: K线数据
            
        Returns:
            True: 退出交易
            False: 继续持有
        """
        return False
    
    # ============ 保护机制 ============
    
    def _protections(self) -> List[Dict]:
        """返回保护配置"""
        return self.config.protections
    
    def ignore_buying_timeout(self, pair: str, current_time: datetime, 
                             current_value: float, **kwargs) -> bool:
        """
        检查是否应该忽略买入超时保护
        
        Returns:
            True: 忽略超时
            False: 应用超时
        """
        return self.config.ignore_protections
    
    def ignore_selling_timeout(self, pair: str, current_time: datetime,
                              current_value: float, **kwargs) -> bool:
        """
        检查是否应该忽略卖出超时保护
        
        Returns:
            True: 忽略超时
            False: 应用超时
        """
        return self.config.ignore_protections
    
    # ============ 仓位和资金管理 ============
    
    def stake_amount(self, pair: str, current_time: datetime,
                    current_rate: float, proposed_stake: float,
                    min_stake: float, max_stake: float, 
                    current_entry: float, current_close: float,
                    **kwargs) -> float:
        """
        计算仓位大小
        
        Args:
            pair: 交易对
            current_time: 当前时间
            current_rate: 当前价格
            proposed_stake: 建议的仓位
            min_stake: 最小仓位
            max_stake: 最大仓位
            current_entry: 入场价格
            current_close: 当前收盘价
            
        Returns:
            最终仓位大小
        """
        return proposed_stake
    
    def min_stake_amount(self, pair: str, current_time: datetime,
                        current_rate: float, **kwargs) -> float:
        """最小仓位"""
        return self.config.default_stake_amount
    
    def max_stake_amount(self, pair: str, current_time: datetime,
                        current_rate: float, **kwargs) -> float:
        """最大仓位"""
        return float('inf')
    
    def custom_stake_amount(self, pair: str, current_time: datetime,
                           current_rate: float, current_profit: float,
                           min_stake: float, max_stake: float,
                           current_stake_amount: float, **kwargs) -> float:
        """
        自定义仓位计算
        
        Returns:
            仓位大小
        """
        return current_stake_amount
    
    # ============ 工具方法 ============
    
    def load_pair_history(self, pair: str, timeframe: Optional[str] = None,
                         since: Optional[datetime] = None) -> pd.DataFrame:
        """
        加载交易对历史数据
        
        Args:
            pair: 交易对
            timeframe: 时间框架
            since: 开始时间
            
        Returns:
            K线数据 DataFrame
        """
        # 子类应实现此方法
        return pd.DataFrame()
    
    def get_informative_pairs(self) -> List[tuple]:
        """
        返回 informative pairs（用于多时间框架）
        
        Returns:
            [(pair, timeframe), ...] 列表
        """
        return []
    
    def timeframe_to_seconds(self, timeframe: str) -> int:
        """
        将时间框架转换为秒数
        
        Args:
            timeframe: 时间框架字符串（如 "1m", "5m", "1h", "1d"）
            
        Returns:
            秒数
        """
        timeframe_map = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "12h": 43200,
            "1d": 86400,
            "1w": 604800,
        }
        return timeframe_map.get(timeframe, 86400)
    
    def seconds_to_timeframe(self, seconds: int) -> str:
        """
        将秒数转换为时间框架
        
        Args:
            seconds: 秒数
            
        Returns:
            时间框架字符串
        """
        if seconds >= 86400 * 7:
            return "1w"
        elif seconds >= 86400:
            return "1d"
        elif seconds >= 43200:
            return "12h"
        elif seconds >= 21600:
            return "6h"
        elif seconds >= 14400:
            return "4h"
        elif seconds >= 7200:
            return "2h"
        elif seconds >= 3600:
            return "1h"
        elif seconds >= 1800:
            return "30m"
        elif seconds >= 900:
            return "15m"
        elif seconds >= 300:
            return "5m"
        elif seconds >= 180:
            return "3m"
        else:
            return "1m"


# ============ 类型别名 ============

StrategyType = TypeVar('StrategyType', bound=IStrategy)


# ============ 便利函数 ============

def create_strategy(config: Dict[str, Any]) -> StrategyConfig:
    """从字典创建策略配置"""
    return StrategyConfig(**config)


def merge_strategy_config(base: StrategyConfig, override: Dict[str, Any]) -> StrategyConfig:
    """合并策略配置"""
    config_dict = base.__dict__.copy()
    config_dict.update(override)
    return StrategyConfig(**config_dict)

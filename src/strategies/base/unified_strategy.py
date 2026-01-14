"""
统一策略接口
支持回测和实盘的统一策略定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type
import pandas as pd

from src.core.types import Signal, OrderIntent, PortfolioState, RiskState, OrderSide, OrderType
from src.core.context import RunContext
from src.utils.logger import get_logger

logger = get_logger("strategy.unified")


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_id: str = ""
    strategy_name: str = ""
    timeframe: str = "1d"
    params: Dict[str, Any] = field(default_factory=dict)
    category: str = ""
    auto_register: bool = True


@dataclass
class StrategyContext:
    """策略执行上下文"""
    mode: str = "backtest"  # "backtest" 或 "live"
    current_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_adapter: Any = None
    portfolio_state: Optional[PortfolioState] = None
    risk_state: Optional[RiskState] = None
    config: Dict[str, Any] = field(default_factory=dict)
    cross_section: Optional[Dict[str, Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedStrategy(ABC):
    """统一策略接口（简化版）"""

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig()
        self.config = config
        self.strategy_id = config.strategy_id or self.__class__.__name__.lower()
        self.strategy_name = config.strategy_name or self.__class__.__name__
        self.timeframe = config.timeframe
        self.params = config.params
        self.category = getattr(self, "category", "")
        self.logger = get_logger(f"strategy.{self.strategy_id}")
        self._informative_data: Dict[str, pd.DataFrame] = {}

    # ============ 向量化方法（可选实现） ============

    def prepare_data(
        self, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        [向量化] 批量计算指标
        默认直接返回数据，可重写添加技术指标
        """
        return data

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """[向量化] 批量计算指标（可选重写）"""
        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """[向量化] 批量计算进场信号（可选重写）"""
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """[向量化] 批量计算离场信号（可选重写）"""
        return dataframe

    # ============ 事件驱动方法（必须实现） ============

    @abstractmethod
    def generate_signals(
        self,
        market_data: Any,
        ctx: RunContext,
    ) -> List[Signal]:
        """
        [事件驱动] 生成策略信号（必须实现）
        """
        pass

    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        """
        默认等权分配仓位
        可重写自定义仓位计算
        """
        if not signals:
            # 清空所有持仓
            orders = []
            for symbol, qty in portfolio.positions.items():
                if abs(qty) > 1e-6:
                    orders.append(OrderIntent(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=abs(qty),
                        order_type=OrderType.MKT,
                        strategy_id=self.strategy_id,
                        metadata={"reason": "no_signal"},
                    ))
            return orders

        # 等权买入
        cash = portfolio.cash
        per_position = cash / len(signals)
        orders = []

        for signal in signals:
            price = signal.metadata.get("price", 0)
            if price <= 0:
                continue
            qty = per_position / price
            orders.append(OrderIntent(
                ts_utc=ctx.now_utc,
                symbol=signal.symbol,
                side=OrderSide.BUY,
                qty=qty,
                order_type=OrderType.MKT,
                strategy_id=self.strategy_id,
                metadata={"score": signal.strength},
            ))

        return orders

    # ============ 生命周期方法 ============

    def on_start(self, context: StrategyContext):
        """策略启动"""
        self.logger.info(f"策略 {self.strategy_id} 启动")
        pass

    def on_stop(self, context: StrategyContext):
        """策略停止"""
        self.logger.info(f"策略 {self.strategy_id} 停止")
        pass

    def on_error(self, error: Exception, context: StrategyContext):
        """错误处理"""
        self.logger.error(f"策略 {self.strategy_id} 错误: {error}")
        pass

    def on_fill(self, fill_event: Any, context: StrategyContext) -> Dict[str, Any]:
        """
        成交事件处理

        Args:
            fill_event: 成交事件
            context: 策略上下文

        Returns:
            状态更新字典
        """
        return {}

    # ============ 辅助方法 ============

    def set_informative_data(self, bars_map: Dict[str, pd.DataFrame]):
        """设置 informative 数据"""
        self._informative_data = bars_map

    def get_informative_pair(self, timeframe: str, column: str) -> pd.Series:
        """获取 informative 数据"""
        key = f"{column}_{timeframe}"
        if timeframe in self._informative_data:
            df = self._informative_data[timeframe]
            if key in df.columns:
                return df[key]
        return pd.Series()

    def get_parameter_value(self, name: str, default: Any = None) -> Any:
        """获取参数值"""
        return self.params.get(name, default)

    def enable_lookahead_check(self) -> bool:
        """是否开启未来数据检查"""
        return bool(self.params.get("lookahead_check", False))

    @property
    def ta(self):
        """
        提供对 pandas_ta 的直接访问
        允许在策略中通过 self.ta.<indicator>(...) 调用
        """
        import pandas_ta as ta
        return ta

    def get_config(self) -> StrategyConfig:
        """获取策略配置"""
        return self.config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "params": self.params,
            "category": self.category,
        }


# ============ 策略工厂 ============

class StrategyFactory:
    """策略工厂"""

    _strategies: Dict[str, Type[UnifiedStrategy]] = {}

    @classmethod
    def register(cls, strategy_id: str, strategy_class: Type[UnifiedStrategy]):
        """注册策略"""
        cls._strategies[strategy_id] = strategy_class
        logger.info(f"注册策略: {strategy_id}")

    @classmethod
    def create(cls, strategy_id: str, config: Optional[StrategyConfig] = None) -> Optional[UnifiedStrategy]:
        """创建策略实例"""
        if strategy_id in cls._strategies:
            strategy_class = cls._strategies[strategy_id]
            if config is None:
                config = StrategyConfig(strategy_id=strategy_id)
            return strategy_class(config)
        return None

    @classmethod
    def get_strategy_class(cls, strategy_id: str) -> Optional[Type[UnifiedStrategy]]:
        """获取策略类"""
        return cls._strategies.get(strategy_id)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有策略"""
        return list(cls._strategies.keys())


# ============ 便捷函数 ============

def create_strategy(
    strategy_id: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[UnifiedStrategy]:
    """创建策略实例的便捷函数"""
    strategy_class = StrategyFactory.get_strategy_class(strategy_id)
    if strategy_class is None:
        logger.error(f"策略未注册: {strategy_id}")
        return None

    config = StrategyConfig(
        strategy_id=strategy_id,
        params=params or {},
    )
    return strategy_class(config)

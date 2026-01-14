"""
策略适配器
将现有策略适配到统一策略接口
"""

from typing import Any, Dict, List, Optional, Type
from datetime import datetime, timezone
import pandas as pd

from src.core.types import Signal, OrderIntent, PortfolioState, RiskState, MarketData, ActionType, OrderSide
from src.core.context import RunContext
from src.strategies.base.strategy import Strategy as LegacyStrategy
from src.strategies.base.unified_strategy import UnifiedStrategy, StrategyConfig, StrategyContext
from src.utils.logger import get_logger

logger = get_logger("strategy.adapter")


class StrategyAdapter(UnifiedStrategy):
    """
    策略适配器
    将现有 LegacyStrategy 适配到 UnifiedStrategy 接口
    """

    def __init__(
        self,
        legacy_strategy: LegacyStrategy,
        config: Optional[StrategyConfig] = None,
    ):
        # 从 legacy strategy 获取配置
        if config is None:
            config = StrategyConfig(
                strategy_id=legacy_strategy.strategy_id,
                strategy_name=legacy_strategy.__class__.__name__,
                timeframe=getattr(legacy_strategy, "timeframe", "1d"),
                params=legacy_strategy.config.params if hasattr(legacy_strategy, "config") else {},
            )

        super().__init__(config)
        self.legacy_strategy = legacy_strategy
        self.logger = get_logger(f"strategy.adapter.{self.strategy_id}")

    def prepare_data(
        self, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """适配 prepare_data"""
        # 调用 legacy populate_indicators
        result = self.legacy_strategy.populate_indicators(data, metadata)
        return result

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """适配 populate_indicators"""
        return self.legacy_strategy.populate_indicators(dataframe, metadata)

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """适配 populate_entry_trend"""
        return self.legacy_strategy.populate_entry_trend(dataframe, metadata)

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """适配 populate_exit_trend"""
        return self.legacy_strategy.populate_exit_trend(dataframe, metadata)

    def generate_signals(
        self,
        market_data: MarketData,
        ctx: RunContext,
    ) -> List[Signal]:
        """适配 generate_signals"""
        try:
            signals = self.legacy_strategy.generate_signals(market_data, ctx)
            if signals is None:
                return []
            return signals
        except Exception as e:
            self.logger.error(f"生成信号失败: {e}")
            return []

    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        """适配 size_positions"""
        try:
            order_intents = self.legacy_strategy.size_positions(
                signals, portfolio, risk, ctx
            )
            if order_intents is None:
                return []
            return order_intents
        except Exception as e:
            self.logger.error(f"计算仓位失败: {e}")
            return []

    def risk_checks(
        self,
        order_intents: List[OrderIntent],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> tuple[List[OrderIntent], List[Dict]]:
        """适配 risk_checks"""
        if hasattr(self.legacy_strategy, "risk_checks"):
            return self.legacy_strategy.risk_checks(order_intents, portfolio, risk, ctx)
        return order_intents, []

    def on_fill(self, fill_event: Any, ctx: RunContext) -> Dict[str, Any]:
        """适配 on_fill"""
        if hasattr(self.legacy_strategy, "on_fill"):
            return self.legacy_strategy.on_fill(fill_event, ctx)
        return {}


def adapt_legacy_strategy(
    legacy_strategy: LegacyStrategy,
    config: Optional[StrategyConfig] = None,
) -> StrategyAdapter:
    """
    将现有策略适配为统一策略接口

    Args:
        legacy_strategy: 现有策略实例
        config: 可选的策略配置

    Returns:
        适配后的策略实例
    """
    return StrategyAdapter(legacy_strategy, config)


def create_adapter_from_class(
    strategy_class: Type[LegacyStrategy],
    params: Optional[Dict[str, Any]] = None,
) -> Optional[StrategyAdapter]:
    """
    从策略类创建适配器

    Args:
        strategy_class: 策略类
        params: 策略参数

    Returns:
        适配后的策略实例
    """
    try:
        legacy_strategy = strategy_class()
        if params:
            legacy_strategy.set_params(params)
        return adapt_legacy_strategy(legacy_strategy)
    except Exception as e:
        logger.error(f"创建策略适配器失败: {strategy_class.__name__}: {e}")
        return None


# ============ 向量化到事件驱动的桥接 ============

class VectorizedToEventAdapter(UnifiedStrategy):
    """
    向量化策略到事件驱动策略的桥接适配器
    允许向量化策略在事件驱动模式下运行
    """

    def __init__(
        self,
        vectorized_strategy: "VectorizedStrategyBridge",
        config: Optional[StrategyConfig] = None,
    ):
        super().__init__(config)
        self.vectorized_strategy = vectorized_strategy
        self._cached_signals: Dict[str, pd.DataFrame] = {}

    def prepare_data(
        self, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """准备数据"""
        # 使用向量化方法计算指标
        result = self.vectorized_strategy.populate_indicators(data, metadata)
        result = self.vectorized_strategy.populate_entry_trend(result, metadata)
        result = self.vectorized_strategy.populate_exit_trend(result, metadata)
        return result

    def generate_signals(
        self,
        market_data: MarketData,
        ctx: RunContext,
    ) -> List[Signal]:
        """从向量化结果生成信号"""
        signals = []
        # 从 datetime 属性获取当前时间
        current_time = getattr(ctx, 'now_utc', datetime.now(timezone.utc)) if ctx else datetime.now(timezone.utc)
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # 从缓存的向量化结果中获取当前时间的信号
        for symbol in market_data.bars["symbol"].unique():
            v_data = self._cached_signals.get(symbol)
            if v_data is not None and current_time in v_data.index:
                row = v_data.loc[current_time]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

                # 检查进场信号
                enter_long = row.get("enter_long", 0)
                if enter_long == 1:
                    ts_utc = current_time or datetime.now(timezone.utc)
                    signals.append(Signal(
                        ts_utc=ts_utc,
                        symbol=symbol,
                        strategy_id=self.strategy_id,
                        action=ActionType.LONG,
                        strength=float(row.get("momentum_score", 1.0)),
                    ))

                # 检查离场信号
                exit_long = row.get("exit_long", 0)
                if exit_long == 1:
                    ts_utc = current_time or datetime.now(timezone.utc)
                    signals.append(Signal(
                        ts_utc=ts_utc,
                        symbol=symbol,
                        strategy_id=self.strategy_id,
                        action=ActionType.FLAT,
                        strength=1.0,
                    ))

        return signals

    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        """计算仓位"""
        # 可以使用横截面数据进行轮动
        return []

    def cache_signals(self, symbol: str, data: pd.DataFrame):
        """缓存向量化结果"""
        self._cached_signals[symbol] = data


class VectorizedStrategyBridge:
    """
    向量化策略桥接基类
    用于兼容现有的向量化策略
    """

    def __init__(self):
        self.strategy_id = self.__class__.__name__.lower()
        self.timeframe = "1d"
        self.logger = get_logger(f"strategy.vectorized.{self.strategy_id}")

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """计算指标"""
        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """计算进场信号"""
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """计算离场信号"""
        return dataframe

    def create_event_adapter(self) -> VectorizedToEventAdapter:
        """创建事件驱动适配器"""
        return VectorizedToEventAdapter(self)

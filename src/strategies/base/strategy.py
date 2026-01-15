"""
策略基类
Freqtrade 风格：支持类属性配置，无需 __init__
"""
from __future__ import annotations  # 延迟类型注解求值，避免循环导入

from abc import ABC, abstractmethod
import inspect
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Type, Union
import pandas as pd

if TYPE_CHECKING:
    from src.execution.order_engine import OrderEngine

from src.core.types import Signal, OrderIntent, MarketData, PortfolioState, RiskState, ActionType, OrderSide, OrderType
from src.core.context import RunContext
from src.utils.logger import get_logger
from .parameters import Parameter, IntParameter, DecimalParameter, BooleanParameter

# 延迟导入，避免循环依赖
_order_engine = None

def _get_order_engine():
    global _order_engine
    if _order_engine is None:
        from src.execution.order_engine import OrderEngine, create_order, buy, sell
        _order_engine = (OrderEngine, create_order, buy, sell)
    return _order_engine


# 先定义 Config 类，再添加类方法
class StrategyConfig:
    """策略配置"""
    def __init__(self, strategy_id: str = "", timeframe: str = "1d", params: Optional[Dict[str, Any]] = None):
        self.strategy_id = strategy_id
        self.timeframe = timeframe
        self.params = params or {}
    
    @classmethod
    def from_class(cls, strategy_cls: Any) -> 'StrategyConfig':
        """从策略类自动创建配置"""
        strategy_id = getattr(strategy_cls, "strategy_id", strategy_cls.__name__.lower())
        timeframe = getattr(strategy_cls, "timeframe", "1d")
        
        # 收集类级别的参数
        params = {}
        for attr_name in dir(strategy_cls):
            if attr_name.startswith("_"):
                continue
            attr = getattr(strategy_cls, attr_name, None)
            if attr is not None and not callable(attr):
                if attr_name not in ["strategy_id", "timeframe", "auto_register"]:
                    params[attr_name] = attr
        
        return cls(
            strategy_id=strategy_id,
            timeframe=timeframe,
            params=params
        )


class Strategy(ABC):
    """策略基类"""

    auto_register: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "auto_register", True):
            return
        if inspect.isabstract(cls):
            return
        registry = globals().get("strategy_registry")
        if registry is None:
            return
        try:
            instance = cls()
        except TypeError as exc:
            get_logger("strategy_registry").debug(
                f"跳过自动注册 {cls.__name__}: {exc}"
            )
            return
        if registry.get_strategy(instance.strategy_id) is None:
            registry.register(instance)
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            # 从类属性获取配置
            cls = self.__class__
            strategy_id = getattr(cls, "strategy_id", cls.__name__.lower())
            timeframe = getattr(cls, "timeframe", "1d")
            
            # 收集类属性作为参数
            params = {}
            for attr_name in dir(cls):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(cls, attr_name, None)
                if attr is not None and not callable(attr):
                    if attr_name not in ["strategy_id", "timeframe", "auto_register"]:
                        params[attr_name] = attr
            
            config = StrategyConfig(
                strategy_id=strategy_id,
                timeframe=timeframe,
                params=params
            )
        self.config = config
        self.strategy_id = config.strategy_id
        self.timeframe = config.timeframe
        self.logger = get_logger(f"strategy.{config.strategy_id}")
        
        self._informative_data: Dict[str, pd.DataFrame] = {}
        self._informative_results: Dict[str, pd.DataFrame] = {}
        self.informative_timeframes: List[str] = []
        self._vectorized_data: Dict[str, pd.DataFrame] = {}  # 存储向量化预计算后的数据
        
        # 自动收集参数
        self._parameters: Dict[str, Parameter] = {}
        
        # 订单引擎（必须在 _collect_parameters 之前初始化）
        # 使用字符串注解避免循环导入
        self._order_engine = None  # type: Optional['OrderEngine']
        
        self._collect_parameters()
    
    def _collect_parameters(self) -> None:
        """自动收集类定义中的参数对象"""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                if attr.name != attr_name:
                    # 参数名与属性名不一致时，更新参数名
                    attr.name = attr_name
                self._parameters[attr_name] = attr
                self.logger.debug(f"发现参数: {attr_name} = {attr.value}")
    
    def get_parameters(self) -> Dict[str, Parameter]:
        """获取所有参数定义"""
        return self._parameters.copy()
    
    def get_parameter_value(self, name: str) -> Any:
        """获取单个参数值"""
        if name in self._parameters:
            return self._parameters[name].value
        # 回退到 config.params
        return self.config.params.get(name)
    
    def set_parameter_values(self, param_dict: Dict[str, Any]) -> None:
        """设置参数值（供优化器使用）"""
        for name, value in param_dict.items():
            if name in self._parameters:
                self._parameters[name].value = value
                self.logger.debug(f"更新参数: {name} = {value}")
            else:
                # 未知参数，记录警告
                self.logger.warning(f"未知参数: {name}")
    
    def _has_vectorized_methods(self) -> bool:
        """检查是否有向量化方法被重写实现"""
        # 检查每个方法是否在子类中被重写
        vectorized_methods = [
            'populate_indicators',
            'populate_entry_trend', 
            'populate_exit_trend'
        ]
        for method_name in vectorized_methods:
            method = getattr(self.__class__, method_name, None)
            # 检查方法是否被显式定义（不是从 Strategy 基类继承的默认实现）
            if method is not None:
                # 如果方法不在 Strategy 基类中定义，说明被重写了
                base_method = Strategy.__dict__.get(method_name)
                if base_method is not method:
                    return True
        return False
    
    def set_informative_data(self, bars_map: Dict[str, pd.DataFrame]):
        self._informative_data = bars_map
        self._main_dataframe_with_informative = bars_map.get(self.timeframe, pd.DataFrame())
    
    def get_informative_methods(self) -> List[Any]:
        """
        获取所有@informative装饰的方法
        
        Returns:
            装饰的方法列表
        """
        methods = []
        for name in dir(self):
            attr = getattr(self, name)
            if callable(attr) and hasattr(attr, '_is_informative'):
                methods.append(attr)
        return methods
    
    def get_informative_pair(self, timeframe: str, column: str) -> Union[pd.Series, pd.DataFrame]:
        key = f"{column}_{timeframe}"
        
        if not hasattr(self, '_informative_results'):
            self.logger.warning(f"未找到informative数据: {timeframe}/{column}")
            return pd.Series()
        
        informative_df = self._informative_results.get(timeframe)
        if informative_df is None or informative_df.empty:
            self.logger.warning(f"未找到informative数据: {timeframe}/{column}")
            return pd.Series()
        
        if key in informative_df.columns:
            return informative_df[key]
        
        self.logger.warning(f"未找到informative数据: {timeframe}/{column}")
        return pd.Series()
    
    @property
    def ta(self):
        """
        提供对 pandas_ta 的直接访问
        允许在策略中通过 self.ta.<indicator>(...) 调用
        如果pandas_ta不可用，返回一个模拟对象
        """
        try:
            import pandas_ta as ta
            return ta
        except ImportError:
            # 返回一个模拟对象，避免ImportError
            class MockTA:
                def __getattr__(self, name):
                    def mock_indicator(*args, **kwargs):
                        # 返回一个空Series或DataFrame
                        import pandas as pd
                        return pd.Series()
                    return mock_indicator
            return MockTA()
    
    def populate_indicators(self, dataframe: Any, metadata: Dict[str, Any]) -> Any:
        """
        [Vectorized] 批量计算指标
        参考 Freqtrade: 一次性计算所有技术指标
        
        Args:
            dataframe: 原始价格 DataFrame
            metadata: 元数据 (symbol, etc.)
            
        Returns:
            带有指标的 DataFrame
        """
        return dataframe

    def populate_entry_trend(self, dataframe: Any, metadata: Dict[str, Any]) -> Any:
        """
        [Vectorized] 批量计算进场信号
        在 'enter_long' 或 'enter_short' 列标记 1
        
        Args:
            dataframe: 带有指标的 DataFrame
            metadata: 元数据
            
        Returns:
            带有进场信号的 DataFrame
        """
        return dataframe

    def populate_exit_trend(self, dataframe: Any, metadata: Dict[str, Any]) -> Any:
        """
        [Vectorized] 批量计算离场信号
        在 'exit_long' 或 'exit_short' 列标记 1
        
        Args:
            dataframe: 带有指标的 DataFrame
            metadata: 元数据
            
        Returns:
            带有离场信号的 DataFrame
        """
        return dataframe

    @abstractmethod
    def generate_signals(
        self,
        md: MarketData,
        ctx: RunContext,
    ) -> List[Signal]:
        """
        生成策略信号
        
        Args:
            md: 市场数据
            ctx: 运行上下文
            
        Returns:
            信号列表
        """
        pass
    
    @abstractmethod
    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        """
        根据信号和组合状态计算目标仓位（订单意图）
        
        Args:
            signals: 策略信号列表
            portfolio: 组合状态
            risk: 风险状态
            ctx: 运行上下文（可通过 ctx.cross_section 获取当前时刻的横截面数据）
            
        Returns:
            订单意图列表
        """
        pass
    
    def risk_checks(
        self,
        order_intents: List[OrderIntent],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> tuple[List[OrderIntent], List[Dict[str, Any]]]:
        """
        风险检查（可选，默认实现）
        
        Args:
            order_intents: 订单意图列表
            portfolio: 组合状态
            risk: 风险状态
            ctx: 运行上下文
            
        Returns:
            (approved_intents, blocks) 元组
        """
        approved = []
        blocks = []
        
        for intent in order_intents:
            # 检查黑名单
            if intent.symbol in risk.blacklist:
                blocks.append({
                    "order_intent": intent,
                    "reason": f"标的 {intent.symbol} 在黑名单中",
                })
                continue
            
            # 检查最大持仓数
            if risk.max_positions is not None:
                current_positions = len([s for s in portfolio.positions.values() if abs(s) > 1e-8])
                if intent.side == OrderSide.BUY and intent.symbol not in portfolio.positions:
                    if current_positions >= risk.max_positions:
                        blocks.append({
                            "order_intent": intent,
                            "reason": f"已达到最大持仓数限制: {risk.max_positions}",
                        })
                        continue
            
            # 检查单标的最大仓位
            if risk.max_position_size is not None:
                # 这里需要知道目标仓位大小，简化处理
                pass
            
            approved.append(intent)
        
        return approved, blocks
    
    def on_fill(self, fill_event, ctx: RunContext) -> Dict[str, Any]:
        """
        成交事件处理（可选）
        
        Args:
            fill_event: 成交事件（Fill对象）
            ctx: 运行上下文
            
        Returns:
            状态更新字典
        """
        return {}
    
    def set_params(self, params: Dict[str, Any]):
        """
        设置策略参数
        
        Args:
            params: 参数字典
        """
        self.config.params.update(params)
        self.logger.info(f"更新策略参数: {params}")

    def enable_lookahead_check(self) -> bool:
        """是否开启未来数据检查（由策略配置控制）"""
        return bool(self.config.params.get("lookahead_check", False))

    # ===== 订单引擎集成方法（延迟导入）=====

    @property
    def order_engine(self):
        """获取订单引擎（懒加载）"""
        from src.execution.order_engine import OrderEngine
        if not hasattr(self, '_cached_order_engine'):
            self._cached_order_engine = OrderEngine(self.strategy_id)
        return self._cached_order_engine
    
    def update_order_engine_context(self, ctx: RunContext):
        """更新订单引擎的运行时上下文"""
        if hasattr(self, '_cached_order_engine'):
            self._cached_order_engine.update_context(ctx.now_utc)
    
    def create_buy_order(self, symbol: str, qty: float, ctx: RunContext, **metadata) -> OrderIntent:
        """快速创建买入订单（便捷方法）"""
        from src.execution.order_engine import buy
        return buy(symbol, qty, self.strategy_id, ctx.now_utc, **metadata)
    
    def create_sell_order(self, symbol: str, qty: float, ctx: RunContext, **metadata) -> OrderIntent:
        """快速创建卖出订单（便捷方法）"""
        from src.execution.order_engine import sell
        return sell(symbol, qty, self.strategy_id, ctx.now_utc, **metadata)
    
    def create_close_order(self, symbol: str, ctx: RunContext, qty: Optional[float] = None, **metadata) -> OrderIntent:
        """快速创建平仓订单（便捷方法）"""
        from src.execution.order_engine import create_order
        metadata["reason"] = "close"
        return create_order(
            "SELL", symbol, qty if qty is not None else 0,
            self.strategy_id, ctx.now_utc, **metadata
        )
    

class StrategyRegistry:
    """策略注册表"""
    
    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        self.logger = get_logger("strategy_registry")
    
    def register(self, strategy: Strategy):
        """注册策略"""
        self._strategies[strategy.strategy_id] = strategy
        self.logger.debug(f"注册策略: {strategy.strategy_id}")
    
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """获取策略"""
        return self._strategies.get(name)
    
    def list_strategies(self, category: Optional[str] = None) -> List[str]:
        """列出策略"""
        if category:
            return [name for name, strategy in self._strategies.items() 
                   if getattr(strategy, 'category', None) == category]
        return list(self._strategies.keys())
    
    def get_strategies_by_category(self, category: str) -> Dict[str, Strategy]:
        """按分类获取策略"""
        return {name: strategy for name, strategy in self._strategies.items()
                if getattr(strategy, 'category', None) == category}
    
    def get_all_strategies(self) -> Dict[str, Strategy]:
        """获取所有策略"""
        return self._strategies.copy()


# 全局策略注册表
strategy_registry = StrategyRegistry()

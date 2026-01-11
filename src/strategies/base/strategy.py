"""
策略基类
统一接口：generate_signals -> size_positions
"""

from abc import ABC, abstractmethod
import inspect
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.core.types import Signal, OrderIntent, MarketData, PortfolioState, RiskState, ActionType, OrderSide, OrderType
from src.core.context import RunContext
from src.utils.logger import get_logger


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_id: str
    timeframe: str
    params: Dict = field(default_factory=dict)


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
            # 为自动注册提供默认配置
            config = StrategyConfig(
                strategy_id=self.__class__.__name__.lower(),
                timeframe="1d",
                params={}
            )
        self.config = config
        self.strategy_id = config.strategy_id
        self.timeframe = config.timeframe
        self.logger = get_logger(f"strategy.{config.strategy_id}")
    
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
            ctx: 运行上下文
            
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
    ) -> tuple[List[OrderIntent], List[dict]]:
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
    
    def on_fill(self, fill_event, ctx: RunContext) -> dict:
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

"""
简洁策略框架
让用户只关心核心逻辑，其他都由框架处理

设计原则：
1. 策略代码<100行
2. 清晰的信号生成逻辑
3. 自动仓位管理
4. 参数配置简单
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from src.core.types import Signal, OrderIntent, PortfolioState, RiskState, ActionType, OrderSide, OrderType
from src.core.context import RunContext
from src.utils.logger import get_logger

logger = get_logger("strategy.simple")


class DataAccessor:
    """数据访问器（类似聚宽的data对象）"""
    
    def __init__(self, bars_map: Dict[str, pd.DataFrame]):
        self.bars_map = bars_map
        
    def history(self, symbol: str, count: int, field: str = "close") -> pd.Series:
        """获取历史数据"""
        if "1d" not in self.bars_map:
            return pd.Series()
        
        bars = self.bars_map["1d"]
        symbol_bars = bars[bars["symbol"] == symbol]
        if symbol_bars.empty:
            return pd.Series()
            
        return symbol_bars[field].iloc[-count:]
    
    def current(self, symbol: str, field: str = "close") -> float:
        """获取当前价格"""
        hist = self.history(symbol, 1, field)
        return float(hist.iloc[-1]) if not hist.empty else 0.0
    
    def get_all_symbols(self) -> List[str]:
        """获取所有标的"""
        if "1d" not in self.bars_map:
            return []
        return list(self.bars_map["1d"]["symbol"].unique())


class SimpleStrategy(ABC):
    """
    简洁策略基类
    
    使用方式：
    1. 继承此类
    2. 实现 generate_signals 方法
    3. 可选：重写 adjust_positions 方法
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.strategy_id = self.__class__.__name__.lower()
        self.logger = get_logger(f"strategy.{self.strategy_id}")
        
    @abstractmethod
    def generate_signals(self, data: DataAccessor, context: RunContext) -> Dict[str, float]:
        """
        生成信号（必须实现）
        
        Args:
            data: 数据访问器
            context: 运行上下文
            
        Returns:
            信号字典：{symbol: weight}
            weight范围：-1.0 到 1.0，表示做空到做多的权重
        """
        pass
    
    def adjust_positions(
        self, 
        signals: Dict[str, float], 
        portfolio: PortfolioState,
        risk: RiskState,
        context: RunContext
    ) -> List[OrderIntent]:
        """
        调整仓位（可选重写）
        
        默认实现：等权分配，权重归一化
        """
        if not signals:
            # 清空所有持仓
            orders = []
            for symbol, qty in portfolio.positions.items():
                if abs(qty) > 1e-6:
                    orders.append(self._create_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=abs(qty),
                        context=context,
                        reason="no_signal"
                    ))
            return orders
        
        # 归一化权重
        total_weight = sum(abs(w) for w in signals.values())
        if total_weight == 0:
            return []
            
        normalized = {s: w / total_weight for s, w in signals.items()}
        
        # 计算目标仓位
        orders = []
        total_equity = portfolio.equity
        
        for symbol, weight in normalized.items():
            current_qty = portfolio.positions.get(symbol, 0)
            current_price = self._get_price(symbol, data=None)  # 实际使用时需要传入data
            
            if current_price <= 0:
                continue
                
            target_value = total_equity * weight
            target_qty = target_value / current_price
            
            # 计算调整量
            qty_diff = target_qty - current_qty
            
            if abs(qty_diff) < 1e-6:
                continue
                
            side = OrderSide.BUY if qty_diff > 0 else OrderSide.SELL
            orders.append(self._create_order(
                symbol=symbol,
                side=side,
                qty=abs(qty_diff),
                context=context,
                reason="weight_adjustment"
            ))
            
        return orders
    
    def _create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        context: RunContext,
        reason: str = ""
    ) -> OrderIntent:
        """创建订单"""
        return OrderIntent(
            ts_utc=context.now_utc,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MKT,
            strategy_id=self.strategy_id,
            metadata={"reason": reason}
        )
    
    def _get_price(self, symbol: str, data: Optional[DataAccessor] = None) -> float:
        """获取价格（简化实现）"""
        return 100.0  # 实际应该从data获取
    
    # ============ 辅助方法 ============
    
    def calculate_momentum(self, prices: pd.Series, lookback: int) -> float:
        """计算动量"""
        if len(prices) < lookback:
            return 0.0
        return float(prices.iloc[-1] / prices.iloc[0] - 1)
    
    def calculate_returns(self, prices: pd.Series, lookback: int) -> float:
        """计算收益率"""
        if len(prices) < lookback:
            return 0.0
        return float(np.log(prices.iloc[-1] / prices.iloc[0]))
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """归一化权重"""
        if not weights:
            return {}
        
        total = sum(abs(w) for w in weights.values())
        if total == 0:
            return {s: 0.0 for s in weights.keys()}
            
        return {s: w / total for s, w in weights.items()}
    
    def rank_by_score(self, scores: Dict[str, float], top_n: int = 1) -> Dict[str, float]:
        """按得分排名，选择Top N"""
        if not scores:
            return {}
            
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]
        
        # 只保留正得分的
        result = {s: w for s, w in top_items if w > 0}
        
        # 归一化
        return self.normalize_weights(result)


# ============ 策略示例 ============

class ETFMomentumStrategy(SimpleStrategy):
    """
    ETF动量轮动策略（<50行代码）
    
    核心逻辑：
    1. 计算ETF的N日动量
    2. 选择动量最强的ETF
    3. 等权分配
    """
    
    def __init__(self):
        params = {
            "etf_pool": ["QQQ", "SPY", "IWM", "TLT", "GLD"],
            "lookback_days": 20,
            "target_positions": 1,
        }
        super().__init__(params)
        
    def generate_signals(self, data: DataAccessor, context: RunContext) -> Dict[str, float]:
        signals = {}
        etf_pool = self.params["etf_pool"]
        lookback = self.params["lookback_days"]
        
        for symbol in etf_pool:
            # 获取历史数据
            hist = data.history(symbol, lookback, "close")
            if len(hist) < lookback:
                continue
                
            # 计算动量
            momentum = self.calculate_momentum(hist, lookback)
            signals[symbol] = momentum
            
        # 选择Top N
        top_n = self.params["target_positions"]
        return self.rank_by_score(signals, top_n)


class DualMovingAverageStrategy(SimpleStrategy):
    """
    双均线策略（<40行代码）
    
    核心逻辑：
    1. 计算快慢均线
    2. 金叉买入，死叉卖出
    """
    
    def __init__(self):
        params = {
            "fast_period": 10,
            "slow_period": 30,
            "symbols": ["AAPL", "MSFT", "GOOGL"],
        }
        super().__init__(params)
        
    def generate_signals(self, data: DataAccessor, context: RunContext) -> Dict[str, float]:
        signals = {}
        fast = self.params["fast_period"]
        slow = self.params["slow_period"]
        
        for symbol in self.params["symbols"]:
            # 获取足够的历史数据
            hist = data.history(symbol, slow + 10, "close")
            if len(hist) < slow + 10:
                continue
                
            # 计算均线
            ma_fast = hist.rolling(fast).mean().iloc[-1]
            ma_slow = hist.rolling(slow).mean().iloc[-1]
            
            # 生成信号
            if ma_fast > ma_slow:
                signals[symbol] = 1.0  # 做多
            elif ma_fast < ma_slow:
                signals[symbol] = -1.0  # 做空
            else:
                signals[symbol] = 0.0
                
        return signals


class MeanReversionStrategy(SimpleStrategy):
    """
    均值回归策略（<60行代码）
    
    核心逻辑：
    1. 计算Z-Score
    2. 超卖买入，超买卖出
    """
    
    def __init__(self):
        params = {
            "lookback": 60,
            "entry_z": -2.0,
            "exit_z": 0.0,
            "symbols": ["SPY", "QQQ"],
        }
        super().__init__(params)
        
    def generate_signals(self, data: DataAccessor, context: RunContext) -> Dict[str, float]:
        signals = {}
        lookback = self.params["lookback"]
        entry_z = self.params["entry_z"]
        exit_z = self.params["exit_z"]
        
        for symbol in self.params["symbols"]:
            hist = data.history(symbol, lookback + 10, "close")
            if len(hist) < lookback + 10:
                continue
                
            # 计算Z-Score
            recent = hist.iloc[-lookback:]
            mean = recent.mean()
            std = recent.std()
            
            if std == 0:
                continue
                
            current_price = hist.iloc[-1]
            z_score = (current_price - mean) / std
            
            # 生成信号
            if z_score < entry_z:
                signals[symbol] = 1.0  # 超卖，买入
            elif z_score > abs(entry_z):
                signals[symbol] = -1.0  # 超买，卖出
            elif abs(z_score) < exit_z:
                signals[symbol] = 0.0  # 平仓
                
        return self.normalize_weights(signals)


# ============ 策略工厂 ============

class StrategyRegistry:
    """策略注册表"""
    
    _strategies: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        cls._strategies[name] = strategy_class
        logger.info(f"注册策略: {name}")
        
    @classmethod
    def create(cls, name: str, params: Optional[Dict] = None) -> Optional[SimpleStrategy]:
        if name not in cls._strategies:
            return None
        return cls._strategies[name](params)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        return list(cls._strategies.keys())


# 注册示例策略
StrategyRegistry.register("etf_momentum", ETFMomentumStrategy)
StrategyRegistry.register("dual_ma", DualMovingAverageStrategy)
StrategyRegistry.register("mean_reversion", MeanReversionStrategy)


# ============ 使用示例 ============

if __name__ == "__main__":
    print("可用策略:")
    for name in StrategyRegistry.list_strategies():
        print(f"  - {name}")
        
    # 创建策略
    strategy = StrategyRegistry.create("etf_momentum")
    print(f"\n策略ID: {strategy.strategy_id}")
    print(f"参数: {strategy.params}")
    
    # 创建双均线策略
    ma_strategy = DualMovingAverageStrategy()
    print(f"\n双均线策略参数: {ma_strategy.params}")

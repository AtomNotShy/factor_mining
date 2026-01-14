"""
ETF 轮动策略
基于动量和相对强度轮动的ETF配置策略

策略逻辑：
1. 选取多个ETF标的
2. 计算各标的的动量得分（n日收益率）
3. 选取动量最强的标的进行轮动
4. 定期调仓
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.strategies.base.unified_strategy import UnifiedStrategy, StrategyConfig
from src.core.types import (
    Signal,
    OrderIntent,
    PortfolioState,
    RiskState,
    MarketData,
    ActionType,
    OrderSide,
    OrderType,
)
from src.core.context import RunContext
from src.execution.order_engine import buy, sell, create_order


class ETFRotationStrategy(UnifiedStrategy):
    """
    ETF 轮动策略

    特点：
    - 基于动量效应的ETF轮动
    - 多标的等权配置
    - 自动止损保护
    """

    # ===== 默认配置 =====
    DEFAULT_ETF_POOL = [
        "QQQ",  # 纳斯达克100
        "SPY",  # 标普500
        "IWM",  # 罗素2000
        "TLT",  # 20年期国债
        "AGG",  # 投资级债券
        "GLD",  # 黄金
        "USO",  # 原油
        "VNQ",  # 房地产
    ]

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        lookback_days: int = 20,
        top_n: int = 3,
        min_momentum: float = 0.0,
    ):
        # 默认配置
        if config is None:
            config = StrategyConfig(
                strategy_id="etf_rotation",
                strategy_name="ETF轮动策略",
                timeframe="1d",
                params={
                    "etf_pool": self.DEFAULT_ETF_POOL.copy(),
                    "lookback_days": lookback_days,
                    "top_n": top_n,
                    "min_momentum": min_momentum,
                },
            )

        super().__init__(config)

        # 策略参数
        self.lookback_days = config.params.get("lookback_days", 20)
        self.top_n = config.params.get("top_n", 3)
        self.min_momentum = config.params.get("min_momentum", 0.0)
        self.etf_pool = config.params.get("etf_pool", self.DEFAULT_ETF_POOL)

    def get_parameters(self) -> Dict[str, Dict]:
        """获取策略参数"""
        return {
            "lookback_days": {
                "value": self.lookback_days,
                "min": 5,
                "max": 60,
                "step": 5,
                "description": "动量计算回看天数",
            },
            "top_n": {
                "value": self.top_n,
                "min": 1,
                "max": 5,
                "step": 1,
                "description": "持有动量最强的标的数量",
            },
            "min_momentum": {
                "value": self.min_momentum,
                "min": -0.1,
                "max": 0.1,
                "step": 0.01,
                "description": "最小动量阈值（负值表示允许持有下跌标的）",
            },
        }

    def prepare_data(
        self, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        批量计算指标

        Args:
            data: 原始价格 DataFrame
            metadata: 元数据 (symbol, etc.)

        Returns:
            带有指标的 DataFrame
        """
        if data.empty:
            return data

        symbol = metadata.get("symbol", "")

        # 计算动量得分
        lookback = min(self.lookback_days, len(data))
        if lookback > 0:
            close = data["close"]
            momentum = (close.iloc[-1] / close.iloc[-lookback]) - 1
            data["momentum"] = momentum
        else:
            data["momentum"] = 0.0

        # 计算波动率
        returns = data["close"].pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
            data["volatility"] = volatility
        else:
            data["volatility"] = 1.0

        # 风险调整后的动量
        data["risk_adjusted_momentum"] = (
            data["momentum"] / data["volatility"].replace(0, 1)
        )

        return data

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        计算技术指标（可重写）

        Args:
            dataframe: K线数据
            metadata: 元数据

        Returns:
            包含指标的DataFrame
        """
        return self.prepare_data(dataframe, metadata)

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        进场信号

        Args:
            dataframe: K线数据
            metadata: 元数据

        Returns:
            包含信号的DataFrame
        """
        # 动量大于阈值时进场
        dataframe["enter_long"] = np.where(
            (dataframe["momentum"] > self.min_momentum)
            & (dataframe["momentum"].rank(ascending=False) <= self.top_n),
            1,
            0,
        )
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        离场信号

        Args:
            dataframe: K线数据
            metadata: 元数据

        Returns:
            包含信号的DataFrame
        """
        # 动量排名下降或低于阈值时离场
        dataframe["exit_long"] = np.where(
            (dataframe["momentum"] <= self.min_momentum)
            | (dataframe["momentum"].rank(ascending=False) > self.top_n),
            1,
            0,
        )
        return dataframe

    def generate_signals(
        self, md: MarketData, ctx: RunContext
    ) -> List[Signal]:
        """
        生成交易信号

        Args:
            md: 市场数据
            ctx: 运行上下文

        Returns:
            信号列表
        """
        signals = []

        if md.bars_all is None or md.bars_all.empty:
            return signals

        # 获取所有ETF数据
        etf_data = md.bars_all[md.bars_all["symbol"].isin(self.etf_pool)]

        if etf_data.empty:
            return signals

        # 计算横截面动量排名
        latest_data = etf_data.groupby("symbol").last().reset_index()
        if latest_data.empty:
            return signals

        # 计算排名
        latest_data["momentum_rank"] = latest_data["momentum"].rank(ascending=False)

        # 按标的分组处理
        for _, row in latest_data.iterrows():
            symbol = row["symbol"]
            momentum = row.get("momentum", 0)
            rank = row.get("momentum_rank", 0)
            price = row.get("close", 0)
            volatility = row.get("volatility", 0)
            risk_adjusted = row.get("risk_adjusted_momentum", 0)

            # 进场信号
            if momentum > self.min_momentum and rank <= self.top_n:
                signals.append(
                    Signal(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        strategy_id=self.strategy_id,
                        action=ActionType.LONG,
                        strength=momentum,
                        metadata={
                            "momentum": momentum,
                            "volatility": volatility,
                            "risk_adjusted": risk_adjusted,
                            "rank": rank,
                            "price": price,
                        },
                    )
                )

            # 离场信号
            elif momentum <= self.min_momentum or rank > self.top_n:
                signals.append(
                    Signal(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        strategy_id=self.strategy_id,
                        action=ActionType.FLAT,
                        strength=0,
                        metadata={
                            "momentum": momentum,
                            "rank": rank,
                            "price": price,
                        },
                    )
                )

        return signals

    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        """
        计算仓位并生成订单

        Args:
            signals: 信号列表
            portfolio: 当前组合状态
            risk: 风险状态
            ctx: 运行上下文

        Returns:
            订单意图列表
        """
        orders = []

        if not signals:
            return orders

        # 筛选有效的 LONG 信号
        long_signals = [
            s for s in signals if s.action == ActionType.LONG
        ]

        # 按动量得分排序
        long_signals.sort(key=lambda x: x.strength, reverse=True)

        # 选取 Top N
        target_symbols = {s.symbol for s in long_signals[: self.top_n]}

        # 1. 卖出不在目标中的持仓（使用工厂函数）
        for symbol, qty in portfolio.positions.items():
            if abs(qty) > 1e-6 and symbol not in target_symbols:
                orders.append(
                    sell(
                        symbol=symbol,
                        qty=abs(qty),
                        strategy_id=self.strategy_id,
                        now_utc=ctx.now_utc,
                        reason="rotation_exit",
                    )
                )

        # 2. 买入目标标的（使用工厂函数）
        held_symbols = set(portfolio.positions.keys())
        available_cash = portfolio.cash

        # 预留缓冲资金
        buffer = available_cash * 0.02
        available_cash -= buffer

        if available_cash > 0:
            slots = len(target_symbols - held_symbols)

            if slots > 0:
                cash_per_slot = available_cash / slots

                for signal in long_signals:
                    if signal.symbol in held_symbols:
                        continue
                    if signal.symbol not in target_symbols:
                        continue

                    price = signal.metadata.get("price", 0)
                    if price <= 0:
                        continue

                    qty = cash_per_slot / price

                    # 最小交易金额检查
                    if qty * price < 100:
                        continue

                    orders.append(
                        buy(
                            symbol=signal.symbol,
                            qty=qty,
                            strategy_id=self.strategy_id,
                            now_utc=ctx.now_utc,
                            reason="rotation_entry",
                            momentum=signal.strength,
                        )
                    )

        return orders


# ===== 测试代码 =====
if __name__ == "__main__":
    # 创建策略实例
    strategy = ETFRotationStrategy(
        lookback_days=20,
        top_n=3,
    )

    print(f"策略ID: {strategy.strategy_id}")
    print(f"策略名称: {strategy.strategy_name}")
    print(f"ETF池: {strategy.etf_pool}")
    print()

    # 获取参数
    print("策略参数:")
    for name, param in strategy.get_parameters().items():
        print(f"  {name}: {param['value']} ({param['description']})")

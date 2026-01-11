"""
简单均线策略示例
- 简单移动平均线交叉
- 金叉做多
"""

from typing import List

from src.core.context import RunContext
from src.core.types import (
    ActionType, MarketData, OrderIntent,
    OrderSide, OrderType, PortfolioState, RiskState, Signal
)
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.utils.logger import get_logger


class SimpleMAStrategy(Strategy):
    """简单均线策略 - 入门示例"""

    def __init__(self, config=None):
        if config is None:
            config = StrategyConfig(
                strategy_id="simple_ma",
                timeframe="1d",
                params={
                    "symbol": "SPY",
                    "fast_ma": 10,
                    "slow_ma": 30,
                }
            )
        super().__init__(config)

    def _ma(self, prices, period):
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        symbol = self.config.params.get("symbol", "SPY")
        fast_period = int(self.config.params.get("fast_ma", 10))
        slow_period = int(self.config.params.get("slow_ma", 30))

        if md.bars is None or md.bars.empty:
            return []

        try:
            if 'symbol' in md.bars.columns:
                bars = md.bars[md.bars['symbol'] == symbol]
            else:
                bars = md.bars

            if bars.empty:
                return []

            prices = bars['close'].values
            current_price = float(prices[-1])

            fast_ma = self._ma(prices, fast_period)
            slow_ma = self._ma(prices, slow_period)

            if fast_ma is None or slow_ma is None:
                return []

            prev_fast = self._ma(prices[:-1], fast_period)
            prev_slow = self._ma(prices[:-1], slow_period)

            if prev_fast is None or prev_slow is None:
                return []

            if prev_fast <= prev_slow and fast_ma > slow_ma:
                return [Signal(
                    ts_utc=ctx.now_utc,
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    action=ActionType.LONG,
                    strength=1.0,
                    metadata={"price": current_price}
                )]

            return []

        except Exception:
            return []

    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        orders = []

        for signal in signals:
            price = float(signal.metadata.get("price", 0))
            if price <= 0:
                continue

            qty = portfolio.equity / price

            orders.append(OrderIntent(
                ts_utc=ctx.now_utc,
                symbol=signal.symbol,
                side=OrderSide.BUY,
                qty=float(qty),
                order_type=OrderType.MKT,
                strategy_id=self.strategy_id,
                metadata={"reason": "ma_crossover"}
            ))

        return orders

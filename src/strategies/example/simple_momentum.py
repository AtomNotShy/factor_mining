"""
简单动量策略示例
- 比较过去 N 天的收益率
- 买入收益率最高的标的
"""

from typing import List

from src.core.context import RunContext
from src.core.types import (
    ActionType, MarketData, OrderIntent,
    OrderSide, OrderType, PortfolioState, RiskState, Signal
)
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.utils.logger import get_logger


class SimpleMomentumStrategy(Strategy):
    """简单动量策略 - 入门示例"""

    def __init__(self, config=None):
        if config is None:
            config = StrategyConfig(
                strategy_id="simple_momentum",
                timeframe="1d",
                params={
                    "symbols": ["SPY", "QQQ", "TLT"],
                    "lookback_days": 20,
                    "target_positions": 1,
                }
            )
        super().__init__(config)

    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        symbols = self.config.params.get("symbols", [])
        lookback = int(self.config.params.get("lookback_days", 20))

        if md.bars_all is None or md.bars_all.empty:
            return []

        signals = []
        scores = []

        for symbol in symbols:
            try:
                bars = md.bars_all[md.bars_all['symbol'] == symbol]
                if len(bars) < lookback + 1:
                    continue

                prices = bars['close'].values
                current_price = float(prices[-1])
                past_price = float(prices[-lookback])

                momentum = (current_price / past_price) - 1

                scores.append({
                    "symbol": symbol,
                    "momentum": momentum,
                    "price": current_price,
                })
            except Exception:
                continue

        if scores:
            scores.sort(key=lambda x: x["momentum"], reverse=True)
            top_n = int(self.config.params.get("target_positions", 1))

            for item in scores[:top_n]:
                if item["momentum"] > 0:
                    signals.append(Signal(
                        ts_utc=ctx.now_utc,
                        symbol=item["symbol"],
                        strategy_id=self.strategy_id,
                        action=ActionType.LONG,
                        strength=item["momentum"],
                        metadata={"price": item["price"]}
                    ))

        return signals

    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        orders = []

        if not signals:
            for symbol, qty in portfolio.positions.items():
                if abs(qty) > 0:
                    orders.append(OrderIntent(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=float(abs(qty)),
                        order_type=OrderType.MKT,
                        strategy_id=self.strategy_id,
                        metadata={"reason": "no_signal"}
                    ))
            return orders

        weight = 1.0 / len(signals)
        target_symbols = {s.symbol for s in signals}

        for symbol, qty in portfolio.positions.items():
            if abs(qty) > 0 and symbol not in target_symbols:
                orders.append(OrderIntent(
                    ts_utc=ctx.now_utc,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=float(abs(qty)),
                    order_type=OrderType.MKT,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "rebalance"}
                ))

        for signal in signals:
            price = float(signal.metadata.get("price", 0))
            if price <= 0:
                continue

            target_qty = (portfolio.equity * weight) / price
            current_qty = portfolio.positions.get(signal.symbol, 0)
            delta = target_qty - current_qty

            if abs(delta) > 0.001:
                orders.append(OrderIntent(
                    ts_utc=ctx.now_utc,
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    qty=float(abs(delta)),
                    order_type=OrderType.MKT,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "signal"}
                ))

        return orders

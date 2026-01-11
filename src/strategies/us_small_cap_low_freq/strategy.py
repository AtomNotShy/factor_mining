"""
美股小市值低频量化示例策略
- 月度再平衡（默认），可选周度/日度
- 6-1 月动量 + 流动性过滤 + 长期趋势过滤
- 低频风格：非再平衡日只做风险退出
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.context import RunContext
from src.core.types import ActionType, MarketData, OrderIntent, OrderSide, OrderType, PortfolioState, RiskState, Signal
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.utils.logger import get_logger


@dataclass
class _Candidate:
    symbol: str
    score: float
    price: float
    long_return: float
    short_return: float
    volatility: float


class USSmallCapLowFreqStrategy(Strategy):
    """美股小市值低频量化策略（示例）"""

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                strategy_id="us_smallcap_lowfreq",
                timeframe="1d",
                params={
                    "small_cap_pool": [
                        "IWM",  # Russell 2000
                        "IJR",  # S&P SmallCap 600
                        "VB",   # Vanguard Small-Cap
                        "VBR",  # Vanguard Small-Cap Value
                        "SCHA", # Schwab U.S. Small-Cap
                        "IWN",  # Russell 2000 Value
                        "IWO",  # Russell 2000 Growth
                    ],
                    "benchmark_symbol": "IWM",
                    "target_positions": 3,
                    "rebalance_frequency": "monthly",  # daily/weekly/monthly
                    "risk_replace": False,  # 非再平衡日是否补位
                    "exit_rank": 5,  # 跌出前N名触发退出
                    "min_score": 0.0,
                    "long_lookback": 126,  # ~6个月
                    "short_lookback": 21,  # ~1个月
                    "vol_lookback": 20,
                    "trend_filter_days": 200,
                    "min_price": 5.0,
                    "min_dollar_volume": 5_000_000,
                    "liquidity_lookback": 20,
                },
            )

        super().__init__(config)
        self.logger = get_logger("us_smallcap_lowfreq")
        self._last_scores: Dict[str, float] = {}
        self._last_rank: Dict[str, int] = {}
        self._last_candidates: List[_Candidate] = []

    def _is_rebalance_day(self, ctx: RunContext) -> bool:
        frequency = str(self.config.params.get("rebalance_frequency", "monthly")).lower()
        if frequency == "daily":
            return True
        if frequency == "weekly":
            current_date = ctx.now_utc.date()
            week_start = current_date - timedelta(days=current_date.weekday())
            try:
                trading_days = ctx.trading_calendar.trading_days_between(week_start, current_date)
                if trading_days:
                    return current_date == trading_days[0]
            except Exception as exc:
                self.logger.debug(f"周度再平衡日期判断失败，回退到周一: {exc}")
            return current_date.weekday() == 0
        if frequency == "monthly":
            current_date = ctx.now_utc.date()
            month_start = current_date.replace(day=1)
            try:
                trading_days = ctx.trading_calendar.trading_days_between(month_start, current_date)
                if trading_days:
                    return current_date == trading_days[0]
            except Exception as exc:
                self.logger.debug(f"月度再平衡日期判断失败，回退到月初: {exc}")
            return current_date.day == 1
        return True

    def _get_symbol_history(self, md: MarketData, symbol: str) -> Optional[pd.DataFrame]:
        if md.bars is None or md.bars.empty:
            return None
        if "symbol" not in md.bars.columns:
            return md.bars.copy()
        df = md.bars[md.bars["symbol"] == symbol].copy()
        if df.empty:
            return None
        return df.sort_index()

    def _calculate_score(self, symbol: str, df: pd.DataFrame) -> Optional[_Candidate]:
        params = self.config.params
        long_lookback = int(params.get("long_lookback", 126))
        short_lookback = int(params.get("short_lookback", 21))
        vol_lookback = int(params.get("vol_lookback", 20))
        trend_filter_days = int(params.get("trend_filter_days", 200))
        min_price = float(params.get("min_price", 0.0))
        min_dollar_volume = float(params.get("min_dollar_volume", 0.0))
        liquidity_lookback = int(params.get("liquidity_lookback", 20))

        min_history = max(long_lookback, short_lookback, vol_lookback, trend_filter_days) + 1
        if len(df) < min_history:
            return None

        close = df["close"].astype(float)
        price = float(close.iloc[-1])
        if price < min_price:
            return None

        if min_dollar_volume > 0 and "volume" in df.columns:
            dollar_volume = (df["close"].astype(float) * df["volume"].astype(float)).tail(
                liquidity_lookback
            ).mean()
            if np.isnan(dollar_volume) or dollar_volume < min_dollar_volume:
                return None

        if trend_filter_days > 0:
            sma = close.rolling(trend_filter_days).mean().iloc[-1]
            if np.isnan(sma) or price < float(sma):
                return None

        long_return = price / float(close.iloc[-long_lookback]) - 1.0
        short_return = price / float(close.iloc[-short_lookback]) - 1.0

        returns = close.pct_change().dropna().tail(vol_lookback)
        volatility = float(returns.std()) if not returns.empty else 0.0

        raw_score = long_return - short_return
        score = raw_score / volatility if volatility > 0 else raw_score

        return _Candidate(
            symbol=symbol,
            score=float(score),
            price=price,
            long_return=float(long_return),
            short_return=float(short_return),
            volatility=float(volatility),
        )

    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        signals: List[Signal] = []
        pool = list(self.config.params.get("small_cap_pool", []))
        if not pool:
            return signals

        candidates: List[_Candidate] = []
        for symbol in pool:
            history = self._get_symbol_history(md, symbol)
            if history is None:
                continue
            candidate = self._calculate_score(symbol, history)
            if candidate is None:
                continue
            candidates.append(candidate)

        candidates.sort(key=lambda x: x.score, reverse=True)
        self._last_candidates = candidates
        self._last_scores = {c.symbol: c.score for c in candidates}
        self._last_rank = {c.symbol: rank + 1 for rank, c in enumerate(candidates)}

        if not candidates:
            return signals

        target_positions = int(self.config.params.get("target_positions", 1))
        rebalance_day = self._is_rebalance_day(ctx)
        risk_replace = bool(self.config.params.get("risk_replace", False))

        if not rebalance_day and not risk_replace:
            return signals

        for candidate in candidates[:target_positions]:
            signals.append(
                Signal(
                    ts_utc=ctx.now_utc,
                    symbol=candidate.symbol,
                    strategy_id=self.strategy_id,
                    action=ActionType.LONG,
                    strength=float(candidate.score),
                    metadata={
                        "current_price": candidate.price,
                        "long_return": candidate.long_return,
                        "short_return": candidate.short_return,
                        "volatility": candidate.volatility,
                        "score": candidate.score,
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
        orders: List[OrderIntent] = []

        target_positions = int(self.config.params.get("target_positions", 1))
        exit_rank = int(self.config.params.get("exit_rank", target_positions))
        min_score = float(self.config.params.get("min_score", 0.0))
        risk_replace = bool(self.config.params.get("risk_replace", False))

        exit_symbols: Dict[str, str] = {}
        for symbol, qty in portfolio.positions.items():
            if abs(qty) < 1e-8:
                continue
            score = self._last_scores.get(symbol)
            rank = self._last_rank.get(symbol)
            if score is None or rank is None:
                continue
            if score < min_score:
                exit_symbols[symbol] = "risk_score"
            elif rank > exit_rank:
                exit_symbols[symbol] = "risk_rank"

        for symbol, reason in sorted(exit_symbols.items()):
            qty = abs(portfolio.positions.get(symbol, 0.0))
            if qty <= 0:
                continue
            orders.append(
                OrderIntent(
                    ts_utc=ctx.now_utc,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=float(qty),
                    order_type=OrderType.MKT,
                    strategy_id=self.strategy_id,
                    metadata={"exit_reason": reason},
                )
            )

        remaining_symbols = {
            symbol
            for symbol, qty in portfolio.positions.items()
            if abs(qty) >= 1e-8 and symbol not in exit_symbols
        }

        rebalance_day = self._is_rebalance_day(ctx)
        if not rebalance_day:
            if not risk_replace:
                return orders

            desired_symbols = [c.symbol for c in self._last_candidates[:target_positions]]
            desired_count = min(len(desired_symbols), target_positions)
            if desired_count <= 0:
                return orders
            target_weight = 1.0 / desired_count
            candidate_map = {c.symbol: c for c in self._last_candidates}

            for symbol in desired_symbols:
                if len(remaining_symbols) >= desired_count:
                    break
                if symbol in remaining_symbols or symbol in exit_symbols:
                    continue
                candidate = candidate_map.get(symbol)
                if candidate is None or candidate.price <= 0:
                    continue
                qty = (portfolio.equity * target_weight) / candidate.price
                orders.append(
                    OrderIntent(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        side=OrderSide.BUY,
                        qty=float(qty),
                        order_type=OrderType.MKT,
                        strategy_id=self.strategy_id,
                        metadata={"entry_tag": "risk_replace"},
                    )
                )
                remaining_symbols.add(symbol)
            return orders

        if not signals:
            for symbol, qty in portfolio.positions.items():
                if abs(qty) < 1e-8 or symbol in exit_symbols:
                    continue
                orders.append(
                    OrderIntent(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=float(abs(qty)),
                        order_type=OrderType.MKT,
                        strategy_id=self.strategy_id,
                        metadata={"exit_reason": "no_signal"},
                    )
                )
            return orders

        target_symbols = {signal.symbol for signal in signals}
        target_weight = 1.0 / len(signals)

        for symbol, qty in portfolio.positions.items():
            if abs(qty) < 1e-8 or symbol in exit_symbols:
                continue
            if symbol not in target_symbols:
                orders.append(
                    OrderIntent(
                        ts_utc=ctx.now_utc,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=float(abs(qty)),
                        order_type=OrderType.MKT,
                        strategy_id=self.strategy_id,
                        metadata={"exit_reason": "rebalance_remove"},
                    )
                )

        for signal in signals:
            price = float(signal.metadata.get("current_price", 0.0) or 0.0)
            if price <= 0:
                continue
            target_value = portfolio.equity * target_weight
            current_qty = portfolio.positions.get(signal.symbol, 0.0)
            target_qty = target_value / price
            delta = target_qty - current_qty
            if abs(delta) < 1e-8:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            entry_tag = "rebalance_add" if side == OrderSide.BUY else "rebalance_trim"
            metadata = {"entry_tag": entry_tag}
            if side == OrderSide.SELL:
                metadata["exit_reason"] = "rebalance_trim"
            orders.append(
                OrderIntent(
                    ts_utc=ctx.now_utc,
                    symbol=signal.symbol,
                    side=side,
                    qty=float(abs(delta)),
                    order_type=OrderType.MKT,
                    strategy_id=self.strategy_id,
                    metadata=metadata,
                )
            )

        return orders

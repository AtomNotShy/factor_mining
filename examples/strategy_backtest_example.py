"""
策略回测示例（v2）
使用合成数据演示 BacktestEngine + Strategy 接口
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from src.core.calendar import TradingCalendar
from src.core.context import RunContext, Environment
from src.evaluation.backtesting.engine import BacktestEngine
from src.strategies.etf_momentum_us.strategy import USETFMomentumStrategy


def make_synthetic_bars(symbols: List[str], days: int = 260) -> pd.DataFrame:
    """构造合成日线数据（含 symbol 列）"""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    frames = []
    for i, symbol in enumerate(symbols):
        base_price = 100 + i * 5
        trend = np.linspace(0, 0.25, days)
        noise = np.random.normal(0, 0.01, days)
        close = base_price * (1 + trend + noise)
        high = close * (1 + 0.01)
        low = close * (1 - 0.01)
        open_ = close * (1 - 0.002)
        volume = np.random.randint(1000000, 2000000, days)

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "symbol": symbol,
            },
            index=dates,
        )
        frames.append(frame)

    return pd.concat(frames).sort_index()


async def main():
    symbols = ["SPY", "QQQ", "IWM"]
    bars = make_synthetic_bars(symbols, days=260)

    strategy = USETFMomentumStrategy()
    strategy.config.params.update(
        {
            "etf_pool": symbols,
            "auto_adjust_lookback": False,
            "default_lookback_days": 20,
            "min_score": -1e9,
            "max_score": 1e9,
        }
    )

    engine = BacktestEngine(
        initial_capital=100000.0,
        commission_rate=0.0005,
        slippage_rate=0.0002,
    )

    ctx = RunContext.create(
        env=Environment.RESEARCH,
        config=strategy.config.params,
        trading_calendar=TradingCalendar(),
    )

    start = bars.index.min().date()
    end = bars.index.max().date()

    print("=" * 60)
    print("Strategy Backtest Example (v2)")
    print("=" * 60)
    print(f"Universe: {symbols}")
    print(f"Period: {start} ~ {end}")
    print(f"Initial Capital: ${engine.initial_capital:,.2f}")
    print("=" * 60)

    result = await engine.run(
        strategies=[strategy],
        universe=symbols,
        start=start,
        end=end,
        ctx=ctx,
        bars=bars,
        auto_download=False,
    )

    if "error" in result:
        raise SystemExit(f"❌ Backtest error: {result['error']}")

    print("\n✅ Backtest completed")
    print(f"Final Equity: ${result['final_equity']:,.2f}")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Signals: {len(result['signals'])}")
    print(f"Orders: {len(result['orders'])}")
    print(f"Fills: {len(result['fills'])}")


if __name__ == "__main__":
    asyncio.run(main())

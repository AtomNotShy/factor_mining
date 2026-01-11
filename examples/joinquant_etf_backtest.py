"""
JoinQuant ETF 轮动示例（v2）

说明：原 JoinQuant v1 策略已移除，本示例使用 v2 动量轮动策略代替。
"""

import asyncio

from src.strategies.etf_momentum_us.strategy import USETFMomentumStrategy
from src.evaluation.backtesting.engine import BacktestEngine
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def make_synthetic_bars(symbols, days=300) -> pd.DataFrame:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    frames = []
    for i, symbol in enumerate(symbols):
        base_price = 90 + i * 6
        trend = np.linspace(0, 0.18, days)
        noise = np.random.normal(0, 0.012, days)
        close = base_price * (1 + trend + noise)
        high = close * (1 + 0.01)
        low = close * (1 - 0.01)
        open_ = close * (1 - 0.002)
        volume = np.random.randint(900000, 1900000, days)

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
    etf_pool = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    bars = make_synthetic_bars(etf_pool, days=360)

    strategy = USETFMomentumStrategy()
    strategy.config.params.update(
        {
            "etf_pool": etf_pool,
            "auto_adjust_lookback": False,
            "default_lookback_days": 25,
            "min_score": -1e9,
            "max_score": 1e9,
        }
    )

    engine = BacktestEngine(
        initial_capital=100000.0,
        commission_rate=0.0002,
        slippage_rate=0.0005,
    )

    ctx = RunContext.create(
        env=Environment.RESEARCH,
        config=strategy.config.params,
        trading_calendar=TradingCalendar(),
    )

    start = bars.index.min().date()
    end = bars.index.max().date()

    print("=" * 60)
    print("JoinQuant ETF 轮动示例 (v2)")
    print("=" * 60)
    print(f"ETF池: {etf_pool}")
    print(f"回测区间: {start} ~ {end}")
    print("=" * 60)

    result = await engine.run(
        strategies=[strategy],
        universe=etf_pool,
        start=start,
        end=end,
        ctx=ctx,
        bars=bars,
        auto_download=False,
    )

    if "error" in result:
        print(f"❌ 回测失败: {result['error']}")
        return

    print("\n✅ 回测完成")
    print(f"最终净值: ${result['final_equity']:,.2f}")
    print(f"总收益率: {result['total_return']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())

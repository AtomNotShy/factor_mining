"""
SPY 1分钟 VWAP 回踩策略信号扫描（v2）

说明：当前 BacktestEngine 以日频为主，未覆盖分钟级撮合。
本示例仅演示如何逐 bar 扫描策略信号。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys


# 允许直接 `python examples/xxx.py` 运行（无需手动设置 PYTHONPATH）
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.core.context import RunContext, Environment
from src.core.types import MarketData
from src.data.collectors.polygon import PolygonCollector
from src.strategies.vwap.vwap_pullback_v2 import VWAPPullbackStrategyV2, VWAPPullbackParams


async def main():
    symbol = "SPY"
    timeframe = "1m"
    days = 10

    params = VWAPPullbackParams(
        max_pullback_bps=15.0,
        stop_loss_bps=25.0,
        take_profit_bps=40.0,
        position_cash_frac=0.2,
    )

    print("=" * 60)
    print("VWAP Pullback Signal Scan (v2)")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Scan Period: {days} days")
    print("=" * 60)

    collector = PolygonCollector()
    try:
        if not await collector.connect():
            raise SystemExit("❌ Failed to connect to Polygon API. Please check POLYGON_API_KEY.")

        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        bars = await collector.get_ohlcv(symbol=symbol, timeframe=timeframe, since=start_date, limit=None)

        if bars.empty:
            raise SystemExit("❌ No data retrieved.")

        bars = bars[["open", "high", "low", "close", "volume"]].dropna().sort_index()
        bars["symbol"] = symbol

        strategy = VWAPPullbackStrategyV2(params)
        ctx = RunContext.create(env=Environment.RESEARCH, config=params.__dict__)
        md = MarketData(bars=bars.iloc[:1], bars_all=bars, features=None)

        signal_times = []
        for i in range(1, len(bars)):
            window = bars.iloc[: i + 1]
            md.bars = window
            ctx.now_utc = window.index[-1].to_pydatetime()
            signals = strategy.generate_signals(md, ctx)
            if signals:
                signal_times.append(window.index[-1])

        print(f"\n✅ 扫描完成: {len(bars)} bars")
        print(f"✅ 触发信号次数: {len(signal_times)}")
        if signal_times:
            print("   最近 5 次信号:")
            for ts in signal_times[-5:]:
                print(f"   - {ts}")

    finally:
        await collector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

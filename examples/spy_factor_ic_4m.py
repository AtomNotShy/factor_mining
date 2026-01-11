"""
分析 SPY 过去 4 个月的因子 IC 表现（本地脚本）

说明：
- 数据来自 Polygon/Massive（会本地缓存到 data/polygon/ohlcv/...）
- 默认用日线（1d），更适合因子 IC 评估；如需分钟线可改 timeframe="1m"
"""

from __future__ import annotations

import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.data.collectors.polygon import PolygonCollector
from src.evaluation.metrics.ic_analysis import ICAnalyzer
from src.factors.base.factor import factor_registry
import src.factors  # noqa: F401 触发因子自动注册（technical/momentum/volatility/reversal）


async def main():
    symbol = "SPY"
    timeframe = "1d"
    days = 120  # 近4个月约 120 天
    periods = [1, 5, 10]

    collector = PolygonCollector()
    try:
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        price_data = await collector.get_ohlcv(symbol=symbol, timeframe=timeframe, since=start_date, limit=None)
        if price_data.empty:
            raise SystemExit("未获取到价格数据：请检查 POLYGON_API_KEY / POLYGON_BASE_URL / 网络。")

        price_data = price_data[["open", "high", "low", "close", "volume"]].dropna().sort_index()

        ic = ICAnalyzer()

        results = []
        for name in factor_registry.list_factors(category="technical"):
            factor = factor_registry.get_factor(name)
            if not factor:
                continue

            fv = factor.calculate(price_data)
            if fv.empty or fv.isna().all():
                continue

            stats = ic.calculate_ic_stats(fv, price_data["close"].pct_change(), periods=periods)
            p1 = stats.get("period_1", {})
            results.append(
                {
                    "factor": name,
                    "ic_1": p1.get("ic"),
                    "ic_ir_1": p1.get("ic_ir"),
                }
            )

        df = pd.DataFrame(results).dropna()
        if df.empty:
            raise SystemExit("没有得到可用的因子 IC 结果（可能数据不足或因子计算失败）。")

        df = df.sort_values(by="ic_ir_1", ascending=False)
        print(f"SPY {timeframe} 近{days}天 | 因子数: {len(df)}")
        print(df.head(20).to_string(index=False))

    finally:
        await collector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())


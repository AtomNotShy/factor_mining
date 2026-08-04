#!/usr/bin/env python3
"""
下载 XLU ETF 的 1分钟数据

用法:
    python3 scripts/download_xlu.py

数据将保存到 data/alpaca/ohlcv/iex/utc/1m/XLU.parquet
"""

import asyncio
from datetime import datetime, timedelta, timezone

from src.data.collectors.alpaca import AlpacaCollector


async def main():
    symbol = "XLU"
    timeframe = "1m"
    days = 740  # 最近30天

    collector = AlpacaCollector()

    # 计算起始时间（从现在起倒数 30 天）
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    print(f"下载 {symbol} {timeframe} 数据")
    print(f"时间范围: {start_time} ~ {end_time}")
    print(f"预计需要请求约 {days * 390} 条 1m bar（美股每天约 390 分钟）")
    print("-" * 50)

    try:
        df = await collector.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_time,
        )

        if df.empty:
            print("未获取到数据")
            return

        print(f"成功下载 {len(df)} 条记录")
        print(f"数据时间范围: {df.index.min()} ~ {df.index.max()}")
        print(f"\n数据预览:")
        print(df.head())
        print(f"\n数据已缓存到: data/alpaca/ohlcv/iex/utc/1m/XLU.parquet")

    except Exception as e:
        print(f"下载失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

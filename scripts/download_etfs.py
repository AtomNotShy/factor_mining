#!/usr/bin/env python3
"""
批量下载多个ETF的1分钟数据

用法:
    python3 scripts/download_etfs.py

支持的ETF:
    IWM (罗素2000), USO (原油), VNQ (房地产), XLF (金融)
    XLE (能源), XLK (科技), XLV (医疗), XLI (工业)
    XLP (消费), XLY (可选消费), XLU (公用事业)
"""

import asyncio
from datetime import datetime, timedelta, timezone

from src.data.collectors.alpaca import AlpacaCollector

ETF_LIST = ["DIA", "EEM", "VWO", "VEA", "USO", "BND"]


def print_progress_bar(
    current: int, total: int, prefix: str = "", bar_length: int = 40
):
    """打印进度条"""
    percent = float(current) / total if total > 0 else 1.0
    filled = int(bar_length * percent)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(
        f"\r{prefix} |{bar}| {current}/{total} ({percent * 100:.1f}%)",
        end="",
        flush=True,
    )


async def download_etf(collector: AlpacaCollector, symbol: str, days: int = 30) -> dict:
    """下载单个ETF的数据"""
    timeframe = "1m"

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    print(f"  正在下载 {symbol}...", end=" ", flush=True)

    try:
        df = await collector.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_time,
        )

        if df.empty:
            print("❌ 无数据")
            return {"symbol": symbol, "count": 0, "status": "no_data"}

        print(f"✓ {len(df)} 条")
        return {
            "symbol": symbol,
            "count": len(df),
            "status": "success",
            "date_range": f"{df.index.min()} ~ {df.index.max()}",
        }

    except Exception as e:
        print(f"❌ 错误: {e}")
        return {"symbol": symbol, "count": 0, "status": "error", "error": str(e)}


async def main():
    days = 700
    total = len(ETF_LIST)

    print(f"批量下载 ETF 1分钟数据（最近 {days} 天）")
    print("=" * 50)

    collector = AlpacaCollector()
    results = []

    try:
        for i, etf in enumerate(ETF_LIST, 1):
            print_progress_bar(i, total, prefix="进度")
            result = await download_etf(collector, etf, days)
            results.append(result)
    finally:
        # 优雅地关闭连接
        await collector.disconnect()
        print("\n✓ 连接已关闭")

    # 清除进度条
    print("\r" + " " * 60 + "\r")

    print("=" * 50)
    print("\n下载汇总:")
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"  成功: {success_count}/{len(ETF_LIST)}")

    # 打印详细表格
    print("\n  {" + "-" * 8 + "}{" + "-" * 10 + "}{" + "-" * 25 + "}")
    print("  {:^6} | {:^8} | {:^23}".format("ETF", "记录数", "时间范围"))
    print("  {" + "-" * 8 + "}{" + "-" * 10 + "}{" + "-" * 25 + "}")

    for r in results:
        if r["status"] == "success":
            date_short = (
                r["date_range"].split(" ~ ")[0][:10]
                + " ~ "
                + r["date_range"].split(" ~ ")[1][:10]
            )
            print(f"  {r['symbol']:^6} | {r['count']:^8,} | {date_short:^23}")
        elif r["status"] == "no_data":
            print(f"  {r['symbol']:^6} | {'无数据':^8} | {'-':^23}")
        else:
            print(f"  {r['symbol']:^6} | {'错误':^8} | {'-':^23}")

    print("  {" + "-" * 8 + "}{" + "-" * 10 + "}{" + "-" * 25 + "}")


if __name__ == "__main__":
    asyncio.run(main())

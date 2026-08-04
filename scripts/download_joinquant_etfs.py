#!/usr/bin/env python3
"""
下载聚宽ETF轮动策略需要的15个ETF日线数据

用法:
    python3 scripts/download_joinquant_etfs.py

时间范围: 2019-01-01 ~ 最新（确保2020-01-01回测有260天warmup）
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.data.collectors.alpaca import AlpacaCollector
from src.data.storage.parquet_store import ParquetDataFrameStore

# 聚宽策略的15个ETF
JOINQUANT_ETFS = [
    "SPY",  # S&P 500 ETF
    "QQQ",  # Nasdaq 100 ETF
    "IWM",  # Russell 2000 ETF
    "TLT",  # 20+ Year Treasury Bond ETF
    "GLD",  # Gold ETF
    "USO",  # Oil ETF
    "VNQ",  # Real Estate ETF
    "XLF",  # Financial Sector ETF
    "XLE",  # Energy Sector ETF
    "XLK",  # Technology Sector ETF
    "XLV",  # Healthcare Sector ETF
    "XLI",  # Industrial Sector ETF
    "XLP",  # Consumer Staples Sector ETF
    "XLY",  # Consumer Discretionary Sector ETF
    "XLU",  # Utilities Sector ETF
]


async def download_etf_daily(
    collector: AlpacaCollector,
    store: ParquetDataFrameStore,
    symbol: str,
    start_date: str = "2019-01-01"
) -> dict:
    """
    下载单个ETF的日线数据

    Args:
        collector: Alpaca数据收集器
        store: Parquet存储
        symbol: ETF代码
        start_date: 开始日期

    Returns:
        下载结果字典
    """
    timeframe = "1d"

    # 解析开始日期
    start_time = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_time = datetime.now(timezone.utc)

    print(f"\n📥 {symbol:<5} - 下载中...", end=" ", flush=True)

    try:
        # 1. 检查现有数据
        existing_path = Path(f"data/bars/{symbol}/1d/data.parquet")
        if existing_path.exists():
            existing_df = store.read(f"bars/{symbol}/1d/data.parquet")
            if not existing_df.empty:
                existing_start = existing_df.index.min()
                existing_end = existing_df.index.max()
                existing_count = len(existing_df)

                # 统一时区（转换为tz-naive以便比较）
                if hasattr(existing_start, 'tz') and existing_start.tz is not None:
                    existing_start = existing_start.tz_localize(None)
                if hasattr(existing_end, 'tz') and existing_end.tz is not None:
                    existing_end = existing_end.tz_localize(None)

                start_time_naive = start_time.replace(tzinfo=None)
                now_naive = datetime.now(timezone.utc).replace(tzinfo=None)

                # 如果现有数据已经足够
                if existing_start <= start_time_naive and existing_end >= now_naive - timedelta(days=7):
                    print(f"✅ 已存在 ({existing_count} bars, {existing_start.date()} ~ {existing_end.date()})")
                    return {
                        "symbol": symbol,
                        "count": existing_count,
                        "status": "skipped",
                        "date_range": f"{existing_start.date()} ~ {existing_end.date()}",
                    }

        # 2. 下载新数据
        df = await collector.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_time,
        )

        if df.empty:
            print("❌ 无数据")
            return {"symbol": symbol, "count": 0, "status": "no_data"}

        # 3. 保存数据
        store.write_dataset(
            dataset="bars",
            df=df,
            partition={"symbol": symbol, "timeframe": timeframe},
        )

        date_start = df.index.min()
        date_end = df.index.max()

        print(f"✅ {len(df)} bars ({date_start.date()} ~ {date_end.date()})")

        return {
            "symbol": symbol,
            "count": len(df),
            "status": "success",
            "date_range": f"{date_start.date()} ~ {date_end.date()}",
        }

    except Exception as e:
        print(f"❌ 错误: {e}")
        return {"symbol": symbol, "count": 0, "status": "error", "error": str(e)}


async def main():
    """主函数"""
    start_date = "2019-01-01"  # 确保2020-01-01回测有260天warmup

    print("=" * 70)
    print("下载聚宽ETF轮动策略的15个ETF日线数据")
    print("=" * 70)
    print(f"时间范围: {start_date} ~ 最新")
    print(f"ETF数量: {len(JOINQUANT_ETFS)}")
    print("=" * 70)

    # 创建收集器和存储
    collector = AlpacaCollector()
    store = ParquetDataFrameStore("data")

    results = []

    try:
        # 顺序下载每个ETF
        for i, symbol in enumerate(JOINQUANT_ETFS, 1):
            print(f"[{i}/{len(JOINQUANT_ETFS)}]", end=" ")
            result = await download_etf_daily(collector, store, symbol, start_date)
            results.append(result)

            # 避免请求过快
            if i < len(JOINQUANT_ETFS):
                await asyncio.sleep(0.5)

    finally:
        await collector.disconnect()
        print("\n\n✅ 连接已关闭")

    # 汇总报告
    print("\n" + "=" * 70)
    print("下载汇总")
    print("=" * 70)

    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] in ("no_data", "error")]

    print(f"✅ 成功下载: {len(success)}")
    print(f"⏭️  已存在: {len(skipped)}")
    print(f"❌ 失败: {len(failed)}")
    print(f"📊 总计: {len(results)}/{len(JOINQUANT_ETFS)}")

    # 详细列表
    if success:
        print(f"\n✅ 成功下载的ETF ({len(success)}):")
        for r in success:
            print(f"   {r['symbol']:<5} - {r['count']:4d} bars ({r['date_range']})")

    if skipped:
        print(f"\n⏭️  已存在的ETF ({len(skipped)}):")
        for r in skipped:
            print(f"   {r['symbol']:<5} - {r['count']:4d} bars ({r['date_range']})")

    if failed:
        print(f"\n❌ 失败的ETF ({len(failed)}):")
        for r in failed:
            if "error" in r:
                print(f"   {r['symbol']:<5} - {r['error']}")
            else:
                print(f"   {r['symbol']:<5} - 无数据")

    print("\n" + "=" * 70)

    # 最终检查
    if len(success) + len(skipped) == len(JOINQUANT_ETFS):
        print("🎉 所有ETF数据已就绪！可以开始回测。")
    else:
        print(f"⚠️  警告：{len(failed)} 个ETF数据缺失，回测结果可能不完整。")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

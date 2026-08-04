#!/usr/bin/env python3
"""
新版数据管理器使用示例

展示如何使用统一的 DataSourceManager 和 BarPathManager
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.storage.path_manager import get_path_manager, DataSource
from src.data.manager_v2 import get_data_source_manager


async def example_usage():
    """示例：使用新版数据管理器"""
    print("=" * 60)
    print("示例：使用新版数据管理器")
    print("=" * 60)

    # 获取管理器
    manager = get_data_source_manager()
    path_manager = get_path_manager()

    # 1. 路径管理演示
    print("\n1. 路径管理演示")
    print("-" * 40)

    unified = path_manager.unified_bar_path("SPY", "1d")
    print(f"统一数据路径: {unified.relative_path}")
    print(f"  - source: {unified.source.value}")
    print(f"  - symbol: {unified.symbol}")
    print(f"  - timeframe: {unified.timeframe}")

    polygon_path = path_manager.source_bar_path(
        "SPY", "1d", source=DataSource.POLYGON, is_adjusted=True
    )
    print(f"\nPolygon 路径: {polygon_path.relative_path}")
    print(f"  - is_adjusted: {polygon_path.is_adjusted}")

    ib_path = path_manager.source_bar_path(
        "SPY", "1d", source=DataSource.IB
    )
    print(f"\nIB 路径: {ib_path.relative_path}")

    # 2. 获取可用数据源
    print("\n2. 检查可用数据源")
    print("-" * 40)

    enabled_sources = manager.get_enabled_sources()
    for name, info in enabled_sources:
        print(f"  - {name}: priority={info.priority}, enabled={info.enabled}")

    # 3. 路径结构说明
    print("\n3. 统一路径结构")
    print("-" * 40)
    print("""
新结构:
    {data_dir}/
        bars/                          # OHLCV 主数据目录
            {symbol}/
                {timeframe}/
                    data.parquet        # 合并后的数据
                    metadata.json       # 元数据
        _sources/                      # 原始数据（可选保留）
            ib/{symbol}/{timeframe}/data.parquet
            polygon/{adjusted|raw}/{symbol}/{timeframe}/data.parquet
            alpaca/{feed}/{symbol}/{timeframe}/data.parquet

旧结构（将被迁移）:
    data/ib/ohlcv/{timeframe}/{symbol}.parquet
    data/polygon/ohlcv/{adjusted|raw}/utc/{timeframe}/{symbol}.parquet
    data/alpaca/ohlcv/{feed}/utc/{timeframe}/{symbol}.parquet
    data/daily/{symbol}_{timeframe}.parquet
    """)

    # 4. 迁移功能演示
    print("\n4. 迁移功能")
    print("-" * 40)
    print("从旧版路径迁移到新版统一路径:")
    print("  python scripts/migrate_data_structure.py --analyze  # 分析")
    print("  python scripts/migrate_data_structure.py --dry-run  # 模拟")
    print("  python scripts/migrate_data_structure.py           # 执行")
    print("  python scripts/migrate_data_structure.py --delete-legacy  # 清理旧数据")

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


def demonstrate_path_manager():
    """演示路径管理器"""
    print("=" * 60)
    print("路径管理器演示")
    print("=" * 60)

    path_manager = get_path_manager()

    # 测试各种路径生成
    symbols = ["SPY", "QQQ", "AAPL", "BRK.B"]
    timeframes = ["1d", "1h", "1m"]

    for symbol in symbols:
        for tf in timeframes:
            unified = path_manager.unified_bar_path(symbol, tf)
            full_path = path_manager.unified_bar_full_path(symbol, tf)
            metadata_path = path_manager.unified_metadata_path(symbol, tf)

            print(f"\n{symbol} {tf}:")
            print(f"  相对路径: {unified.relative_path}")
            print(f"  完整路径: {full_path}")
            print(f"  元数据: {metadata_path}")


def demonstrate_metadata():
    """演示元数据"""
    from src.data.storage.path_manager import BarMetadata
    from datetime import datetime

    print("\n" + "=" * 60)
    print("元数据演示")
    print("=" * 60)

    # 创建元数据
    metadata = BarMetadata(
        symbol="SPY",
        timeframe="1d",
        source="polygon_api",
        source_config={"adjusted": True},
        data_version="v1_20240115",
        ingest_ts_utc=datetime.utcnow().isoformat(),
        row_count=1000,
        date_range=("2024-01-01", "2024-01-15"),
        columns=["open", "high", "low", "close", "volume"],
        checksum="abc123",
    )

    print(f"元数据:")
    for key, value in metadata.to_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # 演示路径管理器
    demonstrate_path_manager()

    # 演示元数据
    demonstrate_metadata()

    # 运行异步示例
    print("\n运行异步示例...")
    asyncio.run(example_usage())

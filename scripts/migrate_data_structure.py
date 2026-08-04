#!/usr/bin/env python3
"""
数据迁移工具
将旧版数据迁移到新版统一路径结构

使用示例：
    # 仅分析当前结构
    python scripts/migrate_data_structure.py --analyze

    # 模拟迁移（不实际执行）
    python scripts/migrate_data_structure.py --dry-run

    # 执行迁移
    python scripts/migrate_data_structure.py

    # 迁移后删除旧数据
    python scripts/migrate_data_structure.py --delete-legacy

    # 迁移指定 symbol
    python scripts/migrate_data_structure.py --symbol SPY --timeframe 1d
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.storage.parquet_store import ParquetDataFrameStore
from src.data.storage.path_manager import (
    BarPathManager,
    DataSource,
    get_path_manager,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("migration")


class DataMigrationTool:
    """数据迁移工具"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.store = ParquetDataFrameStore(str(self.data_dir))
        self.path_manager = get_path_manager(data_dir)

        # 旧版路径模式（symbol 模板）
        self.legacy_patterns = {
            DataSource.IB: [
                "ib/ohlcv/{timeframe}/{symbol}.parquet",
            ],
            DataSource.POLYGON: [
                "polygon/ohlcv/adjusted/utc/{timeframe}/{symbol}.parquet",
                "polygon/ohlcv/raw/utc/{timeframe}/{symbol}.parquet",
            ],
            DataSource.ALPACA: [
                "alpaca/ohlcv/{feed}/utc/{timeframe}/{symbol}.parquet",
            ],
        }

        # 已知 symbol 列表
        self.known_symbols = [
            "SPY",
            "QQQ",
            "IWM",
            "TLT",
            "GLD",
            "XLF",
            "XLK",
            "XLE",
            "XLV",
            "XLI",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK.B",
            "JPM",
            "V",
        ]

        self.timeframes = ["1d", "1h", "1m"]

    def _safe_symbol(self, symbol: str) -> str:
        """安全化的符号名称"""
        return symbol.upper().replace("/", "_").replace(":", "_")

    def check_legacy_paths(
        self, symbol: str, timeframe: str
    ) -> List[tuple]:
        """
        检查旧版路径是否存在

        Returns:
            List of tuples: (source, path_str, full_path)
        """
        existing = []
        safe_symbol = self._safe_symbol(symbol)

        for source, patterns in self.legacy_patterns.items():
            for pattern in patterns:
                path_str = pattern.format(
                    symbol=safe_symbol,
                    timeframe=timeframe,
                    feed="iex",
                    adjusted="adjusted",
                )
                full_path = self.data_dir / path_str
                if full_path.exists():
                    existing.append((source, path_str, full_path))
                    logger.info(f"  找到旧数据: {path_str}")

        return existing

    def migrate_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
        delete_legacy: bool = False,
    ) -> int:
        """
        迁移单个 symbol/timeframe 的数据

        Returns:
            迁移的数据条数
        """
        logger.info(f"迁移 {symbol} {timeframe}...")

        # 检查旧数据
        legacy_files = self.check_legacy_paths(symbol, timeframe)
        if not legacy_files:
            logger.info(f"  无旧数据需要迁移")
            return 0

        # 读取第一个可用数据源
        df = None
        source_name = None
        for source, path_str, full_path in legacy_files:
            try:
                temp_df = self.store.read(path_str)
                if not temp_df.empty:
                    df = temp_df
                    source_name = source.value
                    logger.info(f"  从 {source.value} 读取数据: {len(df)} 条")
                    break
            except Exception as e:
                logger.warning(f"  读取失败 {path_str}: {e}")
                continue

        if df is None:
            logger.warning(f"  无法读取任何数据源")
            return 0

        # 获取统一路径
        unified_path = self.path_manager.unified_bar_path(symbol, timeframe)
        unified_full = self.store.path(unified_path.relative_path)

        # 如果统一数据已存在，合并
        if unified_full.exists():
            existing_df = self.store.read(unified_path.relative_path)
            if not existing_df.empty:
                logger.info(f"  合并现有统一数据: {len(existing_df)} 条")
                # 合并（保留新的）
                merged = pd.concat([existing_df, df]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                df = merged
                logger.info(f"  合并后: {len(df)} 条")

        # 保存到统一路径
        unified_full.parent.mkdir(parents=True, exist_ok=True)
        self.store.write(unified_path.relative_path, df)
        logger.info(f"  已保存到统一路径: {unified_path.relative_path}")

        # 可选：删除旧数据
        if delete_legacy:
            for source, path_str, full_path in legacy_files:
                full_path.unlink()
                logger.info(f"  已删除旧数据: {path_str}")

        return len(df)

    def migrate_all(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        delete_legacy: bool = False,
        dry_run: bool = True,
    ) -> dict:
        """
        迁移所有数据

        Returns:
            Migration results dict
        """
        if symbols is None:
            symbols = self.known_symbols
        if timeframes is None:
            timeframes = self.timeframes

        results = {
            "total_symbols": len(symbols),
            "total_timeframes": len(timeframes),
            "migrated": 0,
            "failed": 0,
            "skipped": 0,
            "details": [],
        }

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    count = self.migrate_symbol_timeframe(
                        symbol, timeframe, delete_legacy
                    )
                    if count > 0:
                        results["migrated"] += 1
                        results["details"].append(
                            {"symbol": symbol, "timeframe": timeframe, "count": count}
                        )
                    else:
                        results["skipped"] += 1
                except Exception as e:
                    logger.error(f"迁移失败 {symbol} {timeframe}: {e}")
                    results["failed"] += 1

        logger.info(f"\n迁移完成:")
        logger.info(f"  成功: {results['migrated']}")
        logger.info(f"  跳过: {results['skipped']}")
        logger.info(f"  失败: {results['failed']}")

        return results

    def analyze_current_structure(self) -> dict:
        """分析当前数据目录结构"""
        analysis = {
            "total_files": 0,
            "by_source": {},
            "by_timeframe": {},
            "symbols_found": set(),
        }

        # 统计各来源文件
        for source, patterns in self.legacy_patterns.items():
            source_files = list((self.data_dir / source.value).rglob("*.parquet"))
            analysis["by_source"][source.value] = len(source_files)
            analysis["total_files"] += len(source_files)

            for f in source_files:
                # 尝试解析 timeframe
                parts = f.parts
                if "ohlcv" in parts:
                    idx = parts.index("ohlcv")
                    if idx + 1 < len(parts):
                        tf = parts[idx + 1]
                        if tf in self.timeframes:
                            analysis["by_timeframe"][tf] = analysis["by_timeframe"].get(
                                tf, 0
                            ) + 1

        # 统计新结构文件
        bars_dir = self.data_dir / "bars"
        if bars_dir.exists():
            for symbol_dir in bars_dir.iterdir():
                if symbol_dir.is_dir():
                    analysis["symbols_found"].add(symbol_dir.name)
                    for tf_dir in symbol_dir.iterdir():
                        if tf_dir.is_dir():
                            analysis["by_timeframe"][tf_dir.name] = (
                                analysis["by_timeframe"].get(tf_dir.name, 0) + 1
                            )

        analysis["symbols_count"] = len(analysis["symbols_found"])
        analysis["symbols_found"] = list(analysis["symbols_found"])[:20]  # 只返回前20个

        return analysis


def main():
    parser = argparse.ArgumentParser(
        description="数据迁移工具 - 将旧版数据迁移到新版统一路径结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅分析当前结构
  python scripts/migrate_data_structure.py --analyze

  # 模拟迁移（不实际执行）
  python scripts/migrate_data_structure.py --dry-run

  # 执行迁移
  python scripts/migrate_data_structure.py

  # 迁移后删除旧数据
  python scripts/migrate_data_structure.py --delete-legacy

  # 迁移指定 symbol
  python scripts/migrate_data_structure.py --symbol SPY --timeframe 1d

  # 从文件读取 symbol 列表
  python scripts/migrate_data_structure.py --symbols-file symbols.txt
        """,
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="数据目录 (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅模拟，不实际执行",
    )
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="迁移后删除旧数据",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="仅分析当前结构",
    )
    parser.add_argument(
        "--symbol",
        help="指定迁移的 symbol",
    )
    parser.add_argument(
        "--timeframe",
        help="指定迁移的 timeframe",
    )
    parser.add_argument(
        "--symbols-file",
        help="从文件读取 symbol 列表（每行一个）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tool = DataMigrationTool(args.data_dir)

    if args.analyze:
        print("\n" + "=" * 50)
        print("=== 当前数据目录结构分析 ===")
        print("=" * 50)
        analysis = tool.analyze_current_structure()

        print(f"\n总文件数: {analysis['total_files']}")
        print(f"\n按数据源:")
        for source, count in analysis.get("by_source", {}).items():
            print(f"  - {source}: {count} 个文件")

        print(f"\n按时间周期:")
        for tf, count in analysis.get("by_timeframe", {}).items():
            print(f"  - {tf}: {count} 个文件")

        print(f"\n已迁移的 symbols: {analysis['symbols_count']}")
        if analysis["symbols_found"]:
            print(f"  示例: {', '.join(analysis['symbols_found'][:10])}")

        return

    # 解析 symbol 列表
    symbols = None
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols_file:
        symbols_file = Path(args.symbols_file)
        if symbols_file.exists():
            with open(symbols_file) as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
        else:
            logger.error(f"Symbol 文件不存在: {args.symbols_file}")
            return

    # 解析 timeframe
    timeframes = None
    if args.timeframe:
        timeframes = [args.timeframe]

    # 执行迁移
    if args.dry_run:
        print("\n" + "=" * 50)
        print("=== DRY RUN - 模拟迁移 ===")
        print("=" * 50)
        results = tool.migrate_all(
            symbols=symbols,
            timeframes=timeframes,
            delete_legacy=False,
            dry_run=True,
        )
        print(f"\n将会迁移 {results['migrated']} 个文件")
        print(f"将会跳过 {results['skipped']} 个文件")

        if results.get("details"):
            print(f"\n迁移详情:")
            for item in results["details"][:10]:
                print(f"  - {item['symbol']} {item['timeframe']}: {item['count']} 条")
    else:
        print("\n" + "=" * 50)
        print("=== 开始迁移 ===")
        print("=" * 50)
        results = tool.migrate_all(
            symbols=symbols,
            timeframes=timeframes,
            delete_legacy=args.delete_legacy,
            dry_run=False,
        )

        # 保存结果
        results_file = PROJECT_ROOT / "migration_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到 {results_file}")


if __name__ == "__main__":
    main()

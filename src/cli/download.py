"""
CLI 数据下载模块
"""

import argparse
import asyncio

from src.data.downloader import Downloader


async def run_download(args: argparse.Namespace) -> None:
    """运行数据下载任务"""
    symbols = args.symbols.split(",")
    downloader = Downloader()
    await downloader.download_symbols(
        symbols=symbols,
        days=args.days,
        timeframe=args.timeframe,
        provider=args.provider
    )

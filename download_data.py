"""
通用数据下载脚本
支持单个或批量下载股票/ETF历史数据

用法:
    # 下载单个标的（默认365天）
    python download_data.py SPY

    # 下载单个标的，指定天数
    python download_data.py SPY --days 180

    # 下载单个标的，指定日期范围
    python download_data.py SPY --start 2024-01-01 --end 2024-12-31

    # 批量下载多个标的
    python download_data.py SPY QQQ IWM --days 365

    # 从文件读取标的列表
    python download_data.py --file tickers.txt

    # 使用IB下载（默认）
    python download_data.py SPY --ib

    # 设置IB连接参数
    python download_data.py SPY --host 127.0.0.1 --port 7497 --client-id 1
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.collectors.ib_history import IBHistoryCollector


def parse_date(date_str: str) -> datetime:
    """解析日期字符串"""
    date_str = date_str.strip()
    if len(date_str) == 10:  # YYYY-MM-DD
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif len(date_str) == 19:  # YYYY-MM-DD HH:MM:SS
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    else:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))


def load_tickers_from_file(filepath: str) -> List[str]:
    """从文件加载标的列表"""
    tickers = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if line and not line.startswith('#'):
                # 分割逗号或空格分隔的标的
                for ticker in line.replace(',', ' ').split():
                    ticker = ticker.strip().upper()
                    if ticker:
                        tickers.append(ticker)
    return tickers


async def download_single_symbol(
    collector: IBHistoryCollector,
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    days: Optional[int] = None,
) -> bool:
    """下载单个标的的数据"""
    now = datetime.now(timezone.utc)
    
    # 计算日期范围
    if days is not None:
        start_dt = now - timedelta(days=days)
        end_dt = now
    elif start_date and end_date:
        start_dt = start_date
        end_dt = end_date
    elif start_date:
        start_dt = start_date
        end_dt = now
    else:
        start_dt = now - timedelta(days=365)
        end_dt = now
    
    print(f"  正在下载 {symbol} {timeframe}...")
    print(f"    日期范围: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
    
    try:
        df = await collector.get_ohlcv_async(
            symbol=symbol,
            timeframe=timeframe,
            since=start_dt,
            end=end_dt,
            use_cache=False,  # 强制重新下载
        )
        
        if df is not None and len(df) > 0:
            print(f"    ✅ 成功: {len(df)} 条数据")
            print(f"       数据范围: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
            return True
        else:
            print(f"    ❌ 无数据")
            return False
            
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return False


async def download_data(
    tickers: List[str],
    timeframe: str = "1d",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    days: Optional[int] = None,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
) -> dict:
    """批量下载数据"""
    print("=" * 60)
    print("数据下载工具")
    print("=" * 60)
    print(f"标的数量: {len(tickers)}")
    print(f"时间周期: {timeframe}")
    print(f"IB连接: {host}:{port} (client_id={client_id})")
    print()
    
    # 创建IB数据收集器
    collector = IBHistoryCollector(
        host=host,
        port=port,
        client_id=client_id,
    )
    
    try:
        # 连接到IB
        print("正在连接到IB Gateway/TWS...")
        ok = await collector.connect()
        if not ok:
            print("❌ 连接IB失败，请确保IB Gateway/TWS已启动")
            return {"success": 0, "failed": 0, "errors": ["连接IB失败"]}
        
        print("✅ 连接IB成功")
        print()
        
        # 批量下载
        success_count = 0
        fail_count = 0
        errors = []
        
        for symbol in tickers:
            print(f"[{tickers.index(symbol) + 1}/{len(tickers)}]", end=" ")
            
            result = await download_single_symbol(
                collector=collector,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                days=days,
            )
            
            if result:
                success_count += 1
            else:
                fail_count += 1
                errors.append(symbol)
            
            print()
        
        print("=" * 60)
        print(f"下载完成: 成功 {success_count}, 失败 {fail_count}")
        print("=" * 60)
        
        return {
            "success": success_count,
            "failed": fail_count,
            "errors": errors,
        }
        
    finally:
        await collector.disconnect()
        print("\n已断开IB连接")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载股票/ETF历史数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 下载单个标的，默认365天
    python download_data.py SPY

    # 下载单个标的，指定180天
    python download_data.py SPY --days 180

    # 下载单个标的，指定日期范围
    python download_data.py SPY --start 2024-01-01 --end 2024-12-31

    # 批量下载多个标的
    python download_data.py SPY QQQ IWM VTI

    # 从文件读取标的列表
    python download_data.py --file tickers.txt

    # 下载分钟级数据
    python download_data.py SPY --timeframe 1h --days 30
        """,
    )
    
    # 位置参数：标的列表（可选）
    parser.add_argument(
        "tickers",
        nargs="*",
        help="要下载的标的代码（可选，可通过--file指定）",
    )
    
    # 文件参数
    parser.add_argument(
        "-f", "--file",
        help="包含标的列表的文件路径（每行一个或逗号分隔）",
    )
    
    # 时间参数
    parser.add_argument(
        "-d", "--days",
        type=int,
        help="下载天数（默认365天）",
    )
    parser.add_argument(
        "-s", "--start",
        help="开始日期（YYYY-MM-DD格式）",
    )
    parser.add_argument(
        "-e", "--end",
        help="结束日期（YYYY-MM-DD格式，默认今天）",
    )
    
    # 数据参数
    parser.add_argument(
        "-t", "--timeframe",
        default="1d",
        choices=["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"],
        help="时间周期（默认1d）",
    )
    
    # IB连接参数
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IB Gateway/TWS主机地址（默认127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="IB Gateway/TWS端口（默认7497）",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=1,
        help="IB客户端ID（默认1）",
    )
    
    # 其他参数
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出",
    )
    
    args = parser.parse_args()
    
    # 收集标的列表
    tickers = []
    
    # 从命令行参数
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    
    # 从文件加载
    if args.file:
        file_tickers = load_tickers_from_file(args.file)
        tickers.extend(file_tickers)
    
    if not tickers:
        parser.error("请提供标的代码（作为位置参数）或使用--file指定文件")
    
    # 去重
    tickers = list(dict.fromkeys(tickers))
    
    # 解析日期
    start_date = parse_date(args.start) if args.start else None
    end_date = parse_date(args.end) if args.end else None
    
    # 运行下载
    result = asyncio.run(download_data(
        tickers=tickers,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        days=args.days,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    ))
    
    # 返回适当的退出码
    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()

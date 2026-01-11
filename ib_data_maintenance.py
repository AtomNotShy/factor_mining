"""
IB 数据同步管家 (Data Maintenance)
支持增量更新 tickers.txt 中的所有标的，适合作为每日定时任务运行。
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.collectors.ib_history import IBHistoryCollector
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("ib_maintenance")

def load_tickers():
    """从 tickers.txt 加载标的"""
    ticker_file = project_root / "tickers.txt"
    if not ticker_file.exists():
        logger.warning("tickers.txt 不存在，将使用默认标的 SPY")
        return ["SPY"]
    
    with open(ticker_file, "r") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

async def run_maintenance():
    settings = get_settings()
    tickers = load_tickers()
    
    logger.info(f"开始增量同步 {len(tickers)} 个标的...")
    
    # 使用专门的 collector_client_id，避免冲突
    collector = IBHistoryCollector(
        host=settings.ib.host,
        port=settings.ib.port,
        client_id=settings.ib.collector_client_id
    )
    
    try:
        ok = await collector.connect()
        if not ok:
            logger.error("无法连接到 IB Gateway/TWS，请检查服务是否启动。")
            return

        for symbol in tickers:
            try:
                logger.info(f"同步 [{symbol}] 中...")
                # 增量更新：自动识别本地缓存日期并从该日期开始下载
                df = await collector.get_ohlcv_incremental_async(symbol, timeframe="1d")
                if not df.empty:
                    logger.info(f"✅ [{symbol}] 同步成功，当前总计 {len(df)} 条记录，最后日期: {df.index.max()}")
                else:
                    logger.warning(f"⚠️ [{symbol}] 未能获取到新数据")
            except Exception as e:
                logger.error(f"❌ [{symbol}] 同步失败: {e}")
                
    finally:
        await collector.disconnect()
        logger.info("同步任务结束。")

if __name__ == "__main__":
    asyncio.run(run_maintenance())

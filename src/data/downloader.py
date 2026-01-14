"""
数据下载服务
CLI 调用的批量数据下载逻辑
"""

import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
from src.data.manager import data_manager
from src.utils.logger import get_logger

logger = get_logger("downloader")

class Downloader:
    """批量数据下载器"""
    
    async def download_symbols(
        self,
        symbols: List[str],
        days: int = 365,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1d",
        provider: str = "auto"
    ):
        """
        批量下载指定标的数据并保存到本地
        
        Args:
            symbols: 标的列表
            days: 追溯天数 (如果未提供 start_date)
            start_date: 开始日期
            end_date: 结束日期 (默认现时)
            timeframe: 时间周期
            provider: 数据源 (auto/ib/polygon)
        """
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=days)
            
        logger.info(f"🚀 开始批量下载数据: {symbols}")
        logger.info(f"   范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   周期: {timeframe}, 数据源: {provider}")

        # 尝试复用 IB 连接（如果涉及 IB）
        ib_opened = False
        if provider in ("ib", "auto"):
            try:
                # 检查 IB 采集器是否能连接
                ib_opened = await data_manager.open_ib_session()
                if ib_opened:
                    logger.info("✅ IB Session 已建立，正在复用连接进行批量下载")
            except Exception as e:
                logger.warning(f"无法建立 IB Session: {e}。将尝试按需连接或切换其他源。")

        success_count = 0
        try:
            for symbol in symbols:
                symbol = symbol.strip().upper()
                if not symbol:
                    continue
                    
                try:
                    df = await data_manager.get_ohlcv(
                        symbol=symbol,
                        start=start_date,
                        end=end_date,
                        timeframe=timeframe,
                        auto_download=True,
                        source_preference=provider,
                        keep_connection=ib_opened
                    )
                    if not df.empty:
                        logger.info(f"✅ {symbol} 处理完成, 总计 {len(df)} 条记录")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ {symbol} 处理结果为空")
                except Exception as e:
                    logger.error(f"❌ {symbol} 下载过程中出错: {e}")
        finally:
            if ib_opened:
                await data_manager.close_ib_session()
                logger.info("🔌 IB Session 已关闭")
                
        logger.info(f"🎉 批量下载任务结束。成功: {success_count}/{len(symbols)}")
        return success_count

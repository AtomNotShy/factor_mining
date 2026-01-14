"""
数据加载器
负责从存储层读取、对齐数据，并处理预热周期
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from src.data.manager import data_manager
from src.utils.logger import get_logger

class HistoryLoader:
    """
    历史数据加载器
    """
    
    def __init__(self, datadir: str, startup_candle_count: int = 200):
        self.datadir = datadir
        self.startup_candle_count = startup_candle_count
        self.logger = get_logger("history_loader")

    async def load_data(
        self,
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        加载并对齐数据
        """
        self.logger.info(f"开始加载数据: {universe}, {start_date} -> {end_date}, TF={timeframe}")
        
        # 计算预热开始时间 (粗略估计，根据 timeframe 调整)
        if timeframe == "1d":
            warmup_days = self.startup_candle_count * 1.5 # 考虑非交易日
            warmup_start = start_date - timedelta(days=warmup_days)
        elif timeframe.endswith("m"):
            mins = int(timeframe[:-1])
            warmup_mins = self.startup_candle_count * mins * 2
            warmup_start = start_date - timedelta(minutes=warmup_mins)
        else:
            warmup_start = start_date - timedelta(days=self.startup_candle_count)

        bars_list = []
        for symbol in universe:
            # 使用 data_manager 获取数据
            bars = await data_manager.get_ohlcv(
                symbol=symbol,
                start=warmup_start,
                end=end_date,
                timeframe=timeframe,
                auto_download=True
            )
            
            if bars is not None and not bars.empty:
                if 'symbol' not in bars.columns:
                    bars['symbol'] = symbol
                bars_list.append(bars)
        
        if not bars_list:
            self.logger.warning("未加载到任何数据")
            return pd.DataFrame()
            
        full_df = pd.concat(bars_list).sort_index()
        
        # 确保时区一致
        if full_df.index.tz is None:
            full_df.index = full_df.index.tz_localize("UTC")
            
        return full_df

    def slice_data(self, df: pd.DataFrame, start_date: datetime) -> pd.DataFrame:
        """
        将预热后的数据切片回用户要求的起始时间
        注意：通常在回测循环开始前不需要切片，因为策略需要历史数据计算指标
        这个方法主要用于评估阶段对齐结果
        """
        if df.empty:
            return df
        return df[df.index >= pd.Timestamp(start_date).tz_localize("UTC")]

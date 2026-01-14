"""
统一数据管理与自动补全模块
实现 "本地优先，缺失自动下载，下载后自动缓存" 的逻辑

支持的数据源优先级：
1. Interactive Brokers - 需要运行 IB TWS/Gateway
2. Polygon.io (美股/ETF) - 需要配置 POLYGON_API_KEY
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple, Any
from pathlib import Path

from src.data.storage.local_bars import load_local_ohlcv
from src.data.collectors.ib_history import IBHistoryCollector
from src.data.collectors.polygon import PolygonCollector
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("data_manager")

# 时间周期到 timedelta 的映射
_TIMEFRAME_DURATION = {
    "1d": timedelta(days=1),
    "1h": timedelta(hours=1),
    "30m": timedelta(minutes=30),
    "15m": timedelta(minutes=15),
    "5m": timedelta(minutes=5),
    "1m": timedelta(minutes=1),
}


class UnifiedDataManager:
    """统一数据管理器"""

    def __init__(self):
        self.settings = get_settings()
        self.data_dir = Path(self.settings.storage.data_dir)
        self._ib_collector: Optional[IBHistoryCollector] = None
        self._polygon_collector: Optional[PolygonCollector] = None

    def _get_polygon_collector(self) -> Optional[PolygonCollector]:
        """获取 Polygon 采集器（如果已配置）"""
        if self._polygon_collector is None:
            if self.settings.polygon.api_key:
                self._polygon_collector = PolygonCollector()
                logger.info("PolygonCollector 已初始化 (POLYGON_API_KEY 已配置)")
            else:
                logger.debug("Polygon API Key 未配置，跳过 Polygon 数据源")
        return self._polygon_collector

    def _get_ib_collector(self) -> IBHistoryCollector:
        """延迟初始化 IB 采集器"""
        if self._ib_collector is None:
            self._ib_collector = IBHistoryCollector(
                host=self.settings.ib.host,
                port=self.settings.ib.port,
                client_id=self.settings.ib.collector_client_id
            )
        return self._ib_collector

    async def open_ib_session(self) -> bool:
        """打开IB连接（用于批量下载复用）"""
        ib_collector = self._get_ib_collector()
        return await ib_collector.connect()

    async def close_ib_session(self) -> None:
        """关闭IB连接"""
        if self._ib_collector is not None:
            await self._ib_collector.disconnect()

    async def _estimate_and_fetch_polygon_data(
        self,
        collector: PolygonCollector,
        symbol: str,
        timeframe: str,
        download_start: datetime,
        download_end: datetime,
    ) -> Optional[pd.DataFrame]:
        """估算并获取 Polygon 数据
        
        改进点：
        1. 提取魔法数字为常量
        2. 简化 NaT 检查逻辑
        3. 添加边界情况处理
        4. 提高代码可读性
        """
        # 常量定义
        DEFAULT_ESTIMATED_BARS = 500
        POLYGON_MAX_LIMIT = 50000
        BUFFER_BARS = 100  # 额外缓冲区
        
        try:
            # 获取时间周期对应的时长
            duration = _TIMEFRAME_DURATION.get(timeframe, timedelta(days=1))
            
            # 安全转换为 Timestamp
            start_ts = pd.Timestamp(download_start)
            end_ts = pd.Timestamp(download_end)
            
            # 检查是否为有效时间戳
            if pd.isna(start_ts) or pd.isna(end_ts):  # type: ignore[comparison-overlap]
                logger.warning(f"无效的时间戳: start={start_ts}, end={end_ts}, 使用默认值")
                estimated_bars = DEFAULT_ESTIMATED_BARS
            else:
                # 计算时间差
                time_diff = end_ts - start_ts
                
                if pd.isna(time_diff) or time_diff <= pd.Timedelta(0):  # type: ignore[comparison-overlap]
                    # 无效或负的时间差
                    estimated_bars = DEFAULT_ESTIMATED_BARS
                else:
                    # 计算预估的K线数量
                    try:
                        # 将 duration 转换为 pd.Timedelta 以兼容除法操作
                        duration_td = pd.Timedelta(duration)
                        estimated_bars = int(time_diff.total_seconds() / duration_td.total_seconds()) + BUFFER_BARS
                    except (ZeroDivisionError, TypeError, ValueError):
                        # 处理除零错误或类型错误
                        estimated_bars = DEFAULT_ESTIMATED_BARS
            
            # 应用限制
            estimated_bars = min(estimated_bars, POLYGON_MAX_LIMIT)
            estimated_bars = max(estimated_bars, 1)  # 确保至少为1
            
            logger.debug(
                f"Polygon 数据估算: {symbol} {timeframe} "
                f"从 {download_start} 到 {download_end}, "
                f"预估 {estimated_bars} 条K线"
            )
            
            # 获取数据
            return await collector.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=download_start,
                limit=estimated_bars,
            )
            
        except Exception as e:
            logger.error(f"估算或获取 Polygon 数据失败: {e}")
            return None

    def _get_best_collector(
        self,
        symbol: str,
        timeframe: str = "1d",
        source_preference: str = "auto",
    ) -> Tuple[Optional[object], str]:
        """获取最佳可用的数据采集器"""
        if source_preference == "ib":
            return self._get_ib_collector(), "ib"
        if source_preference == "polygon":
            polygon = self._get_polygon_collector()
            if polygon is not None:
                return polygon, "polygon"
            return self._get_ib_collector(), "ib"
        polygon = self._get_polygon_collector()
        if polygon is not None:
            return polygon, "polygon"
        return self._get_ib_collector(), "ib"

    async def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
        auto_download: bool = True,
        expected_dates: Optional[List[date]] = None,
        source_preference: str = "auto",
        keep_connection: bool = False,
    ) -> pd.DataFrame:
        """获取 OHLCV 数据，支持自动下载补全

        expected_dates: 交易日列表（用于检测区间内缺口并触发补齐）
        source_preference: 数据源优先级（auto/ib/polygon）
        keep_connection: 是否复用IB连接（适用于批量下载）
        """
        df_full, source = load_local_ohlcv(
            data_dir=self.data_dir,
            symbol=symbol,
            timeframe=timeframe,
            start=datetime(1970, 1, 1),
            end=datetime(2100, 1, 1)
        )

        is_missing = df_full.empty
        is_incomplete = False
        missing_start = False
        missing_end = False
        full_start: Optional[pd.Timestamp] = None
        full_end: Optional[pd.Timestamp] = None
        missing_dates: List[date] = []

        if not df_full.empty:
            idx_min = df_full.index.min()
            idx_max = df_full.index.max()
            
            # 安全获取 Timestamp，处理可能的 NaT
            if isinstance(idx_min, pd.Timestamp):
                full_start = idx_min
            elif idx_min is not None:
                full_start = pd.Timestamp(idx_min)  # type: ignore[arg-type]
            else:
                full_start = None
                
            if isinstance(idx_max, pd.Timestamp):
                full_end = idx_max
            elif idx_max is not None:
                full_end = pd.Timestamp(idx_max)  # type: ignore[arg-type]
            else:
                full_end = None
            
            check_start_ts = pd.Timestamp(start)
            check_end_ts = pd.Timestamp(end)

            if expected_dates:
                normalized_expected: List[date] = []
                for d in expected_dates:
                    if isinstance(d, datetime):
                        normalized_expected.append(d.date())
                    elif isinstance(d, pd.Timestamp):
                        normalized_expected.append(d.date())
                    elif isinstance(d, date):
                        normalized_expected.append(d)
                    else:
                        try:
                            ts = pd.Timestamp(d)
                            if ts is not pd.NaT:
                                normalized_expected.append(ts.date())
                        except Exception:
                            continue
                if normalized_expected:
                    expected_end = max(normalized_expected)
                    check_end_ts = pd.Timestamp(expected_end)

            # 检查缓存是否覆盖请求范围
            if full_start is not None and full_end is not None:
                if full_end < check_end_ts - pd.Timedelta(days=1):  # type: ignore[operator]
                    is_incomplete = True
                    missing_end = True
                    logger.debug(f"{symbol}: 本地数据结束于 {full_end}，但需要到 {check_end_ts}")

                if full_start > check_start_ts + pd.Timedelta(days=1):  # type: ignore[operator]
                    is_incomplete = True
                    missing_start = True
                    logger.debug(f"{symbol}: 本地数据始于 {full_start}，但需要从 {check_start_ts} 开始")

        # 关键：只有当缓存不完整时才检查 expected_dates
        # 如果缓存已覆盖完整范围，则不需要重新下载（缺失的日期是节假日）
        if expected_dates and not df_full.empty and (missing_start or missing_end):
            normalized_expected: List[date] = []
            for d in expected_dates:
                if isinstance(d, datetime):
                    normalized_expected.append(d.date())
                elif isinstance(d, pd.Timestamp):
                    normalized_expected.append(d.date())
                elif isinstance(d, date):
                    normalized_expected.append(d)
                else:
                    try:
                        ts = pd.Timestamp(d)
                        if ts is not pd.NaT:
                            normalized_expected.append(ts.date())
                    except Exception:
                        continue
            
            if normalized_expected:
                expected_set = sorted(set(normalized_expected))
                # 安全获取日期集合
                index_dates = pd.to_datetime(df_full.index).normalize()  # type: ignore[union-attr]
                available_dates = set(d.date() for d in index_dates)  # type: ignore[union-attr]
                
                check_start_date = start.date() if hasattr(start, 'date') else pd.Timestamp(start).date()
                check_end_date = end.date() if hasattr(end, 'date') else pd.Timestamp(end).date()
                
                missing_dates_in_range = [
                    d for d in expected_set 
                    if d >= check_start_date and d <= check_end_date and d not in available_dates
                ]
                
                if missing_dates_in_range:
                    missing_dates = missing_dates_in_range
                    is_incomplete = True
                    logger.debug(f"{symbol}: 缺失 {len(missing_dates)} 个请求范围内的交易日数据")

        if (is_missing or is_incomplete) and auto_download:
            logger.info(f"数据缺失或不全 [{symbol}], 启动自动下载...")
            new_df = None
            collector, source_name = self._get_best_collector(
                symbol, timeframe, source_preference=source_preference
            )
            logger.info(f"使用数据源: {source_name} [{symbol}]")

            try:
                download_start = start
                download_end = end
                if missing_dates:
                    download_start = datetime.combine(missing_dates[0], datetime.min.time())
                    download_end = datetime.combine(missing_dates[-1], datetime.max.time())
                elif missing_start and missing_end:
                    download_start = start
                    download_end = end
                elif missing_start and full_start is not None:
                    download_start = start
                    download_end = datetime.combine(full_start.date(), datetime.max.time())
                elif missing_end and full_end is not None:
                    download_start = datetime.combine(full_end.date(), datetime.min.time())
                    download_end = end

                if source_name == "polygon" and isinstance(collector, PolygonCollector):
                    # Polygon API 需要 limit 参数
                    new_df = await self._estimate_and_fetch_polygon_data(
                        collector=collector,
                        symbol=symbol,
                        timeframe=timeframe,
                        download_start=download_start,
                        download_end=download_end
                    )
                else:
                    # IB 采集器
                    ib_collector = self._get_ib_collector()
                    await ib_collector.connect()

                    if is_missing:
                        logger.info(f"本地数据为空，执行全量下载 [{symbol}] 从 {start} 到 {end}...")
                        new_df = await ib_collector.get_ohlcv_async(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=start,
                            end=end,
                            use_cache=False
                        )
                    elif missing_dates or missing_start:
                        logger.info(
                            f"检测到缺失数据，使用IB补齐 [{symbol}] 从 {download_start} 到 {download_end}..."
                        )
                        new_df = await ib_collector.get_ohlcv_async(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=download_start,
                            end=download_end,
                            use_cache=False,
                        )
                    else:
                        logger.info(f"本地数据存在但不全，执行增量更新 [{symbol}]...")
                        new_df = await ib_collector.get_ohlcv_incremental_async(symbol, timeframe)

                    if not keep_connection:
                        await ib_collector.disconnect()

                if new_df is not None and not new_df.empty:
                    df_full, _ = load_local_ohlcv(
                        data_dir=self.data_dir,
                        symbol=symbol,
                        timeframe=timeframe,
                        start=datetime(1970, 1, 1),
                        end=datetime(2100, 1, 1)
                    )
                    logger.info(f"✅ {symbol} 数据自动下载并本地保存成功 ({source_name})")
                else:
                    logger.warning(f"下载失败或返回空数据: {symbol}")

            except Exception as e:
                logger.error(f"❌ 自动下载 {symbol} 失败 ({source_name}): {e}")

        if not df_full.empty:
            df_final = df_full.loc[(df_full.index >= start) & (df_full.index <= end)]
            if not df_final.empty:
                logger.debug(f"返回 {symbol} 数据: {len(df_final)} 条")
                return df_final

        logger.warning(f"未找到 {symbol} 的任何数据")
        return pd.DataFrame()


# 单例
data_manager = UnifiedDataManager()

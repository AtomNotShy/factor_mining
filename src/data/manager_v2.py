"""
统一数据源管理器
实现 "本地优先，智能选择，自动合并" 的数据获取逻辑

功能：
- 智能数据获取（自动选择最佳数据源）
- 多源数据合并
- 统一路径存储
- 数据血缘追踪
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd

from src.config.settings import get_settings
from src.data.collectors.ib_history import IBHistoryCollector
from src.data.collectors.polygon import PolygonCollector
from src.data.collectors.alpaca import AlpacaCollector
from src.data.storage.parquet_store import ParquetDataFrameStore
from src.data.storage.path_manager import (
    BarPathManager,
    BarMetadata,
    DataSource,
    BarPath,
    get_path_manager,
)
from src.data.storage.versioning import generate_data_version


logger = logging.getLogger("data_source_manager")


@dataclass
class SourceInfo:
    """数据源信息"""

    name: str
    enabled: bool
    priority: int
    config: dict = field(default_factory=dict)


class DataSourceManager:
    """
    统一数据源管理器
    """

    # 默认数据源配置
    DEFAULT_SOURCES = [
        SourceInfo("polygon", True, 1, {"adjusted": True}),
        SourceInfo("ib", True, 2, {}),
        SourceInfo("alpaca", True, 3, {"feed": "iex"}),
    ]

    def __init__(self, data_dir: Optional[str] = None):
        settings = get_settings()
        self.data_dir = data_dir or settings.storage.data_dir
        self.store = ParquetDataFrameStore(self.data_dir)
        self.path_manager = get_path_manager(self.data_dir)

        # 初始化数据源
        self._sources: Dict[str, SourceInfo] = {}
        self._collectors: Dict[str, object] = {}
        self._init_sources()

    def _init_sources(self):
        """初始化数据源配置"""
        for source in self.DEFAULT_SOURCES:
            self._sources[source.name] = source

        # 检查 Polygon 配置
        if get_settings().polygon.api_key:
            self._sources["polygon"].enabled = True
        else:
            self._sources["polygon"].enabled = False
            logger.debug("Polygon API Key 未配置，已禁用")

        # 延迟初始化 collectors
        self._collectors = {}

    def _get_collector(self, source: str):
        """获取指定源的 collector（延迟初始化）"""
        if source not in self._collectors:
            if source == "polygon" and self._sources.get("polygon", {}).enabled:
                self._collectors["polygon"] = PolygonCollector()
            elif source == "ib":
                ib_settings = get_settings().ib
                self._collectors["ib"] = IBHistoryCollector(
                    host=ib_settings.host,
                    port=ib_settings.port,
                    client_id=ib_settings.collector_client_id,
                )
            elif source == "alpaca" and self._sources.get("alpaca", {}).enabled:
                self._collectors["alpaca"] = AlpacaCollector()

        return self._collectors.get(source)

    def get_enabled_sources(self) -> List[Tuple[str, SourceInfo]]:
        """获取已启用的数据源（按优先级排序）"""
        enabled = [
            (name, info) for name, info in self._sources.items() if info.enabled
        ]
        return sorted(enabled, key=lambda x: x[1].priority)

    def _calculate_checksum(self, df: pd.DataFrame) -> str:
        """计算数据校验和"""
        if df.empty:
            return ""
        # 使用列名和最后几行的哈希作为校验
        content = f"{list(df.columns)}{len(df)}{df.index[-3:]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _create_metadata(
        self,
        df: pd.DataFrame,
        source: str,
        source_config: dict,
    ) -> BarMetadata:
        """创建元数据"""
        data_version = generate_data_version(
            source=f"{source}_api",
            timeframe=df.attrs.get("timeframe", "unknown"),
            symbols=[df.attrs.get("symbol", "unknown")],
        )

        return BarMetadata(
            symbol=df.attrs.get("symbol", "unknown"),
            timeframe=df.attrs.get("timeframe", "unknown"),
            source=source,
            source_config=source_config,
            data_version=data_version,
            ingest_ts_utc=datetime.utcnow().isoformat(),
            row_count=len(df),
            date_range=(str(df.index.min()), str(df.index.max())),
            columns=list(df.columns),
            checksum=self._calculate_checksum(df),
        )

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        source_preference: str = "auto",
        merge_strategy: str = "first_available",
    ) -> Tuple[pd.DataFrame, Optional[BarMetadata]]:
        """
        获取 OHLCV 数据（统一接口）

        Args:
            symbol: 标的代码
            timeframe: 时间周期
            start: 起始时间
            end: 结束时间
            source_preference: 数据源偏好 (auto, polygon, ib, alpaca)
            merge_strategy: 合并策略
                - first_available: 使用第一个可用的源
                - merge_all: 合并所有源的最新数据
                - prefer_quality: 优先使用高质量数据源

        Returns:
            (DataFrame, 元数据)
        """
        # 1. 尝试从统一路径读取
        unified_path = self.path_manager.unified_bar_path(symbol, timeframe)
        unified_full_path = self.path_manager.unified_bar_full_path(symbol, timeframe)

        if unified_full_path.exists():
            df = self.store.read(unified_path.relative_path)
            metadata = BarMetadata.from_json(
                unified_full_path.parent / "metadata.json"
            )
            if not df.empty:
                # 裁剪到请求范围
                if start:
                    df = df[df.index >= pd.Timestamp(start)]
                if end:
                    df = df[df.index <= pd.Timestamp(end)]
                return df, metadata

        # 2. 根据偏好获取数据
        if source_preference == "auto":
            sources_to_try = self.get_enabled_sources()
        else:
            source_info = self._sources.get(source_preference)
            if source_info and source_info.enabled:
                sources_to_try = [(source_preference, source_info)]
            else:
                sources_to_try = self.get_enabled_sources()

        # 3. 获取数据
        df = pd.DataFrame()
        metadata: Optional[BarMetadata] = None
        sources_used: List[str] = []

        for source_name, source_info in sources_to_try:
            collector = self._get_collector(source_name)
            if collector is None:
                continue

            try:
                if source_name == "polygon":
                    source_df = await collector.get_ohlcv(
                        symbol, timeframe, since=start, limit=50000
                    )
                elif source_name == "ib":
                    source_df = await collector.get_ohlcv_async(
                        symbol, timeframe, since=start, end=end
                    )
                elif source_name == "alpaca":
                    source_df = await collector.get_ohlcv(
                        symbol, timeframe, since=start
                    )
                else:
                    continue

                if source_df.empty:
                    continue

                sources_used.append(source_name)

                # 格式化数据
                if "data_version" not in source_df.columns:
                    source_df = self._format_bars_schema(source_df, symbol, timeframe)

                if df.empty:
                    df = source_df
                    metadata = self._create_metadata(
                        df, source_name, source_info.config
                    )

                    if merge_strategy == "first_available":
                        break
                else:
                    # 合并数据
                    df = self._merge_data(df, source_df, merge_strategy)

            except Exception as e:
                logger.warning(f"从 {source_name} 获取数据失败: {e}")
                continue

        # 4. 保存到统一路径
        if not df.empty:
            self._save_to_unified(df, symbol, timeframe, metadata)

        return df, metadata

    def _format_bars_schema(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """格式化数据为统一 schema"""
        if df.empty:
            return df

        bars = pd.DataFrame(index=df.index.copy())
        bars["symbol"] = symbol
        bars["timeframe"] = timeframe
        bars["open"] = df["open"].values
        bars["high"] = df["high"].values
        bars["low"] = df["low"].values
        bars["close"] = df["close"].values
        bars["volume"] = df.get("volume", pd.Series(0.0, index=df.index)).values
        bars["source"] = "unified"
        bars["data_version"] = generate_data_version(
            source="unified", timeframe=timeframe, symbols=[symbol]
        )
        bars["ingest_ts_utc"] = datetime.utcnow()
        bars.index.name = "ts_utc"

        return bars

    def _merge_data(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        strategy: str = "prefer_new",
    ) -> pd.DataFrame:
        """合并两个 DataFrame"""
        if df1.empty:
            return df2
        if df2.empty:
            return df1

        # 合并并去重
        merged = pd.concat([df1, df2]).sort_index()

        if strategy == "prefer_new":
            merged = merged[~merged.index.duplicated(keep="last")]
        elif strategy == "prefer_quality":
            merged = merged[~merged.index.duplicated(keep="last")]
        else:
            merged = merged[~merged.index.duplicated(keep="first")]

        return merged

    def _save_to_unified(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        metadata: Optional[BarMetadata],
    ):
        """保存到统一路径"""
        unified_path = self.path_manager.unified_bar_path(symbol, timeframe)
        metadata_path_str = self.path_manager.unified_metadata_path(symbol, timeframe)

        # 保存数据
        self.store.write(unified_path.relative_path, df)

        # 保存元数据
        if metadata is not None:
            metadata_full_path = self.store.path(metadata_path_str)
            metadata.to_json(metadata_full_path)

        logger.info(f"已保存统一数据: {unified_path.relative_path}")

    def get_data_sources(
        self,
        symbol: str,
        timeframe: str,
    ) -> List[BarPath]:
        """获取指定 symbol/timeframe 的所有可用数据源"""
        sources: List[BarPath] = []

        # 检查统一路径
        unified_path = self.path_manager.unified_bar_path(symbol, timeframe)
        if self.store.path(unified_path.relative_path).exists():
            sources.append(unified_path)

        # 检查各来源路径
        for source in [DataSource.IB, DataSource.POLYGON, DataSource.ALPACA]:
            source_path = self.path_manager.source_bar_path(
                symbol, timeframe, source
            )
            if self.store.path(source_path.relative_path).exists():
                sources.append(source_path)

        return sources

    def migrate_from_legacy(self, symbol: str, timeframe: str, source: str):
        """从旧版路径迁移数据到新版路径"""
        # 获取旧版路径
        if source == "ib":
            legacy_path = f"ib/ohlcv/{timeframe}/{symbol}.parquet"
        elif source == "polygon":
            legacy_path = f"polygon/ohlcv/adjusted/utc/{timeframe}/{symbol}.parquet"
        elif source == "alpaca":
            legacy_path = f"alpaca/ohlcv/iex/utc/{timeframe}/{symbol}.parquet"
        else:
            raise ValueError(f"Unknown source: {source}")

        # 读取旧数据
        df = self.store.read(legacy_path)
        if df.empty:
            logger.warning(f"旧数据不存在: {legacy_path}")
            return

        # 格式化并保存
        df = self._format_bars_schema(df, symbol, timeframe)
        self._save_to_unified(df, symbol, timeframe, None)

        logger.info(f"已迁移 {source}: {symbol} {timeframe}")

    def cleanup_duplicate_sources(self, symbol: str, timeframe: str):
        """
        清理重复的来源数据（保留统一路径）
        仅保留统一路径，删除各来源的原始数据
        """
        unified_path = self.path_manager.unified_bar_path(symbol, timeframe)
        unified_full = self.store.path(unified_path.relative_path)

        if not unified_full.exists():
            logger.warning(f"统一数据不存在，无法清理: {symbol} {timeframe}")
            return

        # 删除各来源数据
        for source in [DataSource.IB, DataSource.POLYGON, DataSource.ALPACA]:
            source_path = self.path_manager.source_bar_path(
                symbol, timeframe, source
            )
            source_full = self.store.path(source_path.relative_path)
            if source_full.exists():
                source_full.unlink()
                logger.info(f"已删除来源数据: {source_path.relative_path}")


# 单例
_data_source_manager: Optional[DataSourceManager] = None


def get_data_source_manager(data_dir: Optional[str] = None) -> DataSourceManager:
    """获取 DataSourceManager 单例"""
    global _data_source_manager
    if _data_source_manager is None:
        _data_source_manager = DataSourceManager(data_dir)
    return _data_source_manager

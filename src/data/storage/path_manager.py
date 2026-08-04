"""
统一数据路径管理器
提供标准化的路径结构和访问接口

标准路径结构：
{data_dir}/
    bars/                          # OHLCV 主数据目录（合并后的数据）
        {symbol}/
            {timeframe}/
                data.parquet        # 合并后的数据
                metadata.json       # 数据源、血缘信息
        _sources/                   # 原始数据（按来源保留）
            {source}/
                {symbol}/
                    {timeframe}/
                        data.parquet
                        metadata.json
    corporate_actions/             # 分红、拆股等
    fundamentals/                  # 基本面数据
    cache/                         # 临时缓存
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class DataSource(str, Enum):
    """支持的数据源"""
    IB = "ib"
    POLYGON = "polygon"
    ALPACA = "alpaca"
    CCXT = "ccxt"
    UNIFIED = "unified"  # 合并后的统一数据


@dataclass(frozen=True)
class BarPath:
    """标准化 bar 数据路径"""
    relative_path: str  # 相对于 data_dir 的路径
    source: DataSource
    symbol: str
    timeframe: str
    is_adjusted: Optional[bool] = None  # Polygon 复权
    feed: Optional[str] = None  # Alpaca feed


class BarPathManager:
    """
    统一路径管理器
    """

    # 时间框架映射
    TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"}

    # 数据源优先级（用于自动选择）
    SOURCE_PRIORITY = [DataSource.POLYGON, DataSource.IB, DataSource.ALPACA]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def _safe_symbol(self, symbol: str) -> str:
        """安全化的符号名称（文件名友好）"""
        return symbol.upper().replace("/", "_").replace(":", "_")

    # ==================== 统一路径 ====================

    def unified_bar_path(self, symbol: str, timeframe: str) -> BarPath:
        """
        获取统一路径（合并后的数据）

        返回: bars/{symbol}/{timeframe}/data.parquet
        """
        safe_symbol = self._safe_symbol(symbol)
        relative = f"bars/{safe_symbol}/{timeframe}/data.parquet"
        return BarPath(
            relative_path=relative,
            source=DataSource.UNIFIED,
            symbol=safe_symbol,
            timeframe=timeframe,
        )

    def unified_metadata_path(self, symbol: str, timeframe: str) -> str:
        """获取统一数据的元数据路径"""
        safe_symbol = self._safe_symbol(symbol)
        return f"bars/{safe_symbol}/{timeframe}/metadata.json"

    def unified_bar_full_path(self, symbol: str, timeframe: str) -> Path:
        """获取统一数据的完整路径"""
        return self.data_dir / self.unified_bar_path(symbol, timeframe).relative_path

    # ==================== 来源路径（原始数据） ====================

    def source_bar_path(
        self,
        symbol: str,
        timeframe: str,
        source: DataSource,
        is_adjusted: Optional[bool] = None,
        feed: Optional[str] = None,
    ) -> BarPath:
        """
        获取指定来源的路径

        IB:       sources/ib/{symbol}/{timeframe}/data.parquet
        Polygon:  sources/polygon/{adjusted|raw}/{symbol}/{timeframe}/data.parquet
        Alpaca:   sources/alpaca/{feed}/{symbol}/{timeframe}/data.parquet
        """
        safe_symbol = self._safe_symbol(symbol)

        if source == DataSource.IB:
            relative = f"sources/ib/{safe_symbol}/{timeframe}/data.parquet"
            return BarPath(
                relative_path=relative,
                source=source,
                symbol=safe_symbol,
                timeframe=timeframe,
            )

        elif source == DataSource.POLYGON:
            adjusted = "adjusted" if is_adjusted else "raw"
            relative = f"sources/polygon/{adjusted}/{safe_symbol}/{timeframe}/data.parquet"
            return BarPath(
                relative_path=relative,
                source=source,
                symbol=safe_symbol,
                timeframe=timeframe,
                is_adjusted=is_adjusted,
            )

        elif source == DataSource.ALPACA:
            feed_name = feed or "iex"
            relative = f"sources/alpaca/{feed_name}/{safe_symbol}/{timeframe}/data.parquet"
            return BarPath(
                relative_path=relative,
                source=source,
                symbol=safe_symbol,
                timeframe=timeframe,
                feed=feed,
            )

        elif source == DataSource.CCXT:
            relative = f"sources/ccxt/{safe_symbol}/{timeframe}/data.parquet"
            return BarPath(
                relative_path=relative,
                source=source,
                symbol=safe_symbol,
                timeframe=timeframe,
            )

        raise ValueError(f"Unknown data source: {source}")

    def source_metadata_path(
        self,
        symbol: str,
        timeframe: str,
        source: DataSource,
        is_adjusted: Optional[bool] = None,
        feed: Optional[str] = None,
    ) -> str:
        """获取来源数据的元数据路径"""
        safe_symbol = self._safe_symbol(symbol)

        if source == DataSource.IB:
            return f"sources/ib/{safe_symbol}/{timeframe}/metadata.json"
        elif source == DataSource.POLYGON:
            adjusted = "adjusted" if is_adjusted else "raw"
            return f"sources/polygon/{adjusted}/{safe_symbol}/{timeframe}/metadata.json"
        elif source == DataSource.ALPACA:
            feed_name = feed or "iex"
            return f"sources/alpaca/{feed_name}/{safe_symbol}/{timeframe}/metadata.json"
        elif source == DataSource.CCXT:
            return f"sources/ccxt/{safe_symbol}/{timeframe}/metadata.json"

        raise ValueError(f"Unknown data source: {source}")

    # ==================== 其他数据路径 ====================

    def corporate_actions_path(self, symbol: str) -> str:
        """公司行为数据路径"""
        safe_symbol = self._safe_symbol(symbol)
        return f"corporate_actions/{safe_symbol}.parquet"

    def fundamentals_path(self, symbol: str) -> str:
        """基本面数据路径"""
        safe_symbol = self._safe_symbol(symbol)
        return f"fundamentals/{safe_symbol}.parquet"

    def cache_path(self, key: str) -> str:
        """临时缓存路径"""
        safe_key = key.replace("/", "_").replace(":", "_")
        return f"cache/{safe_key}.parquet"

    # ==================== 兼容性方法 ====================

    def legacy_path(self, symbol: str, timeframe: str, source: DataSource) -> str:
        """
        获取旧版路径（用于迁移）

        IB:       ib/ohlcv/{timeframe}/{symbol}.parquet
        Polygon:  polygon/ohlcv/{adjusted|raw}/utc/{timeframe}/{symbol}.parquet
        Alpaca:   alpaca/ohlcv/{feed}/utc/{timeframe}/{symbol}.parquet
        """
        safe_symbol = self._safe_symbol(symbol)

        if source == DataSource.IB:
            return f"ib/ohlcv/{timeframe}/{safe_symbol}.parquet"
        elif source == DataSource.POLYGON:
            # 注意：需要传入 is_adjusted
            return f"polygon/ohlcv/{{adjusted|raw}}/utc/{timeframe}/{safe_symbol}.parquet"
        elif source == DataSource.ALPACA:
            return f"alpaca/ohlcv/{{feed}}/utc/{timeframe}/{safe_symbol}.parquet"

        raise ValueError(f"Unknown data source: {source}")


@dataclass
class BarMetadata:
    """Bar 数据元信息"""

    symbol: str
    timeframe: str
    source: str  # 数据源标识
    source_config: dict  # 数据源配置（adjusted, feed 等）
    data_version: str
    ingest_ts_utc: str
    row_count: int
    date_range: tuple[str, str]  # (start, end)
    columns: list[str]
    checksum: str  # 数据完整性校验

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source,
            "source_config": self.source_config,
            "data_version": self.data_version,
            "ingest_ts_utc": self.ingest_ts_utc,
            "row_count": self.row_count,
            "date_range": self.date_range,
            "columns": self.columns,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BarMetadata":
        return cls(**data)

    def to_json(self, path: Path) -> None:
        """保存为 JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> Optional["BarMetadata"]:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


# 单例实例
_path_manager: Optional[BarPathManager] = None


def get_path_manager(data_dir: str = "data") -> BarPathManager:
    """获取路径管理器单例"""
    global _path_manager
    if _path_manager is None:
        _path_manager = BarPathManager(data_dir)
    return _path_manager

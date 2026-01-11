from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import os

import pandas as pd


@dataclass(frozen=True)
class ParquetDataFrameStore:
    """
    简单的本地Parquet持久化（面向时间序列DataFrame）。
    支持分层存储和版本化。
    """

    root_dir: str

    def _root(self) -> Path:
        return Path(self.root_dir)

    def path(self, relative_path: str) -> Path:
        relative_path = relative_path.lstrip("/").replace("..", "__")
        return self._root() / relative_path

    def read(self, relative_path: str) -> pd.DataFrame:
        file_path = self.path(relative_path)
        if not file_path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(file_path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Parquet缓存缺少DatetimeIndex: {file_path}")
        return df.sort_index()

    def write(self, relative_path: str, df: pd.DataFrame) -> None:
        file_path = self.path(relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("仅支持DatetimeIndex的DataFrame写入缓存")

        # 避免将Index丢失：写入一列datetime方便跨引擎读取
        out = df.copy()
        out = out.sort_index()
        out.insert(0, "datetime", out.index)

        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        out.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, file_path)

    def merge_time_series(
        self,
        existing: pd.DataFrame,
        new_data: pd.DataFrame,
        *,
        prefer_new: bool = True,
    ) -> pd.DataFrame:
        if existing.empty:
            return new_data.sort_index()
        if new_data.empty:
            return existing.sort_index()

        merged = pd.concat([existing, new_data]).sort_index()
        # 同一时间戳重复时：默认保留新数据（后出现）
        keep = "last" if prefer_new else "first"
        merged = merged[~merged.index.duplicated(keep=keep)]
        return merged.sort_index()

    def clip(self, df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
        if df.empty:
            return df
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df
    
    def write_dataset(
        self,
        dataset: str,  # bars/features/signals/orders/fills/portfolio_daily
        df: pd.DataFrame,
        partition: Optional[Dict[str, str]] = None,  # {"timeframe": "1d", "symbol": "AAPL", "date": "2024-01-01"}
        data_version: Optional[str] = None,
    ) -> str:
        """
        写入数据集（支持分区和版本化）
        
        Args:
            dataset: 数据集名称
            partition: 分区信息
            data_version: 数据版本（如果提供，会加入路径）
            
        Returns:
            相对路径
        """
        # 构建路径
        parts = [dataset]
        if data_version:
            parts.append(data_version)
        if partition:
            for key, value in sorted(partition.items()):
                parts.append(f"{key}={value}")
        
        # 文件名
        if partition and "symbol" in partition and "timeframe" in partition:
            filename = f"{partition['symbol']}_{partition['timeframe']}.parquet"
        else:
            filename = "data.parquet"
        
        relative_path = "/".join(parts) + "/" + filename
        self.write(relative_path, df)
        return relative_path
    
    def read_dataset(
        self,
        dataset: str,
        partition: Optional[Dict[str, str]] = None,
        data_version: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        读取数据集
        
        Args:
            dataset: 数据集名称
            partition: 分区信息
            data_version: 数据版本
            start: 起始时间
            end: 结束时间
            
        Returns:
            DataFrame
        """
        # 构建路径
        parts = [dataset]
        if data_version:
            parts.append(data_version)
        if partition:
            for key, value in sorted(partition.items()):
                parts.append(f"{key}={value}")
        
        # 文件名
        if partition and "symbol" in partition and "timeframe" in partition:
            filename = f"{partition['symbol']}_{partition['timeframe']}.parquet"
        else:
            filename = "data.parquet"
        
        relative_path = "/".join(parts) + "/" + filename
        df = self.read(relative_path)
        
        if not df.empty and (start is not None or end is not None):
            df = self.clip(df, start, end)
        
        return df


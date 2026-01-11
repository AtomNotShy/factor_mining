from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


@dataclass(frozen=True)
class LocalBarsSource:
    """本地 bars 数据来源信息（便于日志展示/排查）。"""

    path: str
    kind: str  # daily_dir | polygon_cache_adjusted | polygon_cache_raw


def _to_utc_naive_ts(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _read_parquet_timeseries(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if df.empty:
        return df

    # 兼容 ParquetDataFrameStore 的写法：有 datetime 列
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    # 兼容直接把 index 写入 parquet 的情况
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()

    # 统一为 tz-naive（约定：代表 UTC 墙钟时间）
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df = df.sort_index()

    # 统一列名
    rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 仅保留回测需要的列
    need = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if len(need) < 4:
        return pd.DataFrame()

    return df[need]


def load_local_ohlcv(
    *,
    data_dir: str | Path,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> Tuple[pd.DataFrame, Optional[LocalBarsSource]]:
    """从本地读取 OHLCV（优先 daily_dir，其次 PolygonCollector 缓存）。"""

    symbol_u = symbol.strip().upper()
    tf = timeframe.strip()
    root = Path(data_dir)

    start_ts = _to_utc_naive_ts(start)
    end_ts = _to_utc_naive_ts(end)

    candidates: List[Tuple[LocalBarsSource, Path]] = []

    # 1) IBCollector 缓存位置：data/ib/ohlcv/{timeframe}/{symbol}.parquet
    candidates.append(
        (
            LocalBarsSource(
                path=str(root / "ib" / "ohlcv" / tf / f"{symbol_u}.parquet"),
                kind="ib_cache",
            ),
            root / "ib" / "ohlcv" / tf / f"{symbol_u}.parquet",
        )
    )

    # 2) 下载脚本的落盘位置：data/daily/AAPL_1d.parquet
    candidates.append(
        (
            LocalBarsSource(path=str(root / "daily" / f"{symbol_u}_{tf}.parquet"), kind="daily_dir"),
            root / "daily" / f"{symbol_u}_{tf}.parquet",
        )
    )

    # 3) PolygonCollector 缓存位置：data/polygon/ohlcv/{adjusted|raw}/utc/{timeframe}/AAPL.parquet
    candidates.append(
        (
            LocalBarsSource(
                path=str(root / "polygon" / "ohlcv" / "adjusted" / "utc" / tf / f"{symbol_u}.parquet"),
                kind="polygon_cache_adjusted",
            ),
            root / "polygon" / "ohlcv" / "adjusted" / "utc" / tf / f"{symbol_u}.parquet",
        )
    )
    candidates.append(
        (
            LocalBarsSource(
                path=str(root / "polygon" / "ohlcv" / "raw" / "utc" / tf / f"{symbol_u}.parquet"),
                kind="polygon_cache_raw",
            ),
            root / "polygon" / "ohlcv" / "raw" / "utc" / tf / f"{symbol_u}.parquet",
        )
    )

    for meta, p in candidates:
        df = _read_parquet_timeseries(p)
        if df.empty:
            continue

        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty:
            continue

        return df, meta

    return pd.DataFrame(), None

"""
Interactive Brokers 历史数据采集器

功能：
- 从IB Gateway/TWS获取OHLCV历史数据
- 支持日线/分钟线
- 支持复权数据（ADJUSTED_LAST）
- 本地Parquet缓存，支持增量更新
- 输出统一格式DataFrame
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from ib_insync import IB, Stock, Contract

from .base import BaseDataCollector
from src.config.settings import get_settings
from src.data.storage.parquet_store import ParquetDataFrameStore


class IBHistoryCollector(BaseDataCollector):
    """IB历史数据采集器（带本地缓存）"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__("ib_history")
        self.host = host
        self.port = port
        self.client_id = client_id
        
        settings = get_settings()
        self.cache_dir = cache_dir or settings.storage.data_dir
        self.store = ParquetDataFrameStore(self.cache_dir)
        
        self.ib: Optional[IB] = None
        self._connected = False
        
        self._cache_subdir = "ib/ohlcv"
        
    def _cache_path(self, symbol: str, timeframe: str) -> str:
        """获取缓存文件路径"""
        return f"{self._cache_subdir}/{timeframe}/{symbol}.parquet"
    
    async def connect(self) -> bool:
        """连接到IB Gateway/TWS"""
        try:
            if self.ib and self._connected:
                return True

            self.ib = IB()
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=10,
            )
            self._connected = True
            self.logger.info(f"已连接到IB: {self.host}:{self.port} (client_id={self.client_id})")
            return True
        except Exception as e:
            self.logger.error(f"连接IB失败: {e}")
            return False

    async def disconnect(self):
        """断开IB连接"""
        if self.ib and self._connected:
            try:
                self.ib.disconnect()
                self.logger.info("已断开IB连接")
            except Exception as e:
                self.logger.warning(f"断开IB连接时出错: {e}")
            finally:
                self.ib = None
                self._connected = False

    def validate_symbol(self, symbol: str) -> bool:
        """验证标的代码格式"""
        return bool(symbol) and symbol.replace(".", "").replace("-", "").isalnum()

    def validate_timeframe(self, timeframe: str) -> bool:
        """验证时间周期"""
        return timeframe in {"1m", "2m", "3m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "1d", "1w", "1M"}

    def _create_contract(self, symbol: str) -> Contract:
        """创建IB合约"""
        contract = Stock(symbol, "SMART", "USD")
        return contract

    def _timeframe_to_ib_duration(self, timeframe: str) -> str:
        """将 timeframe 转换为IB duration格式"""
        if timeframe.endswith("m"):
            minutes = int(timeframe[:-1])
            if minutes < 60:
                return f"{minutes} D"
            else:
                hours = minutes // 60
                return f"{hours} D"
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            if hours < 24:
                return f"{hours} D"
            else:
                days = hours // 24
                return f"{days} D"
        if timeframe.endswith("d"):
            return f"{int(timeframe[:-1])} D"
        if timeframe.endswith("w"):
            return f"{int(timeframe[:-1])} W"
        if timeframe.endswith("M"):
            return f"{int(timeframe[:-1])} M"
        raise ValueError(f"不支持的timeframe: {timeframe}")

    def _timeframe_to_ib_barsize(self, timeframe: str) -> str:
        """将 timeframe 转换为IB barSizeSetting"""
        mapping = {
            "1m": "1 min",
            "2m": "2 mins",
            "3m": "3 mins",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "2h": "2 hours",
            "3h": "3 hours",
            "4h": "4 hours",
            "1d": "1 day",
            "1w": "1 week",
            "1M": "1 month",
        }
        if timeframe not in mapping:
            raise ValueError(f"不支持的timeframe: {timeframe}")
        return mapping[timeframe]

    def _timeframe_duration(self, timeframe: str) -> timedelta:
        """获取timeframe对应的持续时间"""
        if timeframe.endswith("m"):
            return timedelta(minutes=int(timeframe[:-1]))
        if timeframe.endswith("h"):
            return timedelta(hours=int(timeframe[:-1]))
        if timeframe.endswith("d"):
            return timedelta(days=int(timeframe[:-1]))
        if timeframe.endswith("w"):
            return timedelta(weeks=int(timeframe[:-1]))
        if timeframe.endswith("M"):
            return timedelta(days=int(timeframe[:-1]) * 30)
        raise ValueError(f"不支持的timeframe: {timeframe}")

    def _max_chunk_days(self, timeframe: str) -> int:
        """按周期控制分段下载窗口大小（避免IB限制）"""
        if timeframe.endswith("m"):
            return 30
        if timeframe.endswith("h"):
            return 90
        return 365

    def _read_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """读取本地缓存"""
        cache_path = self._cache_path(symbol, timeframe)
        df = self.store.read(cache_path)
        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts_utc" in df.columns:
                df.index = pd.to_datetime(df["ts_utc"], errors="coerce")
                df.index.name = "ts_utc"
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df.index.name = "ts_utc"
        else:
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)

        if "ts_utc" in df.columns:
            df = df.drop(columns=["ts_utc"])
        return df

    def _write_cache(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """写入本地缓存"""
        cache_path = self._cache_path(symbol, timeframe)
        df_copy = self._normalize_ohlcv_df(df)
        # 确保索引是 DatetimeIndex
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index, errors="coerce")
        if getattr(df_copy.index, "tz", None) is not None:
            df_copy.index = df_copy.index.tz_convert("UTC").tz_localize(None)
        # 如果索引没有名字，设置为 ts_utc
        if df_copy.index.name is None:
            df_copy.index.name = "ts_utc"
        # 如果 ts_utc 不是列，添加为列
        if "ts_utc" not in df_copy.columns:
            df_copy["ts_utc"] = df_copy.index
        self.store.write(cache_path, df_copy)
        self.logger.info(f"已缓存 {symbol} {timeframe}: {len(df)} 条数据")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取OHLCV数据（优先使用本地缓存）

        Args:
            symbol: 股票/ETF代码，如 SPY
            timeframe: 1m/5m/15m/30m/1h/1d/1w/1M
            since: 起始时间（可选）
            end: 结束时间（可选，默认当前）
            use_cache: 是否使用本地缓存

        Returns:
            OHLCV DataFrame（统一格式）
        """
        if not self.validate_symbol(symbol) or not self.validate_timeframe(timeframe):
            raise ValueError(f"无效参数: symbol={symbol}, timeframe={timeframe}")

        if use_cache:
            cached_data = self._read_cache(symbol, timeframe)

            if since is not None and len(cached_data) > 0:
                since_ts = pd.Timestamp(since)
                cached_idx = cached_data.index
                if isinstance(cached_idx, pd.DatetimeIndex):
                    if cached_idx.tz is not None and since_ts.tz is None:
                        since_ts = since_ts.tz_localize(cached_idx.tz)
                    elif cached_idx.tz is None and since_ts.tz is not None:
                        since_ts = since_ts.tz_localize(None)
                cached_data = cached_data[cached_data.index >= since_ts]

            if end is not None and len(cached_data) > 0:
                end_ts = pd.Timestamp(end)
                cached_idx = cached_data.index
                if isinstance(cached_idx, pd.DatetimeIndex):
                    if cached_idx.tz is not None and end_ts.tz is None:
                        end_ts = end_ts.tz_localize(cached_idx.tz)
                    elif cached_idx.tz is None and end_ts.tz is not None:
                        end_ts = end_ts.tz_localize(None)
                cached_data = cached_data[cached_data.index <= end_ts]

            if len(cached_data) > 0:
                return cached_data

        return self._fetch_from_ib(symbol, timeframe, since, end)

    async def get_ohlcv_async(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """异步获取OHLCV数据"""
        if not self.validate_symbol(symbol) or not self.validate_timeframe(timeframe):
            raise ValueError(f"无效参数: symbol={symbol}, timeframe={timeframe}")

        if use_cache:
            cached_data = self._read_cache(symbol, timeframe)
            if len(cached_data) > 0:
                # 检查缓存是否覆盖了请求的范围
                cache_start = cached_data.index.min()
                cache_end = cached_data.index.max()
                
                # 统一时区进行比较
                since_ts = pd.Timestamp(since) if since else None
                end_ts = pd.Timestamp(end) if end else None
                
                # Normalize cache dates to Timestamp for comparison
                cache_start_ts = pd.Timestamp(cache_start)
                cache_end_ts = pd.Timestamp(cache_end)
                
                start_covered = since_ts is None or cache_start_ts <= since_ts
                end_covered = end_ts is None or cache_end_ts >= end_ts
                
                if start_covered and end_covered:
                    # 返回切片后的数据
                    result = cached_data
                    if since_ts:
                        result = result[result.index >= since_ts]
                    if end_ts:
                        result = result[result.index <= end_ts]
                    return result
                else:
                    cache_start_str = str(cache_start)[:10]
                    cache_end_str = str(cache_end)[:10]
                    since_str = str(since)[:10] if since else "None"
                    end_str = str(end)[:10] if end else "None"
                    self.logger.info(f"缓存范围不足（{cache_start_str} ~ {cache_end_str}），无法满足 {since_str} ~ {end_str}，将尝试全量更新")

        return await self._fetch_from_ib_async(symbol, timeframe, since, end)

    def _fetch_from_ib(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime],
        end: Optional[datetime],
    ) -> pd.DataFrame:
        """从IB获取数据（同步）"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._fetch_from_ib_async(symbol, timeframe, since, end)
            )
        finally:
            loop.close()

    async def _fetch_from_ib_async(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime],
        end: Optional[datetime],
    ) -> pd.DataFrame:
        """从IB获取数据（异步）"""
        if self.ib is None:
            ok = await self.connect()
            if not ok:
                return pd.DataFrame()
        
        # 使用 assert 告诉类型检查器 self.ib 不为 None
        assert self.ib is not None, "IB connection should be established"
        ib = self.ib

        try:
            now = datetime.now(timezone.utc)
            end_dt = end or now
            
            if since is None:
                since_dt = now - self._timeframe_duration(timeframe) * 500
            else:
                since_dt = since

            start_dt = since_dt.replace(tzinfo=None)
            end_dt = end_dt.replace(tzinfo=None)

            contract = self._create_contract(symbol)

            barsize = self._timeframe_to_ib_barsize(timeframe)
            max_chunk_days = self._max_chunk_days(timeframe)
            
            # 动态计算 duration：如果指定了 since，根据日期范围计算
            if since is not None:
                # 计算日期范围天数
                days = (end_dt - start_dt).days
                if days > max_chunk_days:
                    all_chunks = []
                    chunk_end = end_dt
                    safety = 0
                    while chunk_end > start_dt and safety < 200:
                        safety += 1
                        chunk_start = max(start_dt, chunk_end - timedelta(days=max_chunk_days))
                        duration_days = max((chunk_end - chunk_start).days, 1)
                        duration = f"{duration_days} D"
                        self.logger.info(
                            f"从IB获取 {symbol} {timeframe} 从 {chunk_start} 到 {chunk_end}, duration={duration}"
                        )
                        bars = await ib.reqHistoricalDataAsync(
                            contract,
                            endDateTime=chunk_end,
                            durationStr=duration,
                            barSizeSetting=barsize,
                            whatToShow="TRADES",
                            useRTH=True,
                            formatDate=2,
                            keepUpToDate=False,
                        )
                        if bars:
                            chunk_df = self._parse_ib_bars(bars, symbol, timeframe)
                            if not chunk_df.empty:
                                all_chunks.append(chunk_df)
                        else:
                            self.logger.warning(f"未获取到 {symbol} 的历史数据（分段）")

                        chunk_end = chunk_start - timedelta(days=1)

                    if not all_chunks:
                        return pd.DataFrame()

                    df = pd.concat(all_chunks).sort_index()
                    df = df[~df.index.duplicated(keep="last")]
                    df = self._normalize_ohlcv_df(df)
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

                    if not df.empty:
                        existing = self._read_cache(symbol, timeframe)
                        if not existing.empty:
                            merged = self.store.merge_time_series(existing, df, prefer_new=True)
                            self._write_cache(symbol, timeframe, merged)
                        else:
                            self._write_cache(symbol, timeframe, df)

                    return df

                duration = f"{max(days, 1)} D"  # 至少1天
            else:
                duration = self._timeframe_to_ib_duration(timeframe)

            self.logger.info(f"从IB获取 {symbol} {timeframe} 从 {start_dt} 到 {end_dt}, duration={duration}")

            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=barsize,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                keepUpToDate=False,
            )

            if not bars:
                self.logger.warning(f"未获取到 {symbol} 的历史数据")
                return pd.DataFrame()

            df = self._parse_ib_bars(bars, symbol, timeframe)

            if len(df) > 0:
                # 合并新数据与现有缓存，避免覆盖历史数据
                existing = self._read_cache(symbol, timeframe)
                if not existing.empty:
                    merged = self.store.merge_time_series(existing, df, prefer_new=True)
                    self._write_cache(symbol, timeframe, merged)
                else:
                    self._write_cache(symbol, timeframe, df)

            return df

        except Exception as e:
            self.logger.error(f"IB获取OHLCV失败: {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def _parse_ib_bars(self, bars: List, symbol: str, timeframe: str) -> pd.DataFrame:
        """解析IB bar数据为DataFrame"""
        if not bars:
            return pd.DataFrame()

        data = []
        for bar in bars:
            bar_date = bar.date
            if isinstance(bar_date, str):
                bar_date = pd.to_datetime(bar_date)

            data.append({
                "datetime": bar_date,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            })

        df: pd.DataFrame = pd.DataFrame(data)
        df = df.set_index("datetime").sort_index()
        df.index.name = "ts_utc"

        return self._normalize_ohlcv_df(df)

    def _normalize_ohlcv_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理并标准化 OHLCV 数据（用于下载与缓存）"""
        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df = df.copy()
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.set_index("datetime")
            else:
                df = df.copy()
                df.index = pd.to_datetime(df.index, errors="coerce")

        if getattr(df.index, "tz", None) is not None:
            # 使用类型断言确保Pylance正确识别DatetimeIndex
            idx = df.index
            assert isinstance(idx, pd.DatetimeIndex), "Index should be DatetimeIndex when tz is not None"
            df.index = idx.tz_convert("UTC").tz_localize(None)

        df = df.sort_index()

        rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep_cols] if keep_cols else pd.DataFrame(index=df.index)
        if df.empty:
            return df

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        required = ["open", "high", "low", "close"]
        df = df.dropna(subset=[c for c in required if c in df.columns])

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).clip(lower=0)

        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            df["high"] = df[["high", "open", "close", "low"]].max(axis=1)
            df["low"] = df[["low", "open", "close", "high"]].min(axis=1)

        df = df[~df.index.duplicated(keep="last")]
        df.index.name = "ts_utc"
        return df

    def get_ohlcv_incremental(
        self,
        symbol: str,
        timeframe: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        增量获取OHLCV数据（从缓存的最后日期开始）

        Args:
            symbol: 股票/ETF代码
            timeframe: 时间周期
            use_cache: 是否使用本地缓存

        Returns:
            新数据DataFrame（已合并缓存）
        """
        if not self.validate_symbol(symbol) or not self.validate_timeframe(timeframe):
            raise ValueError(f"无效参数: symbol={symbol}, timeframe={timeframe}")

        existing = self._read_cache(symbol, timeframe) if use_cache else pd.DataFrame()

        last_date: Optional[datetime] = None
        if len(existing) > 0:
            last_ts = existing.index.max()
            if hasattr(last_ts, 'to_pydatetime'):
                last_date = last_ts.to_pydatetime()
            else:
                last_date = pd.Timestamp(last_ts).to_pydatetime()
            self.logger.info(f"{symbol} 缓存最新日期: {last_date}")

        new_data = self._fetch_from_ib(symbol, timeframe, since=last_date, end=None)

        if len(new_data) == 0:
            self.logger.info(f"{symbol} 无新数据")
            return existing

        if len(existing) == 0:
            merged = new_data
        else:
            merged = self.store.merge_time_series(existing, new_data, prefer_new=True)

        if len(merged) > 0:
            self._write_cache(symbol, timeframe, merged)

        return merged

    async def get_ohlcv_incremental_async(
        self,
        symbol: str,
        timeframe: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """增量获取OHLCV数据（异步）"""
        if not self.validate_symbol(symbol) or not self.validate_timeframe(timeframe):
            raise ValueError(f"无效参数: symbol={symbol}, timeframe={timeframe}")

        existing = self._read_cache(symbol, timeframe) if use_cache else pd.DataFrame()

        last_date: Optional[datetime] = None
        if len(existing) > 0:
            last_ts = existing.index.max()
            if hasattr(last_ts, 'to_pydatetime'):
                last_date = last_ts.to_pydatetime()
            else:
                last_date = pd.Timestamp(last_ts).to_pydatetime()
            self.logger.info(f"{symbol} 缓存最新日期: {last_date}")

        new_data = await self._fetch_from_ib_async(symbol, timeframe, since=last_date, end=None)

        if len(new_data) == 0:
            self.logger.info(f"{symbol} 无新数据")
            return existing

        if len(existing) == 0:
            merged = new_data
        else:
            merged = self.store.merge_time_series(existing, new_data, prefer_new=True)

        if len(merged) > 0:
            self._write_cache(symbol, timeframe, merged)

        return merged

    def get_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """批量获取多只标的的OHLCV数据"""
        results = {}
        for symbol in symbols:
            try:
                df = self.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    use_cache=use_cache,
                )
                if len(df) > 0:
                    results[symbol] = df
                    self.logger.info(f"成功获取 {symbol}: {len(df)} 条数据")
                else:
                    self.logger.warning(f"获取 {symbol} 数据为空")
            except Exception as e:
                self.logger.error(f"获取 {symbol} 数据失败: {e}")
        return results

    async def get_multiple_symbols_async(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """批量获取多只标的的OHLCV数据（异步）"""
        results = {}
        for symbol in symbols:
            try:
                df = await self.get_ohlcv_async(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    use_cache=use_cache,
                )
                if len(df) > 0:
                    results[symbol] = df
                    self.logger.info(f"成功获取 {symbol}: {len(df)} 条数据")
                else:
                    self.logger.warning(f"获取 {symbol} 数据为空")
            except Exception as e:
                self.logger.error(f"获取 {symbol} 数据失败: {e}")
        return results

    def update_all_cache(
        self,
        symbols: List[str],
        timeframe: str = "1d",
    ) -> Dict[str, int]:
        """更新所有标的的缓存数据"""
        updated_counts = {}
        for symbol in symbols:
            try:
                df = self.get_ohlcv_incremental(symbol, timeframe, use_cache=True)
                updated_counts[symbol] = len(df)
                self.logger.info(f"已更新 {symbol}: {len(df)} 条数据")
            except Exception as e:
                self.logger.error(f"更新 {symbol} 失败: {e}")
                updated_counts[symbol] = -1
        return updated_counts

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """清除缓存"""
        import shutil
        
        if symbol and timeframe:
            cache_path = self._cache_path(symbol, timeframe)
            full_path = self.store.path(cache_path)
            if full_path.exists():
                full_path.unlink()
                self.logger.info(f"已清除缓存: {cache_path}")
        elif symbol:
            symbol_dir = self.store.path(f"{self._cache_subdir}/{symbol}")
            if symbol_dir.exists():
                shutil.rmtree(symbol_dir)
                self.logger.info(f"已清除缓存: {symbol}")
        else:
            cache_root = self.store.path(self._cache_subdir)
            if cache_root.exists():
                shutil.rmtree(cache_root)
                self.logger.info("已清除所有IB缓存")


async def test_collector():
    """测试IB历史数据采集器"""
    collector = IBHistoryCollector(host="127.0.0.1", port=7497, client_id=1)

    try:
        ok = await collector.connect()
        if not ok:
            print("连接IB失败")
            return

        print("连接IB成功")

        df = collector.get_ohlcv(
            symbol="SPY",
            timeframe="1d",
            since=datetime(2020, 1, 1),
            use_cache=True,
        )

        if len(df) > 0:
            print(f"获取SPY数据: {len(df)} 条")
            print(df.head())
            print(df.tail())
        else:
            print("获取数据为空")

    finally:
        await collector.disconnect()


if __name__ == "__main__":
    asyncio.run(test_collector())

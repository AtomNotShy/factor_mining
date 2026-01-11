"""
Polygon.io 数据采集器（美股/ETF等）

目标：
- 提供与 ExchangeCollector 类似的 get_ohlcv 接口
- 将历史数据本地持久化为 Parquet，避免重复请求
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin
import ssl

import aiohttp
import certifi
import pandas as pd

from .base import BaseDataCollector
from src.config.settings import get_settings
from src.data.storage.parquet_store import ParquetDataFrameStore


class PolygonCollector(BaseDataCollector):
    """Polygon.io OHLCV 采集器（带本地缓存）"""

    def __init__(self):
        super().__init__("polygon")
        settings = get_settings()
        self.polygon_settings = settings.polygon
        self.store = ParquetDataFrameStore(settings.storage.data_dir)
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        if not self.polygon_settings.api_key:
            self.logger.error("未配置 Polygon API Key，请设置环境变量 POLYGON_API_KEY")
            return False
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            connector = aiohttp.TCPConnector(ssl=self._build_ssl_context())
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return True

    async def disconnect(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _build_ssl_context(self):
        """
        macOS 的 python.org 发行版常见证书链缺失，aiohttp 会报 CERTIFICATE_VERIFY_FAILED。
        这里默认使用 certifi 的 CA bundle；必要时可通过环境变量关闭校验或指定自定义 CA。
        """
        if not getattr(self.polygon_settings, "ssl_verify", True):
            return False

        cafile = self.polygon_settings.ssl_ca_bundle or certifi.where()
        context = ssl.create_default_context(cafile=cafile)
        return context

    def validate_symbol(self, symbol: str) -> bool:
        return bool(symbol) and symbol.replace(".", "").replace("-", "").isalnum()

    def validate_timeframe(self, timeframe: str) -> bool:
        return timeframe in {"1m", "5m", "15m", "30m", "1h", "1d"}

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        data_version: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取OHLCV数据（优先读取本地缓存，不足则增量拉取后写入缓存）

        Args:
            symbol: 股票代码，如 AAPL
            timeframe: 1m/5m/15m/30m/1h/1d
            since: 起始时间（本地时区/UTC均可，按绝对时间处理）
            limit: 条数限制；与 since 同时给出时，返回从 since 开始的前 limit 条
        """
        try:
            if not self.validate_symbol(symbol) or not self.validate_timeframe(timeframe):
                raise ValueError(f"无效参数: symbol={symbol}, timeframe={timeframe}")

            ok = await self.connect()
            if not ok:
                return pd.DataFrame()

            start_ts, end_ts = self._resolve_time_range(timeframe=timeframe, since=since, limit=limit)

            cache_key = self._cache_relative_path(symbol=symbol, timeframe=timeframe)
            cached = self.store.read(cache_key)

            cached = self._normalize_ohlcv_df(cached)

            missing_ranges = self._compute_missing_ranges(cached, start_ts, end_ts, timeframe=timeframe)
            if missing_ranges:
                fetched_all: List[pd.DataFrame] = []
                for start_missing, end_missing in missing_ranges:
                    fetched = await self._fetch_aggregates(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_missing,
                        end=end_missing,
                    )
                    fetched_all.append(fetched)

                fetched_df = pd.concat(fetched_all).sort_index() if fetched_all else pd.DataFrame()
                fetched_df = self._normalize_ohlcv_df(fetched_df)

                merged = self.store.merge_time_series(cached, fetched_df, prefer_new=True)
                merged = self._normalize_ohlcv_df(merged)
                self.store.write(cache_key, merged)
                cached = merged

            out = self.store.clip(cached, start_ts, end_ts)

            if since is not None:
                out = out[out.index >= start_ts]
                if limit:
                    out = out.head(limit)
            elif limit:
                out = out.tail(limit)
            
            # 格式化输出为统一schema
            if not out.empty:
                if data_version is None:
                    from src.data.storage.versioning import generate_data_version
                    data_version = generate_data_version(
                        source="polygon_api",
                        timeframe=timeframe,
                        symbols=[symbol],
                    )
                out = self._format_bars_schema(out, symbol, timeframe, data_version)

            return out

        except Exception as e:
            self.logger.error(f"Polygon 获取OHLCV失败: {e}")
            return pd.DataFrame()

    def _cache_relative_path(self, symbol: str, timeframe: str) -> str:
        safe_symbol = symbol.upper().replace("/", "_")
        adj = "adjusted" if self.polygon_settings.adjusted else "raw"
        # 约定：缓存使用 UTC-naive 时间戳（从 tz-aware UTC 直接 tz_localize(None)）
        # 为避免历史缓存（可能是本地时区 naive）混用，这里加一层 utc 目录做版本隔离。
        return f"polygon/ohlcv/{adj}/utc/{timeframe}/{safe_symbol}.parquet"

    def _resolve_time_range(
        self, *, timeframe: str, since: Optional[datetime], limit: Optional[int]
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        now = datetime.now(timezone.utc)
        duration = self._timeframe_duration(timeframe)

        if since is None:
            if not limit:
                # 缺省取最近一段（避免误拉全量）
                limit = 1000
            # 由于非交易时段会导致缺bar，这里放大回溯窗口以覆盖足够bar
            start = now - duration * int(limit * 5)
            end = now
        else:
            since_utc = since.astimezone(timezone.utc) if since.tzinfo else since.replace(tzinfo=timezone.utc)
            start = since_utc
            if limit:
                end = since_utc + duration * int(limit * 5)
            else:
                end = now

        # 保持 UTC 的“墙钟时间”不变，仅去掉 tz 信息（避免转换到本地时区）
        start_ts = pd.Timestamp(start).tz_localize(None)
        end_ts = pd.Timestamp(end).tz_localize(None)
        return start_ts, end_ts

    def _timeframe_duration(self, timeframe: str) -> timedelta:
        if timeframe.endswith("m"):
            return timedelta(minutes=int(timeframe[:-1]))
        if timeframe.endswith("h"):
            return timedelta(hours=int(timeframe[:-1]))
        if timeframe.endswith("d"):
            return timedelta(days=int(timeframe[:-1]))
        raise ValueError(f"不支持的timeframe: {timeframe}")

    def _polygon_timespan(self, timeframe: str) -> Tuple[int, str]:
        if timeframe.endswith("m"):
            return int(timeframe[:-1]), "minute"
        if timeframe.endswith("h"):
            return int(timeframe[:-1]), "hour"
        if timeframe.endswith("d"):
            return int(timeframe[:-1]), "day"
        raise ValueError(f"不支持的timeframe: {timeframe}")

    def _compute_missing_ranges(
        self,
        cached: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        timeframe: str,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        if cached.empty:
            return [(start, end)]

        cached = cached.sort_index()
        min_ts = cached.index.min()
        max_ts = cached.index.max()
        duration = self._timeframe_duration(timeframe)

        missing: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        if start < min_ts:
            missing.append((start, min_ts - duration))
        if end > max_ts:
            missing.append((max_ts + duration, end))
        return missing

    def _normalize_ohlcv_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            else:
                raise ValueError("OHLCV 数据缺少 datetime/index")
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df = df.sort_index()
        # 统一列名（至少包含 open/high/low/close/volume）
        rename_map = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        for col in ["open", "high", "low", "close", "volume", "vw"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        required = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if required:
            df = df.dropna(subset=required)
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).clip(lower=0)
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            df["high"] = df[["high", "open", "close", "low"]].max(axis=1)
            df["low"] = df[["low", "open", "close", "high"]].min(axis=1)
        df = df[~df.index.duplicated(keep="last")]
        return df

    async def _fetch_aggregates(
        self, *, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        if self._session is None or self._session.closed:
            await self.connect()

        multiplier, timespan = self._polygon_timespan(timeframe)

        # Polygon range endpoint 接受 YYYY-MM-DD；对分钟级也可用（按交易日过滤）
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")

        path = f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        url = urljoin(self.polygon_settings.base_url, path)

        params = {
            "adjusted": "true" if self.polygon_settings.adjusted else "false",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.polygon_settings.api_key,
        }

        all_results: List[Dict] = []
        next_url: Optional[str] = url + "?" + urlencode(params)

        # Polygon aggregates 可能会返回 next_url 分页
        for _ in range(1000):
            if not next_url:
                break
            payload = await self._get_json(next_url)
            results = payload.get("results") or []
            all_results.extend(results)

            next_url = payload.get("next_url")
            if next_url:
                # next_url 通常不包含 apiKey，需要补上
                join_char = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{join_char}apiKey={self.polygon_settings.api_key}"

            if not results:
                break

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        # t: unix ms
        # Polygon 聚合时间戳是 UTC；这里保留 UTC 墙钟时间，去掉 tz 信息用于索引与本地缓存
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.set_index("datetime").sort_index()

        # 统一字段：o/h/l/c/v
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        keep_cols = [c for c in ["open", "high", "low", "close", "volume", "vw", "n"] if c in df.columns]
        df = df[keep_cols]
        return df
    
    def _format_bars_schema(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        data_version: str,
    ) -> pd.DataFrame:
        """
        格式化输出为统一的bars schema
        
        Args:
            df: 原始OHLCV数据
            symbol: 标的代码
            timeframe: 时间周期
            data_version: 数据版本
            
        Returns:
            格式化后的bars DataFrame（符合设计文档schema）
        """
        if df.empty:
            return df
        
        # 确保索引是DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("bars数据必须使用DatetimeIndex")
        
        # 创建输出DataFrame（复制索引）
        bars = pd.DataFrame(index=df.index.copy())
        
        # 必需字段
        bars['symbol'] = symbol
        bars['timeframe'] = timeframe
        bars['open'] = df['open'].values
        bars['high'] = df['high'].values
        bars['low'] = df['low'].values
        bars['close'] = df['close'].values
        bars['volume'] = df.get('volume', pd.Series(0.0, index=df.index)).values
        
        # 可选字段
        if 'vw' in df.columns:
            bars['vwap'] = df['vw'].values
        else:
            bars['vwap'] = None
        
        if 'n' in df.columns:
            bars['trades'] = df['n'].values
        else:
            bars['trades'] = None
        
        bars['is_regular'] = True  # 日线默认True，分钟级需要更复杂判断
        
        # 元数据字段
        bars['source'] = 'polygon_api'
        bars['data_version'] = data_version
        bars['ingest_ts_utc'] = datetime.utcnow()
        
        # 索引重命名为ts_utc（但保持为index）
        bars.index.name = 'ts_utc'
        
        return bars
    
    async def get_corporate_actions(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        获取公司行为数据（拆股、分红等）
        
        Args:
            symbol: 标的代码
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            公司行为DataFrame（符合设计文档schema）
        """
        if self._session is None or self._session.closed:
            await self.connect()
        
        # Polygon的参考数据API（需要相应权限）
        # 这里提供简化实现，实际需要根据Polygon API文档调整
        path = f"/v3/reference/dividends"
        url = urljoin(self.polygon_settings.base_url, path)
        
        params = {
            "ticker": symbol.upper(),
            "apiKey": self.polygon_settings.api_key,
        }
        
        if start_date:
            params["ex_dividend_date.gte"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["ex_dividend_date.lte"] = end_date.strftime("%Y-%m-%d")
        
        try:
            payload = await self._get_json(url + "?" + urlencode(params))
            results = payload.get("results", [])
            
            if not results:
                return pd.DataFrame()
            
            actions = []
            for item in results:
                actions.append({
                    'ex_date': pd.Timestamp(item.get('ex_dividend_date')),
                    'symbol': symbol,
                    'action_type': 'DIVIDEND',
                    'split_ratio': None,
                    'dividend_cash': item.get('cash_amount'),
                    'currency': item.get('currency', 'USD'),
                    'source': 'polygon_api',
                    'data_version': datetime.utcnow().strftime("%Y%m%d"),
                })
            
            return pd.DataFrame(actions)
        except Exception as e:
            self.logger.warning(f"获取公司行为数据失败: {e}")
            return pd.DataFrame()

    async def _get_json(self, url: str) -> Dict:
        if not self._session:
            raise RuntimeError("Polygon session未初始化")

        backoff_s = 1.0
        for attempt in range(6):
            async with self._session.get(url) as resp:
                if resp.status == 429:
                    if attempt < 5:
                        await asyncio.sleep(backoff_s)
                        backoff_s *= 2
                        continue
                    else:
                        raise RuntimeError("Polygon 请求频率受限(429)，重试失败")
                
                # 处理403错误（权限问题）
                if resp.status == 403:
                    error_text = await resp.text()
                    self.logger.error(f"Polygon API权限错误 (403): {error_text[:200]}")
                    raise RuntimeError(f"Polygon API权限不足 (403): {error_text[:200]}")
                
                # 处理404错误（数据未找到）
                if resp.status == 404:
                    error_text = await resp.text()
                    self.logger.warning(f"Polygon数据未找到 (404): {error_text[:200]}")
                    # 404不一定是错误，可能只是没有数据，返回空结果
                    return {"results": [], "status": "OK", "resultsCount": 0}
                
                resp.raise_for_status()
                return await resp.json()
        
        raise RuntimeError("Polygon 请求失败，重试次数用尽")

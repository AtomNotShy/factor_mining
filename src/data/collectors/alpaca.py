"""
Alpaca 股票数据采集器（美股/ETF等）

目标：
- 提供与 ExchangeCollector 类似的 get_ohlcv 接口
- 将历史数据本地持久化为 Parquet，避免重复请求
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin
import ssl

import aiohttp
import certifi
import pandas as pd

from .base import BaseDataCollector
from src.config.settings import get_settings
from src.data.storage.parquet_store import ParquetDataFrameStore


class AlpacaCollector(BaseDataCollector):
    """Alpaca OHLCV 采集器（带本地缓存）"""

    # Alpaca 支持的时间周期映射
    TIMEFRAME_MAP = {
        "1m": (1, "Min"),
        "2m": (2, "Min"),
        "3m": (3, "Min"),
        "5m": (5, "Min"),
        "10m": (10, "Min"),
        "15m": (15, "Min"),
        "30m": (30, "Min"),
        "1h": (1, "Hour"),
        "2h": (2, "Hour"),
        "4h": (4, "Hour"),
        "1d": (1, "Day"),
        "1w": (1, "Week"),
        "1M": (1, "Month"),
    }

    def __init__(self):
        super().__init__("alpaca")
        settings = get_settings()
        self.alpaca_settings = settings.alpaca
        self.store = ParquetDataFrameStore(settings.storage.data_dir)
        self._session: Optional[aiohttp.ClientSession] = None

        # 直接使用 settings 中的值（model_validator 已经处理了 PAPER key 兼容）
        self._api_key = self.alpaca_settings.api_key or ""
        self._secret_key = self.alpaca_settings.secret_key or ""
        self._base_url = self.alpaca_settings.base_url or "https://data.alpaca.markets"

    async def connect(self) -> bool:
        if not self._api_key or not self._secret_key:
            self.logger.error(
                "未配置 Alpaca API Key，请设置环境变量 ALPACA_API_KEY 或 ALPACA_PAPER_API_KEY"
            )
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
        if not getattr(self.alpaca_settings, "ssl_verify", True):
            return False
        cafile = getattr(self.alpaca_settings, "ssl_ca_bundle", None) or certifi.where()
        context = ssl.create_default_context(cafile=cafile)
        return context

    def validate_symbol(self, symbol: str) -> bool:
        return bool(symbol) and symbol.replace(".", "").replace("-", "").isalnum()

    def validate_timeframe(self, timeframe: str) -> bool:
        return timeframe in self.TIMEFRAME_MAP

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
            timeframe: 1m/5m/15m/30m/1h/1d/1w/1M
            since: 起始时间（本地时区/UTC均可，按绝对时间处理）
            limit: 条数限制；与 since 同时给出时，返回从 since 开始的前 limit 条
        """
        try:
            if not self.validate_symbol(symbol) or not self.validate_timeframe(timeframe):
                raise ValueError(f"无效参数: symbol={symbol}, timeframe={timeframe}")

            ok = await self.connect()
            if not ok:
                return pd.DataFrame()

            start_ts, end_ts = self._resolve_time_range(
                timeframe=timeframe, since=since, limit=limit
            )

            cache_key = self._cache_relative_path(symbol=symbol, timeframe=timeframe)
            cached = self.store.read(cache_key)
            cached = self._normalize_ohlcv_df(cached)

            missing_ranges = self._compute_missing_ranges(
                cached, start_ts, end_ts, timeframe=timeframe
            )
            if missing_ranges:
                fetched_all: List[pd.DataFrame] = []
                for start_missing, end_missing in missing_ranges:
                    fetched = await self._fetch_bars(
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

                # 直接写入统一的 bars 目录
                unified_path = f"bars/{symbol.upper()}/{timeframe}/data.parquet"
                self.store.write(unified_path, merged)
                self.logger.debug(f"Alpaca: 数据已保存到 {unified_path}")

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
                        source="alpaca_api",
                        timeframe=timeframe,
                        symbols=[symbol],
                    )
                out = self._format_bars_schema(out, symbol, timeframe, data_version)

            return out

        except Exception as e:
            self.logger.error(f"Alpaca 获取OHLCV失败: {e}")
            return pd.DataFrame()

    def _cache_relative_path(self, symbol: str, timeframe: str) -> str:
        safe_symbol = symbol.upper().replace("/", "_")
        feed = self.alpaca_settings.feed or "iex"
        return f"alpaca/ohlcv/{feed}/utc/{timeframe}/{safe_symbol}.parquet"

    def _resolve_time_range(
        self, *, timeframe: str, since: Optional[datetime], limit: Optional[int]
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        now = datetime.now(timezone.utc)
        duration = self._timeframe_duration(timeframe)

        if since is None:
            if not limit:
                limit = 1000
            start = now - duration * int(limit * 5)
            end = now
        else:
            # 处理 date 类型（转换为 datetime）
            if isinstance(since, date) and not isinstance(since, datetime):
                since = datetime.combine(since, datetime.min.time())
            
            since_utc = (
                since.astimezone(timezone.utc)
                if since.tzinfo
                else since.replace(tzinfo=timezone.utc)
            )
            start = since_utc
            if limit:
                end = since_utc + duration * int(limit * 5)
            else:
                end = now

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
        if timeframe.endswith("w"):
            return timedelta(weeks=int(timeframe[:-1]))
        if timeframe.endswith("M"):
            return timedelta(days=int(timeframe[:-1]) * 30)
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

        # 确保索引是 DatetimeIndex
        if not isinstance(cached.index, pd.DatetimeIndex):
            if "ts_utc" in cached.columns:
                cached = cached.set_index("ts_utc")
            elif "datetime" in cached.columns:
                cached = cached.set_index("datetime")
            else:
                return [(start, end)]

        cached = cached.sort_index()
        min_ts = cached.index.min()
        max_ts = cached.index.max()

        # 处理 NaT 情况
        if pd.isna(min_ts) or pd.isna(max_ts):
            return [(start, end)]

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

        # 如果索引不是 DatetimeIndex，尝试从列转换
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts_utc" in df.columns:
                df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
                df = df.set_index("ts_utc")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df = df.set_index("datetime")
            else:
                # 无法识别时间列，返回原始 DataFrame
                return df

        # 移除时区信息，统一使用 UTC naive
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df.sort_index()
        # 统一列名
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

    async def _fetch_bars(
        self, *, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        if self._session is None or self._session.closed:
            await self.connect()

        multiplier, unit = self.TIMEFRAME_MAP.get(timeframe, (1, "Day"))

        # Alpaca v2 API 需要 ISO 8601 格式，不支持毫秒
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 构建 URL 和请求头
        base_url = self._base_url.rstrip("/")
        path = f"/v2/stocks/{symbol.upper()}/bars"
        url = urljoin(base_url, path)

        params = {
            "timeframe": f"{multiplier}{unit}",
            "start": start_iso,
            "end": end_iso,
            "feed": self.alpaca_settings.feed or "iex",
            "limit": 10000,
        }

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

        all_results: List[Dict] = []
        next_token: Optional[str] = None

        # Alpaca 使用分页 token
        for _ in range(100):
            request_params = params.copy()
            if next_token:
                request_params["page_token"] = next_token

            request_url = url + "?" + urlencode(request_params)

            payload = await self._get_json(request_url, headers)
            bars = payload.get("bars") or []
            all_results.extend(bars)

            next_token = payload.get("next_page_token")
            if not next_token:
                break

            if not bars:
                break

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        # Alpaca 返回的是 ISO 8601 时间戳
        df["datetime"] = pd.to_datetime(df["t"], utc=True).dt.tz_localize(None)
        df = df.set_index("datetime").sort_index()

        # 统一字段
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        keep_cols = [
            c for c in ["open", "high", "low", "close", "volume", "vw", "n"] if c in df.columns
        ]
        df = df[keep_cols]
        return df

    def _format_bars_schema(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        data_version: str,
    ) -> pd.DataFrame:
        """格式化输出为统一的bars schema"""
        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("bars数据必须使用DatetimeIndex")

        bars = pd.DataFrame(index=df.index.copy())

        bars["symbol"] = symbol
        bars["timeframe"] = timeframe
        bars["open"] = df["open"].values
        bars["high"] = df["high"].values
        bars["low"] = df["low"].values
        bars["close"] = df["close"].values
        bars["volume"] = df.get("volume", pd.Series(0.0, index=df.index)).values

        if "vw" in df.columns:
            bars["vwap"] = df["vw"].values
        else:
            bars["vwap"] = None

        if "n" in df.columns:
            bars["trades"] = df["n"].values
        else:
            bars["trades"] = None

        bars["is_regular"] = True

        bars["source"] = "alpaca_api"
        bars["data_version"] = data_version
        bars["ingest_ts_utc"] = datetime.utcnow()

        bars.index.name = "ts_utc"

        return bars

    async def get_corporate_actions(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """获取公司行为数据（拆股、分红等）"""
        if self._session is None or self._session.closed:
            await self.connect()

        base_url = self.alpaca_settings.base_url.rstrip("/")
        url = f"{base_url}/v2/corporate_actions"

        params = {"symbol": symbol.upper()}
        if start_date:
            params["since"] = start_date.isoformat()
        if end_date:
            params["until"] = end_date.isoformat()

        headers = {
            "APCA-API-KEY-ID": self.alpaca_settings.api_key,
            "APCA-API-SECRET-KEY": self.alpaca_settings.secret_key,
        }

        try:
            request_url = url + "?" + urlencode(params)
            payload = await self._get_json(request_url, headers)
            actions = payload.get("corporate_actions", [])

            if not actions:
                return pd.DataFrame()

            result = []
            for item in actions:
                result.append(
                    {
                        "ex_date": pd.Timestamp(item.get("ex_date")),
                        "symbol": symbol,
                        "action_type": item.get("type", "UNKNOWN"),
                        "split_ratio": item.get("split_ratio"),
                        "dividend_cash": item.get("dividend"),
                        "currency": "USD",
                        "source": "alpaca_api",
                        "data_version": datetime.utcnow().strftime("%Y%m%d"),
                    }
                )

            return pd.DataFrame(result)

        except Exception as e:
            self.logger.warning(f"获取公司行为数据失败: {e}")
            return pd.DataFrame()

    async def _get_json(self, url: str, headers: Dict[str, str]) -> Any:
        if not self._session:
            raise RuntimeError("Alpaca session未初始化")

        backoff_s = 1.0
        for attempt in range(6):
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 429:
                    if attempt < 5:
                        await asyncio.sleep(backoff_s)
                        backoff_s *= 2
                        continue
                    else:
                        raise RuntimeError("Alpaca 请求频率受限(429)，重试失败")

                if resp.status == 403:
                    error_text = await resp.text()
                    self.logger.error(f"Alpaca API权限错误 (403): {error_text[:200]}")
                    raise RuntimeError(f"Alpaca API权限不足 (403): {error_text[:200]}")

                if resp.status == 404:
                    error_text = await resp.text()
                    self.logger.warning(f"Alpaca数据未找到 (404): {error_text[:200]}")
                    return {"bars": [], "next_page_token": None}

                if resp.status == 422:
                    error_text = await resp.text()
                    self.logger.warning(f"Alpaca请求参数错误 (422): {error_text[:200]}")
                    return {"bars": [], "next_page_token": None}

                resp.raise_for_status()
                return await resp.json()

        raise RuntimeError("Alpaca 请求失败，重试次数用尽")

"""
数据适配器
统一历史数据和实时数据访问接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import logging
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = logging.getLogger("data_adapter")


class DataLoadError(Exception):
    """数据加载失败异常"""
    pass


class DataValidationError(Exception):
    """数据验证失败异常"""
    pass


class DataTimeoutError(Exception):
    """数据获取超时异常"""
    pass


@dataclass
class DataRequest:
    """数据请求"""
    symbol: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    timeframe: str = "1d"
    fields: Optional[List[str]] = None
    include_previous: int = 0
    mode: str = "backtest"  # "backtest" 或 "live"


@dataclass
class DataMetadata:
    """数据元数据"""
    symbol: str = ""
    timeframe: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    record_count: int = 0
    last_updated: Optional[datetime] = None
    source: str = ""


class DataAdapter(ABC):
    """数据适配器基类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics = DataAdapterMetrics()

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取历史数据"""
        pass

    @abstractmethod
    async def subscribe_real_time(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """订阅实时数据"""
        pass

    @abstractmethod
    async def unsubscribe_real_time(self, subscription_id: str) -> bool:
        """取消订阅实时数据"""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        pass

    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """获取可用标的列表"""
        pass

    @abstractmethod
    async def get_data_metadata(
        self, symbol: str, timeframe: str
    ) -> DataMetadata:
        """获取数据元数据"""
        pass

    async def get_data(self, request: DataRequest) -> pd.DataFrame:
        """统一数据获取接口"""
        if request.mode == "backtest":
            return await self._get_backtest_data(request)
        else:
            return await self._get_live_data(request)

    async def _get_backtest_data(self, request: DataRequest) -> pd.DataFrame:
        """获取回测数据"""
        # 1. 确定实际需要的起始时间（考虑warmup）
        start = request.start or datetime.min
        if request.include_previous > 0:
            if request.timeframe.endswith("d"):
                warmup_days = request.include_previous * 260
                warmup_start = start - timedelta(days=warmup_days)
            else:
                warmup_minutes = request.include_previous * int(request.timeframe[:-1])
                warmup_start = start - timedelta(minutes=warmup_minutes)
        else:
            warmup_start = start

        # 2. 加载历史数据
        try:
            data = await self.get_historical_data(
                symbol=request.symbol,
                start=warmup_start,
                end=request.end or datetime.now(timezone.utc),
                timeframe=request.timeframe,
                fields=request.fields,
            )
        except Exception as e:
            raise DataLoadError(f"加载历史数据失败: {e}")

        # 3. 过滤到实际需要的范围
        if request.start and not data.empty:
            data = data[data.index >= request.start]

        return data

    async def _get_live_data(self, request: DataRequest) -> pd.DataFrame:
        """获取实盘数据"""
        current_price = await self.get_current_price(request.symbol)

        if current_price is None:
            # 回退到历史数据
            return await self._get_backtest_data(request)

        # 构建单行DataFrame
        now = datetime.now(timezone.utc)
        data = pd.DataFrame(
            {
                "open": [current_price],
                "high": [current_price],
                "low": [current_price],
                "close": [current_price],
                "volume": [0],
            },
            index=pd.DatetimeIndex([now]),
        )
        data["symbol"] = request.symbol

        return data


class HistoricalDataAdapter(DataAdapter):
    """历史数据适配器"""

    def __init__(self, storage_backend, config: Optional[Dict] = None):
        super().__init__(config)
        self.storage = storage_backend
        self.cache = DataCache(config.get("cache_size", 1000) if config else 1000)

    async def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取历史数据"""
        # 1. 检查缓存
        cache_key = self._generate_cache_key(symbol, start, end, timeframe, fields)
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            self.metrics.cache_hits += 1
            return cached_data

        self.metrics.cache_misses += 1

        # 2. 使用 load_local_ohlcv 从多个可能位置加载数据
        from src.data.storage.local_bars import load_local_ohlcv
        from src.config.settings import get_settings
        
        settings = get_settings()
        data_dir = settings.storage.data_dir
        
        df, source = load_local_ohlcv(
            data_dir=data_dir,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end
        )

        # 3. 数据验证和清洗
        if not df.empty:
            df = await self._clean_and_validate_data(df, symbol, timeframe)
            # 4. 缓存数据
            self.cache.set(cache_key, df)

        return df

    async def _clean_and_validate_data(
        self, data: pd.DataFrame, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """数据清洗和验证"""
        if data.empty:
            return data

        data = data.copy()

        # 1. 检查必要字段
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise DataValidationError(f"缺失必要字段: {col}")

        # 2. 处理缺失值
        for col in required_columns:
            if data[col].isnull().any():
                data[col] = data[col].ffill().bfill()

        # 3. 检查价格合理性
        self._validate_price_ranges(data, symbol)

        # 4. 检查时间连续性
        self._validate_time_continuity(data, timeframe)

        # 5. 添加元数据
        data["symbol"] = symbol
        data["timeframe"] = timeframe

        return data

    def _validate_price_ranges(self, data: pd.DataFrame, symbol: str):
        """验证价格范围合理性"""
        if (data["high"] < data["low"]).any():
            raise DataValidationError(f"价格范围错误: high < low")

        if ((data["close"] < data["low"]) | (data["close"] > data["high"])).any():
            raise DataValidationError(f"收盘价超出高低范围")

        if ((data["open"] < data["low"]) | (data["open"] > data["high"])).any():
            raise DataValidationError(f"开盘价超出高低范围")

    def _validate_time_continuity(self, data: pd.DataFrame, timeframe: str):
        """验证时间连续性（日线数据跳过严格检查）"""
        # 日线数据不进行严格连续性检查，因为周末和节假日会有间隔
        if timeframe.endswith("d"):
            return
        
        if len(data) < 2:
            return

        time_diffs = data.index.to_series().diff().dropna()

        expected_minutes = int(timeframe[:-2])
        expected_seconds = expected_minutes * 60
        tolerance = expected_seconds * 0.1

        # 将time_diffs转换为秒进行比较
        diff_seconds_list = time_diffs.apply(
            lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x)
        )
        
        for diff_seconds in diff_seconds_list:
            if abs(diff_seconds - expected_seconds) > tolerance:
                raise DataValidationError(
                    f"时间连续性错误: 期望间隔{expected_seconds}秒, 实际间隔{diff_seconds}秒"
                )

    def _generate_cache_key(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        fields: Optional[List[str]] = None,
    ) -> str:
        """生成缓存键"""
        fields_str = ",".join(fields) if fields else "all"
        return f"{symbol}_{timeframe}_{start.isoformat()}_{end.isoformat()}_{fields_str}"

    async def subscribe_real_time(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """订阅实时数据（历史适配器不支持）"""
        raise NotImplementedError("HistoricalDataAdapter 不支持实时订阅")

    async def unsubscribe_real_time(self, subscription_id: str) -> bool:
        """取消订阅实时数据"""
        return False

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格（使用最新收盘价）"""
        now = datetime.now(timezone.utc)
        try:
            data = await self.get_historical_data(
                symbol=symbol,
                start=now - timedelta(days=5),
                end=now,
                timeframe="1d",
            )
            if not data.empty:
                return float(data.iloc[-1]["close"])
        except Exception:
            pass
        return None

    async def get_available_symbols(self) -> List[str]:
        """获取可用标的列表"""
        # 简化实现：返回常见标的
        # 在实际项目中，应该从存储中扫描可用文件
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]

    async def get_data_metadata(
        self, symbol: str, timeframe: str
    ) -> DataMetadata:
        """获取数据元数据"""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=365)

        try:
            data = await self.get_historical_data(
                symbol=symbol, start=start, end=now, timeframe=timeframe
            )

            if data.empty:
                return DataMetadata(symbol=symbol, timeframe=timeframe)

            return DataMetadata(
                symbol=symbol,
                timeframe=timeframe,
                start_time=data.index.min(),
                end_time=data.index.max(),
                record_count=len(data),
                last_updated=now,
                source="historical",
            )
        except Exception as e:
            logger.error(f"获取数据元数据失败: {symbol} {timeframe}: {e}")
            return DataMetadata(symbol=symbol, timeframe=timeframe)


class LiveDataAdapter(DataAdapter):
    """实时数据适配器"""

    def __init__(self, data_stream, config: Optional[Dict] = None):
        super().__init__(config)
        self.data_stream = data_stream
        self.subscriptions: Dict[str, Dict] = {}
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_ttl = config.get("cache_ttl", 5) if config else 5  # 秒

    async def subscribe_real_time(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """订阅实时数据"""
        subscription_id = f"{symbol}_{timeframe}_{id(callback)}"

        self.subscriptions[subscription_id] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "callback": callback,
            "created_at": datetime.now(timezone.utc),
        }

        # 启动数据流
        await self._ensure_data_stream(symbol, timeframe)

        return subscription_id

    async def _ensure_data_stream(self, symbol: str, timeframe: str):
        """确保数据流已启动"""
        stream_key = f"{symbol}_{timeframe}"
        # 这里应该启动数据流，实际实现需要根据data_stream接口
        pass

    async def unsubscribe_real_time(self, subscription_id: str) -> bool:
        """取消订阅实时数据"""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            return True
        return False

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 检查缓存
        if symbol in self.price_cache:
            price, timestamp = self.price_cache[symbol]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < self.cache_ttl:
                return price

        # 从数据流获取
        try:
            latest_data = await self.data_stream.get_latest_data(symbol)
            if latest_data and "close" in latest_data:
                price = float(latest_data["close"])
                self.price_cache[symbol] = (price, datetime.now(timezone.utc))
                return price
        except Exception as e:
            logger.error(f"获取当前价格失败: {symbol}: {e}")

        return None

    async def get_available_symbols(self) -> List[str]:
        """获取可用标的列表"""
        return await self.data_stream.get_available_symbols()

    async def get_data_metadata(
        self, symbol: str, timeframe: str
    ) -> DataMetadata:
        """获取数据元数据"""
        return DataMetadata(
            symbol=symbol,
            timeframe=timeframe,
            last_updated=datetime.now(timezone.utc),
            source="live",
        )


class DataCache:
    """数据缓存"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            return data
        return None

    def set(self, key: str, data: pd.DataFrame):
        """设置缓存数据"""
        if len(self.cache) >= self.max_size:
            # 移除最旧的缓存
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = (data, datetime.now(timezone.utc))

    def clear(self):
        """清空缓存"""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
        }


@dataclass
class DataAdapterMetrics:
    """数据适配器指标"""
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    total_requests: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

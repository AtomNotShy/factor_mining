"""
数据源抽象接口
抽象 DataFeed：回测用"历史回放"，实盘用 IB/CCXT；统一发 MarketEvent
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Any, Union
import asyncio
import pandas as pd

from src.core.events import MarketEvent, EventPriority, create_bar_event
from src.utils.logger import get_logger

logger = get_logger("data_feed")


class DataFeed(ABC):
    """
    数据源抽象基类
    
    实现方式：
    - HistoricalDataFeed：历史回放（回测用）
    - IBDataFeed：Interactive Brokers（实盘用）
    - CCXTDataFeed：加密货币交易所（实盘用）
    """
    
    @abstractmethod
    async def initialize(self, symbols: List[str], timeframe: str, **kwargs) -> None:
        """初始化数据源"""
        pass
    
    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """获取历史K线数据"""
        pass
    
    @abstractmethod
    async def stream(self) -> AsyncIterator[MarketEvent]:
        """流式发布市场事件（逐bar/逐tick）"""
        pass
    
    @abstractmethod
    async def get_current_bar(self, symbol: str) -> Dict[str, Any]:
        """获取当前K线数据（用于计算成交价）"""
        pass

    @abstractmethod
    async def get_bars_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str],
        start: datetime,
        end: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """获取多个时间框架的K线数据

        Args:
            symbol: 交易标的
            timeframes: 时间框架列表，如 ["1d", "1m"]
            start: 开始时间
            end: 结束时间

        Returns:
            {timeframe: DataFrame} 的字典
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭数据源"""
        pass


class HistoricalDataFeed(DataFeed):
    """
    历史回放数据源（用于回测）

    特点：
    - 按时间轴逐条发布 MarketEvent
    - 支持多时间框架
    - 可预加载所有历史数据
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        warmup_days: int = 260,
    ):
        self._symbols: List[str] = []
        self._timeframe: str = "1d"
        self._timeframes: List[str] = []  # 支持多时间框架
        self._bars_map: Dict[str, pd.DataFrame] = {}  # {symbol: DataFrame} - 主时间框架
        self._bars_map_by_timeframe: Dict[str, Dict[str, pd.DataFrame]] = {}  # {timeframe: {symbol: DataFrame}}
        self._current_idx: int = 0
        self._timeline: pd.DatetimeIndex = pd.DatetimeIndex([])
        self._warmup_days = warmup_days
        self._closed = False

        # 从数据适配器获取数据
        from src.data.adapter.factory import adapter_factory
        self._adapter = adapter_factory.create_historical_adapter()

        logger.info(f"HistoricalDataFeed 初始化完成 (warmup_days={warmup_days})")

    async def initialize(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """初始化并加载历史数据（单时间框架）"""
        await self.initialize_multi(
            symbols=symbols,
            timeframes=[timeframe],
            start_date=start_date,
            end_date=end_date,
        )

    async def initialize_multi(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """初始化并加载多时间框架历史数据

        Args:
            symbols: 交易标的列表
            timeframes: 时间框架列表，如 ["1d", "1m"]
            start_date: 开始日期
            end_date: 结束日期
        """
        self._symbols = symbols
        self._timeframes = timeframes
        self._timeframe = timeframes[0] if timeframes else "1d"

        # 计算实际需要的日期范围（包含预热期）
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            # 默认回测1年
            start_date = end_date.replace(year=end_date.year - 1)

        # 预热期向前扩展
        from datetime import timedelta
        warmup_delta = timedelta(days=self._warmup_days)
        actual_start = start_date - warmup_delta

        # 初始化多时间框架存储
        self._bars_map_by_timeframe = {tf: {} for tf in timeframes}

        # 加载所有标的的所有时间框架数据
        for symbol in symbols:
            for tf in timeframes:
                try:
                    data = await self._adapter.get_data(
                        symbol=symbol,
                        start=actual_start,
                        end=end_date,
                        timeframe=tf,
                    )

                    if not data.empty:
                        if 'symbol' not in data.columns:
                            data['symbol'] = symbol
                        data = data.sort_index()
                        self._bars_map_by_timeframe[tf][symbol] = data
                        logger.info(f"加载 {symbol} {tf}: {len(data)} bars")
                    else:
                        logger.warning(f"没有找到 {symbol} {tf} 的数据")

                except Exception as e:
                    logger.error(f"加载 {symbol} {tf} 数据失败: {e}")

        # 主时间框架的数据映射
        self._bars_map = self._bars_map_by_timeframe.get(self._timeframe, {})

        # 生成统一的时间轴（使用主时间框架）
        self._timeline = self._generate_timeline()

        logger.info(
            f"HistoricalDataFeed 初始化完成: "
            f"{len(self._symbols)} 标的, {timeframes} 时间框架, {len(self._timeline)} 时间点"
        )
    
    def _generate_timeline(self) -> pd.DatetimeIndex:
        """生成统一时间轴（所有标的的交集）"""
        if not self._bars_map:
            return pd.DatetimeIndex([])
        
        # 找到所有标的共有的交易日
        all_dates: Optional[pd.DatetimeIndex] = None
        
        for symbol, bars in self._bars_map.items():
            if bars.empty:
                continue
            symbol_dates = bars.index
            if all_dates is None:
                all_dates = symbol_dates
            else:
                # 取交集
                all_dates = all_dates.intersection(symbol_dates)
        
        if all_dates is None:
            return pd.DatetimeIndex([])
        
        return all_dates.sort_values()
    
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """获取历史K线数据"""
        if symbol not in self._bars_map:
            return pd.DataFrame()
        
        bars = self._bars_map[symbol]
        mask = (bars.index >= start) & (bars.index <= end)
        return bars[mask].copy()
    
    async def stream(self) -> AsyncIterator[MarketEvent]:
        """流式发布市场事件"""
        if self._timeline.empty:
            logger.warning("时间轴为空，无法发布事件")
            return
        
        for timestamp in self._timeline:
            for symbol in self._symbols:
                if symbol not in self._bars_map:
                    continue
                
                bars = self._bars_map[symbol]
                
                # 获取当前时间点的数据
                current_bars = bars[bars.index == timestamp]
                
                if current_bars.empty:
                    continue
                
                row = current_bars.iloc[0]
                
                # 创建并发布 BarEvent
                bar_event = create_bar_event(
                    symbol=symbol,
                    timeframe=self._timeframe,
                    open_price=float(row.get('open', 0)),
                    high_price=float(row.get('high', 0)),
                    low_price=float(row.get('low', 0)),
                    close_price=float(row.get('close', 0)),
                    volume=float(row.get('volume', 0)),
                    timestamp=timestamp.to_pydatetime(),
                    priority=EventPriority.NORMAL,
                )
                
                yield bar_event
    
    async def get_current_bar(self, symbol: str) -> Dict[str, Any]:
        """获取当前K线数据"""
        if symbol not in self._bars_map:
            return {}
        
        bars = self._bars_map[symbol]
        
        if self._timeline.empty or len(self._timeline) == 0:
            return {}
        
        # 获取最后一个时间点的数据
        last_timestamp = self._timeline[-1]
        current_bars = bars[bars.index == last_timestamp]
        
        if current_bars.empty:
            # 如果没有精确匹配，获取最后一行
            current_bars = bars.iloc[-1:]
        
        if current_bars.empty:
            return {}
        
        row = current_bars.iloc[0]
        return {
            'open': float(row.get('open', 0)),
            'high': float(row.get('high', 0)),
            'low': float(row.get('low', 0)),
            'close': float(row.get('close', 0)),
            'volume': float(row.get('volume', 0)),
            'timestamp': last_timestamp.to_pydatetime() if hasattr(last_timestamp, 'to_pydatetime') else last_timestamp,
        }
    
    async def get_bar_at(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """获取指定时间点的K线数据"""
        if symbol not in self._bars_map:
            return {}
        
        bars = self._bars_map[symbol]
        
        # 尝试精确匹配
        current_bars = bars[bars.index == timestamp]
        
        if current_bars.empty:
            # 如果没有精确匹配，找到最近的时间点
            if len(bars) == 0:
                return {}
            
            # 使用索引找到最近的日期
            idx = bars.index.searchsorted(timestamp)
            if idx >= len(bars):
                idx = len(bars) - 1
            elif idx < 0:
                idx = 0
            
            # 选择索引位置
            if bars.index[idx] > timestamp and idx > 0:
                idx -= 1
            
            current_bars = bars.iloc[[idx]]
        
        if current_bars.empty:
            return {}
        
        row = current_bars.iloc[0]
        return {
            'open': float(row.get('open', 0)),
            'high': float(row.get('high', 0)),
            'low': float(row.get('low', 0)),
            'close': float(row.get('close', 0)),
            'volume': float(row.get('volume', 0)),
            'timestamp': current_bars.index[0].to_pydatetime() if hasattr(current_bars.index[0], 'to_pydatetime') else current_bars.index[0],
        }

    async def get_bars_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str],
        start: datetime,
        end: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """获取多个时间框架的K线数据

        Args:
            symbol: 交易标的
            timeframes: 时间框架列表，如 ["1d", "1m"]
            start: 开始时间
            end: 结束时间

        Returns:
            {timeframe: DataFrame} 的字典
        """
        result: Dict[str, pd.DataFrame] = {}

        for tf in timeframes:
            bars_by_tf = self._bars_map_by_timeframe.get(tf, {})
            if symbol in bars_by_tf:
                bars = bars_by_tf[symbol]
                mask = (bars.index >= start) & (bars.index <= end)
                result[tf] = bars[mask].copy()
                logger.debug(f"获取 {symbol} {tf}: {len(result[tf])} bars")
            else:
                logger.warning(f"没有找到 {symbol} {tf} 的数据")
                result[tf] = pd.DataFrame()

        return result

    async def get_bars_for_timeframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """获取指定时间框架的数据（不指定日期范围，返回全部）

        Args:
            symbol: 交易标的
            timeframe: 时间框架

        Returns:
            DataFrame
        """
        bars_by_tf = self._bars_map_by_timeframe.get(timeframe, {})
        if symbol in bars_by_tf:
            return bars_by_tf[symbol].copy()
        return pd.DataFrame()

    async def close(self) -> None:
        """关闭数据源"""
        self._closed = True
        self._bars_map.clear()
        self._bars_map_by_timeframe.clear()
        logger.info("HistoricalDataFeed 已关闭")


class DataFeedFactory:
    """数据源工厂"""
    
    _feeds: Dict[str, DataFeed] = {}
    
    @classmethod
    def create_historical_feed(
        cls,
        initial_capital: float = 100000.0,
        warmup_days: int = 260,
    ) -> HistoricalDataFeed:
        """创建历史回放数据源（单时间框架）"""
        return HistoricalDataFeed(
            initial_capital=initial_capital,
            warmup_days=warmup_days,
        )

    @classmethod
    def create_historical_feed_multi(
        cls,
        timeframes: List[str],
        initial_capital: float = 100000.0,
        warmup_days: int = 260,
    ) -> HistoricalDataFeed:
        """创建历史回放数据源（多时间框架）

        Args:
            timeframes: 时间框架列表，如 ["1d", "1m"]
            initial_capital: 初始资金
            warmup_days: 预热天数

        Returns:
            HistoricalDataFeed 实例
        """
        feed = HistoricalDataFeed(
            initial_capital=initial_capital,
            warmup_days=warmup_days,
        )
        # 保存 timeframes 以便 initialize_multi 时使用
        feed._timeframes = timeframes
        return feed
    
    @classmethod
    def create_live_feed(cls, source: str = "ib", **kwargs) -> DataFeed:
        """创建实盘数据源（需要外部实现）"""
        if source == "ib":
            from src.data.providers.ib import IBDataFeed
            return IBDataFeed(**kwargs)
        elif source == "ccxt":
            from src.data.providers.ccxt import CCXTDataFeed
            return CCXTDataFeed(**kwargs)
        else:
            raise ValueError(f"未知的数据源: {source}")

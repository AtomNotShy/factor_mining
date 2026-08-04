"""
统一回测引擎

整合所有回测功能，提供单一入口：
- 向量化预计算（Freqtrade 协议）
- 完整的风控管理（止损/ROI/保护机制）
- 插件式特性开关
- 事件驱动架构（可选）
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Union, Set
import uuid
import pandas as pd

from src.core.types import (
    Signal, OrderIntent, Fill, MarketData, PortfolioState, RiskState,
    OrderSide, OrderType, ActionType,
)
from src.core.context import RunContext, Environment
from src.core.events.engine import UnifiedEventEngine
from src.core.calendar import TradingCalendar
from src.core.risk_manager import RiskManager
from src.strategies.base.strategy import Strategy
from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.informative import merge_informative_pairs
from src.utils.logger import get_logger
from src.data.storage.parquet_store import ParquetDataFrameStore
from src.data.adapter.factory import adapter_factory
from src.data.manager import data_manager
from src.data.providers.base import DataFeed, HistoricalDataFeed
from src.execution.providers.base import ExecutionProvider, SimulatedExecutionProvider, ExecutionProviderConfig
from src.core.risk_manager import RiskManager
from src.evaluation.metrics.comprehensive import EnhancedAnalyzer
from .config import UnifiedConfig, FeatureFlag
from .stoploss_manager import StoplossManager
from .cost_model import CostModel
from .config import BacktestResult
from .data_validator import DataValidator


StrategyType = Union[Strategy, FreqtradeStrategy]


@dataclass
class BacktestContext:
    """回测上下文（内部状态）"""
    run_id: str = ""
    strategy_id: str = ""
    current_ts: Optional[datetime] = None
    trading_days: List[date] = field(default_factory=list)
    timeline: List[datetime] = field(default_factory=list)
    all_signals: List[Signal] = field(default_factory=list)
    all_orders: List[OrderIntent] = field(default_factory=list)
    all_fills: List[Fill] = field(default_factory=list)
    portfolio_daily: List[Dict] = field(default_factory=list)
    trades: Dict[str, Dict] = field(default_factory=dict)
    cooldown_periods: Dict[str, int] = field(default_factory=dict)
    daily_losses: Dict[str, List[float]] = field(default_factory=dict)
    vectorized_data: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    seen_fill_ids: Set[str] = field(default_factory=set)


class UnifiedBacktestEngine:
    """
    统一回测引擎
    
    Features:
        - 向量化预计算：启用 VECTORIZED 特性
        - Freqtrade 协议：启用 FREQTRADE_PROTOCOL 特性
        - 止损/ROI：启用 STOPLOSS_MANAGER 特性
        - 保护机制：启用 PROTECTIONS 特性
        - 事件驱动：启用 EVENT_DRIVEN 特性
    """
    
    def __init__(
        self,
        config: Optional[UnifiedConfig] = None,
        data_feed: Optional[DataFeed] = None,
        execution_provider: Optional[ExecutionProvider] = None,
        event_engine: Optional[UnifiedEventEngine] = None,
        store: Optional[ParquetDataFrameStore] = None,
    ):
        self.config = config or UnifiedConfig()
        self.config.validate()
        
        self.logger = get_logger("unified_backtest_engine")
        
        # 组件初始化
        self._data_feed = data_feed or HistoricalDataFeed(
            initial_capital=self.config.trade.initial_capital,
            warmup_days=self.config.time.warmup_days,
        )
        
        exec_config = ExecutionProviderConfig(
            commission_rate=self.config.trade.commission_rate,
            slippage_rate=self.config.trade.slippage_rate,
            fill_price_type=self.config.fill.fill_price,
            initial_capital=self.config.trade.initial_capital,
        )
        self._execution_provider = execution_provider or SimulatedExecutionProvider(
            config=exec_config,
            data_feed=self._data_feed,
            event_engine=event_engine,
        )

        # 市场数据缓存（用于回测最终化时获取收盘价）
        self._bars_map: Optional[Dict[str, pd.DataFrame]] = None
        
        self._event_engine = event_engine or UnifiedEventEngine()
        self._store = store
        
        # 内部状态
        self._context: Optional[BacktestContext] = None
        self._lookahead_warned: set = set()
        
        # 特性组件初始化
        self._init_feature_components()

    def _standardize_bars_timezone(self, bars_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        统一 bars 数据的时区：将所有 DataFrame 的索引转换为 tz-naive UTC

        统一约定：
        - 所有 bars 数据的索引都是 tz-naive UTC（代表 Wall Clock Time）
        - 这样可以避免时区相关的比较和计算问题
        """
        self.logger.info(f"[TIMEZONE] 开始标准化时区，输入有 {len(bars_map)} 个 timeframe")

        standardized = {}
        for tf, bars in bars_map.items():
            if bars is None or bars.empty:
                self.logger.warning(f"[TIMEZONE] {tf} 数据为空，跳过")
                continue

            df = bars.copy()

            # 检查原始时区
            original_tz = getattr(df.index, 'tz', None)
            self.logger.debug(f"[TIMEZONE] {tf} 原始时区: {original_tz}")

            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    self.logger.warning(f"无法将 {tf} 的索引转换为 DatetimeIndex")
                    continue

            # 统一为 tz-naive UTC
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
                self.logger.debug(f"[TIMEZONE] {tf} 已转换为 tz-naive")
            else:
                self.logger.debug(f"[TIMEZONE] {tf} 已经是 tz-naive")

            standardized[tf] = df.sort_index()

        self.logger.info(f"[TIMEZONE] 标准化完成，输出有 {len(standardized)} 个 timeframe")

        if len(standardized) < len(bars_map):
            self.logger.warning(
                f"时区标准化后丢失了部分数据: {len(bars_map)} -> {len(standardized)}"
            )

        return standardized

    def _init_feature_components(self) -> None:
        """初始化特性相关组件"""
        if FeatureFlag.STOPLOSS_MANAGER in self.config.features:
            self._stoploss_manager = StoplossManager(
                commission_rate=self.config.trade.commission_rate,
            )
        else:
            self._stoploss_manager = None
        
        self._cost_model = CostModel(
            commission_rate=self.config.trade.commission_rate,
            slippage_rate=self.config.trade.slippage_rate,
        )
        self._risk_manager: Optional[RiskManager] = None
    
    async def run(
        self,
        strategies: List[StrategyType],
        universe: List[str],
        start: date,
        end: date,
        ctx: Optional[RunContext] = None,
        bars: Optional[pd.DataFrame] = None,
        auto_download: bool = True,
    ) -> BacktestResult:
        """运行回测"""
        self.logger.info("=" * 60)
        self.logger.info("UNIFIED BACKTEST ENGINE")
        self.logger.info(f"Strategies: {[s.strategy_id for s in strategies]}")
        self.logger.info(f"Universe: {universe}")
        self.logger.info(f"Period: {start} ~ {end}")
        self.logger.info(f"Features: {self.config.features}")
        self.logger.info("=" * 60)
        
        # 初始化上下文
        self._context = BacktestContext(
            run_id=str(uuid.uuid4()),
            strategy_id=strategies[0].strategy_id if strategies else "unknown",
        )
        
        # 初始化运行上下文
        trading_calendar = ctx.trading_calendar if ctx else TradingCalendar()
        if ctx is None:
            ctx = RunContext.create(
                env=Environment.RESEARCH,
                trading_calendar=trading_calendar,
                config={},
            )
        
        # 初始化风控管理器
        self._risk_manager = RiskManager(ctx.config)
        
        # 重置执行器
        self._execution_provider.reset(self.config.trade.initial_capital)
        portfolio = self._execution_provider.get_portfolio_state()
        
        # 加载数据
        bars_map = await self._load_data(universe, start, end, ctx, auto_download)
        if bars_map is None or all(df.empty for df in bars_map.values()):
            return BacktestResult(run_id=self._context.run_id)

        # 统一时区处理：所有 bars 数据转换为 tz-naive UTC
        bars_map = self._standardize_bars_timezone(bars_map)

        # 保存 bars_map 以便 _finalize_backtest 使用
        self._bars_map = bars_map
        
        # 向量化预计算
        if FeatureFlag.VECTORIZED in self.config.features:
            self._context.vectorized_data = await self._vectorized_precompute(
                strategies, bars_map, universe
            )
        
        # 生成时间轴
        trading_days = trading_calendar.get_trading_days(start, end)
        # 转换为日期列表（解决 DatetimeIndex 赋值类型不匹配问题）
        self._context.trading_days = list(trading_days)
        self._context.timeline = self._generate_timeline(start, end, ctx, bars_map)
        
        # 初始化策略
        for strategy in strategies:
            await self._initialize_strategy(strategy)
        
        # 主循环
        self.logger.info(f"Phase 2: Backtest Loop ({len(self._context.timeline)} ticks)")
        
        for idx, current_ts in enumerate(self._context.timeline):
            self.logger.debug(f"Tick {idx+1}/{len(self._context.timeline)}: {current_ts}")
            
            await self._process_tick(
                current_ts=current_ts,
                strategies=strategies,
                portfolio=portfolio,
                bars_map=bars_map,
                ctx=ctx,
            )
            
            portfolio = self._execution_provider.get_portfolio_state()
        
        # 生成结果
        result = self._generate_result(portfolio, strategies, ctx)
        
        # 保存结果
        if FeatureFlag.SAVE_RESULTS in self.config.features:
            await self._save_result(result, ctx)
        
        self.logger.info(f"Backtest completed: {result.run_id}")
        
        return result
    
    async def _load_data(
        self,
        universe: List[str],
        start: date,
        end: date,
        ctx: RunContext,
        auto_download: bool,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """加载市场数据"""
        self.logger.info("Phase 1: Loading Data")

        provider = "auto"
        if isinstance(ctx.config, dict):
            provider = ctx.config.get("data", {}).get("provider", provider)
        
        timeframes = {self.config.time.signal_timeframe, self.config.time.execution_timeframe}
        if FeatureFlag.MULTI_TIMEFRAME in self.config.features:
            timeframes.update(self.config.time.informative_timeframes)
        if FeatureFlag.DETAIL_TIMEFRAME in self.config.features and self.config.time.detail_timeframe:
            timeframes.add(self.config.time.detail_timeframe)
        
        bars_map: Dict[str, pd.DataFrame] = {}
        
        for tf in timeframes:
            if tf in bars_map:
                continue
            
            bars_list = []
            warmup_start = start - timedelta(days=self.config.time.warmup_days)
            start_dt = datetime.combine(warmup_start, datetime.min.time())
            end_dt = datetime.combine(end, datetime.max.time())
            
            try:
                for symbol in universe:
                    if auto_download:
                        bars = await data_manager.get_ohlcv(
                            symbol=symbol,
                            start=start_dt,
                            end=end_dt,
                            timeframe=tf,
                            auto_download=True,
                            source_preference=provider,
                        )
                    else:
                        bars = await adapter_factory.get_data(
                            symbol=symbol,
                            start=start_dt,
                            end=end_dt,
                            timeframe=tf,
                            mode="backtest",
                            include_previous=self.config.time.warmup_days,
                        )

                    if not bars.empty:
                        if "symbol" not in bars.columns:
                            bars["symbol"] = symbol
                        bars_list.append(bars)

            except Exception as e:
                self.logger.error(f"加载数据失败 {tf}: {e}")
            
            if bars_list:
                bars_map[tf] = pd.concat(bars_list).sort_index()
                self.logger.info(f"  {tf}: {len(bars_map[tf])} bars")

        # 数据验证
        if bars_map:
            validator = DataValidator(
                min_success_rate=0.5,  # 至少50%的标的有数据
                require_all_symbols=False,  # 不强制要求所有标的
                min_bars_per_symbol=self.config.time.warmup_days,  # 至少要有warmup期的数据
            )

            validation_result = validator.validate(
                universe=universe,
                bars_map=bars_map,
                start_date=start,
                end_date=end,
                warmup_days=self.config.time.warmup_days,
            )

            # 如果验证失败，记录警告（但不中断回测，因为可能部分数据就够用）
            if not validation_result.is_valid:
                self.logger.warning("数据验证未通过，回测结果可能不可靠")

        return bars_map
    
    async def _vectorized_precompute(
        self,
        strategies: List[StrategyType],
        bars_map: Dict[str, pd.DataFrame],
        universe: List[str],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """向量化预计算"""
        self.logger.info("Vectorized Pre-calculation")
        
        vectorized_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        informative_tfs: List[str] = []
        if FeatureFlag.MULTI_TIMEFRAME in self.config.features:
            informative_tfs.extend(self.config.time.informative_timeframes)
        if FeatureFlag.DETAIL_TIMEFRAME in self.config.features and self.config.time.detail_timeframe:
            informative_tfs.append(self.config.time.detail_timeframe)
        if self.config.time.execution_timeframe and self.config.time.execution_timeframe != self.config.time.signal_timeframe:
            informative_tfs.append(self.config.time.execution_timeframe)

        informative_tfs = [tf for tf in dict.fromkeys(informative_tfs) if tf in bars_map]
        
        for strategy in strategies:
            strategy_data: Dict[str, pd.DataFrame] = {}
            main_tf = getattr(strategy, "timeframe", self.config.time.signal_timeframe)
            main_bars = bars_map.get(main_tf, pd.DataFrame())
            if main_bars.empty or "symbol" not in main_bars.columns:
                self.logger.warning(f"No bars for timeframe {main_tf}, skip strategy {strategy.strategy_id}")
                vectorized_data[strategy.strategy_id] = strategy_data
                continue
            
            for symbol in universe:
                symbol_bars = main_bars[main_bars["symbol"] == symbol].copy()
                if symbol_bars.empty:
                    continue

                if informative_tfs:
                    informative_dfs = {
                        tf: bars_map[tf][bars_map[tf]["symbol"] == symbol].copy()
                        for tf in informative_tfs
                        if tf != main_tf and not bars_map[tf].empty
                    }
                    if informative_dfs:
                        symbol_bars = merge_informative_pairs(
                            symbol_bars,
                            informative_dfs,
                            main_tf,
                            ffill=True,
                            lookahead_check=True,
                        )
                
                # Freqtrade 协议
                if hasattr(strategy, 'populate_indicators'):
                    symbol_bars = strategy.populate_indicators(
                        symbol_bars, {"symbol": symbol}
                    )
                if hasattr(strategy, 'populate_entry_trend'):
                    symbol_bars = strategy.populate_entry_trend(
                        symbol_bars, {"symbol": symbol}
                    )
                if hasattr(strategy, 'populate_exit_trend'):
                    symbol_bars = strategy.populate_exit_trend(
                        symbol_bars, {"symbol": symbol}
                    )
                
                strategy_data[symbol] = symbol_bars
            
            vectorized_data[strategy.strategy_id] = strategy_data
            
            # Populate strategy's _vectorized_data for size_positions to use
            # 使用 setattr 绕过 Pylance 的属性类型检查，同时记录日志
            try:
                setattr(strategy, '_vectorized_data', strategy_data)
                self.logger.debug(f"Set _vectorized_data for strategy: {strategy.strategy_id}")
            except Exception as e:
                self.logger.warning(f"Failed to set _vectorized_data for {strategy.strategy_id}: {e}")
        
        return vectorized_data
    
    def _generate_timeline(
        self,
        start: date,
        end: date,
        ctx: RunContext,
        bars_map: Dict[str, pd.DataFrame],
    ) -> List[datetime]:
        """生成时间轴"""
        trading_days = ctx.trading_calendar.get_trading_days(start, end)
        
        if self.config.time.clock_mode == "daily":
            return [d for d in trading_days]
        
        elif self.config.time.clock_mode == "hybrid":
            timeline = []
            for day in trading_days:
                if self.config.time.execution_time:
                    ts = ctx.trading_calendar.session_time(
                        day, self.config.time.execution_time
                    )
                    timeline.append(ts)
                else:
                    schedule = ctx.trading_calendar.get_schedule(day, day)
                    if not schedule.empty:
                        ts = schedule["market_close"].iloc[0]
                        timeline.append(ts)
            return timeline
        
        elif self.config.time.clock_mode == "bar":
            return list(ctx.trading_calendar.get_trading_minutes(start, end))
        
        return []
    
    async def _initialize_strategy(self, strategy: StrategyType) -> None:
        """初始化策略"""
        self.logger.info(f"Initializing strategy: {strategy.strategy_id}")

        # 注入 data_feed（支持多时间框架策略）
        if hasattr(strategy, 'data_feed'):
            strategy.data_feed = self._data_feed
            self.logger.debug(f"Injected data_feed to strategy: {strategy.strategy_id}")

        # 使用 isinstance 检查类型，避免类型检查器报错
        if isinstance(strategy, FreqtradeStrategy):
            await strategy.bot_start()
    
    async def _process_tick(
        self,
        current_ts: datetime,
        strategies: List[StrategyType],
        portfolio: PortfolioState,
        bars_map: Dict[str, pd.DataFrame],
        ctx: RunContext,
    ) -> None:
        """处理单个时间点"""
        if self._context is None:
            return
            
        self._context.current_ts = current_ts
        ctx.now_utc = current_ts
        ctx.dt = pd.Timestamp(current_ts)  # type: ignore
        
        # 构建 MarketData
        md = self._build_market_data(current_ts, bars_map)
        
        # 风控检查（止损/ROI）
        if FeatureFlag.STOPLOSS_MANAGER in self.config.features:
            await self._check_stoploss(portfolio, current_ts, md)
        
        # 保护机制检查
        if FeatureFlag.PROTECTIONS in self.config.features:
            self._check_protections(portfolio, current_ts)
        
        # 策略生成信号
        for strategy in strategies:
            signals = await self._generate_signals(strategy, md, ctx)
            self._context.all_signals.extend(signals)
            
            # 转换信号为订单
            order_intents = await self._convert_to_orders(
                strategy, signals, portfolio, ctx
            )
            self._context.all_orders.extend(order_intents)
            
            # 执行订单
            for intent in order_intents:
                await self._execute_order(intent, current_ts, md)
        
        # 记录每日净值
        self._record_portfolio_daily(portfolio, current_ts)
    
    def _build_market_data(
        self,
        current_ts: datetime,
        bars_map: Dict[str, pd.DataFrame],
    ) -> MarketData:
        """构建 MarketData"""
        signal_tf = self.config.time.signal_timeframe
        signal_bars = bars_map.get(signal_tf, pd.DataFrame())

        # 确保时区一致性：bars_map 是 tz-naive，将 current_ts 也转为 tz-naive
        current_ts_compare: datetime = current_ts
        if hasattr(current_ts, 'tzinfo') and current_ts.tzinfo is not None:
            current_ts_compare = current_ts.replace(tzinfo=None)

        if FeatureFlag.VECTORIZED in self.config.features:
            ts_for_lookup = (
                pd.Timestamp(current_ts_compare)
                if isinstance(current_ts_compare, (date, datetime))
                else current_ts_compare
            )
            if hasattr(ts_for_lookup, 'tz') and ts_for_lookup.tz is not None:
                ts_for_lookup = ts_for_lookup.tz_convert(None)

            visible_bars_map: Dict[str, pd.DataFrame] = {}
            for tf, df in bars_map.items():
                if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                    visible_bars_map[tf] = df
                    continue

                if ts_for_lookup in df.index:
                    rows = df.loc[ts_for_lookup]
                    if isinstance(rows, pd.Series):
                        rows = rows.to_frame().T
                    visible_bars_map[tf] = rows
                else:
                    visible_bars_map[tf] = df.head(0)

            visible_bars = visible_bars_map.get(signal_tf, pd.DataFrame())
            bars_all = pd.concat(visible_bars_map.values()) if visible_bars_map else pd.DataFrame()

            return MarketData(
                bars=visible_bars,
                bars_map=visible_bars_map,
                bars_all=bars_all,
            )

        # 截断到当前时间（避免前视）
        if signal_bars.empty or not isinstance(signal_bars.index, pd.DatetimeIndex):
            visible_bars = signal_bars
        else:
            visible_bars = signal_bars[signal_bars.index <= current_ts_compare]

        # 截断所有时间框架
        visible_bars_map = {}
        for tf, df in bars_map.items():
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                visible_bars_map[tf] = df
            else:
                visible_bars_map[tf] = df[df.index <= current_ts_compare]
        
        return MarketData(
            bars=visible_bars,
            bars_map=visible_bars_map,
            bars_all=pd.concat(visible_bars_map.values()) if visible_bars_map else pd.DataFrame(),
        )
    
    async def _generate_signals(
        self,
        strategy: StrategyType,
        md: MarketData,
        ctx: RunContext,
    ) -> List[Signal]:
        """Generate signals with protective filtering."""
        # Freqtrade 协议
        if FeatureFlag.FREQTRADE_PROTOCOL in self.config.features:
            # 使用 isinstance 检查类型，避免类型检查器报错
            if isinstance(strategy, FreqtradeStrategy):
                await strategy.bot_loop_start()

            # 从向量化结果获取
            if FeatureFlag.VECTORIZED in self.config.features and self._context:
                raw_signals = self._get_signals_from_vectorized(strategy, md)
                if raw_signals:
                    # 防护层：过滤与持仓状态冲突的信号
                    return self._filter_signals_by_position(raw_signals)

        # 回退到原生接口
        if hasattr(strategy, 'generate_signals'):
            raw_signals = strategy.generate_signals(md, ctx)
            # 防护层：过滤与持仓状态冲突的信号
            return self._filter_signals_by_position(raw_signals)

        return []

    def _filter_signals_by_position(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals that conflict with current position state.

        This is a protective layer that prevents:
        - Duplicate LONG signals for symbols already held
        - FLAT signals for symbols not held (no position to close)
        """
        if not signals or self._execution_provider is None:
            return signals

        portfolio = self._execution_provider.get_portfolio_state()
        valid_signals = []

        for signal in signals:
            current_qty = portfolio.positions.get(signal.symbol, 0.0)

            # Skip LONG signal if already holding this symbol
            if signal.action == ActionType.LONG and abs(current_qty) > 1e-8:
                self.logger.debug(
                    f"Filtered duplicate LONG: {signal.symbol} "
                    f"(already holding {current_qty:.2f})"
                )
                continue

            # Skip FLAT signal if not holding this symbol
            if signal.action == ActionType.FLAT and abs(current_qty) < 1e-8:
                self.logger.debug(
                    f"Filtered invalid FLAT: {signal.symbol} "
                    f"(no position to close)"
                )
                continue

            valid_signals.append(signal)

        return valid_signals

    def _select_row_for_ts(
        self,
        bars: pd.DataFrame,
        ts_for_lookup: Any,
        ts_date: Any,
    ) -> Optional[pd.Series]:
        if bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
            return None

        if isinstance(ts_for_lookup, pd.Timestamp):
            if ts_for_lookup in bars.index:
                row = bars.loc[ts_for_lookup]
                return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

        by_date = bars.attrs.get("_by_date_map")
        if by_date is None:
            by_date = {}
            tail_rows = bars.groupby(bars.index.date).tail(1)
            for dt, row in tail_rows.iterrows():
                by_date[dt.date()] = row
            bars.attrs["_by_date_map"] = by_date

        row = by_date.get(ts_date)
        if row is not None:
            return row

        if isinstance(ts_for_lookup, pd.Timestamp):
            pos = bars.index.searchsorted(ts_for_lookup, side="right") - 1
            if pos >= 0:
                return bars.iloc[pos]

        return None
    
    def _get_signals_from_vectorized(
        self,
        strategy: StrategyType,
        md: MarketData,
    ) -> List[Signal]:
        """从向量化结果获取信号"""
        if self._context is None:
            return []

        strategy_data = self._context.vectorized_data.get(
            strategy.strategy_id, {}
        )
        current_ts = self._context.current_ts
        if current_ts is None:
            return []

        # 转换 current_ts 为 Timestamp 以匹配 bars 索引
        ts_for_lookup = pd.Timestamp(current_ts) if isinstance(current_ts, (date, datetime)) else current_ts

        # 移除时区信息以匹配 tz-naive 的 bars 索引
        if hasattr(ts_for_lookup, 'tz') and ts_for_lookup.tz is not None:
            ts_for_lookup = ts_for_lookup.tz_convert(None)

        # 标准化为 date 以匹配可能有不同时间组件的 bars 索引
        # 数据的时间戳可能是 04:00:00 或其他时间，而时间轴是 00:00:00
        # 我们需要找到当天的数据，所以按日期匹配
        ts_date = ts_for_lookup.date() if hasattr(ts_for_lookup, 'date') else ts_for_lookup
        
        # 转换为 datetime
        if isinstance(ts_for_lookup, pd.Timestamp):
            ts_datetime = ts_for_lookup.to_pydatetime()
            if ts_datetime is None or (hasattr(ts_datetime, 'year') is False):
                return []
        else:
            ts_datetime = current_ts

        signals = []

        # 检查是否启用组合选择
        use_portfolio_selection = (
            FeatureFlag.PORTFOLIO_SELECTION in self.config.features and
            hasattr(strategy, 'get_portfolio_selection_signal')
        )

        # 获取当前持仓
        current_holding: Optional[str] = None
        if self._execution_provider:
            portfolio = self._execution_provider.get_portfolio_state()
            for symbol, qty in portfolio.positions.items():
                if abs(qty) > 1e-8:
                    current_holding = symbol
                    break

        # 组合选择：获取应该持有/平仓的标的列表
        portfolio_decisions: Dict[str, str] = {}  # symbol -> action (LONG/FLAT/KEEP)
        if use_portfolio_selection:
            # 使用原始时间戳（而不是日期）传递给策略，让策略自己处理
            decisions = strategy.get_portfolio_selection_signal(
                strategy_data,
                ts_for_lookup,
                current_holding,
            )
            for symbol, decision in decisions.items():
                portfolio_decisions[symbol] = decision.get('action', 'KEEP')

            # Debug logging
            if portfolio_decisions:
                self.logger.debug(f"[{ts_for_lookup}] Portfolio decisions: {portfolio_decisions}")

        # 生成信号
        for symbol, bars in strategy_data.items():
            row = self._select_row_for_ts(bars, ts_for_lookup, ts_date)
            if row is None:
                self.logger.debug(f"[{ts_for_lookup}] Symbol {symbol} has no data for {ts_date}")
                continue

            # 如果启用了组合选择，使用决策结果
            if use_portfolio_selection:
                # 只在有portfolio决策时生成信号（避免在非轮动日使用enter_long/exit_long）
                if symbol in portfolio_decisions:
                    action = portfolio_decisions[symbol]
                    if action == 'LONG':
                        signal = Signal(
                            ts_utc=ts_datetime,
                            symbol=symbol,
                            strategy_id=strategy.strategy_id,
                            action=ActionType.LONG,
                            strength=row.get('enter_tag', 1.0),
                        )
                        signals.append(signal)
                        self.logger.debug(f"[{ts_for_lookup}] Generated LONG signal for {symbol}")
                    elif action == 'FLAT':
                        signal = Signal(
                            ts_utc=ts_datetime,
                            symbol=symbol,
                            strategy_id=strategy.strategy_id,
                            action=ActionType.FLAT,
                            strength=row.get('exit_tag', 1.0),
                        )
                        signals.append(signal)
                        self.logger.debug(f"[{ts_for_lookup}] Generated FLAT signal for {symbol}")
                    # KEEP 不生成信号
            else:
                # 默认行为：每只ETF独立判断（仅在未启用portfolio selection时使用）
                enter_long = row.get('enter_long', 0)
                if enter_long is True or enter_long == 1:
                    signals.append(Signal(
                        ts_utc=ts_datetime,
                        symbol=symbol,
                        strategy_id=strategy.strategy_id,
                        action=ActionType.LONG,
                        strength=row.get('enter_tag', 1.0),
                    ))

                exit_long = row.get('exit_long', 0)
                if exit_long is True or exit_long == 1:
                    signals.append(Signal(
                        ts_utc=ts_datetime,
                        symbol=symbol,
                        strategy_id=strategy.strategy_id,
                        action=ActionType.FLAT,
                        strength=row.get('exit_tag', 1.0),
                    ))

        if signals:
            self.logger.debug(f"[{ts_for_lookup}] Generated {len(signals)} signals")

        return signals
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        if self._context is None or self._context.current_ts is None:
            return 0.0
        
        # 从执行提供者的_current_prices字典获取价格
        if hasattr(self._execution_provider, '_current_prices'):
            current_prices = self._execution_provider._current_prices
            if symbol in current_prices:
                price = current_prices[symbol]
                if price > 0:
                    return price
        
        # 如果无法获取价格，返回默认值
        self.logger.warning(f"无法获取 {symbol} 的当前价格，使用默认值100")
        return 100.0

    async def _convert_to_orders(
        self,
        strategy: StrategyType,
        signals: List[Signal],
        portfolio: PortfolioState,
        ctx: RunContext,
    ) -> List[OrderIntent]:
        """转换信号为订单"""
        if not signals:
            return []

        risk = RiskState(
            daily_loss_limit=None,
            max_position_size=self.config.trade.max_position_size,
            max_positions=self.config.trade.max_positions,
            blacklist=[],
        )

        if hasattr(strategy, 'size_positions'):
            return strategy.size_positions(signals, portfolio, risk, ctx)

        # 默认订单转换：信号 -> 订单意图
        orders = []
        
        # 计算总资产（注意：portfolio.equity = cash + positions_value，不需要再加 cash）
        total_value = portfolio.equity
        position_pct = getattr(strategy, 'max_position_size', 0.2)
        stake_amount = self.config.trade.stake_amount
        if stake_amount is None:
            stake_amount = total_value * position_pct

        for signal in signals:
            # 根据信号类型确定订单方向
            if signal.action == ActionType.LONG:
                side = OrderSide.BUY
            elif signal.action == ActionType.SHORT:
                side = OrderSide.SELL  # 做空：卖出
            else:  # FLAT 或其他
                side = OrderSide.SELL  # 平仓：卖出
            
            # 获取当前价格来计算股票数量
            current_price = self._get_current_price(signal.symbol)
            if current_price <= 0:
                self.logger.warning(f"无法获取 {signal.symbol} 的价格，跳过订单")
                continue
            
            # 获取当前持仓数量
            current_qty = portfolio.positions.get(signal.symbol, 0.0)
            
            # 对于FLAT信号，清空当前持仓；对于LONG信号，使用固定金额
            if signal.action == ActionType.FLAT:
                # 平仓信号：清空当前持仓
                qty = abs(current_qty)
            else:
                # 开仓信号：计算买入数量
                qty = stake_amount / current_price
            
            # 如果没有持仓或数量为0，跳过
            if qty <= 1e-8:
                continue
            
            order = OrderIntent(
                ts_utc=signal.ts_utc,
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                side=side,
                order_type=OrderType.MKT,
                qty=qty,
            )
            orders.append(order)

        return orders
    
    async def _execute_order(
        self,
        intent: OrderIntent,
        current_ts: datetime,
        md: MarketData,
    ) -> None:
        """执行订单"""
        # 收集当前价格
        tick_bars = {}
        for symbol in md.bars["symbol"].unique():
            symbol_rows = md.bars[
                (md.bars.index == current_ts) & (md.bars["symbol"] == symbol)
            ]
            if symbol_rows.empty:
                self.logger.debug(
                    f"Missing bar for {symbol} at {current_ts.strftime('%Y-%m-%d')}"
                )
                continue

            tick_bars[symbol] = symbol_rows.iloc[-1]

        # 先更新价格（供 place_order 使用）
        self._execution_provider.update_account_state(
            current_ts, 
            {s: b['close'] for s, b in tick_bars.items()}
        )

        # 执行订单
        self._execution_provider.place_order(intent, current_ts)
        
        # 撮合
        fills = self._execution_provider.on_tick(current_ts, tick_bars)
        if self._context:
            new_fills: List[Fill] = []
            for fill in fills:
                fill_id = getattr(fill, "fill_id", None)
                if fill_id and fill_id in self._context.seen_fill_ids:
                    continue
                if fill_id:
                    self._context.seen_fill_ids.add(fill_id)
                new_fills.append(fill)
            self._context.all_fills.extend(new_fills)
    
    async def _check_stoploss(
        self,
        portfolio: PortfolioState,
        current_ts: datetime,
        md: MarketData,
    ) -> None:
        """检查止损/ROI"""
        if self._stoploss_manager is None or self._context is None:
            return
            
        for symbol, qty in list(portfolio.positions.items()):
            if abs(qty) < 1e-8:
                continue
            
            # 获取当前价格
            s_bars = md.bars[md.bars['symbol'] == symbol]
            if s_bars.empty:
                continue
            
            current_price = s_bars.iloc[-1]['close']
            avg_price = portfolio.avg_price.get(symbol, 0.0)
            
            # 查找入场时间
            entry_time = current_ts
            for fill in reversed(self._context.all_fills):
                if fill.symbol == symbol and fill.side == OrderSide.BUY:
                    entry_time = fill.ts_fill_utc
                    break
            
            # 检查离场
            exit_reason = self._stoploss_manager.check_exit(
                symbol=symbol,
                current_price=current_price,
                avg_price=avg_price,
                entry_time=entry_time,
                current_time=current_ts,
                roi_table=self.config.stoploss.minimal_roi,
            )
            
            if exit_reason:
                self.logger.info(f"[STOPLEVEL] {symbol} exit: {exit_reason.reason}")
                # 生成平仓订单
                await self._execute_stoploss_exit(symbol, qty, current_ts, md)
    
    async def _execute_stoploss_exit(
        self,
        symbol: str,
        qty: float,
        current_ts: datetime,
        md: MarketData,
    ) -> None:
        """执行止损平仓"""
        intent = OrderIntent(
            ts_utc=current_ts,
            symbol=symbol,
            side=OrderSide.SELL,
            qty=abs(qty),
            order_type=OrderType.MKT,
            strategy_id="stoploss_manager",
        )
        await self._execute_order(intent, current_ts, md)
    
    def _check_protections(
        self,
        portfolio: PortfolioState,
        current_ts: datetime,
    ) -> None:
        """检查保护机制"""
        if self._context is None or not self.config.protection.enabled:
            return
        
        # 冷却期检查
        for symbol in list(self._context.cooldown_periods.keys()):
            if symbol in self._context.cooldown_periods:
                last_sell = None
                for fill in reversed(self._context.all_fills):
                    if fill.symbol == symbol and fill.side == OrderSide.SELL:
                        last_sell = fill.ts_fill_utc
                        break
                
                if last_sell:
                    cooldown_end = last_sell + timedelta(
                        minutes=self._context.cooldown_periods[symbol]
                    )
                    if current_ts < cooldown_end:
                        self.logger.info(f"[COOLDOWN] {symbol} blocked")
    
    def _record_portfolio_daily(
        self,
        portfolio: PortfolioState,
        current_ts: datetime,
    ) -> None:
        """记录每日净值"""
        if self._context is None:
            return
            
        # 处理 current_ts 可能是 date 或 datetime 类型的情况
        if isinstance(current_ts, date) and not isinstance(current_ts, datetime):
            # current_ts 是 date 类型
            current_date = current_ts
        elif hasattr(current_ts, 'date'):
            # current_ts 是 datetime 类型，有 date() 方法
            current_date = current_ts.date()
        else:
            # 其他情况，直接使用
            current_date = current_ts
            
        trading_day_dates = set()
        for d in self._context.trading_days:
            if isinstance(d, date) and not isinstance(d, datetime):
                # d 是 date 类型
                trading_day_dates.add(d)
            elif hasattr(d, 'date'):
                # d 是 datetime 类型，有 date() 方法
                trading_day_dates.add(d.date())
            else:
                # 其他情况，直接使用
                trading_day_dates.add(d)
                
        is_trading_day = current_date in trading_day_dates
        
        if self.config.time.clock_mode == "bar" or is_trading_day:
            self._context.portfolio_daily.append({
                "timestamp": current_ts,
                "date": current_date,
                "equity": portfolio.equity,
                "cash": portfolio.cash,
            })

    def _get_benchmark_symbol(self, ctx: Optional[RunContext]) -> str:
        """Determine benchmark symbol from context or fallback"""
        default_symbol = "SPY"
        if ctx is None:
            return default_symbol

        config = ctx.config
        if not isinstance(config, dict):
            return default_symbol

        data_config = config.get("data")
        if isinstance(data_config, dict):
            return data_config.get("benchmark_symbol", default_symbol) or default_symbol

        return default_symbol

    def _format_fills_for_metrics(self) -> List[Dict[str, Any]]:
        """Convert Fill dataclasses into simple trade dicts"""
        if self._context is None:
            return []

        formatted: List[Dict[str, Any]] = []
        for fill in self._context.all_fills:
            formatted.append({
                "timestamp": fill.ts_fill_utc.isoformat() if fill.ts_fill_utc else None,
                "symbol": fill.symbol,
                "order_type": getattr(fill.side, "value", str(fill.side)).lower(),
                "price": fill.price,
                "size": abs(fill.qty),
                "fill_id": fill.fill_id,
            })
        return formatted

    def _calculate_performance_metrics(self, ctx: Optional[RunContext], benchmark_symbol: str = "SPY", initial_capital: float = 100000) -> Dict[str, Any]:
        """Compute performance metrics via EnhancedAnalyzer"""
        default_metrics = {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
        }

        if self._context is None or not self._context.portfolio_daily:
            return default_metrics

        portfolio_df = pd.DataFrame(self._context.portfolio_daily)
        if portfolio_df.empty or "equity" not in portfolio_df.columns:
            return default_metrics

        if "timestamp" in portfolio_df.columns:
            portfolio_df["timestamp"] = pd.to_datetime(
                portfolio_df["timestamp"], errors="coerce"
            )
            portfolio_df = portfolio_df.dropna(subset=["timestamp"])
            if portfolio_df.empty:
                return default_metrics
            portfolio_df = portfolio_df.sort_values("timestamp").set_index("timestamp")
        elif "date" in portfolio_df.columns:
            portfolio_df["date"] = pd.to_datetime(portfolio_df["date"], errors="coerce")
            portfolio_df = portfolio_df.dropna(subset=["date"])
            if portfolio_df.empty:
                return default_metrics
            portfolio_df = portfolio_df.sort_values("date").set_index("date")
        else:
            portfolio_df = portfolio_df.sort_index()

        equity_series = portfolio_df["equity"].astype(float)
        if equity_series.empty or len(equity_series) < 2:
            return default_metrics

        returns = equity_series.pct_change().dropna()
        if returns.empty:
            return default_metrics

        analyzer = EnhancedAnalyzer(benchmark_symbol=benchmark_symbol)

        # 获取benchmark数据
        benchmark_returns = None
        benchmark_equity = None
        try:
            # 获取benchmark的OHLCV数据，与portfolio_df的日期范围对齐
            start_date = portfolio_df.index.min().strftime("%Y-%m-%d")
            end_date = portfolio_df.index.max().strftime("%Y-%m-%d")

            benchmark_data = analyzer.benchmark_analyzer.get_benchmark_data(
                start_date=start_date,
                end_date=end_date,
                data_source="local"  # 优先使用本地缓存数据
            )

            self.logger.info(f"Benchmark data fetched: {benchmark_data is not None}")
            if benchmark_data is not None:
                if isinstance(benchmark_data, pd.DataFrame) and "close" in benchmark_data.columns:
                    # 确保benchmark_data与portfolio_df索引对齐
                    benchmark_close = benchmark_data["close"].reindex(portfolio_df.index, method='ffill')
                    self.logger.info(f"Benchmark close series length: {len(benchmark_close)}, portfolio index length: {len(portfolio_df)}")
                    benchmark_returns = benchmark_close.pct_change()
                    # 第一个值是NaN，设为0（表示初始无变化）
                    if len(benchmark_returns) > 0:
                        benchmark_returns.iloc[0] = 0.0

                    # 生成benchmark净值曲线（以初始净值为基准）
                    benchmark_equity = (1 + benchmark_returns).cumprod() * initial_capital
                    benchmark_equity = benchmark_equity.tolist()
                    self.logger.info(f"Benchmark equity length: {len(benchmark_equity)}, min: {min(benchmark_equity) if benchmark_equity else 'N/A'}, max: {max(benchmark_equity) if benchmark_equity else 'N/A'}")

                elif isinstance(benchmark_data, pd.Series):
                    # 如果返回的是系列数据
                    benchmark_close = benchmark_data.reindex(portfolio_df.index, method='ffill')
                    self.logger.info(f"Benchmark close series (Series) length: {len(benchmark_close)}")
                    benchmark_returns = benchmark_close.pct_change()
                    if len(benchmark_returns) > 0:
                        benchmark_returns.iloc[0] = 0.0
                    benchmark_equity = (1 + benchmark_returns).cumprod() * initial_capital
                    benchmark_equity = benchmark_equity.tolist()
                    self.logger.info(f"Benchmark equity length: {len(benchmark_equity)}")

        except Exception as exc:
            self.logger.warning(f"获取benchmark数据失败: {exc}")
            # 如果获取失败，不创建假的benchmark数据，返回空列表
            benchmark_equity = []

        try:
            comp_results = analyzer.comprehensive_analysis(
                returns=returns,
                portfolio_value=equity_series,
                trades=self._format_fills_for_metrics(),
                benchmark_returns=benchmark_returns,
                benchmark_equity=benchmark_equity,
            )

            # 调试：检查comp_results
            self.logger.debug(f"comp_results keys: {list(comp_results.keys()) if comp_results else 'None'}")
            if comp_results and 'drawdown_series' in comp_results:
                self.logger.debug(f"drawdown_series length: {len(comp_results['drawdown_series'])}")
        except Exception as exc:
            self.logger.warning(f"性能指标计算失败: {exc}")
            return default_metrics

        result_dict = {
            "annual_return": comp_results.get("annual_return", 0.0),
            "annual_volatility": comp_results.get("annual_volatility", 0.0),
            "sharpe_ratio": comp_results.get("sharpe_ratio", 0.0),
            "sortino_ratio": comp_results.get("sortino_ratio", 0.0),
            "calmar_ratio": comp_results.get("calmar_ratio", 0.0),
            "max_drawdown": comp_results.get("max_drawdown", 0.0),
            "profit_factor": comp_results.get("profit_factor", 0.0),
            "win_rate": comp_results.get("win_rate", 0.0),
            "benchmark_symbol": benchmark_symbol,
            "benchmark_return": comp_results.get("benchmark_return", 0.0),
            "benchmark_volatility": comp_results.get("benchmark_volatility", 0.0),
            "alpha": comp_results.get("alpha", 0.0),
            "beta": comp_results.get("beta", 1.0),
            "information_ratio": comp_results.get("information_ratio", 0.0),
            "r_squared": comp_results.get("r_squared", 0.0),
            "benchmark_equity": benchmark_equity or [],
            "drawdown_series": comp_results.get("drawdown_series", []),
            "timestamps": pd.to_datetime(portfolio_df.index).strftime("%Y-%m-%d").tolist() if not portfolio_df.empty else [],
        }

        self.logger.info(f"[PERF METRICS] benchmark_symbol={benchmark_symbol}, benchmark_return={result_dict['benchmark_return']:.4f}, benchmark_equity length={len(result_dict['benchmark_equity'])}, timestamps length={len(result_dict['timestamps'])}")
        return result_dict
    
    def _calculate_final_equity_with_market_price(
        self,
        portfolio: PortfolioState,
    ) -> float:
        """使用最后一天的市场价格计算最终权益"""
        if not self._context:
            return portfolio.cash

        # 如果没有持仓，直接返回现金
        if not portfolio.positions:
            return portfolio.cash

        # 获取最后一天的收盘价
        def get_last_close_price(symbol: str) -> Optional[float]:
            """获取指定标的最后一天的收盘价"""
            if not self._bars_map:
                return None

            # 优先使用 signal_timeframe 的数据
            tf = self.config.time.signal_timeframe
            if tf not in self._bars_map:
                tf = next(iter(self._bars_map.keys()), None)
                if not tf:
                    return None

            bars = self._bars_map.get(tf)
            if bars is None or bars.empty:
                return None

            # 筛选该 symbol 的数据
            symbol_bars = bars[bars["symbol"] == symbol] if "symbol" in bars.columns else bars
            if symbol_bars.empty:
                return None

            # 获取最后一天的收盘价
            last_close = symbol_bars["close"].iloc[-1]
            return float(last_close) if pd.notna(last_close) else None

        # 使用最后一天收盘价计算持仓价值
        positions_value = 0.0

        # 调试：检查 bars_map 状态
        if not self._bars_map:
            self.logger.warning("[FINAL_EQUITY] bars_map 为空，无法使用收盘价计算")
        else:
            self.logger.debug(f"[FINAL_EQUITY] bars_map 包含 {len(self._bars_map)} 个 timeframe")

        for symbol, qty in portfolio.positions.items():
            if abs(qty) < 1e-8:
                continue

            # 优先使用最后一天收盘价
            final_price = get_last_close_price(symbol)

            if final_price is not None and final_price > 0:
                self.logger.info(f"[FINAL_EQUITY] {symbol}: 使用收盘价 {final_price:.4f}, qty={qty:.4f}")
            else:
                # 如果无法从 bars_map 获取，使用成本价作为 fallback
                avg_price = portfolio.avg_price.get(symbol, 0.0)
                if avg_price > 0:
                    final_price = avg_price
                    self.logger.info(f"[FINAL_EQUITY] {symbol}: 使用成本价 {avg_price:.4f}, qty={qty:.4f}")
                else:
                    # 如果没有成本价，尝试最后成交价
                    for fill in reversed(self._context.all_fills):
                        if fill.symbol == symbol:
                            final_price = fill.price
                            self.logger.info(f"[FINAL_EQUITY] {symbol}: 使用最后成交价 {fill.price:.4f}, qty={qty:.4f}")
                            break

            if final_price and final_price > 0:
                positions_value += abs(qty) * final_price
        
        final_equity = portfolio.cash + positions_value
        
        self.logger.debug(
            f"Final equity calculation: cash={portfolio.cash:.2f}, "
            f"positions_value={positions_value:.2f}, total={final_equity:.2f}"
        )
        
        return final_equity
    
    def _finalize_backtest(
        self,
        portfolio: PortfolioState,
    ) -> tuple[List[Dict], List[Fill]]:
        """
        统一回测最终化处理

        在回测结束时，强制平仓所有未平仓持仓，计算真实盈亏。
        这确保了胜率计算包含所有交易的盈亏，而不仅仅是已平仓交易。

        Returns:
            closed_trades: 包含所有交易（含未平仓强制平仓）的盈亏列表
            final_fills: 强制平仓产生的 Fill 记录列表
        """
        from src.core.types import Fill, OrderSide
        from src.utils.cli_printer import CLIPrinter
        import uuid
        from datetime import datetime

        if self._context is None:
            return [], []

        # 1. 首先匹配已平仓交易（使用原始 fill.price）
        closed_trades, open_lots = CLIPrinter.match_trades(self._context.all_fills)

        # 2. 使用实际每日收盘价重新计算所有交易的 PnL
        def get_close_price(symbol: str, ts: datetime) -> Optional[float]:
            """获取指定日期的收盘价（数据已统一为 tz-naive）"""
            if not self._bars_map:
                return None

            tf = self.config.time.signal_timeframe
            if tf not in self._bars_map:
                tf = next(iter(self._bars_map.keys()), None)
                if not tf:
                    return None

            bars = self._bars_map.get(tf)
            if bars is None or bars.empty:
                return None

            # 筛选该 symbol 的数据
            symbol_bars = bars[bars["symbol"] == symbol] if "symbol" in bars.columns else bars
            if symbol_bars.empty:
                return None

            # 找到最接近该日期的收盘价
            if hasattr(symbol_bars.index, 'date'):
                symbol_bars = symbol_bars.sort_index()

                # 统一时区：使用 pandas Timestamp 进行比较
                # 确保两边都是 tz-naive
                idx = symbol_bars.index
                ts_norm = pd.Timestamp(ts)
                if ts_norm.tz is not None:
                    ts_norm = ts_norm.tz_localize(None)

                # 确保索引也是 tz-naive
                if idx.tz is not None:
                    idx = idx.tz_localize(None)

                # 找到日期 <= ts 的最后一条
                mask = idx <= ts_norm
                if mask.any():
                    return float(symbol_bars.loc[mask, "close"].iloc[-1])

            return None

        # 重新计算每笔交易的 PnL（使用实际收盘价）
        recalculated_trades = []
        for trade in closed_trades:
            symbol = trade.get("symbol")
            entry_ts = trade.get("ts")
            exit_ts = trade.get("exit_ts")

            self.logger.debug(f"[DEBUG] {symbol}: entry_ts={entry_ts} (type={type(entry_ts)}), exit_ts={exit_ts}")

            if entry_ts and exit_ts:
                # 获取实际收盘价
                entry_price = get_close_price(symbol, entry_ts)
                exit_price = get_close_price(symbol, exit_ts)

                if entry_price and exit_price and entry_price > 0:
                    trade["entry_price"] = entry_price
                    trade["exit_price"] = exit_price
                    trade["pnl"] = (exit_price - entry_price) * trade.get("qty", 0)
                    trade["ret_pct"] = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                    self.logger.debug(
                        f"[RECALC] {symbol}: entry={entry_price:.4f} @ {entry_ts}, "
                        f"exit={exit_price:.4f} @ {exit_ts}, pnl={trade['pnl']:.2f}"
                    )

            recalculated_trades.append(trade)

        # 3. 检查是否有未平仓持仓
        final_fills: List[Fill] = []
        if not portfolio.positions:
            return recalculated_trades, final_fills

        # 4. 获取最后一天的日期作为平仓时间
        last_day = None
        if self._context.trading_days:
            last_day = self._context.trading_days[-1]
            if isinstance(last_day, datetime):
                pass
            elif hasattr(last_day, 'date'):
                last_day = datetime.combine(last_day, datetime.min.time())
            else:
                last_day = datetime.combine(last_day, datetime.min.time())
        else:
            last_day = datetime.now()

        last_ts = self._context.all_fills[-1].ts_fill_utc if self._context.all_fills else last_day

        # 5. 处理未平仓持仓（使用最后一天收盘价）
        for symbol, qty in portfolio.positions.items():
            if abs(qty) < 1e-8:
                continue

            # 获取入场时间
            entry_ts = None
            if symbol in open_lots and open_lots[symbol]:
                lot = open_lots[symbol][0]
                entry_ts = lot.get("ts")

            if not entry_ts:
                # 查找最后一次买入时间
                for fill in reversed(self._context.all_fills):
                    if fill.symbol == symbol and fill.side == OrderSide.BUY:
                        entry_ts = fill.ts_fill_utc
                        break

            if not entry_ts:
                self.logger.warning(f"无法获取 {symbol} 的入场时间，跳过强制平仓")
                continue

            # 获取入场和出场收盘价
            entry_price = get_close_price(symbol, entry_ts)
            exit_price = get_close_price(symbol, last_ts)

            if not entry_price or entry_price <= 0:
                self.logger.warning(f"无法获取 {symbol} 的入场价格，跳过强制平仓")
                continue

            if not exit_price or exit_price <= 0:
                exit_price = entry_price  # fallback

            # 创建强制平仓的 Fill 记录
            close_fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=OrderSide.SELL if qty > 0 else OrderSide.BUY,
                qty=abs(qty),
                price=exit_price,
                fee=0.0,
                ts_fill_utc=last_ts,
                metadata={
                    "exit_reason": "FORCE_CLOSE_AT_END",
                    "is_final_close": True,
                }
            )
            final_fills.append(close_fill)

            # 计算未实现盈亏
            pnl = (exit_price - entry_price) * abs(qty) if qty > 0 else (entry_price - exit_price) * abs(qty)
            ret_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

            duration = None
            if entry_ts and last_ts:
                duration = (last_ts - entry_ts).total_seconds()

            trade = {
                "symbol": symbol,
                "qty": abs(qty),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "ret_pct": ret_pct,
                "duration": duration,
                "entry_tag": "LONG",
                "exit_reason": "FORCE_CLOSE_AT_END",
                "exit_ts": last_ts,
                "is_force_close": True,
            }
            recalculated_trades.append(trade)

            self.logger.info(
                f"[FINALIZE] {symbol}: qty={abs(qty):.4f}, entry={entry_price:.4f}, "
                f"exit={exit_price:.4f} (market close), pnl={pnl:.2f}"
            )

        return recalculated_trades, final_fills

    def _generate_result(
        self,
        portfolio: PortfolioState,
        strategies: List[StrategyType],
        ctx: Optional[RunContext],
    ) -> BacktestResult:
        """生成回测结果"""
        if self._context is None:
            return BacktestResult()

        initial_capital = self.config.trade.initial_capital

        # 调试：打印 portfolio 状态
        self.logger.debug(f"[DEBUG] portfolio.cash = {portfolio.cash:.2f}")
        self.logger.debug(f"[DEBUG] portfolio.equity = {portfolio.equity:.2f}")
        self.logger.debug(f"[DEBUG] portfolio.positions = {portfolio.positions}")
        self.logger.debug(f"[DEBUG] portfolio.avg_price = {portfolio.avg_price}")

        # 统一回测最终化：处理未平仓持仓
        closed_trades, final_fills = self._finalize_backtest(portfolio)

        # 将最终化的 fills 添加到 all_fills（用于完整记录）
        all_fills_with_final = list(self._context.all_fills) + final_fills

        # 计算交易统计
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in closed_trades if t.get('pnl', 0) < 0])
        draw_trades = len([t for t in closed_trades if t.get('pnl', 0) == 0])

        # 计算胜率
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # 计算盈亏
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        profits = [t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) > 0]
        losses = [abs(t.get('pnl', 0)) for t in closed_trades if t.get('pnl', 0) < 0]
        profit_factor = sum(profits) / sum(losses) if sum(losses) > 0 else float('inf')

        self.logger.info(
            f"[FINALIZE SUMMARY] total_trades={total_trades}, winning={winning_trades}, "
            f"losing={losing_trades}, draw={draw_trades}, win_rate={win_rate:.2%}, total_pnl={total_pnl:.2f}"
        )

        # 修复：使用最后一天的市场价格重新计算 equity，而非成本价
        final_equity = self._calculate_final_equity_with_market_price(portfolio)

        total_return = (final_equity / initial_capital) - 1

        self.logger.debug(f"[DEBUG] final_equity = {final_equity:.2f}")
        self.logger.debug(f"[DEBUG] total_return = {total_return:.4f}")

        performance = self._calculate_performance_metrics(ctx)
        annualized_return = performance.get("annual_return", 0.0)
        volatility = performance.get("annual_volatility", 0.0)
        sharpe_ratio = performance.get("sharpe_ratio", 0.0)
        sortino_ratio = performance.get("sortino_ratio", 0.0)
        calmar_ratio = performance.get("calmar_ratio", 0.0)
        raw_max_drawdown = performance.get("max_drawdown", 0.0)
        max_drawdown = abs(raw_max_drawdown)
        max_drawdown_pct = max_drawdown
        perf_profit_factor = performance.get("profit_factor", 0.0)

        # 使用引擎计算的 profit_factor（如果有效）
        profit_factor = perf_profit_factor if perf_profit_factor > 0 else profit_factor

        unique_fill_ids = {
            fill.fill_id for fill in all_fills_with_final
            if getattr(fill, "fill_id", None)
        }
        fills_count = len(unique_fill_ids)

        profits = [t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) > 0]
        losses = [abs(t.get('pnl', 0)) for t in closed_trades if t.get('pnl', 0) < 0]
        manual_profit_factor = sum(profits) / sum(losses) if sum(losses) > 0 else float('inf')
        if profit_factor == 0.0:
            profit_factor = manual_profit_factor

        # 格式化 timerange
        trading_days = self._context.trading_days
        if trading_days is not None and len(trading_days) > 0:
            first_day = trading_days[0]
            last_day = trading_days[-1]
            
            # 辅助函数：将日期对象转换为字符串
            def date_to_str(d):
                if isinstance(d, (datetime, pd.Timestamp)):
                    return d.strftime('%Y-%m-%d')
                elif isinstance(d, date):
                    return d.strftime('%Y-%m-%d')
                elif hasattr(d, 'date'):
                    return d.date().strftime('%Y-%m-%d')
                else:
                    return str(d)
            
            start_str = date_to_str(first_day)
            end_str = date_to_str(last_day)
            timerange = f"{start_str}~{end_str}"
        else:
            timerange = ""

        # 获取性能指标（包含benchmark数据）
        benchmark_symbol = self._get_benchmark_symbol(ctx)
        perf_metrics = self._calculate_performance_metrics(ctx, benchmark_symbol, initial_capital)

        return BacktestResult(
            run_id=self._context.run_id,
            strategy_name=strategies[0].strategy_id if strategies else "unknown",
            timeframe=self.config.time.signal_timeframe,
            timerange=timerange,
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return * 100,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            benchmark_symbol=perf_metrics.get("benchmark_symbol", ""),
            benchmark_return=perf_metrics.get("benchmark_return", 0.0),
            benchmark_volatility=perf_metrics.get("benchmark_volatility", 0.0),
            benchmark_equity=perf_metrics.get("benchmark_equity", []),
            drawdown_series=perf_metrics.get("drawdown_series", []),
            alpha=perf_metrics.get("alpha", 0.0),
            beta=perf_metrics.get("beta", 1.0),
            information_ratio=perf_metrics.get("information_ratio", 0.0),
            r_squared=perf_metrics.get("r_squared", 0.0),
            signals=[s.__dict__ for s in self._context.all_signals],
            orders=[o.__dict__ for o in self._context.all_orders],
            fills=[f.__dict__ for f in all_fills_with_final],
            trades=closed_trades,
            portfolio_daily=self._context.portfolio_daily,
        )
    
    async def _save_result(
        self,
        result: BacktestResult,
        ctx: RunContext,
    ) -> None:
        """保存回测结果"""
        if self._store is None:
            return
        
        self._store.write_dataset(
            dataset="backtest_results",
            df=pd.DataFrame(result.portfolio_daily),
            partition={"run_id": result.run_id},
            data_version=getattr(ctx, 'data_version', '1.0'),
        )

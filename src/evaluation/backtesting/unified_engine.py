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
from typing import List, Dict, Optional, Any, Union
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
from src.utils.logger import get_logger
from src.data.storage.parquet_store import ParquetDataFrameStore
from src.data.adapter.factory import adapter_factory
from src.data.providers.base import DataFeed, HistoricalDataFeed
from src.execution.providers.base import ExecutionProvider, SimulatedExecutionProvider, ExecutionProviderConfig
from src.core.risk_manager import RiskManager
from .config import UnifiedConfig, FeatureFlag
from .stoploss_manager import StoplossManager
from .cost_model import CostModel
from .config import BacktestResult


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
        
        self._event_engine = event_engine or UnifiedEventEngine()
        self._store = store
        
        # 内部状态
        self._context: Optional[BacktestContext] = None
        self._lookahead_warned: set = set()
        
        # 特性组件初始化
        self._init_feature_components()
    
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
        
        # 向量化预计算
        if FeatureFlag.VECTORIZED in self.config.features:
            self._context.vectorized_data = await self._vectorized_precompute(
                strategies, bars_map, universe
            )
        
        # 生成时间轴
        trading_days = trading_calendar.get_trading_days(start, end)
        self._context.trading_days = trading_days
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
        result = self._generate_result(portfolio, strategies)
        
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
        
        for strategy in strategies:
            strategy_data: Dict[str, pd.DataFrame] = {}
            main_tf = getattr(strategy, "timeframe", self.config.time.signal_timeframe)
            main_bars = bars_map.get(main_tf, pd.DataFrame())
            
            for symbol in universe:
                symbol_bars = main_bars[main_bars["symbol"] == symbol].copy()
                if symbol_bars.empty:
                    continue
                
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
            if hasattr(strategy, '_vectorized_data'):
                strategy._vectorized_data = strategy_data
        
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
        
        if hasattr(strategy, 'bot_start'):
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

        # 截断到当前时间（避免前视）
        visible_bars = signal_bars[signal_bars.index <= current_ts_compare]

        # 截断所有时间框架
        visible_bars_map = {
            tf: df[df.index <= current_ts_compare]
            for tf, df in bars_map.items()
        }
        
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
        """生成信号"""
        # Freqtrade 协议
        if FeatureFlag.FREQTRADE_PROTOCOL in self.config.features:
            if hasattr(strategy, 'bot_loop_start'):
                await strategy.bot_loop_start()
            
            # 从向量化结果获取
            if FeatureFlag.VECTORIZED in self.config.features and self._context:
                signals = self._get_signals_from_vectorized(strategy, md)
                if signals:
                    return signals
        
        # 回退到原生接口
        if hasattr(strategy, 'generate_signals'):
            return strategy.generate_signals(md, ctx)
        
        return []
    
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
        
        # 转换为 datetime
        if isinstance(ts_for_lookup, pd.Timestamp):
            ts_datetime = ts_for_lookup.to_pydatetime()
            if ts_datetime is None or (hasattr(ts_datetime, 'year') is False):
                return []
        else:
            ts_datetime = current_ts

        signals = []

        for symbol, bars in strategy_data.items():
            if ts_for_lookup not in bars.index:
                continue

            row = bars.loc[ts_for_lookup]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            
            # 检查进场信号
            # 支持 boolean True 和整数 1
            enter_long = row.get('enter_long', 0)
            if enter_long is True or enter_long == 1:
                signals.append(Signal(
                    ts_utc=ts_datetime,  # type: ignore[arg-type]
                    symbol=symbol,
                    strategy_id=strategy.strategy_id,
                    action=ActionType.LONG,
                    strength=row.get('enter_tag', 1.0),
                ))

            # 检查离场信号
            exit_long = row.get('exit_long', 0)
            # 支持 boolean True 和整数 1
            if exit_long is True or exit_long == 1:
                signals.append(Signal(
                    ts_utc=ts_datetime,  # type: ignore[arg-type]
                    symbol=symbol,
                    strategy_id=strategy.strategy_id,
                    action=ActionType.FLAT,
                    strength=row.get('exit_tag', 1.0),
                ))

        return signals
    
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
        
        # 计算总资产
        total_value = portfolio.cash + portfolio.equity
        position_pct = getattr(strategy, 'max_position_size', 0.2)
        stake_amount = self.config.trade.stake_amount
        if stake_amount is None:
            stake_amount = total_value * position_pct

        for signal in signals:
            order = OrderIntent(
                ts_utc=signal.ts_utc,
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                side=OrderSide.BUY if signal.action == ActionType.LONG else OrderSide.SELL,
                order_type=OrderType.MKT,
                qty=stake_amount,
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
        self._execution_provider.place_order(intent, current_ts)
        
        # 撮合
        tick_bars = {}
        for symbol in md.bars["symbol"].unique():
            s_bars = md.bars[md.bars.index == current_ts]
            if not s_bars.empty:
                row = s_bars.iloc[-1]
                tick_bars[symbol] = row
        
        fills = self._execution_provider.on_tick(current_ts, tick_bars)
        if self._context:
            self._context.all_fills.extend(fills)
        
        # 更新净值
        self._execution_provider.update_account_state(
            current_ts, 
            {s: b['close'] for s, b in tick_bars.items()}
        )
    
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
                await self._execute_stoploss_exit(symbol, qty, current_ts)
    
    async def _execute_stoploss_exit(
        self,
        symbol: str,
        qty: float,
        current_ts: datetime,
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
        await self._execute_order(intent, current_ts, MarketData(bars=pd.DataFrame()))
    
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
            
        current_date = current_ts.date()
        trading_day_dates = {d.date() for d in self._context.trading_days}
        is_trading_day = current_date in trading_day_dates
        
        if self.config.time.clock_mode == "bar" or is_trading_day:
            self._context.portfolio_daily.append({
                "timestamp": current_ts,
                "date": current_date,
                "equity": portfolio.equity,
                "cash": portfolio.cash,
            })
    
    def _generate_result(
        self,
        portfolio: PortfolioState,
        strategies: List[StrategyType],
    ) -> BacktestResult:
        """生成回测结果"""
        if self._context is None:
            return BacktestResult()
            
        initial_capital = self.config.trade.initial_capital
        final_equity = portfolio.equity
        total_return = (final_equity / initial_capital) - 1
        
        # 计算交易统计
        sell_fills = [f for f in self._context.all_fills if f.side == OrderSide.SELL]
        total_trades = len(sell_fills)
        winning_trades = 0
        
        # 简化计算盈利交易
        for i, fill in enumerate(sell_fills):
            if i > 0:
                prev_fill = sell_fills[i-1]
                if fill.symbol == prev_fill.symbol:
                    if fill.price > prev_fill.price:
                        winning_trades += 1
        
        return BacktestResult(
            run_id=self._context.run_id,
            strategy_name=strategies[0].strategy_id if strategies else "unknown",
            timeframe=self.config.time.signal_timeframe,
            timerange=f"{self._context.trading_days[0]}~{self._context.trading_days[-1]}" if self._context.trading_days is not None and len(self._context.trading_days) > 0 else "",
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return * 100,
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate=winning_trades / total_trades if total_trades > 0 else 0,
            signals=[s.__dict__ for s in self._context.all_signals],
            orders=[o.__dict__ for o in self._context.all_orders],
            fills=[f.__dict__ for f in self._context.all_fills],
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

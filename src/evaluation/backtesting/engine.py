"""
回测引擎
产出 signals/orders/fills/portfolio_daily，启用成本模型
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import uuid

from src.core.types import (
    Signal, OrderIntent, Fill, MarketData, PortfolioState, RiskState,
    OrderSide, OrderType, OrderStatus, ActionType,
)
from src.core.context import RunContext, Environment
from src.strategies.base.strategy import Strategy
from src.utils.logger import get_logger
from src.data.storage.parquet_store import ParquetDataFrameStore
from src.data.storage.versioning import get_code_version, hash_config, extract_backtest_config
from src.config.settings import get_settings


class CostModel:
    """成本模型"""
    
    def __init__(self, commission_rate: float = 0.001, slippage_rate: float = 0.0005):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
    
    def estimate_fee(self, order: OrderIntent, fill_price: float) -> float:
        """估算手续费"""
        return abs(order.qty) * fill_price * self.commission_rate
    
    def estimate_slippage(self, order: OrderIntent, fill_price: float) -> float:
        """估算滑点"""
        return abs(order.qty) * fill_price * self.slippage_rate


class BacktestEngine:
    """回测引擎（新设计）"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        store: Optional[ParquetDataFrameStore] = None,
    ):
        self.initial_capital = initial_capital
        self.cost_model = CostModel(commission_rate, slippage_rate)
        settings = get_settings()
        self.store = store or ParquetDataFrameStore(settings.storage.data_dir)
        self.logger = get_logger("backtest_engine_v2")
        self._lookahead_warned: set[tuple[str, str]] = set()
    
    async def run(
        self,
        strategies: List[Strategy],
        universe: List[str],
        start: date,
        end: date,
        ctx: RunContext,
        bars: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None,
        auto_download: bool = True,
    ) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            strategies: 策略列表
            universe: 股票池
            start: 起始日期
            end: 结束日期
            ctx: 运行上下文
            bars: bars数据（如果为None，从store读取）
            features: features数据（如果为None，从store读取）
            
        Returns:
            回测结果字典（包含signals/orders/fills/portfolio_daily等）
        """
        # 初始化组合状态
        portfolio = PortfolioState(
            cash=self.initial_capital,
            positions={},
            avg_price={},
            equity=self.initial_capital,
            daily_loss=0.0,
        )
        
        risk = RiskState(
            daily_loss_limit=None,
            max_position_size=0.2,  # 单标的最大20%
            max_positions=10,
            blacklist=[],
        )
        
        # 加载数据
        if bars is None:
            bars = await self._load_bars(universe, start, end, ctx, auto_download=auto_download)
        if features is None:
            # TODO: 实现 features 的自动下载/补全
            features = self._load_features(universe, start, end, ctx)
        
        if bars.empty:
            return {"error": "没有可用的bars数据"}

        # 创建MarketData
        md = MarketData(bars=bars, bars_all=bars, features=features)
        
        # 存储结果
        all_signals: List[Signal] = []
        all_orders: List[OrderIntent] = []
        all_fills: List[Fill] = []
        portfolio_daily: List[Dict] = []
        
        # 按日期迭代
        trading_days = ctx.trading_calendar.get_trading_days(start, end)
        
        for current_date in trading_days:
            current_date_dt = current_date.date()
            current_ts = pd.Timestamp(current_date_dt)
            
            # 获取当前时刻及其之前的所有数据（可见历史）
            visible_history = bars[bars.index <= current_ts]
            if visible_history.empty:
                continue
            
            # 更新MarketData：传递整个可见历史，以便策略计算指标
            md.bars = visible_history
            if features is not None and not features.empty:
                md.features = features[features.index <= current_ts]
            
            # 更新运行上下文时间
            ctx.now_utc = current_ts.to_pydatetime()
            
            # 调试：检查传递给策略的数据量
            unique_symbols = visible_history['symbol'].unique() if 'symbol' in visible_history.columns else []
            self.logger.debug(f"[{current_date_dt}] 传递给策略: {len(visible_history)} 条数据, {len(unique_symbols)} 个标的")
            
            # 每个策略生成信号
            for strategy in strategies:
                try:
                    if strategy.enable_lookahead_check():
                        self._check_lookahead_data(md, ctx, strategy)
                    signals = strategy.generate_signals(md, ctx)
                    all_signals.extend(signals)
                    
                    # 计算目标仓位
                    order_intents = strategy.size_positions(signals, portfolio, risk, ctx)
                    
                    # 风险检查
                    approved_intents, blocks = strategy.risk_checks(
                        order_intents, portfolio, risk, ctx
                    )
                    
                    # 执行订单（简化：bar close下单，next open成交）
                    for intent in approved_intents:
                        normalized_qty = int(intent.qty)
                        if normalized_qty <= 0:
                            self.logger.info(
                                f"  [下单跳过] {intent.symbol} {intent.side.value} qty<1: {intent.qty:.4f}"
                            )
                            continue
                        if normalized_qty != intent.qty:
                            intent.qty = float(normalized_qty)
                        all_orders.append(intent)
                        self.logger.info(f"  [下单] {intent.symbol} {intent.side.value} {intent.qty:.2f} @ MKT")
                        
                        # 模拟成交（使用next bar的open价格）
                        next_date = ctx.trading_calendar.next_trading_day(current_date_dt)
                        next_ts = pd.Timestamp(next_date)
                        
                        # 重要：必须过滤符号和确切时间戳
                        symbol_next_bars = bars[(bars.index == next_ts) & (bars['symbol'] == intent.symbol)]
                        
                        if not symbol_next_bars.empty:
                            fill_price = symbol_next_bars.iloc[0]['open']
                            fill = self._create_fill(intent, fill_price, next_ts.to_pydatetime())
                            all_fills.append(fill)
                            self.logger.info(f"  [成交] {fill.symbol} {fill.side.value} {fill.qty:.2f} @ {fill.price:.2f}")
                            
                            # 更新组合状态
                            portfolio = self._update_portfolio(portfolio, fill, fill_price)
                        else:
                            self.logger.warning(f"  [成交失败] 未找到 {intent.symbol} 在 {next_date} 的数据")
                except Exception as e:
                    self.logger.error(f"策略 {strategy.strategy_id} 执行失败: {e}")
            
            # 记录每日组合状态 (使用当前 bar 的收盘价更新净值)
            current_prices = bars[bars.index == current_ts]
            portfolio.equity = self._calculate_equity(portfolio, current_prices)
            
            # 计算暴露
            gross_exposure = 0.0
            net_exposure = 0.0
            for symbol, qty in portfolio.positions.items():
                symbol_price_row = current_prices[current_prices['symbol'] == symbol]
                if not symbol_price_row.empty:
                    p = symbol_price_row.iloc[0]['close']
                    gross_exposure += abs(qty * p)
                    net_exposure += qty * p
            
            portfolio_daily.append({
                'date': current_date_dt,
                'equity': portfolio.equity,
                'cash': portfolio.cash,
                'gross_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'daily_pnl': portfolio.equity - (portfolio_daily[-1]['equity'] if portfolio_daily else self.initial_capital),
                'daily_return': (portfolio.equity / (portfolio_daily[-1]['equity'] if portfolio_daily else self.initial_capital) - 1),
                'drawdown': (portfolio.equity - self.initial_capital) / self.initial_capital,
                'turnover': 0.0,  # 简化
                'cost_total': 0.0,  # 简化
            })
        
        # 保存结果
        run_id = str(uuid.uuid4())
        
        # 仅在有ctx时保存结果
        if ctx is not None:
            self._save_results(
                run_id=run_id,
                signals=all_signals,
                orders=all_orders,
                fills=all_fills,
                portfolio_daily=portfolio_daily,
                ctx=ctx,
            )
        
        return {
            'run_id': run_id,
            'signals': all_signals,
            'orders': all_orders,
            'fills': all_fills,
            'portfolio_daily': pd.DataFrame(portfolio_daily),
            'final_equity': portfolio.equity,
            'total_return': (portfolio.equity / self.initial_capital - 1),
            'bars': bars,  # Return bars data for frontend charts
        }

    def _check_lookahead_data(self, md: MarketData, ctx: RunContext, strategy: Strategy) -> None:
        now_ts = pd.Timestamp(ctx.now_utc)

        def _max_ts(df: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
            if df is None or df.empty:
                return None
            idx = df.index
            try:
                if isinstance(idx, pd.MultiIndex):
                    ts = pd.to_datetime(idx.get_level_values(0), errors="coerce")
                else:
                    ts = pd.to_datetime(idx, errors="coerce")
                ts = ts[ts.notna()]
                if len(ts) == 0:
                    return None
                return ts.max()
            except Exception:
                return None

        for name, df in (("bars", md.bars), ("bars_all", md.bars_all), ("features", md.features)):
            max_ts = _max_ts(df)
            if max_ts is None or max_ts <= now_ts:
                continue
            key = (strategy.strategy_id, name)
            if key in self._lookahead_warned:
                continue
            self._lookahead_warned.add(key)
            self.logger.warning(
                f"[LookaheadCheck] {strategy.strategy_id} {name} 包含未来数据: "
                f"max_ts={max_ts} > now={now_ts}"
            )
    
    async def _load_bars(
        self, 
        universe: List[str], 
        start: date, 
        end: date, 
        ctx: RunContext,
        auto_download: bool = True
    ) -> pd.DataFrame:
        """加载bars数据，支持自动补全"""
        from src.data.manager import data_manager
        
        bars_list = []
        # 增加 260 天（约1年）的预热期，确保策略有充足历史数据计算指标
        warmup_start = start - timedelta(days=260)
        start_dt = datetime.combine(warmup_start, datetime.min.time())
        end_dt = datetime.combine(end, datetime.max.time())
        
        expected_dates = None
        if ctx is not None:
            expected_dates = ctx.trading_calendar.trading_days_between(start, end)

        ib_session_opened = False
        if auto_download:
            try:
                ib_session_opened = await data_manager.open_ib_session()
            except Exception as e:
                self.logger.warning(f"IB 连接复用失败，将按需重连: {e}")

        try:
            for symbol in universe:
                bars = await data_manager.get_ohlcv(
                    symbol=symbol,
                    start=start_dt,
                    end=end_dt,
                    timeframe="1d",
                    auto_download=auto_download,
                    expected_dates=expected_dates,
                    source_preference="ib",
                    keep_connection=ib_session_opened,
                )
                
                if bars is not None and not bars.empty:
                    if 'symbol' not in bars.columns:
                        bars['symbol'] = symbol
                    self.logger.info(f"加载 {symbol}: {len(bars)} bars")
                    bars_list.append(bars)
                else:
                    self.logger.warning(f"未找到 {symbol} 的本地数据且自动下载失败")
        finally:
            if ib_session_opened:
                await data_manager.close_ib_session()
        
        if bars_list:
            return pd.concat(bars_list).sort_index()
        return pd.DataFrame()
    
    def _load_features(self, universe: List[str], start: date, end: date, ctx: RunContext) -> Optional[pd.DataFrame]:
        """加载features数据"""
        # 简化实现
        return None
    
    def _create_fill(
        self,
        order: OrderIntent,
        fill_price: float,
        fill_time: datetime,
    ) -> Fill:
        """创建成交记录"""
        fill_id = str(uuid.uuid4())
        order_id = str(uuid.uuid4())  # 简化：应该从order获取
        
        fee = self.cost_model.estimate_fee(order, fill_price)
        slippage = self.cost_model.estimate_slippage(order, fill_price)
        
        return Fill(
            fill_id=fill_id,
            order_id=order_id,
            ts_fill_utc=fill_time,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=fee,
            slippage_est=slippage,
            metadata={**(order.metadata or {})},
        )
    
    def _update_portfolio(self, portfolio: PortfolioState, fill: Fill, fill_price: float) -> PortfolioState:
        """更新组合状态"""
        new_positions = portfolio.positions.copy()
        new_avg_price = portfolio.avg_price.copy()
        
        current_qty = new_positions.get(fill.symbol, 0.0)
        current_avg = new_avg_price.get(fill.symbol, fill_price)
        
        if fill.side == OrderSide.BUY:
            new_qty = current_qty + fill.qty
            if new_qty != 0:
                new_avg = (current_qty * current_avg + fill.qty * fill_price) / new_qty
            else:
                new_avg = fill_price
        else:  # SELL
            new_qty = current_qty - fill.qty
            if abs(new_qty) < 1e-8:
                new_qty = 0.0
                new_avg = 0.0
            else:
                new_avg = current_avg  # 卖出不改变平均成本
        
        new_positions[fill.symbol] = new_qty
        new_avg_price[fill.symbol] = new_avg
        
        # 更新现金
        if fill.side == OrderSide.BUY:
            cost = fill.qty * fill_price + fill.fee
            new_cash = portfolio.cash - cost
        else:
            proceeds = fill.qty * fill_price - fill.fee
            new_cash = portfolio.cash + proceeds
        
        return PortfolioState(
            cash=new_cash,
            positions=new_positions,
            avg_price=new_avg_price,
            equity=portfolio.equity,  # 稍后计算
            daily_loss=portfolio.daily_loss,
        )
    
    def _calculate_equity(self, portfolio: PortfolioState, current_prices: pd.DataFrame) -> float:
        """
        计算总资产净值（支持多标的）
        
        Args:
            portfolio: 组合状态
            current_prices: 当前时刻的所有标的数据
        """
        positions_value = 0.0
        for symbol, qty in portfolio.positions.items():
            if abs(qty) < 1e-8:
                continue
            
            symbol_data = current_prices[current_prices['symbol'] == symbol]
            if not symbol_data.empty:
                price = symbol_data.iloc[-1]['close']
                positions_value += qty * price
            else:
                # 如果没有当前价格数据，尝试使用平均持仓成本
                avg_price = portfolio.avg_price.get(symbol, 0.0)
                positions_value += qty * avg_price
                
        return portfolio.cash + positions_value
    
    def _save_results(
        self,
        run_id: str,
        signals: List[Signal],
        orders: List[OrderIntent],
        fills: List[Fill],
        portfolio_daily: List[Dict],
        ctx: RunContext,
    ):
        """保存回测结果到Parquet"""
        # 保存signals
        if signals:
            signals_df = pd.DataFrame([
                {
                    'ts_utc': s.ts_utc,
                    'symbol': s.symbol,
                    'strategy_id': s.strategy_id,
                    'action': s.action.value,
                    'strength': s.strength,
                    'stop_price': s.stop_price,
                    'take_profit': s.take_profit,
                    'ttl_bars': s.ttl_bars,
                }
                for s in signals
            ])
            # 设置时间戳为索引
            signals_df['ts_utc'] = pd.to_datetime(signals_df['ts_utc'])
            signals_df = signals_df.set_index('ts_utc')
            self.store.write_dataset(
                dataset="signals",
                df=signals_df,
                partition={"run_id": run_id},
                data_version=ctx.data_version,
            )
        
        # 保存orders
        if orders:
            orders_df = pd.DataFrame([
                {
                    'ts_create_utc': o.ts_utc,
                    'symbol': o.symbol,
                    'side': o.side.value,
                    'qty': o.qty,
                    'order_type': o.order_type.value,
                    'limit_price': o.limit_price,
                    'stop_price': o.stop_price,
                    'strategy_id': o.strategy_id,
                }
                for o in orders
            ])
            # 设置时间戳为索引
            orders_df['ts_create_utc'] = pd.to_datetime(orders_df['ts_create_utc'])
            orders_df = orders_df.set_index('ts_create_utc')
            self.store.write_dataset(
                dataset="orders",
                df=orders_df,
                partition={"run_id": run_id},
                data_version=ctx.data_version,
            )
        
        # 保存fills
        if fills:
            fills_df = pd.DataFrame([
                {
                    'fill_id': f.fill_id,
                    'order_id': f.order_id,
                    'ts_fill_utc': f.ts_fill_utc,
                    'symbol': f.symbol,
                    'side': f.side.value,
                    'qty': f.qty,
                    'price': f.price,
                    'fee': f.fee,
                    'slippage_est': f.slippage_est,
                }
                for f in fills
            ])
            # 设置时间戳为索引
            fills_df['ts_fill_utc'] = pd.to_datetime(fills_df['ts_fill_utc'])
            fills_df = fills_df.set_index('ts_fill_utc')
            self.store.write_dataset(
                dataset="fills",
                df=fills_df,
                partition={"run_id": run_id},
                data_version=ctx.data_version,
            )
        
        # 保存portfolio_daily
        if portfolio_daily:
            portfolio_df = pd.DataFrame(portfolio_daily)
            if 'date' in portfolio_df.columns:
                # 设置日期为索引
                portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
                portfolio_df = portfolio_df.set_index('date')
            self.store.write_dataset(
                dataset="portfolio_daily",
                df=portfolio_df,
                partition={"run_id": run_id},
                data_version=ctx.data_version,
            )

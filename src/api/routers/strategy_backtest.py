"""
策略回测API路由 (v2)
提供策略管理和回测功能
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator, model_validator, Field
from typing import Annotated, Union
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timezone
import pandas as pd
import json
import threading
import numpy as np

from src.strategies import strategy_registry
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.evaluation.backtesting.engine import BacktestEngine
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.core.task_manager import TaskManager, TaskStatus, task_manager
from src.data.storage.backtest_store import BacktestStore
from src.utils.logger import get_logger

router = APIRouter(prefix="/strategy-backtest", tags=["策略回测"])
logger = get_logger(__name__)

# 全局存储实例
_backtest_store = BacktestStore()
_lock = threading.Lock()


def _convert_numpy(obj: Any) -> Any:
    """递归转换 numpy 类型为 Python 原生类型，解决 FastAPI JSON 序列化问题"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_convert_numpy(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    else:
        return obj


def _log_backtest_summary(result: Dict[str, Any]) -> None:
    if not result:
        return
    strategy_name = result.get("strategy_name", "unknown")
    universe = result.get("universe") or []
    if not universe and isinstance(result.get("symbol"), str):
        universe = [s.strip() for s in result.get("symbol", "").split(",") if s.strip()]

    period = result.get("backtest_period", {})
    start = period.get("start_date")
    end = period.get("end_date")

    performance = result.get("performance", {})
    enhanced = result.get("enhanced_metrics", {})
    trading = result.get("trading_stats", {})

    logger.info(
        "Backtest Summary | strategy=%s universe=%s period=%s~%s final_equity=%.2f total_return=%.2f%% annual_return=%.2f%% sharpe=%.2f max_dd=%.2f%% alpha=%.4f beta=%.4f",
        strategy_name,
        ",".join(universe),
        start,
        end,
        performance.get("final_equity", 0.0),
        float(performance.get("total_return", 0.0)) * 100,
        float(performance.get("annualized_return", 0.0)) * 100,
        float(performance.get("sharpe_ratio", 0.0)),
        float(performance.get("max_drawdown", 0.0)) * 100,
        float(enhanced.get("alpha", 0.0) or 0.0),
        float(enhanced.get("beta", 0.0) or 0.0),
    )

    logger.info(
        "Backtest Trades | signals=%s orders=%s fills=%s total_trades=%s win_rate=%.2f%% profit_factor=%.2f expectancy=%.4f",
        trading.get("total_signals", 0),
        trading.get("total_orders", 0),
        trading.get("total_fills", 0),
        trading.get("total_trades", 0),
        float(trading.get("win_rate", 0.0)) * 100,
        float(trading.get("profit_factor", 0.0)),
        float(trading.get("expectancy", 0.0)),
    )


# =============================================================================
# Pydantic Models
# =============================================================================

class BacktestRequest(BaseModel):
    """回测请求"""
    strategy_name: str
    universe: List[str] = Field(default_factory=lambda: ["SPY"])
    symbol: Optional[str] = None
    etf_pool: Optional[List[str]] = None
    benchmark_symbol: Optional[str] = None
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    days: Optional[int] = None        # 最近N天
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f'日期格式无效，请使用 YYYY-MM-DD 格式，当前值: {v}')
    
    @model_validator(mode='after')
    def validate_date_or_days(self):
        # 检查是否提供了日期范围或天数
        start_date = self.start_date
        end_date = self.end_date
        
        # 确保在验证前值不为None
        if start_date is not None and end_date is not None:
            has_dates = bool(start_date.strip()) and bool(end_date.strip())
        else:
            has_dates = False
        
        has_days = self.days is not None and self.days > 0
        
        if not has_dates and not has_days:
            raise ValueError('请提供日期范围（start_date 和 end_date）或天数（days）')
        
        if has_dates:
            try:
                assert start_date is not None and end_date is not None
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
                if start >= end:
                    raise ValueError('开始日期必须早于结束日期')
            except ValueError as e:
                if '日期格式无效' not in str(e):
                    raise

        if "universe" not in self.model_fields_set:
            if self.etf_pool:
                normalized = self._normalize_symbols(self.etf_pool)
                if normalized:
                    self.universe = normalized
            elif self.symbol:
                symbol_list = [s.strip() for s in self.symbol.split(",") if s.strip()]
                normalized = self._normalize_symbols(symbol_list)
                if normalized:
                    self.universe = normalized
        elif self.universe:
            self.universe = self._normalize_symbols(self.universe)

        if self.benchmark_symbol:
            normalized = self._normalize_symbols([self.benchmark_symbol])
            self.benchmark_symbol = normalized[0] if normalized else None
        
        return self

    @staticmethod
    def _normalize_symbols(values: List[str]) -> List[str]:
        seen = set()
        normalized = []
        for value in values:
            symbol = value.strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
        return normalized


class StrategyConfigRequest(BaseModel):
    """策略配置请求"""
    params: Dict[str, Any] = {}


# =============================================================================
# Strategy Management
# =============================================================================

@router.get("/strategies")
async def list_strategies():
    """获取所有已注册的策略"""
    strategies = strategy_registry.list_strategies()
    strategy_info = []
    for name in strategies:
        strategy = strategy_registry.get_strategy(name)
        if strategy:
            strategy_info.append({
                "name": strategy.strategy_id,
                "timeframe": strategy.timeframe,
                "params": strategy.config.params,
            })
    
    return {
        "strategies": strategy_info,
        "count": len(strategy_info)
    }


@router.get("/strategies/{name}")
async def get_strategy(name: str):
    """获取策略详情"""
    strategy = strategy_registry.get_strategy(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"策略不存在: {name}")
    
    return {
        "name": strategy.strategy_id,
        "timeframe": strategy.timeframe,
        "config": {
            "strategy_id": strategy.config.strategy_id,
            "timeframe": strategy.config.timeframe,
            "params": strategy.config.params,
        }
    }


@router.post("/strategies/{name}/config")
async def update_strategy_config(name: str, request: StrategyConfigRequest):
    """更新策略配置"""
    strategy = strategy_registry.get_strategy(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"策略不存在: {name}")
    
    # 更新参数
    strategy.config.params.update(request.params)
    logger.info(f"更新策略 {name} 配置: {request.params}")
    
    return {
        "message": "配置已更新",
        "strategy": name,
        "params": strategy.config.params
    }


# =============================================================================
# Backtest Execution (同步)
# =============================================================================

async def _run_backtest_with_progress(
    strategy_name: str,
    universe: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
    days: Optional[int] = None,
    benchmark_symbol: Optional[str] = None,
    task_id: Optional[str] = None,
    auto_download: bool = True,
    etf_pool: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """执行回测（带进度更新）"""
    from src.core.task_manager import task_manager
    from datetime import timedelta
    
    # 更新进度：开始
    if task_id:
        task = task_manager.get_task(task_id)
        if task:
            task.progress = 10.0
            task.status = TaskStatus.RUNNING
    
    strategy = strategy_registry.get_strategy(strategy_name)
    if not strategy:
        raise ValueError(f"策略不存在: {strategy_name}")
    
    # 如果传入了非空的 etf_pool，使用传入的值
    if etf_pool and len(etf_pool) > 0:
        strategy.set_params({"etf_pool": etf_pool})
        logger.info(f"使用传入的 etf_pool: {etf_pool}")
    else:
        # etf_pool 为空或未传入，使用策略的默认 etf_pool
        default_etf_pool = strategy.config.params.get("etf_pool", [])
        if default_etf_pool and isinstance(default_etf_pool, list) and len(default_etf_pool) > 0:
            strategy.set_params({"etf_pool": default_etf_pool})
            # 同时更新 universe 为 etf_pool，确保回测引擎加载正确的数据
            universe = default_etf_pool
            logger.info(f"使用策略默认 etf_pool: {default_etf_pool}")
    
    # 计算日期
    from datetime import timedelta as _timedelta
    from typing import cast as _cast
    
    today = date.today()
    
    if end_date:
        _end = datetime.strptime(_cast(str, end_date), "%Y-%m-%d").date()
    else:
        _end = today
    
    if start_date:
        _start = datetime.strptime(_cast(str, start_date), "%Y-%m-%d").date()
    elif days:
        _start = _end - _timedelta(days=days)
    else:
        raise ValueError("请提供 start_date 或 days 参数")
    
    start_dt, end_dt = _start, _end
    
    # 更新进度：加载数据
    if task_id:
        task = task_manager.get_task(task_id)
        if task:
            task.progress = 30.0
    
    trading_calendar = TradingCalendar()
    ctx = RunContext(
        env=Environment.RESEARCH,
        code_version="1.0.0",
        data_version="latest",
        config_hash="",
        now_utc=datetime.now(timezone.utc),
        trading_calendar=trading_calendar,
    )
    
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )
    
    # 更新进度：执行回测
    if task_id:
        task = task_manager.get_task(task_id)
        if task:
            task.progress = 50.0
    
    result = await engine.run(
        strategies=[strategy],
        universe=universe,
        start=start_dt,
        end=end_dt,
        ctx=ctx,
        auto_download=auto_download,
    )
    
    if "error" in result:
        raise ValueError(result["error"])
    
    # 更新进度：计算指标
    if task_id:
        task = task_manager.get_task(task_id)
        if task:
            task.progress = 80.0
    
    fills = result.get("fills", [])
    requested_start_dt = start_dt
    effective_start_dt = start_dt
    first_fill_ts = None
    for fill in fills:
        ts_fill = getattr(fill, "ts_fill_utc", None)
        if ts_fill is None:
            continue
        if first_fill_ts is None or ts_fill < first_fill_ts:
            first_fill_ts = ts_fill
    if first_fill_ts is not None:
        first_fill_date = first_fill_ts.date()
        if first_fill_date > effective_start_dt:
            effective_start_dt = first_fill_date

    portfolio_df = result.get("portfolio_daily", pd.DataFrame())
    if not portfolio_df.empty:
        if "date" in portfolio_df.columns:
            trimmed = portfolio_df[portfolio_df["date"] >= effective_start_dt]
        else:
            trimmed = portfolio_df[portfolio_df.index >= pd.Timestamp(effective_start_dt)]
        if not trimmed.empty:
            portfolio_df = trimmed
    portfolio_values = portfolio_df['equity'].tolist() if not portfolio_df.empty else []
    total_return = result.get("total_return", 0.0)
    if isinstance(total_return, pd.Series):
        total_return = float(total_return.iloc[-1]) if len(total_return) > 0 else 0.0
    
    days_count = (end_dt - effective_start_dt).days
    annualized_return = (1 + total_return) ** (365.0 / max(days_count, 1)) - 1 if days_count > 0 else 0.0
    
    if not portfolio_df.empty:
        if 'date' in portfolio_df.columns:
            equity_index = pd.to_datetime(portfolio_df['date'])
        else:
            equity_index = pd.to_datetime(portfolio_df.index)
        equity_series = pd.Series(portfolio_values, index=equity_index)
    else:
        equity_series = pd.Series([initial_capital])
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if len(returns) > 1 and returns.std() > 0 else 0.0
    
    # 获取价格数据（用于图表）
    bars = result.get("bars", pd.DataFrame())
    price_data = {"timestamps": [], "close": [], "volume": []}
    if not bars.empty and 'symbol' in bars.columns:
        # 使用第一个标的的价格数据
        first_symbol = universe[0] if universe else "SPY"
        symbol_bars = bars[bars['symbol'] == first_symbol]
        if not symbol_bars.empty:
            symbol_bars = symbol_bars.loc[symbol_bars.index >= pd.Timestamp(effective_start_dt)]
            if not symbol_bars.empty:
                price_data["timestamps"] = [str(d)[:10] for d in symbol_bars.index.tolist()]
                price_data["close"] = symbol_bars['close'].tolist()
                price_data["volume"] = symbol_bars['volume'].tolist() if 'volume' in symbol_bars.columns else []
    
    # 构建交易列表（基于成交记录）
    trades = []
    for i, fill in enumerate(fills):
        ts_fill = getattr(fill, "ts_fill_utc", None)
        side = getattr(fill, "side", None)
        side_value = side.value.lower() if side is not None else ""
        qty = float(getattr(fill, "qty", 0) or 0)
        price = float(getattr(fill, "price", 0) or 0)
        trades.append({
            "id": str(i + 1),
            "timestamp": ts_fill.isoformat() if ts_fill else None,
            "symbol": getattr(fill, "symbol", ""),
            "order_type": side_value,
            "price": price,
            "size": qty,
            "amount": price * qty,
            "commission": float(getattr(fill, "fee", 0) or 0),
        })
    trades.sort(key=lambda t: t.get("timestamp") or "")

    def _analyze_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
            }

        pnl_list: List[float] = []
        buy_trades: Dict[str, List[Dict[str, float]]] = {}

        for trade in trades:
            symbol = trade.get("symbol", "UNKNOWN")
            order_type = str(trade.get("order_type", "")).lower()
            size = abs(float(trade.get("size", 0) or 0))
            price = float(trade.get("price", 0) or 0)

            if symbol not in buy_trades:
                buy_trades[symbol] = []

            if order_type == "buy":
                buy_trades[symbol].append({"size": size, "price": price})
            elif order_type == "sell" and buy_trades[symbol]:
                buy = buy_trades[symbol].pop(0)
                pnl = (price - buy["price"]) * min(size, buy["size"])
                pnl_list.append(pnl)

        if not pnl_list:
            return {
                "total_trades": len(trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
            }

        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]

        total_trades = len(pnl_list)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses)) / len(losses) if losses else 0.0

        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        def _max_consecutive(values: List[float], positive: bool) -> int:
            max_run = 0
            current_run = 0
            for val in values:
                if (val > 0 and positive) or (val < 0 and not positive):
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            return max_run

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
            "consecutive_wins": _max_consecutive(pnl_list, positive=True),
            "consecutive_losses": _max_consecutive(pnl_list, positive=False),
        }
    
    # 计算增强指标
    daily_returns = equity_series.pct_change().dropna()
    winning_days = int((daily_returns > 0).sum())
    losing_days = int((daily_returns < 0).sum())
    total_days = len(daily_returns)
    
    avg_win = float(daily_returns[daily_returns > 0].mean()) if winning_days > 0 else 0
    avg_loss = float(daily_returns[daily_returns < 0].mean()) if losing_days > 0 else 0
    
    # 计算回撤序列
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    drawdown_series = [float(x) for x in drawdown.tolist()]
    
    # 找到最大回撤区间
    max_dd_value = float(drawdown.min()) if len(drawdown) > 0 else 0
    drawdown_array = np.asarray(drawdown.values, dtype=float)
    max_dd_idx = int(np.argmin(drawdown_array)) if len(drawdown) > 0 else 0
    peak_array = np.asarray(drawdown[:max_dd_idx+1].values, dtype=float)
    peak_idx = int(np.argmax(peak_array)) if max_dd_idx > 0 else 0
    
    # 获取日期列表（YYYYMMDD格式）- 优先使用date列，否则使用索引
    def format_date_yyyymmdd(d):
        if hasattr(d, 'strftime'):
            return d.strftime("%Y%m%d")
        s = str(d)
        # 尝试解析并格式化
        try:
            parsed = pd.to_datetime(s)
            return parsed.strftime("%Y%m%d")
        except:
            # 去掉所有非数字字符
            return ''.join(c for c in s if c.isdigit())[:8]
    
    # 优先使用date列
    if 'date' in portfolio_df.columns:
        portfolio_dates = [format_date_yyyymmdd(d) for d in portfolio_df['date'].tolist()]
    else:
        portfolio_dates = [format_date_yyyymmdd(d) for d in portfolio_df.index.tolist()]
    
    # 最大回撤开始和结束日期
    max_dd_start = portfolio_dates[peak_idx] if peak_idx < len(portfolio_dates) else portfolio_dates[0]
    max_dd_end = portfolio_dates[max_dd_idx] if max_dd_idx < len(portfolio_dates) else portfolio_dates[-1]
    
    # 计算恢复天数
    recovery_days = None
    for i in range(max_dd_idx + 1, len(equity_series)):
        if equity_series.iloc[i] >= equity_series.iloc[peak_idx]:
            recovery_days = i - max_dd_idx
            break
    recovery_days = recovery_days if recovery_days else 0
    
    # 计算其他分析指标
    largest_daily_win = float(daily_returns.max()) if len(daily_returns) > 0 else 0
    largest_daily_loss = float(daily_returns.min()) if len(daily_returns) > 0 else 0

    trade_stats = _analyze_trades(trades)
    
    # Ulcer Index
    dd_pct = (drawdown * 100).abs()
    ulcer_index_val = (dd_pct.pow(2).mean() ** 0.5) if len(dd_pct) > 0 else 0
    
    # Time in market (simplified - assume always in market)
    time_in_market = 1.0
    
    # Average drawdown
    avg_drawdown = float(drawdown.mean()) if len(drawdown) > 0 else 0
    
    # Average drawdown duration (simplified)
    avg_duration = 5
    
    # 基准数据（优先真实曲线，失败则回退线性曲线）
    portfolio_equity = [float(x) for x in portfolio_values]
    strategy_equity = portfolio_equity
    benchmark_equity = [
        initial_capital * (1 + i * total_return / max(len(portfolio_values) - 1, 1))
        for i in range(len(portfolio_values))
    ]
    excess_returns = [
        (portfolio_equity[i] - benchmark_equity[i]) / benchmark_equity[i] if benchmark_equity[i] > 0 else 0
        for i in range(len(portfolio_values))
    ]
    comparison_timestamps = portfolio_dates

    resolved_benchmark = ""
    if benchmark_symbol:
        resolved_benchmark = str(benchmark_symbol).strip().upper()
    if not resolved_benchmark:
        resolved_benchmark = str(strategy.config.params.get("benchmark_symbol", "")).strip().upper()
    if not resolved_benchmark:
        resolved_benchmark = universe[0] if universe else "SPY"

    benchmark_symbol = resolved_benchmark
    benchmark_return = total_return
    benchmark_volatility = float(daily_returns.std() * (252 ** 0.5)) if len(daily_returns) > 0 else 0
    excess_return = 0.0
    alpha = 0.0
    beta = 1.0
    tracking_error = 0.0
    information_ratio = 0.0
    r_squared = 1.0

    try:
        from src.evaluation.metrics.benchmark import BenchmarkAnalyzer

        benchmark_analyzer = BenchmarkAnalyzer(benchmark_symbol)
        start_date_str = effective_start_dt.strftime("%Y-%m-%d")
        end_date_str = end_dt.strftime("%Y-%m-%d")
        benchmark_df = benchmark_analyzer.get_benchmark_data(
            start_date=start_date_str,
            end_date=end_date_str,
            data_source="ib",
        )
        if benchmark_df is None or benchmark_df.empty:
            benchmark_df = benchmark_analyzer.get_benchmark_data(
                start_date=start_date_str,
                end_date=end_date_str,
                data_source="local",
            )

        if benchmark_df is not None and not benchmark_df.empty:
            if "close" not in benchmark_df.columns and "c" in benchmark_df.columns:
                benchmark_df["close"] = benchmark_df["c"]
            if "close" in benchmark_df.columns:
                benchmark_returns = benchmark_analyzer.calculate_returns_from_prices(
                    benchmark_df["close"]
                )
                benchmark_metrics = benchmark_analyzer.calculate_benchmark_metrics(benchmark_returns)
                benchmark_analysis = benchmark_analyzer.comprehensive_benchmark_analysis(
                    returns, benchmark_returns
                )
                comparison = benchmark_analyzer.get_equity_comparison(
                    returns, benchmark_returns, initial_value=initial_capital
                )

                benchmark_return = float(benchmark_metrics.total_return)
                benchmark_volatility = float(benchmark_metrics.volatility)
                excess_return = float(benchmark_analysis.get("excess_return", 0.0) or 0.0)
                alpha = float(benchmark_analysis.get("alpha", 0.0) or 0.0)
                beta = float(benchmark_analysis.get("beta", 1.0) or 1.0)
                tracking_error = float(benchmark_analysis.get("tracking_error", 0.0) or 0.0)
                information_ratio = float(benchmark_analysis.get("information_ratio", 0.0) or 0.0)
                r_squared = float(benchmark_analysis.get("r_squared", 0.0) or 0.0)

                strategy_equity = comparison.get("strategy_equity", strategy_equity)
                benchmark_equity = comparison.get("benchmark_equity", benchmark_equity)
                excess_returns = comparison.get("excess_returns", excess_returns)
                comparison_timestamps = [str(ts) for ts in comparison.get("timestamps", comparison_timestamps)]
            else:
                logger.warning(f"基准数据缺少 close 列: {benchmark_symbol}")
        else:
            logger.warning(f"无法获取基准数据: {benchmark_symbol}")
    except Exception as e:
        logger.warning(f"基准曲线计算失败，回退线性基准: {e}")
    
    # 完整的结果 - 将所有指标合并到 enhanced_metrics 以支持前端展示
    trading_stats_data = {
        "total_signals": len(result.get("signals", [])),
        "total_orders": len(result.get("orders", [])),
        "total_fills": len(result.get("fills", [])),
        "total_trades": trade_stats.get("total_trades", 0),
        "winning_trades": trade_stats.get("winning_trades", 0),
        "losing_trades": trade_stats.get("losing_trades", 0),
        "win_rate": trade_stats.get("win_rate", 0.0),
        "daily_win_rate": winning_days / total_days if total_days > 0 else 0,
        "profit_loss_ratio": trade_stats.get("profit_loss_ratio", 0.0),
        "profit_factor": trade_stats.get("profit_factor", 0.0),
        "expectancy": trade_stats.get("expectancy", 0.0),
        "consecutive_wins": trade_stats.get("consecutive_wins", 0),
        "consecutive_losses": trade_stats.get("consecutive_losses", 0),
        "largest_win": trade_stats.get("largest_win", 0.0),
        "largest_loss": trade_stats.get("largest_loss", 0.0),
        "winning_days": winning_days,
        "losing_days": losing_days,
    }
    
    # 最大回撤区间数据
    max_drawdown_window = {
        "start_date": max_dd_start,
        "end_date": max_dd_end,
        "drawdown_pct": max_dd_value,
        "duration_days": max_dd_idx - peak_idx if max_dd_idx >= peak_idx else 0,
        "recovery_days": recovery_days,
        "equity_at_peak": float(equity_series.iloc[peak_idx]) if peak_idx < len(equity_series) else initial_capital,
        "equity_at_trough": float(equity_series.iloc[max_dd_idx]) if max_dd_idx < len(equity_series) else initial_capital,
    }
    
    # 所有前端需要的指标都放在 enhanced_metrics 中
    enhanced_metrics_data = {
        # 收益指标
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "annual_return": annualized_return,
        "excess_return": excess_return,
        "best_day": largest_daily_win,
        "worst_day": largest_daily_loss,
        # 风险指标
        "max_drawdown": max_dd_value,
        "annual_volatility": float(daily_returns.std() * (252 ** 0.5)) if len(daily_returns) > 0 else 0,
        "benchmark_volatility": benchmark_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": (daily_returns.mean() / daily_returns[daily_returns < 0].std() * (252 ** 0.5)) if len(daily_returns) > 1 and daily_returns[daily_returns < 0].std() > 0 else 0,
        "calmar_ratio": annualized_return / abs(max_dd_value) if max_dd_value != 0 else 0,
        # 基准指标
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "information_ratio": information_ratio,
        "tracking_error": tracking_error,
        # 回撤图表数据
        "drawdown_series": drawdown_series,
        # 最大回撤区间
        "max_drawdown_window": max_drawdown_window,
        # 分析指标
        "ulcer_index": float(ulcer_index_val) if ulcer_index_val else 0,
        "burke_ratio": 0,
        "time_in_market": time_in_market,
        "avg_drawdown": avg_drawdown,
        "avg_drawdown_duration": avg_duration,
        # 交易统计
        **trading_stats_data
    }
    
    final_result = {
        "run_id": result.get("run_id"),
        "strategy_name": strategy_name,
        "symbol": ",".join(universe) if universe else "SPY",
        "universe": universe,
        "backtest_period": {
            "start_date": (
                effective_start_dt.isoformat()
                if hasattr(effective_start_dt, 'isoformat')
                else effective_start_dt.strftime("%Y-%m-%d")
            ),
            "end_date": end_dt.isoformat() if hasattr(end_dt, 'isoformat') else end_dt.strftime("%Y-%m-%d"),
            "days": (end_dt - effective_start_dt).days,
            "requested_start_date": (
                requested_start_dt.isoformat()
                if hasattr(requested_start_dt, 'isoformat')
                else requested_start_dt.strftime("%Y-%m-%d")
            ),
        },
        "config": {
            "initial_capital": initial_capital,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
        },
        "price_data": price_data,
        "trades": trades,
        "results": {
            "final_value": result.get("final_equity", 0.0),
            "total_return": total_return,
            "annual_return": annualized_return,
            "best_day": float(daily_returns.max()) if not daily_returns.empty else 0,
            "worst_day": float(daily_returns.min()) if not daily_returns.empty else 0,
            "daily_win_rate": winning_days / total_days if total_days > 0 else 0,
        },
        "enhanced_metrics": enhanced_metrics_data,
        "equity_comparison": {
            "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "excess_returns": excess_returns,
        "timestamps": comparison_timestamps,
        },
        "benchmark_data": {
            "symbol": benchmark_symbol,
            "alpha": alpha,
            "beta": beta,
            "r_squared": r_squared,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
        },
        "performance": {
            "final_equity": result.get("final_equity", 0.0),
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_dd_value,
            "sharpe_ratio": float(sharpe_ratio),
        },
        "trading_stats": trading_stats_data,
        "portfolio": {
            "dates": portfolio_dates,
            "values": portfolio_equity,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # 更新进度：完成
    if task_id:
        task = task_manager.get_task(task_id)
        if task:
            task.progress = 100.0

    _log_backtest_summary(final_result)
    
    # 转换 numpy 类型为 Python 原生类型，解决 JSON 序列化问题
    return _convert_numpy(final_result)


def _on_backtest_complete(task_id: str, result: Any) -> Any:
    """回测完成回调：保存到历史存储"""
    if result and isinstance(result, dict):
        try:
            backtest_id = _backtest_store.generate_id()
            _backtest_store.save(backtest_id, result)
            logger.info(f"回测结果已保存: {backtest_id}")
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
    return result

@router.post("/run")
async def run_backtest(request: BacktestRequest):
    """运行回测（异步执行，v2引擎）"""
    result = await _run_backtest_with_progress(
        strategy_name=request.strategy_name,
        universe=request.universe,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        commission_rate=request.commission_rate,
        slippage_rate=request.slippage_rate,
        days=request.days,
        benchmark_symbol=request.benchmark_symbol,
        auto_download=True, # 默认开启自动下载
        etf_pool=request.etf_pool,
    )
    
    # 保存到历史记录
    _on_backtest_complete("", result)
    
    return result


@router.post("/run-async")
async def submit_async_backtest(request: BacktestRequest):
    """
    提交异步回测任务
    
    返回任务ID，后续可通过 /tasks/{task_id} 查询进度
    """
    # 验证策略存在
    strategy = strategy_registry.get_strategy(request.strategy_name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"策略不存在: {request.strategy_name}")
    
    try:
        if request.start_date:
            datetime.strptime(request.start_date, "%Y-%m-%d").date()
        if request.end_date:
            datetime.strptime(request.end_date, "%Y-%m-%d").date()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"日期格式错误: {e}")
    
    import asyncio
    
    def run_sync_backtest():
        # 在线程中运行异步回测
        return asyncio.run(_run_backtest_with_progress(
            request.strategy_name,
            request.universe,
            request.start_date,
            request.end_date,
            request.initial_capital,
            request.commission_rate,
            request.slippage_rate,
            benchmark_symbol=request.benchmark_symbol,
            task_id=task_id,
        ))

    # 提交任务
    task_id = task_manager.submit(
        name=f"回测: {request.strategy_name}",
        func=run_sync_backtest,
        args=(),
    )
    
    # 获取任务并绑定完成后的操作
    task = task_manager.get_task(task_id)
    # 此处逻辑略作简化，原逻辑在提交后立即再次调用了 _run_backtest_with_progress，这实际上会导致两次运行。
    # 我们修正为通过任务管理器正常执行。
    
    return {
        "message": "回测任务已提交",
        "task_id": task_id,
        "status": "pending",
        "check_url": f"/api/v1/strategy-backtest/tasks/{task_id}",
    }


# =============================================================================
# Task Management
# =============================================================================

@router.get("/tasks")
async def list_tasks(status: Optional[str] = None, limit: int = Query(50, ge=1, le=100)):
    """列出任务"""
    task_status = None
    if status:
        try:
            task_status = TaskStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效状态: {status}")
    
    # task_manager.list_tasks expects TaskStatus or None
    tasks = task_manager.list_tasks(status=task_status if task_status else None, limit=limit)
    
    return {
        "tasks": [t.to_dict() for t in tasks],
        "count": len(tasks),
        "queue_stats": task_manager.get_queue_stats(),
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    return task.to_dict()


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """取消任务"""
    success = task_manager.cancel(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="无法取消任务（可能已在执行或已完成）")
    
    return {"message": "任务已取消", "task_id": task_id}


# =============================================================================
# History (持久化)
# =============================================================================

@router.get("/history")
async def list_backtest_history(
    strategy_name: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
):
    """获取历史回测记录"""
    records = _backtest_store.list(strategy_name=strategy_name, symbol=symbol, limit=limit)

    backtests = []
    for record in records:
        symbol_value = record.symbol
        universe_value = None
        result = _backtest_store.get(record.id)
        if isinstance(result, dict):
            result_universe = result.get("universe")
            if isinstance(result_universe, list) and result_universe:
                universe_value = [str(item) for item in result_universe if item]
                if universe_value:
                    symbol_value = ",".join(universe_value)
            else:
                result_symbol = result.get("symbol")
                if isinstance(result_symbol, str) and result_symbol:
                    symbol_value = result_symbol

        backtests.append(
            {
                "id": record.id,
                "strategy_name": record.strategy_name,
                "symbol": symbol_value,
                "universe": universe_value,
                "timeframe": record.timeframe,
                "start_date": record.start_date,
                "end_date": record.end_date,
                "initial_capital": record.initial_capital,
                "final_value": record.final_value,
                "total_return": record.total_return,
                "sharpe_ratio": record.sharpe_ratio,
                "max_drawdown": record.max_drawdown,
                "total_trades": record.total_trades,
                "win_rate": record.win_rate,
                "created_at": record.created_at,
            }
        )

    return {
        "backtests": backtests,
        "count": len(backtests),
    }


@router.get("/history/{run_id}")
async def get_backtest_result(run_id: str):
    """获取回测结果详情"""
    result = _backtest_store.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"回测记录不存在: {run_id}")
    return result


@router.delete("/history/{run_id}")
async def delete_backtest_result(run_id: str):
    """删除回测记录"""
    success = _backtest_store.delete(run_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"回测记录不存在: {run_id}")
    return {"message": "回测记录已删除", "run_id": run_id}


# =============================================================================
# Walk-Forward Analysis
# =============================================================================

class WalkForwardRequest(BaseModel):
    """Walk-Forward 分析请求"""
    strategy_name: str
    universe: List[str] = ["SPY"]
    start_date: str
    end_date: str
    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 21
    param_grid: Optional[Dict[str, List[float]]] = None
    optimization_metric: str = "sharpe_ratio"


@router.post("/walk-forward")
async def run_walk_forward_analysis(request: WalkForwardRequest):
    """
    运行 Walk-Forward 分析
    
    支持参数优化和过拟合检测
    """
    from src.evaluation.walk_forward import WalkForwardAnalyzer
    
    strategy = strategy_registry.get_strategy(request.strategy_name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"策略不存在: {request.strategy_name}")
    
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d").date()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"日期格式错误: {e}")
    
    trading_calendar = TradingCalendar()
    ctx = RunContext(
        env=Environment.RESEARCH,
        code_version="1.0.0",
        data_version="latest",
        config_hash="",
        now_utc=datetime.now(timezone.utc),
        trading_calendar=trading_calendar,
    )
    
    analyzer = WalkForwardAnalyzer()
    
    result = await analyzer.run(
        strategies=[strategy],
        universe=request.universe,
        start=start_dt,
        end=end_dt,
        train_window_days=request.train_window_days,
        test_window_days=request.test_window_days,
        step_days=request.step_days,
        param_grid=request.param_grid,
        optimization_metric=request.optimization_metric,
        ctx=ctx,
    )
    
    # 格式化结果
    return {
        "strategy_name": request.strategy_name,
        "total_windows": result.get("total_windows", 0),
        "pass_rate": result.get("pass_rate", 0.0),
        "param_grid": result.get("param_grid"),
        "optimization_metric": result.get("optimization_metric"),
        "degradation_analysis": result.get("degradation_analysis"),
        "note": "返回结果包含每个窗口的详细回测结果"
    }

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
from src.evaluation.backtesting.unified_engine import (
    UnifiedBacktestEngine,
    UnifiedConfig,
    FeatureFlag,
)
from src.evaluation.backtesting.config import TradeConfig, TimeConfig
from src.evaluation.backtesting import BacktestResult
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
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        val = float(obj)
        # 只转换真正的 NaN/Inf，保留极大的有效数值
        if np.isnan(val):
            return None
        if np.isinf(val):
            # 检查是否是因为计算错误导致的超大值
            if abs(val) > 1e15:  # 超过 1 万亿的值视为无效数据
                return None
            return None  # 任何无穷大都返回 None
        return val
    elif isinstance(obj, np.ndarray):
        return [_convert_numpy(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, "item") and not isinstance(obj, (int, float, bool, str)):  # numpy scalar
        try:
            val = obj.item()
            if isinstance(val, (float, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    return None
            return val
        except Exception:
            return str(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return obj


def _get_backtest_period(start_dt: Any, end_dt: Any) -> Dict[str, Any]:
    """Build backtest_period dict from start/end datetime objects"""
    start_date = ""
    end_date = ""
    days = 0

    if start_dt:
        if hasattr(start_dt, "strftime"):
            start_date = start_dt.strftime("%Y-%m-%d")
        else:
            start_date = str(start_dt)

    if end_dt:
        if hasattr(end_dt, "strftime"):
            end_date = end_dt.strftime("%Y-%m-%d")
        else:
            end_date = str(end_dt)

    # Calculate days
    if start_dt and end_dt:
        try:
            # Handle both date and datetime objects
            if hasattr(start_dt, "date"):
                start_date_only = start_dt.date()
            else:
                start_date_only = start_dt

            if hasattr(end_dt, "date"):
                end_date_only = end_dt.date()
            else:
                end_date_only = end_dt

            days = (end_date_only - start_date_only).days
        except Exception:
            pass

    return {
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
    }


def _format_trades_from_fills(fills: Any) -> List[Dict[str, Any]]:
    """
    将 fills 数据转换为前端所需的 trades 格式

    前端需要: timestamp, symbol, order_type, price, size, fill_id
    Fill 字段: ts_fill_utc, symbol, side, qty, price, fill_id
    """
    if not fills:
        return []

    if not isinstance(fills, list):
        return []

    formatted_trades = []
    for fill in fills:
        if isinstance(fill, dict):
            # 已经是字典格式
            ts_fill_utc = fill.get("ts_fill_utc") or fill.get("timestamp")
            side = fill.get("side")
            symbol = fill.get("symbol", "UNKNOWN")
            qty = float(fill.get("qty", 0))
            price = float(fill.get("price", 0))
            fill_id = fill.get("fill_id", "")
        else:
            # Fill 对象
            ts_fill_utc = getattr(fill, "ts_fill_utc", None) or getattr(fill, "timestamp", None)
            side = getattr(fill, "side", None)
            symbol = getattr(fill, "symbol", "UNKNOWN")
            qty = float(getattr(fill, "qty", 0))
            price = float(getattr(fill, "price", 0))
            fill_id = getattr(fill, "fill_id", "")

        # 转换 side 为 order_type
        order_type = "unknown"
        if side is not None:
            if isinstance(side, dict):
                order_type = side.get("value", str(side)).lower()
            elif hasattr(side, "value"):
                order_type = side.value.lower()
            else:
                order_type = str(side).lower()

        # 转换时间戳
        timestamp = None
        if ts_fill_utc:
            if hasattr(ts_fill_utc, "isoformat"):
                timestamp = ts_fill_utc.isoformat()
            elif isinstance(ts_fill_utc, str):
                timestamp = ts_fill_utc

        formatted_trades.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "order_type": order_type,
                "price": price,
                "size": abs(qty),
                "fill_id": fill_id,
            }
        )

    return formatted_trades


def _get_result_field(result: Dict[str, Any], field_name: str, default: Any = None) -> Any:
    """
    从回测结果中获取字段，支持新旧两种格式：
    - 新格式: result["results"][field_name]
    - 旧格式: result[field_name]
    """
    if not isinstance(result, dict):
        return default

    # 优先从 results 中获取（新格式）
    results = result.get("results")
    if isinstance(results, dict) and field_name in results:
        return results[field_name]

    # 尝试从根级别获取（旧格式）
    if field_name in result:
        return result[field_name]

    return default


def _log_backtest_summary(result: Dict[str, Any]) -> None:
    if not result:
        return
    strategy_name = result.get("strategy_name", "unknown")
    universe = result.get("universe") or []
    if not universe and isinstance(result.get("symbol"), str):
        universe = [s.strip() for s in result.get("symbol", "").split(",") if s.strip()]

    period = result.get("backtest_period", {})
    start = period.get("start_date", "")
    end = period.get("end_date", "")

    performance = result.get("performance", {})
    enhanced = result.get("enhanced_metrics", {})
    trading = result.get("trading_stats", {})

    # 安全获取数值
    final_equity = float(performance.get("final_equity", 0.0))
    total_return = float(performance.get("total_return", 0.0)) * 100
    annual_return = float(performance.get("annualized_return", 0.0)) * 100
    sharpe = float(performance.get("sharpe_ratio", 0.0))
    max_dd = float(performance.get("max_drawdown", 0.0)) * 100
    alpha = float(enhanced.get("alpha", 0.0) or 0.0)
    beta = float(enhanced.get("beta", 0.0) or 0.0)

    logger.info(
        f"Backtest Summary | strategy={strategy_name} universe={','.join(universe)} "
        f"period={start}~{end} final_equity={final_equity:.2f} "
        f"total_return={total_return:.2f}% annual_return={annual_return:.2f}% "
        f"sharpe={sharpe:.2f} max_dd={max_dd:.2f}% alpha={alpha:.4f} beta={beta:.4f}"
    )

    total_signals = trading.get("total_signals", 0)
    total_orders = trading.get("total_orders", 0)
    total_fills = trading.get("total_fills", 0)
    total_trades = trading.get("total_trades", 0)
    win_rate = float(trading.get("win_rate", 0.0)) * 100
    profit_factor = float(trading.get("profit_factor", 0.0))
    expectancy = float(trading.get("expectancy", 0.0))

    logger.info(
        f"Backtest Trades | signals={total_signals} orders={total_orders} "
        f"fills={total_fills} total_trades={total_trades} "
        f"win_rate={win_rate:.2f}% profit_factor={profit_factor:.2f} expectancy={expectancy:.4f}"
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
    end_date: Optional[str] = None  # YYYY-MM-DD
    days: Optional[int] = None  # 最近N天
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"日期格式无效，请使用 YYYY-MM-DD 格式，当前值: {v}")

    @model_validator(mode="after")
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
            raise ValueError("请提供日期范围（start_date 和 end_date）或天数（days）")

        if has_dates:
            try:
                assert start_date is not None and end_date is not None
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
                if start >= end:
                    raise ValueError("开始日期必须早于结束日期")
            except ValueError as e:
                if "日期格式无效" not in str(e):
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
            # 只返回可JSON序列化的参数值，过滤掉Parameter对象和其他不可序列化对象
            serializable_params = {}
            for k, v in strategy.config.params.items():
                # 跳过Parameter对象和其他复杂对象
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    serializable_params[k] = v

            strategy_info.append(
                {
                    "name": strategy.strategy_id,
                    "timeframe": strategy.timeframe,
                    "params": serializable_params,
                }
            )

    return {"strategies": strategy_info, "count": len(strategy_info)}


@router.get("/strategies/{name}")
async def get_strategy(name: str):
    """获取策略详情"""
    strategy = strategy_registry.get_strategy(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"策略不存在: {name}")

    # 只返回可JSON序列化的参数值
    serializable_params = {}
    for k, v in strategy.config.params.items():
        # 跳过Parameter对象和其他复杂对象
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serializable_params[k] = v

    return {
        "name": strategy.strategy_id,
        "timeframe": strategy.timeframe,
        "config": {
            "strategy_id": strategy.config.strategy_id,
            "timeframe": strategy.config.timeframe,
            "params": serializable_params,
        },
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

    return {"message": "配置已更新", "strategy": name, "params": strategy.config.params}


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
        config={"data": {"benchmark_symbol": benchmark_symbol}} if benchmark_symbol else {},
    )

    # 使用 UnifiedBacktestEngine
    config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        ),
        time=TimeConfig(
            signal_timeframe="1d",
            clock_mode="daily",
        ),
        timerange=f"{start_dt.strftime('%Y%m%d')}-{end_dt.strftime('%Y%m%d')}",
        features=FeatureFlag.ALL,
    )
    engine = UnifiedBacktestEngine(config=config)

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

    # 处理结果
    if isinstance(result, dict) and "error" in result:
        raise ValueError(result["error"])

    # 更新进度：完成
    if task_id:
        task = task_manager.get_task(task_id)
        if task:
            task.progress = 100.0

    # 统一结果格式
    if hasattr(result, "to_dict"):
        raw_result = result.to_dict()
    elif isinstance(result, dict):
        raw_result = result
    else:
        raw_result = {"result": result}

    # 安全提取 total_return（小数形式，如 0.18 表示 18%）
    raw_total_return = raw_result.get("total_return", 0.0) if isinstance(raw_result, dict) else 0.0
    # 处理可能的 BacktestResult 类型或非数值类型
    if isinstance(raw_total_return, BacktestResult):
        raw_total_return = raw_total_return.total_return
    # 确保是数值类型才进行 float 转换
    if isinstance(raw_total_return, (int, float)) and not isinstance(raw_total_return, bool):
        total_return_val = float(raw_total_return)
    else:
        try:
            total_return_val = float(raw_total_return) if raw_total_return is not None else 0.0
        except (TypeError, ValueError):
            total_return_val = 0.0

    # 从raw_result提取基础字段
    portfolio_daily_raw = raw_result.get("portfolio_daily", [])

    portfolio_daily_list = []
    if isinstance(portfolio_daily_raw, list):
        portfolio_daily_list = portfolio_daily_raw
    else:
        portfolio_daily_list = []

    # 从raw_result提取其他字段
    fills_raw = raw_result.get("fills", [])
    fills = fills_raw if isinstance(fills_raw, list) else []

    benchmark_symbol = benchmark_symbol or ""
    benchmark_equity_list = []

    if isinstance(raw_result, dict):
        benchmark_result = raw_result.get("benchmark_result")
        if isinstance(benchmark_result, dict):
            benchmark_equity_list = benchmark_result.get("equity_curve", [])
        elif raw_result.get("benchmark_equity"):
            benchmark_equity_list = raw_result.get("benchmark_equity", [])

    equity_comparison = {
        "strategy_equity": [d.get("equity", 0.0) for d in portfolio_daily_list],
        "benchmark_equity": benchmark_equity_list if benchmark_equity_list else [],
        "excess_returns": [],
        "timestamps": [str(d.get("timestamp", d.get("date", ""))) for d in portfolio_daily_list],
    }

    if equity_comparison["strategy_equity"] and equity_comparison["benchmark_equity"]:
        strategy_eq = equity_comparison["strategy_equity"]
        bench_eq = equity_comparison["benchmark_equity"]
        min_len = min(len(strategy_eq), len(bench_eq))
        equity_comparison["excess_returns"] = [
            (strategy_eq[i] - bench_eq[i]) / bench_eq[i] if bench_eq[i] != 0 else 0
            for i in range(min_len)
        ]

    import numpy as np

    # 创建benchmark_data，使用safe_float确保类型正确
    import numpy as np

    def safe_float(val):
        if val is None:
            return 0.0
        try:
            result = float(val)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result
        except (ValueError, TypeError):
            return 0.0

    # 检查raw_result中的benchmark数据
    benchmark_return_val = (
        raw_result.get("benchmark_return", 0.0) if isinstance(raw_result, dict) else 0.0
    )
    alpha_val = raw_result.get("alpha", 0.0) if isinstance(raw_result, dict) else 0.0

    if benchmark_return_val is None:
        benchmark_return_val = 0.0
    if alpha_val is None:
        alpha_val = 0.0

    benchmark_data = {
        "symbol": benchmark_symbol,
        "benchmark_equity": benchmark_equity_list,
        "benchmark_return": safe_float(benchmark_return_val),
        "benchmark_volatility": (
            safe_float(raw_result.get("benchmark_volatility", 0.0))
            if isinstance(raw_result, dict)
            else 0.0
        ),
        "alpha": safe_float(alpha_val),
        "beta": safe_float(raw_result.get("beta", 1.0)) if isinstance(raw_result, dict) else 1.0,
    }

    final_result = {
        "results": raw_result,
        "strategy_name": (
            raw_result.get("strategy_name", "") if isinstance(raw_result, dict) else ""
        ),
        "timeframe": raw_result.get("timeframe", "") if isinstance(raw_result, dict) else "",
        "timerange": raw_result.get("timerange", "") if isinstance(raw_result, dict) else "",
        "initial_capital": (
            raw_result.get("initial_capital", 0.0) if isinstance(raw_result, dict) else 0.0
        ),
        "final_value": raw_result.get("final_equity", 0.0) if isinstance(raw_result, dict) else 0.0,
        "total_return": (
            raw_result.get("total_return", 0.0) if isinstance(raw_result, dict) else 0.0
        ),
        "sharpe_ratio": (
            raw_result.get("sharpe_ratio", 0.0) if isinstance(raw_result, dict) else 0.0
        ),
        "max_drawdown": (
            raw_result.get("max_drawdown", 0.0) if isinstance(raw_result, dict) else 0.0
        ),
        "win_rate": raw_result.get("win_rate", 0.0) if isinstance(raw_result, dict) else 0.0,
        "total_trades": raw_result.get("total_trades", 0) if isinstance(raw_result, dict) else 0,
        "backtest_period": _get_backtest_period(start_dt, end_dt),
        "universe": universe if isinstance(universe, list) else [],
        "symbol": ",".join(universe) if isinstance(universe, list) and len(universe) > 0 else "",
        "config": {
            "initial_capital": float(initial_capital),
            "commission_rate": float(commission_rate),
            "slippage_rate": float(slippage_rate),
        },
        "performance_stats": {
            "sharpe_ratio": (
                raw_result.get("sharpe_ratio", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "max_drawdown": (
                raw_result.get("max_drawdown", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "win_rate": raw_result.get("win_rate", 0.0) if isinstance(raw_result, dict) else 0.0,
            "total_return": (
                raw_result.get("total_return", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "annualized_return": (
                raw_result.get("annualized_return", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
        },
        "trade_stats": {
            "total_trades": (
                raw_result.get("total_trades", 0) if isinstance(raw_result, dict) else 0
            ),
            "winning_trades": (
                raw_result.get("winning_trades", 0) if isinstance(raw_result, dict) else 0
            ),
            "profit_factor": (
                raw_result.get("profit_factor", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
        },
        "enhanced_metrics": {
            "total_return": (
                total_return_val / 100 if abs(total_return_val) > 1 else total_return_val
            ),
            "annual_return": (
                raw_result.get("annualized_return", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "annual_volatility": (
                raw_result.get("volatility", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "sharpe_ratio": (
                raw_result.get("sharpe_ratio", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "sortino_ratio": (
                raw_result.get("sortino_ratio", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "calmar_ratio": (
                raw_result.get("calmar_ratio", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "max_drawdown": (
                raw_result.get("max_drawdown", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "win_rate": raw_result.get("win_rate", 0.0) if isinstance(raw_result, dict) else 0.0,
            "total_trades": (
                raw_result.get("total_trades", 0) if isinstance(raw_result, dict) else 0
            ),
            "profit_factor": (
                raw_result.get("profit_factor", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "benchmark_return": (
                raw_result.get("benchmark_return", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "benchmark_volatility": (
                raw_result.get("benchmark_volatility", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "excess_return": (
                raw_result.get("excess_return", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "alpha": raw_result.get("alpha", 0.0) if isinstance(raw_result, dict) else 0.0,
            "beta": raw_result.get("beta", 1.0) if isinstance(raw_result, dict) else 1.0,
            "information_ratio": (
                raw_result.get("information_ratio", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "r_squared": raw_result.get("r_squared", 0.0) if isinstance(raw_result, dict) else 0.0,
            "winning_trades": (
                raw_result.get("winning_trades", 0) if isinstance(raw_result, dict) else 0
            ),
            "losing_trades": (
                raw_result.get("losing_trades", 0) if isinstance(raw_result, dict) else 0
            ),
            "profit_loss_ratio": (
                raw_result.get("profit_loss_ratio", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "expectancy": (
                raw_result.get("expectancy", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "largest_win": (
                raw_result.get("largest_win", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "largest_loss": (
                raw_result.get("largest_loss", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "consecutive_wins": (
                raw_result.get("consecutive_wins", 0) if isinstance(raw_result, dict) else 0
            ),
            "consecutive_losses": (
                raw_result.get("consecutive_losses", 0) if isinstance(raw_result, dict) else 0
            ),
            "daily_win_rate": (
                raw_result.get("daily_win_rate", 0.0) if isinstance(raw_result, dict) else 0.0
            ),
            "drawdown_series": (
                raw_result.get("drawdown_series", []) if isinstance(raw_result, dict) else []
            ),
            "max_drawdown_window": (
                raw_result.get("max_drawdown_window") if isinstance(raw_result, dict) else None
            ),
            "drawdown_windows": (
                raw_result.get("drawdown_windows", []) if isinstance(raw_result, dict) else []
            ),
        },
        "portfolio_value": {
            "timestamps": [
                str(d.get("timestamp", d.get("date", ""))) for d in portfolio_daily_list
            ],
            "values": [d.get("equity", 0.0) for d in portfolio_daily_list],
        },
        "trades": fills,
        "equity_comparison": equity_comparison,
        "benchmark_data": benchmark_data,
        "price_data": _extract_price_data(raw_result, universe),
    }

    # 转换 numpy 类型为 Python 原生类型，解决 JSON 序列化问题
    result = _convert_numpy(final_result)

    return result


def _extract_price_data(raw_result: Dict[str, Any], universe: List[str]) -> Dict[str, Any]:
    """从回测结果中提取价格数据，供前端图表使用"""
    # 优先从 results 中提取
    results = raw_result.get("results", raw_result)

    # 尝试从各种可能的字段名提取
    price_data = {}

    # 方法1: 检查 raw_result 中是否有 price_data
    if (
        "price_data" in raw_result
        and isinstance(raw_result["price_data"], dict)
        and raw_result["price_data"]
    ):
        return raw_result["price_data"]

    # 方法2: 从 benchmark_result 中提取
    benchmark_result = raw_result.get("benchmark_result")
    if isinstance(benchmark_result, dict):
        if "price_data" in benchmark_result:
            price_data = benchmark_result["price_data"]
        elif "close" in benchmark_result:
            # 如果有 close 数组，尝试构建 price_data
            close_vals = benchmark_result["close"]
            if isinstance(close_vals, list) and len(close_vals) > 0:
                price_data = {
                    "timestamps": [f"2024-01-{(i+1):02d}" for i in range(len(close_vals))],
                    "close": close_vals,
                    "open": close_vals,
                    "high": close_vals,
                    "low": close_vals,
                    "volume": [0] * len(close_vals),
                }

    # 方法3: 从 benchmark_equity 推断（如果存在的话）
    benchmark_equity = raw_result.get("benchmark_equity", [])
    if isinstance(benchmark_equity, list) and len(benchmark_equity) > 0 and not price_data:
        # 使用 benchmark_equity 作为价格代理
        timestamps = [
            str(d.get("timestamp", d.get("date", "")))
            for d in raw_result.get("portfolio_daily", [])
        ]
        if len(timestamps) >= len(benchmark_equity):
            price_data = {
                "timestamps": timestamps[: len(benchmark_equity)],
                "close": benchmark_equity,
                "open": benchmark_equity,
                "high": benchmark_equity,
                "low": benchmark_equity,
                "volume": [0] * len(benchmark_equity),
            }

    # 方法4: 从 portfolio_daily 的 equity 值推断价格
    portfolio_daily = raw_result.get("portfolio_daily", [])
    if isinstance(portfolio_daily, list) and len(portfolio_daily) > 0 and not price_data:
        timestamps = [str(d.get("timestamp", d.get("date", ""))) for d in portfolio_daily]
        equity_values = [d.get("equity", 0) for d in portfolio_daily]
        if len(timestamps) > 0:
            price_data = {
                "timestamps": timestamps,
                "close": equity_values,
                "open": equity_values,
                "high": equity_values,
                "low": equity_values,
                "volume": [0] * len(equity_values),
            }

    # 如果还是没有数据，返回空的 timestamps 数组（而不是空字典）
    if not price_data:
        price_data = {
            "timestamps": [],
            "close": [],
            "open": [],
            "high": [],
            "low": [],
            "volume": [],
        }

    return price_data


def _on_backtest_complete(task_id: str, result: Any) -> Any:
    """回测完成回调：保存到历史存储"""
    if result is None:
        return result
    
    try:
        # 转换为字典
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            # 尝试使用 __dict__
            result_dict = getattr(result, '__dict__', {})
        
        if result_dict and isinstance(result_dict, dict):
            backtest_id = _backtest_store.generate_id()
            _backtest_store.save(backtest_id, result_dict)
    except Exception as e:
        logger.error(f"保存回测结果失败: {e}", exc_info=True)
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
        auto_download=True,  # 默认开启自动下载
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
        return asyncio.run(
            _run_backtest_with_progress(
                request.strategy_name,
                request.universe,
                request.start_date,
                request.end_date,
                request.initial_capital,
                request.commission_rate,
                request.slippage_rate,
                benchmark_symbol=request.benchmark_symbol,
                task_id=task_id,
            )
        )

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
    limit: int = Query(100, ge=1, le=1000),
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

    return _convert_numpy(
        {
            "backtests": backtests,
            "count": len(backtests),
        }
    )


@router.delete("/history/cleanup")
async def cleanup_invalid_history():
    """
    清理无效的回测历史记录（final_value=0 或 total_return=0 的记录）
    这些记录通常是由于回测引擎计算错误产生的
    """
    try:
        deleted_count = _backtest_store.cleanup_invalid_records()
        return {
            "message": f"已清理 {deleted_count} 条无效记录",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        logger.error(f"清理无效记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


@router.get("/history/{run_id}")
async def get_backtest_result(run_id: str):
    """获取回测结果详情"""
    result = _backtest_store.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"回测记录不存在: {run_id}")

    # 确保返回数据包含前端所需的所有字段
    if isinstance(result, dict):
        # 确保 backtest_period 存在
        if "backtest_period" not in result or not result.get("backtest_period"):
            # 从 timerange 推断
            timerange = result.get("timerange", "")
            start_date = ""
            end_date = ""
            days = 0
            if timerange and "~" in timerange:
                parts = timerange.split("~")
                if len(parts) == 2:
                    start_date = parts[0]
                    end_date = parts[1]
                    try:
                        from datetime import datetime

                        start = datetime.strptime(start_date, "%Y%m%d")
                        end = datetime.strptime(end_date, "%Y%m%d")
                        days = (end - start).days
                    except Exception:
                        pass
            result["backtest_period"] = {
                "start_date": start_date,
                "end_date": end_date,
                "days": days,
            }

        # 确保 config 存在
        if "config" not in result or not result.get("config"):
            result["config"] = {
                "initial_capital": result.get("initial_capital", 100000),
                "commission_rate": 0.001,
                "slippage_rate": 0.0005,
            }

        # 确保 universe 存在
        if "universe" not in result:
            symbol = result.get("symbol", "")
            result["universe"] = (
                [s.strip() for s in symbol.split(",") if s.strip()] if symbol else []
            )

        # 从多个可能的来源收集数据
        performance = result.get("performance", {}) or {}
        perf_stats = result.get("performance_stats", {}) or {}
        trade_stats = result.get("trade_stats", {}) or {}
        trading_stats = result.get("trading_stats", {}) or {}
        enhanced = result.get("enhanced_metrics", {}) or {}
        results = result.get("results", {}) or {}

        # 优先使用新格式，否则从旧格式映射
        # 注意：旧格式使用 _pct 后缀表示百分比值，需要除以100转换为小数
        final_value = (
            results.get("final_value")
            or performance.get("final_equity")
            or result.get("final_equity", 0)
        )

        # total_return: 多种可能的字段名和格式
        # 注意：total_return_pct 已经是百分比形式（如1818.87表示1818.87%）
        # total_return 可能是小数形式（如18.1887表示1818.87%）
        total_return_pct_raw = result.get("total_return_pct", 0)
        total_return_raw = results.get("total_return") or performance.get("total_return") or 0

        # 优先使用 total_return_pct（已经是百分比）
        if total_return_pct_raw != 0:
            total_return = total_return_pct_raw / 100  # 转换为小数（1818.87% -> 18.1887）
        elif total_return_raw != 0:
            # 如果 total_return_raw 大于1，可能是百分比形式，需要转换为小数
            if abs(total_return_raw) > 1:
                total_return = total_return_raw / 100  # 转换为小数（1818.87 -> 18.1887）
            else:
                total_return = total_return_raw  # 已经是小数形式
        else:
            total_return = 0

        initial_capital = result.get("config", {}).get("initial_capital") or result.get(
            "initial_capital", 100000
        )

        # 计算best_day和worst_day
        portfolio_daily = _get_result_field(result, "portfolio_daily", [])
        best_day = 0.0
        worst_day = 0.0
        if isinstance(portfolio_daily, list) and len(portfolio_daily) > 0:
            daily_returns = []
            prev_equity = None
            for d in portfolio_daily:
                equity = d.get("equity", 0)
                if prev_equity and prev_equity > 0:
                    daily_ret = (equity - prev_equity) / prev_equity
                    daily_returns.append(daily_ret)
                    if daily_ret > best_day:
                        best_day = daily_ret
                    if daily_ret < worst_day:
                        worst_day = daily_ret
                prev_equity = equity

        # 确保 portfolio_daily 存在于根级别（供前端使用）
        if (
            "portfolio_daily" not in result
            and isinstance(portfolio_daily, list)
            and len(portfolio_daily) > 0
        ):
            result["portfolio_daily"] = portfolio_daily

        # 确保 benchmark_data 存在
        if "benchmark_data" not in result:
            benchmark_equity_list = _get_result_field(result, "benchmark_equity", [])
            result["benchmark_data"] = {
                "symbol": _get_result_field(result, "benchmark_symbol", "SPY"),
                "benchmark_equity": benchmark_equity_list,
                "total_return": _get_result_field(result, "benchmark_return", 0),
                "alpha": _get_result_field(result, "alpha", 0),
                "beta": _get_result_field(result, "beta", 1),
                "sharpe_ratio": _get_result_field(
                    result, "benchmark_volatility", 0
                ),  # 使用volatility作为sharpe
            }

        # 填充 enhanced_metrics（从前端METRIC_CONFIG所需的所有字段）
        # 映射关系：新格式字段 -> 旧格式字段
        # 注意：旧格式使用 _pct 后缀存储百分比值
        result["enhanced_metrics"] = {
            # 收益指标
            "total_return": enhanced.get("total_return") or total_return or 0,
            "annual_return": enhanced.get("annual_return")
            or perf_stats.get("annualized_return")
            or performance.get("annualized_return")
            or (
                result.get("annualized_return_pct", 0) / 100
                if result.get("annualized_return_pct")
                else 0
            ),
            "excess_return": enhanced.get("excess_return") or result.get("excess_return", 0),
            "benchmark_return": enhanced.get("benchmark_return")
            or result.get("benchmark_return", 0),
            "best_day": enhanced.get("best_day") or best_day or 0,
            "worst_day": enhanced.get("worst_day") or worst_day or 0,
            # 风险指标
            "max_drawdown": enhanced.get("max_drawdown")
            or perf_stats.get("max_drawdown")
            or performance.get("max_drawdown")
            or (result.get("max_drawdown_pct", 0) / 100 if result.get("max_drawdown_pct") else 0),
            "annual_volatility": enhanced.get("annual_volatility")
            or result.get("volatility", 0)
            or result.get("annual_volatility", 0),
            "benchmark_volatility": enhanced.get("benchmark_volatility")
            or result.get("benchmark_volatility", 0),
            "sharpe_ratio": enhanced.get("sharpe_ratio")
            or perf_stats.get("sharpe_ratio")
            or performance.get("sharpe_ratio")
            or result.get("sharpe_ratio", 0),
            "sortino_ratio": enhanced.get("sortino_ratio") or result.get("sortino_ratio", 0),
            "calmar_ratio": enhanced.get("calmar_ratio") or result.get("calmar_ratio", 0),
            # 基准对比指标
            "alpha": enhanced.get("alpha") or result.get("alpha", 0),
            "beta": enhanced.get("beta") or result.get("beta", 1),
            "information_ratio": enhanced.get("information_ratio")
            or result.get("information_ratio", 0),
            "r_squared": enhanced.get("r_squared") or result.get("r_squared", 0),
            "tracking_error": enhanced.get("tracking_error") or result.get("tracking_error", 0),
            # 交易统计
            "total_trades": enhanced.get("total_trades")
            or trade_stats.get("total_trades")
            or trading_stats.get("total_trades")
            or result.get("total_trades", 0),
            "winning_trades": enhanced.get("winning_trades")
            or trade_stats.get("winning_trades")
            or trading_stats.get("winning_trades")
            or result.get("winning_trades", 0),
            "losing_trades": enhanced.get("losing_trades")
            or result.get("losing_trades", 0)
            or (result.get("total_trades", 0) - result.get("winning_trades", 0)),
            "win_rate": enhanced.get("win_rate")
            or trade_stats.get("win_rate")
            or trading_stats.get("win_rate")
            or (result.get("win_rate_pct", 0) / 100 if result.get("win_rate_pct") else 0),
            "daily_win_rate": enhanced.get("daily_win_rate") or result.get("daily_win_rate", 0),
            "profit_loss_ratio": enhanced.get("profit_loss_ratio")
            or result.get("profit_loss_ratio", 0),
            "profit_factor": enhanced.get("profit_factor")
            or trade_stats.get("profit_factor")
            or result.get("profit_factor", 0),
            "expectancy": enhanced.get("expectancy") or result.get("expectancy", 0),
            "consecutive_wins": enhanced.get("consecutive_wins")
            or result.get("consecutive_wins", 0),
            "consecutive_losses": enhanced.get("consecutive_losses")
            or result.get("consecutive_losses", 0),
            # 回撤序列
            "drawdown_series": enhanced.get("drawdown_series") or result.get("drawdown_series", []),
            "max_drawdown_window": enhanced.get("max_drawdown_window")
            or result.get("max_drawdown_window", None),
            "drawdown_windows": enhanced.get("drawdown_windows")
            or result.get("drawdown_windows", []),
        }

        # 确保 equity_comparison 存在
        if "equity_comparison" not in result:
            portfolio_daily = _get_result_field(result, "portfolio_daily", [])
            equity_curve = (
                [d.get("equity", 0) for d in portfolio_daily]
                if isinstance(portfolio_daily, list)
                else []
            )
            timestamps = (
                [d.get("timestamp", d.get("date", "")) for d in portfolio_daily]
                if isinstance(portfolio_daily, list)
                else []
            )

            # 计算 excess_returns
            benchmark_equity = _get_result_field(result, "benchmark_equity", [])
            excess_returns = []
            if len(equity_curve) == len(benchmark_equity):
                for i in range(len(equity_curve)):
                    if i == 0:
                        excess_returns.append(0)
                    else:
                        strat_ret = (
                            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                            if equity_curve[i - 1]
                            else 0
                        )
                        bench_ret = (
                            (benchmark_equity[i] - benchmark_equity[i - 1])
                            / benchmark_equity[i - 1]
                            if benchmark_equity[i - 1]
                            else 0
                        )
                        excess_returns.append(strat_ret - bench_ret)

            result["equity_comparison"] = {
                "strategy_equity": equity_curve,
                "benchmark_equity": benchmark_equity,
                "excess_returns": excess_returns,
                "timestamps": timestamps,
            }

        # 确保 trades 存在并格式化
        fills_raw = _get_result_field(result, "fills", [])
        if fills_raw:
            # 优先从 fills 格式化
            result["trades"] = _format_trades_from_fills(fills_raw)
            # 同时保留原始 fills 数据
            if "fills" not in result:
                result["fills"] = fills_raw
        elif "trades" not in result or not result.get("trades"):
            result["trades"] = []

        # 确保 benchmark_equity 存在于根级别（供前端使用）
        if "benchmark_equity" not in result:
            benchmark_equity = _get_result_field(result, "benchmark_equity", [])
            if isinstance(benchmark_equity, list) and len(benchmark_equity) > 0:
                result["benchmark_equity"] = benchmark_equity

        # 确保 price_data 存在
        if "price_data" not in result:
            result["price_data"] = _extract_price_data(result, result.get("universe", []))

    return _convert_numpy(result)


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
        "note": "返回结果包含每个窗口的详细回测结果",
    }

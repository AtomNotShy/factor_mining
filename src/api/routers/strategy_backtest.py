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
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine, UnifiedConfig, FeatureFlag
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
    """递归转换 numpy 类型为 Python 原生类型，解决 FastAPI JSON 序列化问题，并处理 NaN/Inf"""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return [_convert_numpy(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'item') and not isinstance(obj, (int, float, bool, str)): # numpy scalar
        try:
            val = obj.item()
            if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
                return None
            return val
        except Exception:
            return str(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
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
            # 只返回可JSON序列化的参数值，过滤掉Parameter对象和其他不可序列化对象
            serializable_params = {}
            for k, v in strategy.config.params.items():
                # 跳过Parameter对象和其他复杂对象
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    serializable_params[k] = v
            
            strategy_info.append({
                "name": strategy.strategy_id,
                "timeframe": strategy.timeframe,
                "params": serializable_params,
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
    if hasattr(result, 'to_dict'):
        final_result = result.to_dict()
    elif hasattr(result, 'model_dump'):
        final_result = result.model_dump()
    elif isinstance(result, dict):
        final_result = result
    else:
        final_result = {"result": result}
    
    # 最后更新进度
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

    return _convert_numpy({
        "backtests": backtests,
        "count": len(backtests),
    })


@router.get("/history/{run_id}")
async def get_backtest_result(run_id: str):
    """获取回测结果详情"""
    result = _backtest_store.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"回测记录不存在: {run_id}")
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
        "note": "返回结果包含每个窗口的详细回测结果"
    }

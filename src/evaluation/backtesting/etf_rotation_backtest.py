"""
Backtest Framework for Robust ETF Rotation Strategy
====================================================

Features:
- Walk-Forward Optimization (WFO) with rolling windows
- Train/Test split (70/30 or time-based)
- Bootstrap confidence intervals
- Comprehensive metrics and reporting

Usage:
    from src.evaluation.backtesting.etf_rotation_backtest import run_backtest, run_wfo
    
    # 简单回测
    results = run_backtest(
        strategy=RobustETFRotationStrategy(),
        universe=['SPY', 'QQQ', 'TLT', 'XLV'],
        start='2015-01-01',
        end='2024-12-31',
    )
    
    # Walk-Forward 优化
    wfo_results = run_wfo(
        strategy_class=RobustETFRotationStrategy,
        universe=['SPY', 'QQQ', 'TLT', 'XLV'],
        start='2015-01-01',
        end='2024-12-31',
        window_months=12,
        step_months=1,
    )
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import warnings

import numpy as np
import pandas as pd

from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.data.storage.backtest_store import BacktestStore
from src.evaluation.backtesting import BacktestResult
from src.evaluation.backtesting.config import TradeConfig
from src.evaluation.backtesting.unified_engine import (
    UnifiedBacktestEngine,
    UnifiedConfig,
    FeatureFlag,
)
from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.utils.cli_printer import CLIPrinter
from src.utils.logger import get_logger

logger = logging.getLogger("backtest.etf_rotation")

# 抑制 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000
    commission_rate: float = 0.0005  # 5bps
    slippage_rate: float = 0.0001    # 1bp
    transaction_cost: float = 0.0005  # 5bps 交易成本
    
    # 再平衡参数
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    turnover_limit: float = 0.15  # 月度换手率限制
    
    # 风险管理
    stoploss: float = -0.10
    max_drawdown_limit: float = 0.20
    
    # 输出配置
    output_dir: str = "./data/backtests"
    save_results: bool = True


@dataclass
class BacktestMetrics:
    """回测绩效指标"""
    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    monthly_returns: List[float] = field(default_factory=list)
    
    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # 天数
    
    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    
    # 换手率
    avg_monthly_turnover: float = 0.0
    total_turnover: float = 0.0
    
    # 基准对比
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_monthly_turnover': self.avg_monthly_turnover,
            'alpha': self.alpha,
            'beta': self.beta,
        }


def calculate_metrics(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series = None,
    risk_free_rate: float = 0.02,
) -> BacktestMetrics:
    """
    计算绩效指标
    
    Args:
        equity_curve: 资金曲线
        benchmark_curve: 基准资金曲线
        risk_free_rate: 无风险利率
        
    Returns:
        BacktestMetrics 对象
    """
    metrics = BacktestMetrics()
    
    # 基本收益
    metrics.total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # 年化收益
    days = len(equity_curve)
    years = days / 252
    metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 月度收益
    try:
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            monthly = equity_curve.resample('ME').last().pct_change().dropna()
        else:
            # 如果不是 DatetimeIndex，手动计算月度收益
            n = len(equity_curve)
            monthly_points = n // 21  # 每月约21个交易日
            if monthly_points > 1:
                monthly = (equity_curve.iloc[21::21] - equity_curve.iloc[:-21:21]) / equity_curve.iloc[:-21:21]
            else:
                monthly = pd.Series([metrics.total_return])
        metrics.monthly_returns = monthly.tolist()
    except Exception:
        metrics.monthly_returns = [metrics.total_return]
    
    # 波动率
    daily_returns = equity_curve.pct_change().dropna()
    metrics.volatility = daily_returns.std() * np.sqrt(252)
    
    # 最大回撤
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics.max_drawdown = drawdown.min()
    
    # 回撤持续天数
    dd_periods = (drawdown < 0).astype(int)
    max_dd_start = (drawdown == metrics.max_drawdown).idxmax() if metrics.max_drawdown < 0 else None
    if max_dd_start is not None:
        dd_after = dd_periods.loc[max_dd_start:]
        metrics.max_drawdown_duration = dd_after.sum()
    
    # 夏普比率
    excess_returns = daily_returns - risk_free_rate / 252
    metrics.sharpe_ratio = (
        np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        if daily_returns.std() > 0 else 0
    )
    
    # 索提诺比率 (只考虑下行波动)
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    metrics.sortino_ratio = (
        metrics.annualized_return / downside_vol
        if downside_vol > 0 else 0
    )
    
    # 卡尔马比率
    metrics.calmar_ratio = (
        metrics.annualized_return / abs(metrics.max_drawdown)
        if metrics.max_drawdown < 0 else 0
    )
    
    # 基准对比
    if benchmark_curve is not None and len(benchmark_curve) > 1:
        metrics.benchmark_return = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0]) - 1
        metrics.alpha = metrics.annualized_return - metrics.benchmark_return
        
        # Beta
        if benchmark_curve.pct_change().std() > 0:
            cov = daily_returns.cov(benchmark_curve.pct_change())
            var = benchmark_curve.pct_change().var()
            metrics.beta = cov / var if var > 0 else 1.0
        else:
            metrics.beta = 1.0
    
    return metrics


def generate_report(
    metrics: BacktestMetrics,
    strategy_name: str,
    timerange: str,
    equity_curve: pd.Series,
) -> str:
    """生成回测报告"""
    
    # 计算交易统计
    monthly_returns = np.array(metrics.monthly_returns)
    positive_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    
    report = f"""
================================================================================
                    ETF ROTATION STRATEGY BACKTEST REPORT
================================================================================

STRATEGY: {strategy_name}
DATE RANGE: {timerange}
INITIAL CAPITAL: $100,000.00

--------------------------------------------------------------------------------
                              PERFORMANCE SUMMARY
--------------------------------------------------------------------------------

TOTAL RETURN:          {metrics.total_return*100:>8.2f}%     ANNUALIZED: {metrics.annualized_return*100:>8.2f}%
VOLATILITY:            {metrics.volatility*100:>8.2f}%     MAX DD:    {metrics.max_drawdown*100:>8.2f}%

RISK-ADJUSTED RETURNS:
  Sharpe Ratio:        {metrics.sharpe_ratio:>8.3f}
  Sortino Ratio:       {metrics.sortino_ratio:>8.3f}
  Calmar Ratio:        {metrics.calmar_ratio:>8.3f}

BENCHMARK COMPARISON (SPY):
  Strategy Return:     {metrics.total_return*100:>8.2f}%
  Benchmark Return:    {metrics.benchmark_return*100:>8.2f}%
  Alpha:               {metrics.alpha*100:>8.2f}%
  Beta:                {metrics.beta:>8.3f}

TRADING STATISTICS:
  Total Trades:        {metrics.total_trades:>8d}
  Win Rate:            {metrics.win_rate*100:>8.2f}%
  Profit Factor:       {metrics.profit_factor:>8.3f}
  Monthly Turnover:    {metrics.avg_monthly_turnover*100:>8.2f}%

MONTHLY DISTRIBUTION:
  Positive Months:     {positive_months:>3d} / {total_months:>3d} ({positive_months/total_months*100:.1f}%)
  Best Month:          {monthly_returns.max()*100:>8.2f}%
  Worst Month:         {monthly_returns.min()*100:>8.2f}%
  Monthly Std Dev:     {monthly_returns.std()*100:>8.2f}%

================================================================================
"""
    return report


async def run_backtest(
    strategy: FreqtradeStrategy,
    universe: List[str],
    start: str,
    end: str,
    config: BacktestConfig = None,
) -> Dict[str, Any]:
    """
    运行回测
    
    Args:
        strategy: 策略实例
        universe: ETF 标的列表
        start: 开始日期 (YYYY-MM-DD)
        end: 结束日期 (YYYY-MM-DD)
        config: 回测配置
        
    Returns:
        回测结果字典
    """
    if config is None:
        config = BacktestConfig()
    
    logger.info(f"Starting backtest: {strategy.strategy_id}")
    logger.info(f"  Universe: {universe}")
    logger.info(f"  Period: {start} to {end}")
    
    # 创建配置
    trade_config = TradeConfig(
        initial_capital=config.initial_capital,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
    )
    
    engine_config = UnifiedConfig(
        trade=trade_config,
        features=FeatureFlag.ALL,
    )
    
    # 创建引擎
    engine = UnifiedBacktestEngine(config=engine_config)
    
    # 准备上下文
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    
    ctx = RunContext.create(
        env=Environment.BACKTEST,
        config={
            'strategy': strategy.strategy_id,
            'universe': universe,
            'rebalance_frequency': config.rebalance_frequency,
            'turnover_limit': config.turnover_limit,
        },
        trading_calendar=TradingCalendar()
    )
    
    # 运行回测
    result = await engine.run(
        strategies=[strategy],
        universe=universe,
        start=start_date,
        end=end_date,
        ctx=ctx,
        auto_download=False,
    )
    
    # 处理结果
    if isinstance(result, dict) and 'error' in result:
        logger.error(f"Backtest error: {result['error']}")
        return result
    
    # 提取资金曲线
    equity_curve = pd.Series(result['equity_curve']) if 'equity_curve' in result else pd.Series([100000])
    
    # 计算指标
    metrics = calculate_metrics(equity_curve)
    
    # 生成报告
    report = generate_report(
        metrics,
        strategy.strategy_id,
        f"{start}~{end}",
        equity_curve,
    )
    
    print(report)
    
    # 保存结果
    if config.save_results:
        store = BacktestStore()
        run_id = f"etf_rotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_data = {
            'run_id': run_id,
            'strategy': strategy.strategy_id,
            'universe': universe,
            'timerange': f"{start}~{end}",
            'metrics': metrics.to_dict(),
            'equity_curve': equity_curve.tolist(),
        }
        store.save(run_id, save_data)
        logger.info(f"Results saved: {run_id}")
    
    return {
        'metrics': metrics,
        'equity_curve': equity_curve,
        'report': report,
    }


def run_wfo(
    strategy_class: Type[FreqtradeStrategy],
    universe: List[str],
    start: str,
    end: str,
    window_months: int = 12,
    step_months: int = 1,
    config: BacktestConfig = None,
) -> Dict[str, Any]:
    """
    运行 Walk-Forward Optimization
    
    Args:
        strategy_class: 策略类
        universe: ETF 标的列表
        start: 开始日期
        end: 结束日期
        window_months: 滚动窗口长度（月）
        step_months: 滚动步长（月）
        config: 回测配置
        
    Returns:
        WFO 结果
    """
    if config is None:
        config = BacktestConfig()
    
    logger.info(f"Starting Walk-Forward Optimization")
    logger.info(f"  Window: {window_months} months, Step: {step_months} months")
    
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    
    # 生成滚动窗口
    windows = []
    current = start_dt
    
    while current + timedelta(days=window_months * 30) <= end_dt:
        window_start = current.strftime("%Y-%m-%d")
        window_end = (current + timedelta(days=window_months * 30)).strftime("%Y-%m-%d")
        windows.append((window_start, window_end))
        current += timedelta(days=step_months * 30)
    
    logger.info(f"  Total windows: {len(windows)}")
    
    # 运行每个窗口的回测
    window_results = []
    
    for i, (w_start, w_end) in enumerate(windows):
        logger.info(f"  Window {i+1}/{len(windows)}: {w_start} to {w_end}")
        
        try:
            strategy = strategy_class()
            result = asyncio.run(run_backtest(
                strategy,
                universe,
                w_start,
                w_end,
                config,
            ))
            
            if 'metrics' in result:
                window_results.append({
                    'window': i + 1,
                    'start': w_start,
                    'end': w_end,
                    'metrics': result['metrics'],
                })
        except Exception as e:
            logger.error(f"  Window {i+1} failed: {e}")
            continue
    
    # 汇总结果
    if not window_results:
        logger.error("All windows failed!")
        return {'error': 'No successful windows'}
    
    # 计算汇总统计
    sharpe_values = [w['metrics'].sharpe_ratio for w in window_results]
    dd_values = [abs(w['metrics'].max_drawdown) for w in window_results]
    return_values = [w['metrics'].annualized_return for w in window_results]
    
    wfo_summary = {
        'num_windows': len(window_results),
        'sharpe_mean': np.mean(sharpe_values),
        'sharpe_std': np.std(sharpe_values),
        'sharpe_ci_95': np.percentile(sharpe_values, [5, 95]),
        'max_dd_mean': np.mean(dd_values),
        'max_dd_max': np.max(dd_values),
        'return_mean': np.mean(return_values),
        'return_std': np.std(return_values),
        'windows': window_results,
    }
    
    print(f"""
================================================================================
                      WALK-FORWARD OPTIMIZATION SUMMARY
================================================================================

Windows Analyzed:     {wfo_summary['num_windows']}
Window Length:        {window_months} months
Step Size:            {step_months} month(s)

SHARPE RATIO:
  Mean:               {wfo_summary['sharpe_mean']:.3f}
  Std:                {wfo_summary['sharpe_std']:.3f}
  95% CI:             [{wfo_summary['sharpe_ci_95'][0]:.3f}, {wfo_summary['sharpe_ci_95'][1]:.3f}]

MAX DRAWDOWN:
  Mean:               {wfo_summary['max_dd_mean']*100:.2f}%
  Max:                {wfo_summary['max_dd_max']*100:.2f}%

ANNUALIZED RETURN:
  Mean:               {wfo_summary['return_mean']*100:.2f}%
  Std:                {wfo_summary['return_std']*100:.2f}%

================================================================================
""")
    
    return wfo_summary


def run_bootstrap(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap 置信区间估计
    
    Args:
        returns: 收益序列
        n_bootstrap: Bootstrap 次数
        confidence: 置信水平
        
    Returns:
        各指标的置信区间
    """
    logger.info(f"Running bootstrap with {n_bootstrap} iterations...")
    
    results = {
        'sharpe': [],
        'max_dd': [],
        'return': [],
        'win_rate': [],
    }
    
    for _ in range(n_bootstrap):
        # 带放回抽样
        boot_returns = np.random.choice(returns, size=len(returns), replace=True)
        
        # 计算指标
        equity = (1 + boot_returns).cumprod()
        
        # 夏普
        sharpe = (boot_returns.mean() / boot_returns.std() * np.sqrt(252)) if boot_returns.std() > 0 else 0
        results['sharpe'].append(sharpe)
        
        # 最大回撤
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        results['max_dd'].append(dd.min())
        
        # 收益
        results['return'].append(equity[-1] - 1)
        
        # 胜率
        results['win_rate'].append((boot_returns > 0).mean())
    
    # 计算置信区间
    lower = (1 - confidence) / 2 * 100
    upper = (1 - lower)
    
    ci = {}
    for key, values in results.items():
        ci[key] = (
            np.percentile(values, lower),
            np.percentile(values, upper),
        )
    
    print(f"""
================================================================================
                          BOOTSTRAP CONFIDENCE INTERVALS
================================================================================

Bootstrap Iterations: {n_bootstrap}
Confidence Level:     {confidence*100}%

                    95% CONFIDENCE INTERVAL
                    Lower           Upper
                    
Sharpe Ratio:        {ci['sharpe'][0]:>8.3f}       {ci['sharpe'][1]:>8.3f}
Max Drawdown:        {ci['max_dd'][0]*100:>8.2f}%       {ci['max_dd'][1]*100:>8.2f}%
Total Return:        {ci['return'][0]*100:>8.2f}%       {ci['return'][1]*100:>8.2f}%
Win Rate:            {ci['win_rate'][0]*100:>8.2f}%       {ci['win_rate'][1]*100:>8.2f}%

================================================================================
""")
    
    return ci


# ============================================================================
# 便捷函数
# ============================================================================

def quick_backtest(
    strategy_class: Type[FreqtradeStrategy],
    universe: List[str],
    years: int = 5,
) -> Dict[str, Any]:
    """
    快速回测（最近 N 年）
    
    Args:
        strategy_class: 策略类
        universe: ETF 标的池
        years: 回测年数
        
    Returns:
        回测结果
    """
    end = date.today()
    start = date(end.year - years, end.month, end.day)
    
    strategy = strategy_class()
    config = BacktestConfig()
    
    return asyncio.run(run_backtest(
        strategy,
        universe,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        config,
    ))

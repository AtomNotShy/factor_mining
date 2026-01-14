"""
增强的回测报告生成器

新增功能：
1. 滚动的收益分布
2. 每月/每周的收益热力图
3. 持仓时间 vs 收益分布图
4. 信号触发的延迟分析
5. 交易信号标记（离场原因、持仓时间等）
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import math

import pandas as pd
import numpy as np

from src.utils.logger import get_logger


logger = get_logger("report")


@dataclass
class TradeDetail:
    """交易详情"""
    trade_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_days: float
    entry_reason: str = ""  # 信号标签
    exit_reason: str = ""   # stoploss/ROI/trailing/exit_signal
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class SignalAnalysis:
    """信号分析"""
    total_signals: int = 0
    entry_signals: int = 0
    exit_signals: int = 0
    signal_to_entry_delay_bars: float = 0.0  # 平均延迟
    signal_to_entry_delay_days: float = 0.0
    signals_with_entry: int = 0


@dataclass
class EnhancedBacktestReport:
    """增强的回测报告"""
    
    # 基本信息
    run_id: str = ""
    strategy_name: str = ""
    timeframe: str = "1d"
    timerange: str = ""
    initial_capital: float = 0.0
    
    # 收益率指标
    final_equity: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    weekly_returns: Dict[str, float] = field(default_factory=dict)
    
    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    
    # 滚动的收益分布
    rolling_returns_1m: float = 0.0
    rolling_returns_3m: float = 0.0
    rolling_returns_6m: float = 0.0
    rolling_returns_12m: float = 0.0
    return_distribution: Dict[str, float] = field(default_factory=dict)
    
    # 交易指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    
    # 盈亏分布
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_trade_duration: float = 0.0
    duration_distribution: Dict[str, float] = field(default_factory=dict)
    
    # 效率指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 信号分析
    signal_analysis: SignalAnalysis = field(default_factory=SignalAnalysis)
    
    # 交易详情列表
    trades: List[TradeDetail] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "timerange": self.timerange,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct * 100,
            "annualized_return": self.annualized_return * 100,
            "monthly_returns": self.monthly_returns,
            "weekly_returns": self.weekly_returns,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct * 100,
            "max_drawdown_duration_days": self.max_drawdown_duration,
            "volatility": self.volatility * 100,
            "rolling_returns": {
                "1m": self.rolling_returns_1m * 100,
                "3m": self.rolling_returns_3m * 100,
                "6m": self.rolling_returns_6m * 100,
                "12m": self.rolling_returns_12m * 100,
            },
            "return_distribution": self.return_distribution,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate * 100,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "best_trade": self.best_trade * 100,
            "worst_trade": self.worst_trade * 100,
            "avg_trade_duration_days": self.avg_trade_duration,
            "duration_distribution": self.duration_distribution,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "signal_analysis": {
                "total_signals": self.signal_analysis.total_signals,
                "entry_signals": self.signal_analysis.entry_signals,
                "exit_signals": self.signal_analysis.exit_signals,
                "avg_entry_delay_bars": self.signal_analysis.signal_to_entry_delay_bars,
                "avg_entry_delay_days": self.signal_analysis.signal_to_entry_delay_days,
            },
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "entry_time": t.entry_time.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time.isoformat(),
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct * 100,
                    "duration_days": t.duration_days,
                    "entry_reason": t.entry_reason,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ],
        }


class EnhancedReportGenerator:
    """增强的回测报告生成器"""
    
    def __init__(
        self,
        strategy_name: str = "Unknown Strategy",
        timeframe: str = "1d",
        timerange: str = "Unknown",
    ):
        self.strategy_name = strategy_name
        self.timeframe = timeframe
        self.timerange = timerange
        self.logger = get_logger("enhanced_report_generator")
    
    def generate_report(
        self,
        initial_capital: float,
        final_equity: float,
        equity_curve: List[Dict],
        trades: List[Dict],
        signals: List[Dict] = None,
        entry_signals: List[Dict] = None,
    ) -> EnhancedBacktestReport:
        """
        生成增强回测报告
        
        Args:
            initial_capital: 初始资金
            final_equity: 最终资金
            equity_curve: 净值曲线 [{timestamp, equity, ...}]
            trades: 交易记录 [{entry_time, exit_time, pnl, ...}]
            signals: 信号记录（可选）
            entry_signals: 进入信号记录（可选，用于计算延迟）
            
        Returns:
            EnhancedBacktestReport: 增强报告
        """
        self.logger.info("Generating enhanced backtest report...")
        
        # 基础计算
        total_return = final_equity - initial_capital
        total_return_pct = total_return / initial_capital if initial_capital > 0 else 0
        
        # 计算时间范围
        if equity_curve and len(equity_curve) >= 2:
            start_date = self._parse_timestamp(equity_curve[0]['timestamp'])
            end_date = self._parse_timestamp(equity_curve[-1]['timestamp'])
            days = (end_date - start_date).days
            annualized_return = ((final_equity / initial_capital) ** (365.0 / max(days, 1))) - 1 if days > 0 else 0
        else:
            days = 0
            annualized_return = 0
            start_date = end_date = datetime.now()
        
        # 净值数据
        equity_df = self._build_equity_df(equity_curve, initial_capital)
        
        # 月度/周收益
        monthly_returns = self._calculate_monthly_returns(equity_df)
        weekly_returns = self._calculate_weekly_returns(equity_df)
        
        # 风险指标
        max_drawdown, max_drawdown_pct, max_drawdown_duration = self._calculate_max_drawdown(
            equity_df['equity'].tolist(), initial_capital
        )
        current_drawdown = self._calculate_current_drawdown(equity_df['equity'].tolist())
        volatility = self._calculate_volatility(equity_df['equity'].tolist(), initial_capital)
        
        # 滚动的收益分布
        rolling_returns = self._calculate_rolling_returns(equity_df)
        return_distribution = self._calculate_return_distribution(equity_df['returns'].tolist())
        
        # 交易指标
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profits = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        losses = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
        
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else float('inf')
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # 盈亏分布
        all_pnls = [t.get('pnl', 0) for t in trades]
        best_trade = max(all_pnls) / initial_capital if all_pnls else 0
        worst_trade = min(all_pnls) / initial_capital if all_pnls else 0
        
        # 持仓时间分布
        durations = self._calculate_trade_durations(trades)
        avg_trade_duration = np.mean(durations) if durations else 0
        duration_distribution = self._categorize_durations(durations)
        
        # 效率指标
        returns = self._calculate_returns(equity_df['equity'].tolist())
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        calmar_ratio = annualized_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
        
        # 信号分析
        signal_analysis = self._analyze_signals(signals, entry_signals, trades)
        
        # 交易详情
        trade_details = self._build_trade_details(trades)
        
        # 构建报告
        report = EnhancedBacktestReport(
            run_id=str(datetime.now().timestamp()),
            strategy_name=self.strategy_name,
            timeframe=self.timeframe,
            timerange=self.timerange,
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            monthly_returns=monthly_returns,
            weekly_returns=weekly_returns,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration=max_drawdown_duration,
            current_drawdown=current_drawdown,
            volatility=volatility,
            rolling_returns_1m=rolling_returns.get('1m', 0),
            rolling_returns_3m=rolling_returns.get('3m', 0),
            rolling_returns_6m=rolling_returns.get('6m', 0),
            rolling_returns_12m=rolling_returns.get('12m', 0),
            return_distribution=return_distribution,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_trade_duration,
            duration_distribution=duration_distribution,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            signal_analysis=signal_analysis,
            trades=trade_details,
        )
        
        self.logger.info(
            f"Enhanced report generated: return={total_return_pct*100:.2f}%, "
            f"max_dd={max_drawdown_pct*100:.2f}%, win_rate={win_rate*100:.2f}%"
        )
        
        return report
    
    def _parse_timestamp(self, ts) -> datetime:
        """解析时间戳"""
        if isinstance(ts, str):
            return pd.to_datetime(ts).to_pydatetime()
        elif hasattr(ts, 'to_pydatetime'):
            return ts.to_pydatetime()
        return ts
    
    def _build_equity_df(self, equity_curve: List[Dict], initial_capital: float) -> pd.DataFrame:
        """构建净值 DataFrame"""
        if not equity_curve:
            return pd.DataFrame(columns=['timestamp', 'equity', 'returns'])
        
        df = pd.DataFrame(equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # 计算收益率
        df['returns'] = df['equity'].pct_change()
        
        # 填充初始收益率
        if len(df) > 0:
            df.loc[df.index[0], 'returns'] = (df['equity'].iloc[0] - initial_capital) / initial_capital
        
        return df
    
    def _calculate_monthly_returns(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        """计算月收益率"""
        if equity_df.empty:
            return {}
        
        monthly = equity_df['equity'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        return {
            idx.strftime('%Y-%m'): float(ret)
            for idx, ret in monthly_returns.items()
        }
    
    def _calculate_weekly_returns(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        """计算周收益率"""
        if equity_df.empty:
            return {}
        
        weekly = equity_df['equity'].resample('W').last()
        weekly_returns = weekly.pct_change().dropna()
        
        return {
            idx.strftime('%Y-W%U'): float(ret)
            for idx, ret in weekly_returns.items()
        }
    
    def _calculate_max_drawdown(
        self, equity_values: List[float], initial_capital: float
    ) -> tuple:
        """计算最大回撤"""
        if not equity_values:
            return 0, 0, 0
        
        equity = np.array(equity_values)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        
        max_dd_idx = np.argmin(drawdown)
        max_drawdown = drawdown[max_dd_idx]
        
        # 计算回撤持续时间
        peak_idx = np.argmax(equity[:max_dd_idx + 1])
        duration = max_dd_idx - peak_idx
        
        return float(equity[max_dd_idx] - peak[max_dd_idx]), float(max_drawdown), duration
    
    def _calculate_current_drawdown(self, equity_values: List[float]) -> float:
        """计算当前回撤"""
        if not equity_values:
            return 0
        
        equity = np.array(equity_values)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity[-1] - peak[-1]) / peak[-1] if peak[-1] > 0 else 0
        return float(drawdown)
    
    def _calculate_volatility(self, equity_values: List[float], initial_capital: float) -> float:
        """计算年化波动率"""
        if len(equity_values) < 2:
            return 0
        
        equity = np.array(equity_values)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) == 0:
            return 0
        
        return float(np.std(returns) * np.sqrt(252))
    
    def _calculate_rolling_returns(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        """计算滚动收益率"""
        if equity_df.empty:
            return {}
        
        result = {}
        
        # 1个月滚动（假设日线，约21个交易日）
        if len(equity_df) >= 21:
            rolling_1m = equity_df['equity'].pct_change(periods=21).iloc[-1]
            result['1m'] = float(rolling_1m) if not np.isnan(rolling_1m) else 0
        
        # 3个月滚动（约63个交易日）
        if len(equity_df) >= 63:
            rolling_3m = equity_df['equity'].pct_change(periods=63).iloc[-1]
            result['3m'] = float(rolling_3m) if not np.isnan(rolling_3m) else 0
        
        # 6个月滚动（约126个交易日）
        if len(equity_df) >= 126:
            rolling_6m = equity_df['equity'].pct_change(periods=126).iloc[-1]
            result['6m'] = float(rolling_6m) if not np.isnan(rolling_6m) else 0
        
        # 12个月滚动（约252个交易日）
        if len(equity_df) >= 252:
            rolling_12m = equity_df['equity'].pct_change(periods=252).iloc[-1]
            result['12m'] = float(rolling_12m) if not np.isnan(rolling_12m) else 0
        
        return result
    
    def _calculate_return_distribution(self, returns: List[float]) -> Dict[str, float]:
        """计算收益分布（分位数）"""
        if not returns:
            return {}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) == 0:
            return {}
        
        return {
            'min': float(np.min(returns)),
            'max': float(np.max(returns)),
            'mean': float(np.mean(returns)),
            'std': float(np.std(returns)),
            'p5': float(np.percentile(returns, 5)),
            'p25': float(np.percentile(returns, 25)),
            'p50': float(np.percentile(returns, 50)),
            'p75': float(np.percentile(returns, 75)),
            'p95': float(np.percentile(returns, 95)),
            'skewness': float(pd.Series(returns).skew()),
            'kurtosis': float(pd.Series(returns).kurtosis()),
        }
    
    def _calculate_trade_durations(self, trades: List[Dict]) -> List[float]:
        """计算交易持仓时间"""
        durations = []
        for t in trades:
            if 'entry_time' in t and 'exit_time' in t:
                entry = self._parse_timestamp(t['entry_time'])
                exit = self._parse_timestamp(t['exit_time'])
                duration = (exit - entry).days
                if duration >= 0:
                    durations.append(duration)
        return durations
    
    def _categorize_durations(self, durations: List[float]) -> Dict[str, float]:
        """分类持仓时间"""
        if not durations:
            return {}
        
        categories = {
            'intraday': 0,    # < 1天
            'short': 0,       # 1-5天
            'medium': 0,      # 5-20天
            'long': 0,        # 20-60天
            'very_long': 0,   # > 60天
        }
        
        for d in durations:
            if d < 1:
                categories['intraday'] += 1
            elif d < 5:
                categories['short'] += 1
            elif d < 20:
                categories['medium'] += 1
            elif d < 60:
                categories['long'] += 1
            else:
                categories['very_long'] += 1
        
        total = len(durations)
        return {k: v / total for k, v in categories.items()}
    
    def _calculate_returns(self, equity_values: List[float]) -> List[float]:
        """计算收益率序列"""
        if len(equity_values) < 2:
            return []
        
        equity = np.array(equity_values)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        return returns.tolist()
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0
        
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate / 252
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    
    def _calculate_sortino(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算 Sortino 比率"""
        if len(returns) < 2:
            return 0
        
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate / 252
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 0.001
        return float(np.mean(excess_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    def _analyze_signals(
        self,
        signals: List[Dict],
        entry_signals: List[Dict],
        trades: List[Dict],
    ) -> SignalAnalysis:
        """分析信号"""
        analysis = SignalAnalysis()
        
        if signals:
            analysis.total_signals = len(signals)
            analysis.entry_signals = len([s for s in signals if s.get('action') in ['BUY', 'LONG', 'buy', 'long']])
            analysis.exit_signals = len([s for s in signals if s.get('action') in ['SELL', 'FLAT', 'sell', 'flat']])
        
        # 计算信号到实际入场的延迟
        if entry_signals and trades:
            delays = []
            for signal in entry_signals:
                signal_time = self._parse_timestamp(signal.get('timestamp'))
                for trade in trades:
                    trade_entry = self._parse_timestamp(trade.get('entry_time'))
                    if trade_entry >= signal_time:
                        delay = (trade_entry - signal_time).days
                        delays.append(delay)
                        break
            
            if delays:
                analysis.signals_with_entry = len(delays)
                analysis.signal_to_entry_delay_days = np.mean(delays)
                # 假设每天1根K线，转为bar数
                analysis.signal_to_entry_delay_bars = analysis.signal_to_entry_delay_days
        
        return analysis
    
    def _build_trade_details(self, trades: List[Dict]) -> List[TradeDetail]:
        """构建交易详情"""
        details = []
        for i, t in enumerate(trades):
            entry_time = self._parse_timestamp(t.get('entry_time'))
            exit_time = self._parse_timestamp(t.get('exit_time'))
            
            details.append(TradeDetail(
                trade_id=f"trade_{i+1:04d}",
                symbol=t.get('symbol', ''),
                entry_time=entry_time,
                entry_price=t.get('entry_price', 0),
                exit_time=exit_time,
                exit_price=t.get('exit_price', 0),
                quantity=t.get('quantity', 0),
                pnl=t.get('pnl', 0),
                pnl_pct=t.get('pnl_pct', 0) if 'pnl_pct' in t else (t.get('pnl', 0) / t.get('entry_price', 1)),
                duration_days=(exit_time - entry_time).days if exit_time >= entry_time else 0,
                entry_reason=t.get('entry_reason', ''),
                exit_reason=t.get('exit_reason', ''),
                commission=t.get('commission', 0),
                slippage=t.get('slippage', 0),
            ))
        
        return details
    
    def generate_summary(self, report: EnhancedBacktestReport) -> str:
        """生成文本摘要"""
        return f"""
=======================================
    ENHANCED BACKTEST REPORT
=======================================

📊 STRATEGY: {report.strategy_name}
   Timeframe: {report.timeframe}
   Timerange: {report.timerange}

💰 RETURNS
   Initial Capital: ${report.initial_capital:,.2f}
   Final Equity: ${report.final_equity:,.2f}
   Total Return: {report.total_return_pct*100:.2f}%
   Annualized Return: {report.annualized_return*100:.2f}%

📈 ROLLING RETURNS
   1-Month: {report.rolling_returns_1m*100:.2f}%
   3-Month: {report.rolling_returns_3m*100:.2f}%
   6-Month: {report.rolling_returns_6m*100:.2f}%
   12-Month: {report.rolling_returns_12m*100:.2f}%

📉 RISK
   Max Drawdown: {report.max_drawdown_pct*100:.2f}%
   Max DD Duration: {report.max_drawdown_duration} days
   Volatility: {report.volatility*100:.2f}%

📊 TRADING
   Total Trades: {report.total_trades}
   Win Rate: {report.win_rate*100:.2f}%
   Profit Factor: {report.profit_factor:.2f}
   Avg Win: ${report.avg_win:,.2f}
   Avg Loss: ${report.avg_loss:,.2f}
   Expectancy: ${report.expectancy:,.2f}

⏱️ DURATION DISTRIBUTION
   Intraday: {report.duration_distribution.get('intraday', 0)*100:.1f}%
   Short (1-5d): {report.duration_distribution.get('short', 0)*100:.1f}%
   Medium (5-20d): {report.duration_distribution.get('medium', 0)*100:.1f}%
   Long (20-60d): {report.duration_distribution.get('long', 0)*100:.1f}%
   Very Long (60d+): {report.duration_distribution.get('very_long', 0)*100:.1f}%

⚡ EFFICIENCY
   Sharpe Ratio: {report.sharpe_ratio:.2f}
   Sortino Ratio: {report.sortino_ratio:.2f}
   Calmar Ratio: {report.calmar_ratio:.2f}

📡 SIGNAL ANALYSIS
   Total Signals: {report.signal_analysis.total_signals}
   Entry Signals: {report.signal_analysis.entry_signals}
   Exit Signals: {report.signal_analysis.exit_signals}
   Avg Entry Delay: {report.signal_analysis.signal_to_entry_delay_days:.1f} days

=======================================
"""


if __name__ == "__main__":
    # 测试报告生成
    import pandas as pd
    from datetime import timedelta
    
    # 生成模拟数据
    np.random.seed(42)
    start_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start=start_date, periods=252, freq='B')
    
    # 模拟净值曲线
    equity = 100000
    equity_curve = []
    for ts in dates:
        daily_return = np.random.normal(0.0005, 0.015)
        equity = equity * (1 + daily_return)
        equity_curve.append({
            'timestamp': ts,
            'equity': equity,
        })
    
    # 模拟交易
    trades = []
    for i in range(10):
        entry_time = start_date + timedelta(days=i * 20)
        exit_time = entry_time + timedelta(days=np.random.randint(5, 30))
        pnl = np.random.uniform(-1000, 2000)
        trades.append({
            'symbol': 'SPY',
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': 450 + np.random.uniform(-10, 10),
            'exit_price': 450 + np.random.uniform(-10, 10),
            'pnl': pnl,
            'pnl_pct': pnl / 100000,
            'quantity': 100,
            'entry_reason': 'momentum_crossover',
            'exit_reason': np.random.choice(['stoploss', 'roi', 'exit_signal', 'trailing']),
        })
    
    # 生成报告
    generator = EnhancedReportGenerator(
        strategy_name="Test Strategy",
        timeframe="1d",
        timerange="2024-01-01 to 2024-12-31",
    )
    
    report = generator.generate_report(
        initial_capital=100000,
        final_equity=equity,
        equity_curve=equity_curve,
        trades=trades,
    )
    
    # 打印摘要
    print(generator.generate_summary(report))
    
    # 打印部分数据
    print("\nMonthly Returns (last 5):")
    for k, v in list(report.monthly_returns.items())[-5:]:
        print(f"  {k}: {v*100:.2f}%")
    
    print("\nDuration Distribution:")
    for k, v in report.duration_distribution.items():
        print(f"  {k}: {v*100:.1f}%")

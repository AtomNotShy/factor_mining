"""
综合性能分析模块
整合回撤分析、基准对比、交易统计等所有增强指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from src.utils.logger import get_logger


@dataclass
class EnhancedMetrics:
    """增强版回测指标"""
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_loss_ratio: float
    
    excess_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    r_squared: Optional[float] = None
    
    ulcer_index: Optional[float] = None
    burke_ratio: Optional[float] = None
    time_in_market: Optional[float] = None
    
    avg_daily_return: Optional[float] = None
    best_day: Optional[float] = None
    worst_day: Optional[float] = None
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    
    def to_dict(self) -> Dict:
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                result[key] = None
            else:
                result[key] = value
        return result


@dataclass
class BenchmarkData:
    """基准数据"""
    symbol: str
    equity_curve: List[float]
    timestamps: List[str]
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DrawdownWindowResult:
    """回撤区间结果"""
    start_date: str
    end_date: str
    peak_date: str
    trough_date: str
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EnhancedAnalyzer:
    """增强版性能分析器"""
    
    def __init__(self, benchmark_symbol: str = "QQQ"):
        self.logger = get_logger("enhanced_analyzer")
        self.benchmark_symbol = benchmark_symbol
        
        from src.evaluation.metrics.performance import PerformanceAnalyzer
        from src.evaluation.metrics.benchmark import BenchmarkAnalyzer
        from src.evaluation.metrics.drawdown import DrawdownAnalyzer
        
        self.perf_analyzer = PerformanceAnalyzer()
        self.benchmark_analyzer = BenchmarkAnalyzer(benchmark_symbol)
        self.drawdown_analyzer = DrawdownAnalyzer()
    
    def analyze_trades(
        self,
        trades: List[Dict],
        portfolio_value: pd.Series
    ) -> Dict:
        """详细分析交易记录"""
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'profit_loss_ratio': 0.0,
                    'gross_profit': 0.0,
                    'gross_loss': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'consecutive_wins': 0,
                    'consecutive_losses': 0,
                    'expectancy': 0.0,
                    'profit_factor': 0.0
                }
            
            pnl_list = []
            buy_trades = {}
            
            for trade in trades:
                symbol = trade.get('symbol', 'UNKNOWN')
                order_type = str(trade.get('order_type', '')).lower()
                size = abs(trade.get('size', 0))
                price = trade.get('price', 0)
                
                if symbol not in buy_trades:
                    buy_trades[symbol] = []
                
                if order_type == 'buy':
                    buy_trades[symbol].append({
                        'size': size,
                        'price': price,
                        'timestamp': trade.get('timestamp')
                    })
                elif order_type == 'sell' and buy_trades[symbol]:
                    buy = buy_trades[symbol].pop(0)
                    pnl = (price - buy['price']) * min(size, buy['size'])
                    pnl_list.append({
                        'pnl': pnl,
                        'timestamp': trade.get('timestamp')
                    })
            
            if not pnl_list:
                return {
                    'total_trades': len(trades),
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'profit_loss_ratio': 0.0,
                    'gross_profit': 0.0,
                    'gross_loss': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'consecutive_wins': 0,
                    'consecutive_losses': 0,
                    'expectancy': 0.0,
                    'profit_factor': 0.0
                }
            
            wins = [p['pnl'] for p in pnl_list if p['pnl'] > 0]
            losses = [p['pnl'] for p in pnl_list if p['pnl'] < 0]
            
            total_trades = len(pnl_list)
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            
            largest_win = max(wins) if wins else 0.0
            largest_loss = min(losses) if losses else 0.0
            
            profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0
            
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
            
            consecutive_wins = self._calculate_max_consecutive(pnl_list, positive=True)
            consecutive_losses = self._calculate_max_consecutive(pnl_list, positive=False)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'expectancy': expectancy,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            self.logger.error(f"交易分析失败: {e}")
            return {}
    
    def _calculate_max_consecutive(
        self,
        pnl_list: List[Dict],
        positive: bool
    ) -> int:
        """计算最大连续盈利/亏损次数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl_dict in pnl_list:
            pnl = pnl_dict['pnl']
            if positive:
                if pnl > 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            else:
                if pnl < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
        
        return max_consecutive
    
    def get_benchmark_data(
        self,
        start_date: str,
        end_date: str
    ) -> Optional[BenchmarkData]:
        """获取基准数据"""
        try:
            benchmark_df = self.benchmark_analyzer.get_benchmark_data(
                start_date=start_date,
                end_date=end_date,
                data_source="local"
            )
            
            # 如果本地没有，尝试从 IB 获取
            if benchmark_df is None or benchmark_df.empty:
                benchmark_df = self.benchmark_analyzer.get_benchmark_data(
                    start_date=start_date,
                    end_date=end_date,
                    data_source="ib"
                )
            
            if benchmark_df is None or benchmark_df.empty:
                self.logger.warning(f"无法获取基准数据: {self.benchmark_symbol}")
                return None
            
            benchmark_returns = self.benchmark_analyzer.calculate_returns_from_prices(
                pd.Series(benchmark_df['close'])
            )
            
            equity_curve = self.benchmark_analyzer.calculate_cumulative_equity(
                benchmark_returns
            ).tolist()
            
            timestamps = [str(ts) for ts in benchmark_df.index]
            
            total_return = (1 + benchmark_returns).cumprod().iloc[-1] - 1
            annual_return = benchmark_returns.mean() * 252
            annual_volatility = benchmark_returns.std() * np.sqrt(252)
            sharpe = (annual_return - 0.02) / annual_volatility if annual_volatility > 0 else 0.0
            
            dd_info = self.perf_analyzer.calculate_max_drawdown(benchmark_returns)
            max_drawdown = dd_info.get('max_drawdown', 0.0)
            
            return BenchmarkData(
                symbol=self.benchmark_symbol,
                equity_curve=equity_curve,
                timestamps=timestamps,
                total_return=total_return,
                annual_return=annual_return,
                annual_volatility=annual_volatility,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            self.logger.error(f"获取基准数据失败: {e}")
            return None
    
    def comprehensive_analysis(
        self,
        returns: pd.Series,
        portfolio_value: pd.Series,
        trades: List[Dict],
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_equity: Optional[List[float]] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Dict:
        """
        完整性能分析
        
        Args:
            returns: 策略收益率序列
            portfolio_value: 组合净值序列
            trades: 交易记录列表
            benchmark_returns: 基准收益率序列
            benchmark_equity: 基准净值序列
            risk_free_rate: 无风险收益率
            periods_per_year: 年化周期数
            
        Returns:
            完整分析结果
        """
        try:
            total_return = (1 + returns).cumprod().iloc[-1] - 1
            annual_return = returns.mean() * periods_per_year
            annual_volatility = returns.std() * np.sqrt(periods_per_year)
            sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
            
            sortino = self.perf_analyzer.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
            calmar = self.perf_analyzer.calculate_calmar_ratio(returns, periods_per_year)
            
            dd_info = self.perf_analyzer.calculate_max_drawdown(returns)
            max_drawdown = dd_info.get('max_drawdown', 0.0)
            max_drawdown_duration = dd_info.get('max_drawdown_duration', 0)
            
            win_rate = self.perf_analyzer.calculate_win_rate(returns)
            pl_ratio = self.perf_analyzer.calculate_profit_loss_ratio(returns)
            
            avg_daily_return = returns.mean()
            best_day = returns.max()
            worst_day = returns.min()
            
            trade_stats = self.analyze_trades(trades, portfolio_value)
            
            result = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'win_rate': win_rate,
                'profit_loss_ratio': pl_ratio,
                'avg_daily_return': avg_daily_return,
                'best_day': best_day,
                'worst_day': worst_day,
                **trade_stats
            }
            
            if benchmark_returns is not None:
                benchmark_analysis = self.benchmark_analyzer.comprehensive_benchmark_analysis(
                    returns, benchmark_returns, risk_free_rate
                )
                
                # 计算基准收益和波动率
                benchmark_total_return = (1 + benchmark_returns).cumprod().iloc[-1] - 1
                benchmark_annual_volatility = benchmark_returns.std() * np.sqrt(periods_per_year)
                
                # 计算日胜率
                daily_wins = (returns > 0).sum()
                daily_total = len(returns)
                daily_win_rate = daily_wins / daily_total if daily_total > 0 else 0.0
                
                result.update({
                    'benchmark_return': benchmark_total_return,
                    'benchmark_volatility': benchmark_annual_volatility,
                    'excess_return': benchmark_analysis.get('excess_return'),
                    'alpha': benchmark_analysis.get('alpha'),
                    'beta': benchmark_analysis.get('beta'),
                    'tracking_error': benchmark_analysis.get('tracking_error'),
                    'information_ratio': benchmark_analysis.get('information_ratio'),
                    'r_squared': benchmark_analysis.get('r_squared'),
                    'daily_win_rate': daily_win_rate
                })
                
                if benchmark_equity is not None:
                    result['benchmark_equity'] = benchmark_equity
            
            drawdown_analysis = self.drawdown_analyzer.comprehensive_drawdown_analysis(
                portfolio_value, risk_free_rate
            )

            # 调试：检查drawdown_analysis
            self.logger.debug(f"Drawdown analysis keys: {list(drawdown_analysis.keys()) if drawdown_analysis else 'None'}")
            if drawdown_analysis and 'drawdown_series' in drawdown_analysis:
                drawdown_series_val = drawdown_analysis['drawdown_series']
                self.logger.debug(f"Drawdown series type: {type(drawdown_series_val)}")
                self.logger.debug(f"Drawdown series length: {len(drawdown_series_val) if drawdown_series_val is not None else 'None'}")
                if drawdown_series_val:
                    self.logger.debug(f"Drawdown series first 3: {drawdown_series_val[:3]}")
            else:
                self.logger.debug("Drawdown analysis missing drawdown_series")

            # 调试：检查最终结果
            self.logger.debug(f"Final result will include drawdown_series: {'drawdown_series' in result}")
            if 'drawdown_series' in result:
                self.logger.debug(f"Result drawdown_series length: {len(result['drawdown_series']) if result['drawdown_series'] else 'Empty'}")

            drawdown_series_val = drawdown_analysis.get('drawdown_series')

            result.update({
                'ulcer_index': drawdown_analysis.get('ulcer_index'),
                'burke_ratio': drawdown_analysis.get('burke_ratio'),
                'time_in_market': drawdown_analysis.get('time_in_drawdown_pct'),
                'max_drawdown_window': drawdown_analysis.get('max_drawdown_window'),
                'drawdown_windows': drawdown_analysis.get('drawdown_windows', []),
                'drawdown_series': drawdown_series_val
            })

            return result
            
        except Exception as e:
            self.logger.error(f"综合分析失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def generate_equity_comparison(
        self,
        strategy_equity: List[float],
        benchmark_equity: Optional[List[float]] = None,
        timestamps: Optional[List[Any]] = None
    ) -> Dict:
        """生成净值对比数据"""
        try:
            n = len(strategy_equity)
            if timestamps is None or len(timestamps) != n:
                timestamps = list(range(n))

            result = {
                'strategy_equity': strategy_equity,
                'timestamps': [str(ts) if ts is not None else f"t{i}" for i, ts in enumerate(timestamps)]
            }

            if benchmark_equity is not None:
                result['benchmark_equity'] = benchmark_equity

                min_len = min(len(strategy_equity), len(benchmark_equity))
                excess_returns = []
                for i in range(min_len):
                    if benchmark_equity[i] > 0:
                        excess = (strategy_equity[i] / benchmark_equity[i] - 1) * 100
                        excess_returns.append(excess)
                    else:
                        excess_returns.append(0.0)

                result['excess_returns'] = excess_returns

            return result

        except Exception as e:
            self.logger.error(f"生成净值对比失败: {e}")
            return {}

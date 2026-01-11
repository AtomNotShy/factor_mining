"""
性能分析器
提供策略回测性能分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.utils.logger import get_logger


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.logger = get_logger("performance_analyzer")
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """计算收益率"""
        return prices.pct_change().fillna(0)
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """计算累计收益率"""
        return (1 + returns).cumprod() - 1
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险收益率（年化）
            periods_per_year: 年化周期数
        """
        try:
            if len(returns) == 0:
                return np.nan
            
            # 年化收益率
            annual_return = returns.mean() * periods_per_year
            
            # 年化波动率
            annual_volatility = returns.std() * np.sqrt(periods_per_year)
            
            if annual_volatility == 0:
                return np.nan
            
            # 夏普比率
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            return sharpe
            
        except Exception as e:
            self.logger.error(f"计算夏普比率失败: {e}")
            return np.nan
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Dict:
        """
        计算最大回撤
        
        Returns:
            包含最大回撤、开始时间、结束时间等信息的字典
        """
        try:
            if len(returns) == 0:
                return {}
            
            # 计算累计收益率
            cumulative = self.calculate_cumulative_returns(returns)
            
            # 计算累计最高点
            running_max = cumulative.expanding().max()
            
            # 计算回撤
            drawdown = (cumulative - running_max) / (1 + running_max)
            
            # 最大回撤
            max_drawdown = drawdown.min()
            
            # 找到最大回撤的时间点
            max_dd_end = drawdown.idxmin()
            max_dd_start = cumulative.loc[:max_dd_end].idxmax()
            
            # 计算回撤持续时间
            if pd.notna(max_dd_start) and pd.notna(max_dd_end):
                try:
                    if isinstance(max_dd_start, str):
                        max_dd_start = pd.to_datetime(max_dd_start)
                    if isinstance(max_dd_end, str):
                        max_dd_end = pd.to_datetime(max_dd_end)
                    
                    # 如果索引是整数，则使用索引差值
                    if isinstance(max_dd_start, (int, np.integer)) or isinstance(max_dd_end, (int, np.integer)):
                        duration = int(max_dd_end) - int(max_dd_start)
                    else:
                        # 如果是datetime，计算天数差
                        duration = (max_dd_end - max_dd_start).days
                except:
                    duration = 0
            else:
                duration = 0
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_start': max_dd_start,
                'max_drawdown_end': max_dd_end,
                'max_drawdown_duration': duration,
                'drawdown_series': drawdown
            }
            
        except Exception as e:
            self.logger.error(f"计算最大回撤失败: {e}")
            return {}
    
    def calculate_calmar_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """计算卡玛比率"""
        try:
            annual_return = returns.mean() * periods_per_year
            max_dd_info = self.calculate_max_drawdown(returns)
            
            if not max_dd_info or max_dd_info.get('max_drawdown', 0) == 0:
                return np.nan
            
            calmar = annual_return / abs(max_dd_info['max_drawdown'])
            return calmar
            
        except Exception as e:
            self.logger.error(f"计算卡玛比率失败: {e}")
            return np.nan
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """计算索提诺比率"""
        try:
            annual_return = returns.mean() * periods_per_year
            
            # 下行标准差
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() * np.sqrt(periods_per_year)
            
            if downside_std == 0:
                return np.nan
            
            sortino = (annual_return - risk_free_rate) / downside_std
            return sortino
            
        except Exception as e:
            self.logger.error(f"计算索提诺比率失败: {e}")
            return np.nan
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """计算胜率"""
        try:
            if len(returns) == 0:
                return np.nan
            
            win_rate = (returns > 0).mean()
            return win_rate
            
        except Exception as e:
            self.logger.error(f"计算胜率失败: {e}")
            return np.nan
    
    def calculate_profit_loss_ratio(self, returns: pd.Series) -> float:
        """计算盈亏比"""
        try:
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return np.nan
            
            avg_profit = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            if avg_loss == 0:
                return np.nan
            
            pl_ratio = avg_profit / avg_loss
            return pl_ratio
            
        except Exception as e:
            self.logger.error(f"计算盈亏比失败: {e}")
            return np.nan
    
    def calculate_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """计算年化波动率"""
        try:
            if len(returns) == 0:
                return np.nan
            
            volatility = returns.std() * np.sqrt(periods_per_year)
            return volatility
            
        except Exception as e:
            self.logger.error(f"计算波动率失败: {e}")
            return np.nan
    
    def calculate_skewness(self, returns: pd.Series) -> float:
        """计算偏度"""
        try:
            if len(returns) < 3:
                return np.nan
            
            skewness = returns.skew()
            return skewness
            
        except Exception as e:
            self.logger.error(f"计算偏度失败: {e}")
            return np.nan
    
    def calculate_kurtosis(self, returns: pd.Series) -> float:
        """计算峰度"""
        try:
            if len(returns) < 4:
                return np.nan
            
            kurtosis = returns.kurtosis()
            return kurtosis
            
        except Exception as e:
            self.logger.error(f"计算峰度失败: {e}")
            return np.nan
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """计算风险价值 (VaR)"""
        try:
            if len(returns) == 0:
                return np.nan
            
            var = returns.quantile(confidence_level)
            return var
            
        except Exception as e:
            self.logger.error(f"计算VaR失败: {e}")
            return np.nan
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """计算条件风险价值 (CVaR)"""
        try:
            if len(returns) == 0:
                return np.nan
            
            var = self.calculate_var(returns, confidence_level)
            cvar = returns[returns <= var].mean()
            return cvar
            
        except Exception as e:
            self.logger.error(f"计算CVaR失败: {e}")
            return np.nan
    
    def calculate_information_ratio(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """计算信息比率"""
        try:
            # 超额收益
            excess_returns = strategy_returns - benchmark_returns
            
            if len(excess_returns) == 0:
                return np.nan
            
            # 跟踪误差
            tracking_error = excess_returns.std()
            
            if tracking_error == 0:
                return np.nan
            
            # 信息比率
            ir = excess_returns.mean() / tracking_error
            return ir
            
        except Exception as e:
            self.logger.error(f"计算信息比率失败: {e}")
            return np.nan
    
    def calculate_beta(
        self, 
        strategy_returns: pd.Series, 
        market_returns: pd.Series
    ) -> float:
        """计算贝塔系数"""
        try:
            # 对齐数据
            aligned_data = pd.concat([strategy_returns, market_returns], axis=1).dropna()
            if len(aligned_data) < 2:
                return np.nan
            
            strategy_aligned = aligned_data.iloc[:, 0]
            market_aligned = aligned_data.iloc[:, 1]
            
            # 计算贝塔
            covariance = np.cov(strategy_aligned, market_aligned)[0, 1]
            market_variance = np.var(market_aligned)
            
            if market_variance == 0:
                return np.nan
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            self.logger.error(f"计算贝塔系数失败: {e}")
            return np.nan
    
    def calculate_cagr(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        计算年化复合增长率 (CAGR)
        
        Args:
            returns: 收益率序列
            periods_per_year: 年化周期数
            
        Returns:
            CAGR
        """
        try:
            if len(returns) == 0:
                return np.nan
            
            cumulative = self.calculate_cumulative_returns(returns)
            total_return = cumulative.iloc[-1]
            
            if total_return <= -1:
                return np.nan
            
            # 计算年数
            if len(returns) > 1:
                years = len(returns) / periods_per_year
                if years > 0:
                    cagr = (1 + total_return) ** (1 / years) - 1
                    return cagr
            
            return np.nan
            
        except Exception as e:
            self.logger.error(f"计算CAGR失败: {e}")
            return np.nan
    
    def calculate_expectancy(
        self,
        trades: List,
        order_type_field: str = "order_type"
    ) -> Dict:
        """
        计算期望值 (Expectancy)
        
        Args:
            trades: 交易记录列表
            order_type_field: 订单类型字段名
            
        Returns:
            期望值统计
        """
        try:
            if not trades:
                return {}
            
            # 配对买卖交易计算盈亏
            pnl_list = []
            buy_trades = []
            
            for trade in trades:
                order_type = getattr(trade, order_type_field, None) or trade.get(order_type_field)
                size = getattr(trade, 'size', None) or trade.get('size', 0)
                price = getattr(trade, 'price', None) or trade.get('price', 0)
                
                if (hasattr(order_type, 'value') and order_type.value == 'buy') or \
                   (isinstance(order_type, str) and order_type.lower() == 'buy'):
                    buy_trades.append({'size': size, 'price': price, 'trade': trade})
                elif (hasattr(order_type, 'value') and order_type.value == 'sell') or \
                     (isinstance(order_type, str) and order_type.lower() == 'sell'):
                    if buy_trades:
                        buy = buy_trades.pop(0)
                        pnl = (price - buy['price']) * min(abs(size), abs(buy['size']))
                        pnl_list.append(pnl)
            
            if not pnl_list:
                return {
                    'expectancy': np.nan,
                    'avg_win': np.nan,
                    'avg_loss': np.nan,
                    'win_rate': np.nan,
                }
            
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            win_rate = len(wins) / len(pnl_list) if pnl_list else 0
            
            # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            return {
                'expectancy': expectancy,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_rate': win_rate,
                'total_trades': len(pnl_list),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
            }
            
        except Exception as e:
            self.logger.error(f"计算期望值失败: {e}")
            return {}
    
    def calculate_profit_factor(self, trades: List, order_type_field: str = "order_type") -> float:
        """
        计算盈利因子 (Profit Factor)
        
        Args:
            trades: 交易记录列表
            order_type_field: 订单类型字段名
            
        Returns:
            盈利因子
        """
        try:
            expectancy_info = self.calculate_expectancy(trades, order_type_field)
            
            avg_win = expectancy_info.get('avg_win', 0)
            avg_loss = expectancy_info.get('avg_loss', 0)
            
            if avg_loss == 0:
                return np.nan if avg_win == 0 else np.inf
            
            profit_factor = avg_win / avg_loss
            return profit_factor
            
        except Exception as e:
            self.logger.error(f"计算盈利因子失败: {e}")
            return np.nan
    
    def comprehensive_analysis(
        self, 
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        trades: Optional[List] = None,
    ) -> Dict:
        """
        综合性能分析
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列（可选）
            risk_free_rate: 无风险收益率
            periods_per_year: 年化周期数
            
        Returns:
            综合分析结果字典
        """
        try:
            if len(returns) == 0:
                return {}
            
            # 基础统计
            total_return = self.calculate_cumulative_returns(returns).iloc[-1]
            cagr = self.calculate_cagr(returns, periods_per_year)
            annual_return = returns.mean() * periods_per_year
            volatility = self.calculate_volatility(returns, periods_per_year)
            
            # 风险调整指标
            sharpe = self.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
            sortino = self.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
            calmar = self.calculate_calmar_ratio(returns, periods_per_year)
            
            # 回撤分析
            drawdown_info = self.calculate_max_drawdown(returns)
            
            # 分布特征
            skewness = self.calculate_skewness(returns)
            kurtosis = self.calculate_kurtosis(returns)
            
            # 风险指标
            var_5 = self.calculate_var(returns, 0.05)
            var_1 = self.calculate_var(returns, 0.01)
            cvar_5 = self.calculate_cvar(returns, 0.05)
            cvar_1 = self.calculate_cvar(returns, 0.01)
            
            # 胜率和盈亏比
            win_rate = self.calculate_win_rate(returns)
            pl_ratio = self.calculate_profit_loss_ratio(returns)
            
            results = {
                'total_return': total_return,
                'cagr': cagr,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': drawdown_info.get('max_drawdown', np.nan),
                'max_drawdown_duration': drawdown_info.get('max_drawdown_duration', np.nan),
                'win_rate': win_rate,
                'profit_loss_ratio': pl_ratio,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'var_5pct': var_5,
                'var_1pct': var_1,
                'cvar_5pct': cvar_5,
                'cvar_1pct': cvar_1,
            }
            
            # 如果提供了交易记录，计算交易相关指标
            if trades:
                expectancy_info = self.calculate_expectancy(trades)
                profit_factor = self.calculate_profit_factor(trades)
                trade_stats = self.calculate_trade_analysis(trades, returns)
                
                results.update({
                    'expectancy': expectancy_info.get('expectancy', np.nan),
                    'profit_factor': profit_factor,
                    'avg_trade': expectancy_info.get('avg_win', 0) - expectancy_info.get('avg_loss', 0),
                    'total_trades': trade_stats.get('total_trades', 0),
                    'trades_per_day': trade_stats.get('trades_per_day', np.nan),
                    'turnover': np.nan,  # 需要portfolio_values，稍后补充
                })
            
            # 如果提供了基准，计算相对指标
            if benchmark_returns is not None:
                try:
                    ir = self.calculate_information_ratio(returns, benchmark_returns)
                    beta = self.calculate_beta(returns, benchmark_returns)
                    
                    results.update({
                        'information_ratio': ir,
                        'beta': beta
                    })
                except Exception as e:
                    self.logger.warning(f"计算相对指标失败: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"综合性能分析失败: {e}")
            return {}
    
    def calculate_turnover_rate(
        self,
        trades: List,
        portfolio_values: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        计算换手率
        
        Args:
            trades: 交易记录列表
            portfolio_values: 投资组合价值序列
            periods_per_year: 年化周期数
            
        Returns:
            年化换手率
        """
        try:
            if not trades or len(portfolio_values) == 0:
                return np.nan
            
            # 计算总交易金额
            total_trade_value = sum(abs(t.size * t.price) for t in trades)
            
            # 平均投资组合价值
            avg_portfolio_value = portfolio_values.mean()
            
            if avg_portfolio_value == 0:
                return np.nan
            
            # 换手率
            turnover = total_trade_value / avg_portfolio_value
            
            # 年化
            if len(portfolio_values) > 1:
                days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
                if days > 0:
                    annual_turnover = turnover * (periods_per_year / days)
                    return annual_turnover
            
            return turnover
            
        except Exception as e:
            self.logger.error(f"计算换手率失败: {e}")
            return np.nan
    
    def calculate_holding_period(
        self,
        trades: List,
        order_type_field: str = "order_type"
    ) -> Dict:
        """
        计算平均持仓时间
        
        Args:
            trades: 交易记录列表
            order_type_field: 订单类型字段名
            
        Returns:
            包含平均持仓时间等信息的字典
        """
        try:
            if not trades:
                return {}
            
            # 按标的分组，计算持仓时间
            positions = {}  # {symbol: {entry_time, entry_size, exit_times: []}}
            holding_periods = []
            
            for trade in trades:
                symbol = getattr(trade, 'symbol', None) or trade.get('symbol', 'UNKNOWN')
                timestamp = getattr(trade, 'timestamp', None) or trade.get('timestamp')
                order_type = getattr(trade, order_type_field, None) or trade.get(order_type_field)
                size = getattr(trade, 'size', None) or trade.get('size', 0)
                
                if symbol not in positions:
                    positions[symbol] = {
                        'entry_time': None,
                        'entry_size': 0,
                        'exit_times': []
                    }
                
                pos = positions[symbol]
                
                # 买入
                if (hasattr(order_type, 'value') and order_type.value == 'buy') or \
                   (isinstance(order_type, str) and order_type.lower() == 'buy'):
                    if pos['entry_time'] is None:
                        pos['entry_time'] = timestamp
                        pos['entry_size'] = size
                    else:
                        pos['entry_size'] += size
                
                # 卖出
                elif (hasattr(order_type, 'value') and order_type.value == 'sell') or \
                     (isinstance(order_type, str) and order_type.lower() == 'sell'):
                    if pos['entry_time'] is not None:
                        # 计算持仓时间
                        if isinstance(timestamp, str):
                            timestamp = pd.to_datetime(timestamp)
                        if isinstance(pos['entry_time'], str):
                            pos['entry_time'] = pd.to_datetime(pos['entry_time'])
                        
                        holding_days = (timestamp - pos['entry_time']).days
                        if holding_days >= 0:
                            holding_periods.append(holding_days)
                        
                        # 更新持仓
                        pos['entry_size'] -= size
                        if pos['entry_size'] <= 0:
                            pos['entry_time'] = None
                            pos['entry_size'] = 0
            
            if len(holding_periods) == 0:
                return {
                    'avg_holding_period_days': np.nan,
                    'median_holding_period_days': np.nan,
                    'min_holding_period_days': np.nan,
                    'max_holding_period_days': np.nan,
                    'total_positions': 0
                }
            
            return {
                'avg_holding_period_days': np.mean(holding_periods),
                'median_holding_period_days': np.median(holding_periods),
                'min_holding_period_days': np.min(holding_periods),
                'max_holding_period_days': np.max(holding_periods),
                'total_positions': len(holding_periods)
            }
            
        except Exception as e:
            self.logger.error(f"计算持仓时间失败: {e}")
            return {}
    
    def calculate_trade_analysis(
        self,
        trades: List,
        returns: pd.Series
    ) -> Dict:
        """
        交易分析
        
        Args:
            trades: 交易记录列表
            returns: 收益率序列
            
        Returns:
            交易分析结果
        """
        try:
            if not trades:
                return {}
            
            buy_trades = []
            sell_trades = []
            
            for trade in trades:
                order_type = getattr(trade, 'order_type', None) or trade.get('order_type')
                
                if (hasattr(order_type, 'value') and order_type.value == 'buy') or \
                   (isinstance(order_type, str) and order_type.lower() == 'buy'):
                    buy_trades.append(trade)
                elif (hasattr(order_type, 'value') and order_type.value == 'sell') or \
                     (isinstance(order_type, str) and order_type.lower() == 'sell'):
                    sell_trades.append(trade)
            
            # 计算交易统计
            total_trades = len(trades)
            buy_count = len(buy_trades)
            sell_count = len(sell_trades)
            
            # 计算平均交易金额
            buy_amounts = [abs(t.size * t.price) for t in buy_trades if hasattr(t, 'size') and hasattr(t, 'price')]
            sell_amounts = [abs(t.size * t.price) for t in sell_trades if hasattr(t, 'size') and hasattr(t, 'price')]
            
            avg_buy_amount = np.mean(buy_amounts) if buy_amounts else 0
            avg_sell_amount = np.mean(sell_amounts) if sell_amounts else 0
            
            # 计算总手续费
            total_commission = sum(
                getattr(t, 'commission', 0) or t.get('commission', 0) for t in trades
            )
            
            # 计算交易频率（每日交易次数）
            if len(returns) > 0:
                trading_days = len(returns)
                trades_per_day = total_trades / trading_days if trading_days > 0 else 0
            else:
                trades_per_day = 0
            
            return {
                'total_trades': total_trades,
                'buy_trades': buy_count,
                'sell_trades': sell_count,
                'avg_buy_amount': avg_buy_amount,
                'avg_sell_amount': avg_sell_amount,
                'total_commission': total_commission,
                'trades_per_day': trades_per_day,
                'trade_ratio': buy_count / sell_count if sell_count > 0 else np.nan
            }
            
        except Exception as e:
            self.logger.error(f"交易分析失败: {e}")
            return {}
    
    def rolling_analysis(
        self,
        returns: pd.Series,
        window: int = 252,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        滚动性能分析
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
            metrics: 需要计算的指标列表
            
        Returns:
            滚动指标DataFrame
        """
        try:
            if metrics is None:
                metrics = ['sharpe_ratio', 'volatility', 'max_drawdown']
            
            rolling_results = pd.DataFrame(index=returns.index)
            
            for metric in metrics:
                if metric == 'sharpe_ratio':
                    rolling_results[metric] = returns.rolling(window).apply(
                        lambda x: self.calculate_sharpe_ratio(x), raw=False
                    )
                elif metric == 'volatility':
                    rolling_results[metric] = returns.rolling(window).apply(
                        lambda x: self.calculate_volatility(x), raw=False
                    )
                elif metric == 'max_drawdown':
                    rolling_results[metric] = returns.rolling(window).apply(
                        lambda x: self.calculate_max_drawdown(x).get('max_drawdown', np.nan), 
                        raw=False
                    )
            
            return rolling_results
            
        except Exception as e:
            self.logger.error(f"滚动分析失败: {e}")
            return pd.DataFrame() 
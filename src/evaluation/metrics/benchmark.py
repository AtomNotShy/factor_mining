"""
基准对比分析模块
提供策略与基准（如SPY）的对比分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import get_logger


@dataclass
class BenchmarkMetrics:
    """基准指标"""
    symbol: str
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    equity_curve: List[float]


@dataclass 
class ComparisonMetrics:
    """对比指标"""
    excess_return: float
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float
    r_squared: float
    correlation: float


class BenchmarkAnalyzer:
    """基准对比分析器"""

    def __init__(self, benchmark_symbol: str = "QQQ"):
        self.logger = get_logger("benchmark_analyzer")
        self.benchmark_symbol = benchmark_symbol
        self._benchmark_cache: Dict[str, pd.DataFrame | pd.Series] = {}
    
    def get_benchmark_data(
        self,
        start_date: str,
        end_date: str,
        data_source: str = "local"
    ) -> Optional[pd.DataFrame | pd.Series]:
        """
        获取基准数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            data_source: 数据来源 (local/ib)

        Returns:
            包含基准价格数据的DataFrame
        """
        try:
            cache_key = f"{self.benchmark_symbol}_{start_date}_{end_date}"

            if cache_key in self._benchmark_cache:
                return self._benchmark_cache[cache_key]

            df: Optional[pd.DataFrame | pd.Series] = None

            if data_source == "ib":
                from datetime import datetime, timezone as dt_timezone
                from src.data.collectors.ib_history import IBHistoryCollector
                collector = IBHistoryCollector()
                start_dt = datetime.fromisoformat(start_date).replace(tzinfo=dt_timezone.utc)
                end_dt = datetime.fromisoformat(end_date).replace(tzinfo=dt_timezone.utc)
                df = collector.get_ohlcv(
                    symbol=self.benchmark_symbol,
                    timeframe="1d",
                    since=start_dt,
                    end=end_dt,
                    use_cache=True
                )
            else:
                from src.config.settings import get_settings
                import os
                safe_symbol = self.benchmark_symbol.upper().replace("/", "_")
                data_dir = get_settings().storage.data_dir
                cache_path = os.path.join(data_dir, "polygon/ohlcv/adjusted/utc/1d", f"{safe_symbol}.parquet")
                if os.path.exists(cache_path):
                    df = pd.read_parquet(cache_path)
                    if "datetime" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        df = df.set_index("datetime")
                    if not df.empty:
                        start_ts = pd.Timestamp(start_date)
                        end_ts = pd.Timestamp(end_date)
                        df = df[(df.index >= start_ts) & (df.index <= end_ts)]

            if df is not None and len(df) > 0:
                self._benchmark_cache[cache_key] = df
                return df

            self.logger.warning(f"无法获取基准数据: {self.benchmark_symbol}")
            return None

        except Exception as e:
            self.logger.error(f"获取基准数据失败: {e}")
            return None
    
    def calculate_returns_from_prices(self, prices: pd.Series) -> pd.Series:
        """从价格序列计算收益率"""
        if len(prices) == 0:
            return pd.Series(dtype=float)
        return prices.pct_change().fillna(0)
    
    def calculate_cumulative_equity(self, returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
        """计算累计净值曲线"""
        cumulative = (1 + returns).cumprod()
        return cumulative * initial_value
    
    def align_returns(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        对齐策略和基准的收益率序列
        
        Returns:
            对齐后的策略收益率和基准收益率
        """
        try:
            if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
                return strategy_returns, benchmark_returns
            
            combined_index = strategy_returns.index.union(benchmark_returns.index)
            strategy_aligned = strategy_returns.reindex(combined_index).dropna()
            benchmark_aligned = benchmark_returns.reindex(combined_index).dropna()
            
            common_index = strategy_aligned.index.intersection(benchmark_aligned.index)
            return strategy_aligned.reindex(common_index), benchmark_aligned.reindex(common_index)
            
        except Exception as e:
            self.logger.error(f"对齐收益率序列失败: {e}")
            return strategy_returns, benchmark_returns
    
    def calculate_alpha_jensen(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        计算 Jensen's Alpha
        
        Alpha = Rp - [Rf + Beta * (Rm - Rf)]
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            risk_free_rate: 无风险收益率
            periods_per_year: 年化周期数
            
        Returns:
            Alpha值
        """
        try:
            strategy_aligned, benchmark_aligned = self.align_returns(
                strategy_returns, benchmark_returns
            )
            
            if len(strategy_aligned) < 2:
                return np.nan
            
            beta = self.calculate_beta(strategy_aligned, benchmark_aligned)
            if np.isnan(beta):
                return np.nan
            
            annual_strategy = strategy_aligned.mean() * periods_per_year
            annual_benchmark = benchmark_aligned.mean() * periods_per_year
            
            alpha = annual_strategy - (risk_free_rate + beta * (annual_benchmark - risk_free_rate))
            return alpha
            
        except Exception as e:
            self.logger.error(f"计算Alpha失败: {e}")
            return np.nan
    
    def calculate_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算 Beta 系数
        
        Beta = Cov(Rp, Rm) / Var(Rm)
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            
        Returns:
            Beta系数
        """
        try:
            strategy_aligned, benchmark_aligned = self.align_returns(
                strategy_returns, benchmark_returns
            )
            
            if len(strategy_aligned) < 2:
                return np.nan
            
            covariance = np.cov(strategy_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            
            if benchmark_variance == 0:
                return np.nan
            
            return covariance / benchmark_variance
            
        except Exception as e:
            self.logger.error(f"计算Beta失败: {e}")
            return np.nan
    
    def calculate_excess_return(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算超额收益
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            
        Returns:
            超额收益（总收益差）
        """
        try:
            strategy_aligned, benchmark_aligned = self.align_returns(
                strategy_returns, benchmark_returns
            )
            
            if len(strategy_aligned) == 0:
                return np.nan
            
            strategy_cumulative = (1 + strategy_aligned).cumprod().iloc[-1] - 1
            benchmark_cumulative = (1 + benchmark_aligned).cumprod().iloc[-1] - 1
            
            return strategy_cumulative - benchmark_cumulative
            
        except Exception as e:
            self.logger.error(f"计算超额收益失败: {e}")
            return np.nan
    
    def calculate_tracking_error(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算跟踪误差
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            
        Returns:
            跟踪误差（年化）
        """
        try:
            strategy_aligned, benchmark_aligned = self.align_returns(
                strategy_returns, benchmark_returns
            )
            
            if len(strategy_aligned) < 2:
                return np.nan
            
            excess_returns = strategy_aligned - benchmark_aligned
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            return tracking_error
            
        except Exception as e:
            self.logger.error(f"计算跟踪误差失败: {e}")
            return np.nan
    
    def calculate_information_ratio(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        计算信息比率
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            risk_free_rate: 无风险收益率
            
        Returns:
            信息比率
        """
        try:
            strategy_aligned, benchmark_aligned = self.align_returns(
                strategy_returns, benchmark_returns
            )
            
            if len(strategy_aligned) < 2:
                return np.nan
            
            excess_returns = strategy_aligned - benchmark_aligned
            tracking_error = excess_returns.std()
            
            if tracking_error == 0:
                return np.nan
            
            ir = excess_returns.mean() / tracking_error
            return ir * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"计算信息比率失败: {e}")
            return np.nan
    
    def calculate_r_squared(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算 R平方（决定系数）
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            
        Returns:
            R平方值
        """
        try:
            # 使用安全的相关系数计算
            correlation = self.calculate_correlation(strategy_returns, benchmark_returns)
            
            if np.isnan(correlation):
                return np.nan
            
            return correlation ** 2
            
        except Exception as e:
            self.logger.error(f"计算R平方失败: {e}")
            return np.nan
    
    def calculate_correlation(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算与基准的相关系数
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            
        Returns:
            相关系数
        """
        try:
            strategy_aligned, benchmark_aligned = self.align_returns(
                strategy_returns, benchmark_returns
            )
            
            if len(strategy_aligned) < 2:
                return np.nan
            
            # 检查数据是否有效（对齐后可能还有NaN）
            if strategy_aligned.isna().any() or benchmark_aligned.isna().any():
                return np.nan
            
            # 检查标准差是否为零
            if strategy_aligned.std() == 0 or benchmark_aligned.std() == 0:
                return np.nan
            
            # 使用安全的相关系数计算
            with np.errstate(invalid='ignore', divide='ignore'):
                correlation = np.corrcoef(strategy_aligned, benchmark_aligned)[0, 1]
            
            # 检查结果是否有效
            if np.isnan(correlation) or np.isinf(correlation):
                return np.nan
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"计算相关系数失败: {e}")
            return np.nan
    
    def calculate_benchmark_metrics(
        self,
        benchmark_returns: pd.Series
    ) -> BenchmarkMetrics:
        """
        计算基准的完整指标
        
        Args:
            benchmark_returns: 基准收益率
            
        Returns:
            基准指标
        """
        try:
            from src.evaluation.metrics.performance import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer()
            
            total_return = (1 + benchmark_returns).cumprod().iloc[-1] - 1
            annual_return = benchmark_returns.mean() * 252
            volatility = benchmark_returns.std() * np.sqrt(252)
            sharpe = (annual_return - 0.02) / volatility if volatility > 0 else np.nan
            
            max_dd_info = analyzer.calculate_max_drawdown(benchmark_returns)
            max_drawdown = max_dd_info.get('max_drawdown', np.nan)
            
            equity_curve = self.calculate_cumulative_equity(benchmark_returns).tolist()
            
            return BenchmarkMetrics(
                symbol=self.benchmark_symbol,
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                equity_curve=equity_curve
            )
            
        except Exception as e:
            self.logger.error(f"计算基准指标失败: {e}")
            return BenchmarkMetrics(
                symbol=self.benchmark_symbol,
                total_return=np.nan,
                annual_return=np.nan,
                volatility=np.nan,
                sharpe_ratio=np.nan,
                max_drawdown=np.nan,
                equity_curve=[]
            )
    
    def comprehensive_benchmark_analysis(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        完整的基准对比分析
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            risk_free_rate: 无风险收益率
            
        Returns:
            完整的对比分析结果
        """
        try:
            if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
                return {}
            
            alpha = self.calculate_alpha_jensen(
                strategy_returns, benchmark_returns, risk_free_rate
            )
            beta = self.calculate_beta(strategy_returns, benchmark_returns)
            excess_return = self.calculate_excess_return(
                strategy_returns, benchmark_returns
            )
            tracking_error = self.calculate_tracking_error(
                strategy_returns, benchmark_returns
            )
            ir = self.calculate_information_ratio(
                strategy_returns, benchmark_returns, risk_free_rate
            )
            r_squared = self.calculate_r_squared(strategy_returns, benchmark_returns)
            correlation = self.calculate_correlation(
                strategy_returns, benchmark_returns
            )
            
            return {
                'benchmark_symbol': self.benchmark_symbol,
                'excess_return': excess_return,
                'alpha': alpha,
                'beta': beta,
                'tracking_error': tracking_error,
                'information_ratio': ir,
                'r_squared': r_squared,
                'correlation': correlation
            }
            
        except Exception as e:
            self.logger.error(f"基准对比分析失败: {e}")
            return {}
    
    def get_equity_comparison(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        initial_value: float = 1.0
    ) -> Dict[str, List]:
        """
        获取净值对比数据
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            initial_value: 初始净值
            
        Returns:
            包含策略净值、基准净值、超额收益的字典
        """
        try:
            strategy_aligned, benchmark_aligned = self.align_returns(
                strategy_returns, benchmark_returns
            )
            
            if len(strategy_aligned) == 0:
                return {
                    'strategy_equity': [],
                    'benchmark_equity': [],
                    'excess_returns': []
                }
            
            strategy_equity = self.calculate_cumulative_equity(
                strategy_aligned, initial_value
            )
            benchmark_equity = self.calculate_cumulative_equity(
                benchmark_aligned, initial_value
            )
            
            excess_returns = (strategy_equity / benchmark_equity - 1).tolist()
            
            return {
                'strategy_equity': strategy_equity.tolist(),
                'benchmark_equity': benchmark_equity.tolist(),
                'excess_returns': excess_returns,
                'timestamps': strategy_equity.index.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"获取净值对比失败: {e}")
            return {
                'strategy_equity': [],
                'benchmark_equity': [],
                'excess_returns': []
            }
    
    def clear_cache(self):
        """清除基准数据缓存"""
        self._benchmark_cache.clear()

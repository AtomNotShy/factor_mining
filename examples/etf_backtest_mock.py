"""
ETF回测示例框架 - 简化版本

用于在没有IB连接时测试策略逻辑
"""

import asyncio
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.core.types import ActionType, Signal


logger = get_logger(__name__)


def load_config():
    """加载ETF配置"""
    config_path = Path(__file__).parent / "etf_config.py"
    spec = importlib.util.spec_from_file_location("etf_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {config_path}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


config = load_config()
ETF_LIST = config.ETF_LIST
ETFBacktestConfig = config.ETFBacktestConfig


class MockDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def generate_price(
        self,
        start_price: float,
        start_date: datetime,
        end_date: datetime,
        volatility: float = 0.02,
        drift: float = 0.0001,
    ) -> pd.Series:
        """生成模拟价格序列"""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n = len(dates)
        
        returns = np.random.normal(drift / 252, volatility / np.sqrt(252), n - 1)
        prices = [start_price]
        
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        return pd.Series(prices, index=dates, name="close")
    
    def generate_etf_prices(
        self,
        etfs: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.Series]:
        """生成多只ETF的模拟价格"""
        np.random.seed(42)
        
        etf_params = {
            "EFA": {"start": 75, "vol": 0.018, "drift": 0.0002},
            "EEM": {"start": 48, "vol": 0.022, "drift": 0.0001},
            "USMV": {"start": 75, "vol": 0.012, "drift": 0.0002},
            "XLE": {"start": 88, "vol": 0.025, "drift": -0.0001},
            "VNQ": {"start": 95, "vol": 0.020, "drift": 0.0001},
            "TLT": {"start": 130, "vol": 0.015, "drift": -0.0001},
            "IEF": {"start": 110, "vol": 0.010, "drift": 0.00005},
            "SHY": {"start": 85, "vol": 0.005, "drift": 0.0001},
            "LQD": {"start": 115, "vol": 0.008, "drift": 0.0001},
            "HYG": {"start": 85, "vol": 0.012, "drift": 0.00005},
            "DBC": {"start": 25, "vol": 0.018, "drift": -0.00005},
            "USO": {"start": 70, "vol": 0.035, "drift": -0.0003},
            "MTUM": {"start": 165, "vol": 0.018, "drift": 0.00025},
        }
        
        result = {}
        for etf in etfs:
            params = etf_params.get(etf, {"start": 100, "vol": 0.02, "drift": 0.0001})
            prices = self.generate_price(
                start_price=params["start"],
                start_date=start_date,
                end_date=end_date,
                volatility=params["vol"],
                drift=params["drift"],
            )
            result[etf] = prices
        
        return result


class ETFScreener:
    """ETF筛选器"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.selected_etfs: List[str] = []
        
    def calculate_momentum(self, prices: pd.DataFrame, periods: int = 21) -> pd.DataFrame:
        """计算动量因子"""
        return prices.pct_change(periods=periods)
    
    def select_etfs(
        self,
        prices: pd.DataFrame,
        n_select: int = 5,
    ) -> List[str]:
        """基于动量筛选ETF"""
        if prices.empty:
            return []
        
        momentum = self.calculate_momentum(prices, periods=21)
        latest = momentum.iloc[-1]
        
        available_etfs = [c for c in latest.index if not pd.isna(latest[c])]
        if not available_etfs:
            return []
        
        scores = -latest[available_etfs]
        scores = scores.dropna()
        if scores.empty:
            return []
        
        selected = scores.sort_values().head(n_select).index.tolist()
        return selected


class SimpleBacktest:
    """简化回测引擎"""
    
    def __init__(self, config: ETFBacktestConfig):
        self.config = config
        self.screener = ETFScreener()
        self.prices: Dict[str, pd.Series] = {}
        
    def fetch_data(self) -> pd.DataFrame:
        """获取模拟数据"""
        logger.info(f"生成 {len(self.config.etfs)} 只ETF的模拟数据...")
        
        generator = MockDataGenerator(seed=42)
        self.prices = generator.generate_etf_prices(
            self.config.etfs,
            self.config.start_date,
            self.config.end_date,
        )
        
        prices_df = pd.DataFrame(self.prices)
        prices_df = prices_df.sort_index()
        
        logger.info(f"数据范围: {prices_df.index.min()} 到 {prices_df.index.max()}")
        
        return prices_df
    
    def run(self) -> Tuple[pd.DataFrame, dict]:
        """运行回测"""
        prices = self.fetch_data()
        
        logger.info("运行回测...")
        
        cash = self.config.initial_capital
        positions: Dict[str, float] = {}
        equity_curve = []
        dates = prices.index.tolist()
        
        rebalance_interval = 21
        
        for i, date in enumerate(dates):
            if i < self.screener.lookback_period:
                continue
            
            if (i - self.screener.lookback_period) % rebalance_interval != 0:
                equity = cash + sum(
                    positions.get(sym, 0) * prices.loc[date, sym]
                    for sym in positions
                    if sym in prices.columns
                )
                equity_curve.append({"date": date, "equity": equity})
                continue
            
            price_slice = prices.loc[:date]
            
            new_selected = self.screener.select_etfs(price_slice, n_select=5)
            
            if not new_selected:
                equity = cash + sum(
                    positions.get(sym, 0) * prices.loc[date, sym]
                    for sym in positions
                    if sym in prices.columns
                )
                equity_curve.append({"date": date, "equity": equity})
                continue
            
            current_etfs = set(positions.keys())
            new_set = set(new_selected)
            
            for sym in current_etfs - new_set:
                if sym in positions and sym in prices.columns:
                    qty = positions[sym]
                    price = prices.loc[date, sym]
                    cash += qty * price * (1 - self.config.commission_rate)
                    del positions[sym]
            
            total_equity = cash + sum(
                positions.get(sym, 0) * prices.loc[date, sym]
                for sym in positions
                if sym in prices.columns
            )
            
            target_per_etf = total_equity / len(new_selected)
            
            for sym in new_selected:
                if sym not in prices.columns:
                    continue
                price = prices.loc[date, sym]
                if price <= 0:
                    continue
                qty = target_per_etf / price
                qty = int(qty)
                
                if qty > 0:
                    cost = qty * price * (1 + self.config.commission_rate)
                    if cost <= cash:
                        cash -= cost
                        positions[sym] = qty
            
            self.screener.selected_etfs = new_selected
            
            equity = cash + sum(
                positions.get(sym, 0) * prices.loc[date, sym]
                for sym in positions
                if sym in prices.columns
            )
            equity_curve.append({"date": date, "equity": equity})
        
        portfolio_df = pd.DataFrame(equity_curve)
        portfolio_df = portfolio_df.set_index("date")
        
        metrics = self.calculate_metrics(portfolio_df)
        
        return portfolio_df, metrics
    
    def calculate_metrics(self, portfolio_df: pd.DataFrame) -> dict:
        """计算绩效指标"""
        if portfolio_df.empty:
            return {}
        
        equity = portfolio_df["equity"]
        returns = equity.pct_change().dropna()
        
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        
        days = (pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).days
        n_years = days / 365.25
        cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        
        annual_vol = returns.std() * np.sqrt(252) * 100
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        
        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "annual_volatility": annual_vol,
            "win_rate": win_rate,
            "trading_days": len(returns),
            "final_equity": equity.iloc[-1],
        }
    
    def print_results(self, metrics: dict):
        """打印结果"""
        print("\n" + "=" * 60)
        print("ETF回测结果 (模拟数据)")
        print("=" * 60)
        print(f"总收益率:     {metrics.get('total_return', 0):.2f}%")
        print(f"年化收益率:   {metrics.get('cagr', 0):.2f}%")
        print(f"夏普比率:     {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤:     {metrics.get('max_drawdown', 0):.2f}%")
        print(f"年化波动率:   {metrics.get('annual_volatility', 0):.2f}%")
        print(f"胜率:         {metrics.get('win_rate', 0):.2f}%")
        print(f"交易天数:     {metrics.get('trading_days', 0)}")
        print(f"最终资金:     ${metrics.get('final_equity', 0):.2f}")
        print("=" * 60)
    
    def run_with_results(self):
        """运行并输出结果"""
        try:
            portfolio_df, metrics = self.run()
            
            if metrics:
                self.print_results(metrics)
            
            return portfolio_df, metrics
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETF回测（模拟数据）")
    parser.add_argument("--capital", type=float, default=1000.0, help="初始资金")
    parser.add_argument("--start", type=str, default="2020-01-01", help="开始日期")
    args = parser.parse_args()
    
    config = ETFBacktestConfig(
        start_date=datetime.strptime(args.start, "%Y-%m-%d"),
        initial_capital=args.capital,
    )
    
    backtest = SimpleBacktest(config)
    backtest.run_with_results()


if __name__ == "__main__":
    main()

"""
ETF回测示例框架

功能：
- 从IB获取历史数据
- 使用本地缓存，避免重复请求
- 实现简单的ETF筛选策略
- 运行回测并输出绩效指标
"""

import asyncio
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.data.collectors.ib_history import IBHistoryCollector
from src.evaluation.backtesting.engine import BacktestEngine
from src.core.types import MarketData, Signal, ActionType
from src.core.context import RunContext, Environment
from src.utils.logger import get_logger


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
ETF_NAMES = config.ETF_NAMES


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


class ETFBacktest:
    """ETF回测器（使用本地缓存）"""

    def __init__(self, config: ETFBacktestConfig = None):
        self.config = config or ETFBacktestConfig()
        self.collector = IBHistoryCollector(
            host=self.config.ib_host,
            port=self.config.ib_port,
            client_id=self.config.ib_client_id,
        )
        self.strategy = ETFScreener()
        self.prices: Dict[str, pd.DataFrame] = {}

    def fetch_data(self) -> pd.DataFrame:
        """获取所有ETF数据（优先使用本地缓存）"""

        all_data: Dict[str, pd.DataFrame] = {}

        for symbol in self.config.etfs:
            try:
                df = self.collector.get_ohlcv(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    since=self.config.start_date,
                    use_cache=True,
                )
                if len(df) > 0:
                    if "close" in df.columns:
                        all_data[symbol] = pd.DataFrame(df["close"])
                    else:
                        all_data[symbol] = pd.DataFrame(df.iloc[:, 0])
                    logger.info(f"  {symbol}: {len(df)} 条数据")
                else:
                    logger.warning(f"  {symbol}: 无数据")
            except Exception as e:
                logger.error(f"  {symbol}: 获取失败 - {e}")

        if not all_data:
            raise ValueError("未能获取任何数据")

        prices = pd.DataFrame(all_data)
        prices = prices.sort_index()

        self.prices = all_data
        logger.info(f"数据范围: {prices.index.min()} 到 {prices.index.max()}")

        return prices

    async def run_backtest(self, prices: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """运行回测"""
        logger.info("运行回测...")

        ctx = RunContext.create(
            env=Environment.RESEARCH,
            config={"backtest_id": f"etf_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"},
        )

        engine = BacktestEngine(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
        )

        signals_list = []
        dates = prices.index.tolist()

        rebalance_interval = 21
        if self.config.rebalance_freq == "weekly":
            rebalance_interval = 5
        elif self.config.rebalance_freq == "quarterly":
            rebalance_interval = 63

        for i, date in enumerate(dates):
            if i < self.strategy.lookback_period:
                continue

            if (i - self.strategy.lookback_period) % rebalance_interval != 0:
                continue

            price_slice = prices.loc[:date]
            signals = self._generate_signals(price_slice, date)
            signals_list.extend(signals)

        logger.info(f"生成 {len(signals_list)} 个信号")

        if not signals_list:
            logger.warning("没有生成任何信号")
            return pd.DataFrame(), {}

        md = MarketData(bars=prices)

        start_date = prices.index.min()
        end_date = prices.index.max()

        results = await engine.run(
            strategies=[],
            universe=list(prices.columns),
            start=start_date,
            end=end_date,
            ctx=ctx,
            bars=md.bars,
            features=None,
        )

        portfolio_daily = results.get("portfolio_daily")

        if portfolio_daily is None or portfolio_daily.empty:
            logger.warning("回测结果中没有 portfolio_daily 数据")
            return pd.DataFrame(), {}

        metrics = self.calculate_metrics(portfolio_daily)

        return portfolio_daily, metrics

    def _generate_signals(self, prices: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """生成信号"""
        signals = []

        available_symbols = [c for c in prices.columns if c in prices.columns]
        if not available_symbols:
            return signals

        try:
            prices_subset = prices[available_symbols].dropna(axis=1, how="all")
            if prices_subset.empty:
                return signals

            new_selected = self.strategy.select_etfs(prices_subset, n_select=5)

            for symbol in new_selected:
                if symbol not in self.strategy.selected_etfs:
                    signals.append(Signal(
                        ts_utc=timestamp,
                        symbol=symbol,
                        strategy_id="etf_screener",
                        action=ActionType.LONG,
                        strength=1.0,
                    ))

            for symbol in self.strategy.selected_etfs:
                if symbol not in new_selected:
                    signals.append(Signal(
                        ts_utc=timestamp,
                        symbol=symbol,
                        strategy_id="etf_screener",
                        action=ActionType.FLAT,
                        strength=0.0,
                    ))

            self.strategy.selected_etfs = new_selected

        except Exception as e:
            logger.error(f"生成信号失败: {e}")

        return signals

    def calculate_metrics(self, portfolio_daily: pd.DataFrame) -> dict:
        """计算绩效指标"""
        if portfolio_daily is None or portfolio_daily.empty:
            return {}

        equity = portfolio_daily["equity"]
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

        metrics = {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "annual_volatility": annual_vol,
            "win_rate": win_rate,
            "trading_days": len(returns),
            "final_equity": equity.iloc[-1],
        }

        return metrics

    def print_results(self, metrics: dict):
        """打印结果"""
        print("\n" + "=" * 60)
        print("ETF回测结果")
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

    def show_cache_status(self):
        """显示缓存状态"""
        logger.info("缓存目录: " + self.collector.store.path(self.collector._cache_subdir).__str__())
        for symbol in self.config.etfs:
            cache_path = self.collector._cache_path(symbol, "1d")
            full_path = self.collector.store.path(cache_path)
            if full_path.exists():
                size = full_path.stat().st_size / 1024
                logger.info(f"  {symbol}: 已缓存 ({size:.1f} KB)")
            else:
                logger.info(f"  {symbol}: 未缓存")

    async def run(self):
        """运行完整回测"""
        try:
            prices = self.fetch_data()
            portfolio_daily, metrics = await self.run_backtest(prices)

            if metrics:
                self.print_results(metrics)

            return portfolio_daily, metrics

        except Exception as e:
            logger.error(f"回测失败: {e}")
            raise
        finally:
            await self.collector.disconnect()


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ETF回测（使用本地缓存）")
    parser.add_argument("--port", type=int, default=7497, help="IB端口 (7497模拟, 7496实盘)")
    parser.add_argument("--capital", type=float, default=1000.0, help="初始资金")
    parser.add_argument("--start", type=str, default="2020-01-01", help="开始日期")
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
    parser.add_argument("--clear-cache", action="store_true", help="清除缓存后退出")
    args = parser.parse_args()

    config = ETFBacktestConfig(
        start_date=datetime.strptime(args.start, "%Y-%m-%d"),
        initial_capital=args.capital,
        ib_port=args.port,
    )

    collector = IBHistoryCollector(
        host=config.ib_host,
        port=config.ib_port,
        client_id=config.ib_client_id,
    )

    if args.clear_cache:
        collector.clear_cache()
        print("缓存已清除")
        return

    backtest = ETFBacktest(config)
    backtest.show_cache_status()
    await backtest.run()


if __name__ == "__main__":
    asyncio.run(main())

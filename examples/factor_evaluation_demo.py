"""
因子评估演示脚本（v2）
展示 IC 分析与性能指标计算（不包含旧版因子回测）
"""

import asyncio
from datetime import datetime, timedelta

import pandas as pd

from src.evaluation.metrics.ic_analysis import ICAnalyzer
from src.evaluation.metrics.performance import PerformanceAnalyzer
from src.data.collectors.exchange import MultiExchangeCollector
from src.factors.base.factor import factor_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def demo_ic_analysis():
    print("\n" + "=" * 50)
    print("IC 分析演示")
    print("=" * 50)

    ic_analyzer = ICAnalyzer()
    data_collector = MultiExchangeCollector()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    price_data = await data_collector.get_ohlcv(
        symbol="BTC/USDT",
        timeframe="1d",
        since=start_date,
        limit=90,
    )

    if price_data.empty:
        print("❌ 无法获取价格数据")
        return None

    test_factors = ["momentum_5", "momentum_20", "rsi_momentum_14"]
    ic_results = {}

    for factor_name in test_factors:
        factor = factor_registry.get_factor(factor_name)
        if not factor:
            print(f"⚠️  因子 {factor_name} 不存在")
            continue

        factor_values = factor.calculate(price_data)
        if factor_values.empty or factor_values.isna().all():
            print(f"⚠️  因子 {factor_name} 计算失败")
            continue

        results = ic_analyzer.comprehensive_analysis(
            factor_values=factor_values,
            price_data=price_data,
            periods=[1, 5, 10],
        )
        ic_results[factor_name] = results

        basic_stats = results.get("basic_ic_stats", {})
        period_1 = basic_stats.get("period_1", {})
        print(
            f"  {factor_name:<20} IC: {period_1.get('ic', 0):>8.4f}  "
            f"IC_IR: {period_1.get('ic_ir', 0):>8.4f}"
        )

    print(f"\n✅ IC 分析完成，共分析 {len(ic_results)} 个因子")
    return ic_results


async def demo_performance_analysis():
    print("\n" + "=" * 50)
    print("性能指标演示")
    print("=" * 50)

    data_collector = MultiExchangeCollector()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)

    price_data = await data_collector.get_ohlcv(
        symbol="BTC/USDT",
        timeframe="1d",
        since=start_date,
        limit=120,
    )

    if price_data.empty:
        print("❌ 无法获取价格数据")
        return None

    returns = price_data["close"].pct_change().dropna()
    analyzer = PerformanceAnalyzer()
    stats = analyzer.comprehensive_analysis(returns)

    print(f"✅ Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}")
    print(f"✅ Max Drawdown: {stats.get('max_drawdown', 0):.2%}")
    return stats


async def main():
    await demo_ic_analysis()
    await demo_performance_analysis()


if __name__ == "__main__":
    asyncio.run(main())

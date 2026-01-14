"""
回测性能测试
测试统一事件驱动架构的回测引擎性能
"""

import asyncio
import time
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import cProfile
import pstats
import io
from pathlib import Path

from src.evaluation.backtesting.event_engine import BacktestEventEngine, BacktestConfig
from src.strategies.base.unified_strategy import UnifiedStrategy, StrategyConfig
from src.core.types import Signal, OrderIntent, PortfolioState, RiskState, ActionType, OrderSide, OrderType
from src.core.types import OrderType as CoreOrderType
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar

logger = logging.getLogger("performance_test")


class DummyStrategy(UnifiedStrategy):
    """用于性能测试的简单策略"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                strategy_id="dummy_perf",
                strategy_name="Dummy Performance Test",
                timeframe="1d",
                params={"lookback": 20, "threshold": 0.0}
            )
        super().__init__(config)
        self.signal_count = 0
        self.order_count = 0
    
    def prepare_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """准备数据"""
        # 简单移动平均
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        return data
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """计算指标"""
        return self.prepare_data(dataframe, metadata)
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """进场信号"""
        dataframe['enter_long'] = (dataframe['sma_20'] > dataframe['sma_50']).astype(int)
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """离场信号"""
        dataframe['exit_long'] = (dataframe['sma_20'] < dataframe['sma_50']).astype(int)
        return dataframe
    
    def generate_signals(self, market_data: Any, ctx: RunContext) -> List[Signal]:
        """生成信号"""
        self.signal_count += 1
        # 简单信号逻辑
        signals = []
        if self.signal_count % 10 == 0:  # 每10个bar生成一个信号
            signals.append(Signal(
                ts_utc=datetime.now(),
                symbol="AAPL",
                strategy_id=self.strategy_id,
                action=ActionType.LONG,
                strength=1.0,
                metadata={"test": True}
            ))
        return signals
    
    def size_positions(self, signals: List[Signal], portfolio: PortfolioState, 
                      risk: RiskState, ctx: RunContext) -> List[OrderIntent]:
        """计算仓位"""
        self.order_count += 1
        intents = []
        for signal in signals:
            intents.append(OrderIntent(
                order_id=f"test_order_{self.order_count}",
                symbol=signal.symbol,
                side=OrderSide.BUY,
                qty=100,
                order_type=CoreOrderType.MKT,
                strategy_id=self.strategy_id,
                metadata={"performance_test": True}
            ))
        return intents


def generate_test_data(symbols: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
    """生成测试数据"""
    np.random.seed(42)
    start_date = date(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=days, freq='B')
    
    bars_map = {}
    for timeframe in ["1d"]:
        for symbol in symbols:
            # 生成随机价格数据
            price_start = 100.0
            returns = np.random.normal(0.0005, 0.02, days)
            prices = price_start * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.01, days)),
                'high': prices * (1 + np.abs(np.random.normal(0.02, 0.01, days))),
                'low': prices * (1 - np.abs(np.random.normal(0.02, 0.01, days))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, days)
            }, index=dates)
            
            key = f"{symbol}_{timeframe}" if len(symbols) > 1 else timeframe
            bars_map[key] = df
    
    return bars_map


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self
    
    def record_metric(self, name: str, value: Any):
        """记录指标"""
        self.metrics[name] = value
    
    def calculate(self, total_events: int, total_trades: int) -> Dict[str, Any]:
        """计算所有指标"""
        if self.start_time is None or self.end_time is None:
            elapsed = 0.0
        else:
            elapsed = self.end_time - self.start_time
        
        return {
            "total_time_seconds": elapsed,
            "events_per_second": total_events / elapsed if elapsed > 0 else 0,
            "trades_per_second": total_trades / elapsed if elapsed > 0 else 0,
            "memory_usage_mb": self._get_memory_usage(),
            "cpu_usage": self._estimate_cpu_usage(),
            "metrics": self.metrics
        }
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def _estimate_cpu_usage(self) -> float:
        """估计CPU使用率"""
        import psutil
        return psutil.cpu_percent(interval=0.1)


async def run_performance_test(
    symbol_count: int = 10,
    days: int = 252,
    strategy_count: int = 3
) -> Dict[str, Any]:
    """运行性能测试"""
    print(f"开始性能测试: {symbol_count}个标的, {days}天, {strategy_count}个策略")
    
    # 1. 生成测试数据
    symbols = [f"SYMBOL_{i}" for i in range(symbol_count)]
    bars_map = generate_test_data(symbols, days)
    
    # 2. 创建策略
    strategies = []
    for i in range(strategy_count):
        config = StrategyConfig(
            strategy_id=f"strategy_{i}",
            strategy_name=f"Test Strategy {i}",
            timeframe="1d",
            params={"lookback": 20 + i * 5}
        )
        strategies.append(DummyStrategy(config))
    
    # 3. 配置回测引擎
    backtest_config = BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        signal_timeframe="1d",
        clock_mode="daily",
        warmup_days=50
    )
    
    # 4. 运行性能测试
    metrics = PerformanceMetrics().start()
    
    engine = BacktestEventEngine(backtest_config)
    
    # 运行回测
    start_date = date(2024, 1, 1)
    end_date = start_date + timedelta(days=days)
    
    result = await engine.run(
        strategies=strategies,
        universe=symbols,
        start_date=start_date,
        end_date=end_date,
        bars_map=bars_map
    )
    
    metrics.stop()
    
    # 5. 收集结果
    total_events = days * symbol_count * strategy_count
    total_trades = result.get("total_trades", 0)
    
    performance_data = metrics.calculate(total_events, total_trades)
    
    # 合并回测结果
    performance_data.update({
        "test_config": {
            "symbol_count": symbol_count,
            "days": days,
            "strategy_count": strategy_count
        },
        "backtest_results": {
            "final_equity": result.get("final_equity"),
            "total_return": result.get("total_return"),
            "total_trades": total_trades
        }
    })
    
    return performance_data


def run_profiling_test():
    """运行性能分析测试"""
    print("开始性能分析测试...")
    
    # 创建性能分析器
    pr = cProfile.Profile()
    pr.enable()
    
    # 运行测试
    asyncio.run(run_performance_test(
        symbol_count=5,
        days=100,
        strategy_count=2
    ))
    
    pr.disable()
    
    # 输出分析结果
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    # 保存到文件
    output_file = Path("reports/performance_profile.txt")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(s.getvalue())
    
    print(f"性能分析结果已保存到: {output_file}")
    
    # 打印热点函数
    print("\n热点函数 (按累计时间排序):")
    ps.print_stats(10)


async def run_scalability_test():
    """运行可扩展性测试"""
    print("开始可扩展性测试...")
    
    test_cases = [
        {"symbols": 5, "days": 100, "strategies": 1},
        {"symbols": 10, "days": 100, "strategies": 2},
        {"symbols": 20, "days": 100, "strategies": 3},
        {"symbols": 10, "days": 252, "strategies": 2},
        {"symbols": 20, "days": 252, "strategies": 3},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {test_case}")
        
        try:
            result = await run_performance_test(
                symbol_count=test_case["symbols"],
                days=test_case["days"],
                strategy_count=test_case["strategies"]
            )
            
            results.append({
                "test_case": test_case,
                "performance": result
            })
            
            print(f"  完成时间: {result['total_time_seconds']:.2f}秒")
            print(f"  事件处理速度: {result['events_per_second']:.1f} 事件/秒")
            
        except Exception as e:
            print(f"  测试失败: {e}")
            results.append({
                "test_case": test_case,
                "error": str(e)
            })
    
    # 生成报告
    report = generate_scalability_report(results)
    
    # 保存报告
    output_file = Path("reports/scalability_report.md")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n可扩展性报告已保存到: {output_file}")
    
    return results


def generate_scalability_report(results: List[Dict]) -> str:
    """生成可扩展性报告"""
    report = "# 回测引擎可扩展性测试报告\n\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## 测试结果汇总\n\n"
    report += "| 测试用例 | 标的数 | 天数 | 策略数 | 总时间(秒) | 事件/秒 | 内存使用(MB) |\n"
    report += "|---------|--------|------|--------|------------|---------|--------------|\n"
    
    for i, result in enumerate(results):
        if "performance" in result:
            perf = result["performance"]
            test_case = result["test_case"]
            
            report += f"| {i+1} | {test_case['symbols']} | {test_case['days']} | {test_case['strategies']} | "
            report += f"{perf['total_time_seconds']:.2f} | {perf['events_per_second']:.1f} | {perf['memory_usage_mb']:.1f} |\n"
        else:
            test_case = result["test_case"]
            report += f"| {i+1} | {test_case['symbols']} | {test_case['days']} | {test_case['strategies']} | "
            report += "失败 | - | - |\n"
    
    report += "\n## 性能分析\n\n"
    
    # 计算性能变化
    if len(results) >= 2 and "performance" in results[0] and "performance" in results[1]:
        base_time = results[0]["performance"]["total_time_seconds"]
        last_time = results[-1]["performance"]["total_time_seconds"]
        
        report += f"- 基础测试用例处理时间: {base_time:.2f}秒\n"
        report += f"- 最大规模测试用例处理时间: {last_time:.2f}秒\n"
        report += f"- 时间增长倍数: {last_time/base_time:.2f}x\n"
        
        # 分析可扩展性
        total_events_base = (
            results[0]["test_case"]["symbols"] * 
            results[0]["test_case"]["days"] * 
            results[0]["test_case"]["strategies"]
        )
        total_events_last = (
            results[-1]["test_case"]["symbols"] * 
            results[-1]["test_case"]["days"] * 
            results[-1]["test_case"]["strategies"]
        )
        
        event_growth = total_events_last / total_events_base
        time_growth = last_time / base_time
        
        report += f"- 事件增长倍数: {event_growth:.2f}x\n"
        report += f"- 时间增长倍数: {time_growth:.2f}x\n"
        
        if time_growth <= event_growth * 1.2:
            report += "- ✅ 可扩展性良好: 时间增长与事件增长基本线性\n"
        else:
            report += f"- ⚠️ 可扩展性需优化: 时间增长({time_growth:.2f}x)超过事件增长({event_growth:.2f}x)的20%\n"
    
    report += "\n## 建议\n\n"
    report += "1. **事件处理优化**: 如果事件处理速度低于预期，考虑优化事件队列和处理器\n"
    report += "2. **内存管理**: 监控内存使用，避免内存泄漏\n"
    report += "3. **异步优化**: 确保异步任务正确调度，避免阻塞\n"
    report += "4. **缓存策略**: 优化数据缓存，减少重复计算\n"
    
    return report


async def main():
    """主函数"""
    print("=" * 60)
    print("回测引擎性能测试套件")
    print("=" * 60)
    
    # 创建输出目录
    Path("reports").mkdir(exist_ok=True)
    
    # 1. 运行基础性能测试
    print("\n1. 运行基础性能测试...")
    basic_result = await run_performance_test(
        symbol_count=10,
        days=100,
        strategy_count=2
    )
    
    print(f"   总时间: {basic_result['total_time_seconds']:.2f}秒")
    print(f"   事件处理速度: {basic_result['events_per_second']:.1f} 事件/秒")
    print(f"   内存使用: {basic_result['memory_usage_mb']:.1f} MB")
    
    # 2. 运行可扩展性测试
    print("\n2. 运行可扩展性测试...")
    scalability_results = await run_scalability_test()
    
    # 3. 运行性能分析
    print("\n3. 运行性能分析...")
    run_profiling_test()
    
    print("\n" + "=" * 60)
    print("性能测试完成!")
    print("=" * 60)
    
    # 生成最终报告
    final_report = generate_final_report(basic_result, scalability_results)
    
    with open("reports/final_performance_report.md", 'w') as f:
        f.write(final_report)
    
    print(f"\n最终报告已保存到: reports/final_performance_report.md")


def generate_final_report(basic_result: Dict, scalability_results: List[Dict]) -> str:
    """生成最终报告"""
    report = "# 统一事件驱动架构回测引擎性能测试报告\n\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## 执行摘要\n\n"
    report += "本次测试评估了统一事件驱动架构回测引擎的性能和可扩展性。\n\n"
    
    report += "## 1. 基础性能测试结果\n\n"
    report += f"- **测试配置**: {basic_result['test_config']['symbol_count']}个标的, "
    report += f"{basic_result['test_config']['days']}天, "
    report += f"{basic_result['test_config']['strategy_count']}个策略\n"
    report += f"- **总处理时间**: {basic_result['total_time_seconds']:.2f}秒\n"
    report += f"- **事件处理速度**: {basic_result['events_per_second']:.1f} 事件/秒\n"
    report += f"- **内存使用**: {basic_result['memory_usage_mb']:.1f} MB\n"
    report += f"- **总交易次数**: {basic_result['backtest_results']['total_trades']}\n\n"
    
    report += "## 2. 可扩展性测试结果\n\n"
    
    if scalability_results and len(scalability_results) > 0:
        report += "详细结果见: reports/scalability_report.md\n\n"
    
    report += "## 3. 性能分析结果\n\n"
    report += "详细性能分析见: reports/performance_profile.txt\n\n"
    
    report += "## 4. 结论与建议\n\n"
    report += "### 性能评估\n\n"
    
    # 性能评估逻辑
    events_per_second = basic_result['events_per_second']
    if events_per_second > 1000:
        report += "- ✅ **事件处理性能优秀**: 超过1000事件/秒\n"
    elif events_per_second > 500:
        report += "- ⚠️ **事件处理性能良好**: 500-1000事件/秒\n"
    else:
        report += "- ❌ **事件处理性能需优化**: 低于500事件/秒\n"
    
    memory_usage = basic_result['memory_usage_mb']
    if memory_usage < 500:
        report += f"- ✅ **内存使用合理**: {memory_usage:.1f} MB\n"
    elif memory_usage < 1000:
        report += f"- ⚠️ **内存使用偏高**: {memory_usage:.1f} MB\n"
    else:
        report += f"- ❌ **内存使用过高**: {memory_usage:.1f} MB\n"
    
    report += "\n### 优化建议\n\n"
    report += "1. **事件队列优化**: 如果事件处理速度不足，考虑优化事件队列数据结构\n"
    report += "2. **异步任务调度**: 确保异步任务正确调度，避免阻塞\n"
    report += "3. **内存管理**: 监控内存使用，及时清理缓存\n"
    report += "4. **数据缓存**: 优化数据缓存策略，减少重复计算\n"
    report += "5. **向量化计算**: 尽可能使用向量化操作替代循环\n\n"
    
    report += "## 5. 后续步骤\n\n"
    report += "1. 根据性能分析结果优化热点代码\n"
    report += "2. 运行更大规模的测试验证可扩展性\n"
    report += "3. 对比新旧系统性能差异\n"
    report += "4. 持续监控生产环境性能\n"
    
    return report
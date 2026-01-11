"""
美股 ETF 动量轮动策略 - 使用示例

展示如何使用 us_etf_momentum 策略进行回测
"""

import asyncio
from datetime import datetime, date
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.etf_momentum_us.strategy import USETFMomentumStrategy
from src.evaluation.backtesting.engine import BacktestEngine
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar


async def run_etf_momentum_backtest():
    """运行美股 ETF 动量轮动策略回测"""
    
    # 创建策略实例
    strategy = USETFMomentumStrategy()
    
    print("=" * 60)
    print("美股 ETF 动量轮动策略回测")
    print("=" * 60)
    print(f"策略ID: {strategy.strategy_id}")
    print(f"ETF池: {strategy.config.params['etf_pool']}")
    print(f"持仓数量: {strategy.config.params['target_positions']}")
    print(f"动态回溯: {strategy.config.params['auto_adjust_lookback']}")
    print()
    
    # 设置回测参数
    start_date = date(2023, 1, 1)
    end_date = date(2024, 12, 31)
    initial_capital = 100000.0
    
    print(f"回测期间: {start_date} ~ {end_date}")
    print(f"初始资金: ${initial_capital:,.2f}")
    print()
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.0002,  # 0.02% 手续费
        slippage_rate=0.0005,    # 0.05% 滑点
    )
    
    # 创建运行上下文
    trading_calendar = TradingCalendar()
    ctx = RunContext(
        env=Environment.RESEARCH,
        code_version="1.0.0",
        data_version="latest",
        config_hash="",
        now_utc=datetime.now(),
        trading_calendar=trading_calendar,
    )
    
    print("开始回测...")
    print()
    
    # 运行回测
    result = await engine.run(
        strategies=[strategy],
        universe=strategy.config.params['etf_pool'],
        start=start_date,
        end=end_date,
        ctx=ctx,
    )
    
    # 显示结果
    if "error" in result:
        print(f"❌ 回测失败: {result['error']}")
        return
    
    print("=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"最终净值: ${result['final_equity']:,.2f}")
    print(f"总收益率: {result['total_return']:.2%}")
    print(f"信号数量: {len(result['signals'])}")
    print(f"订单数量: {len(result['orders'])}")
    print(f"成交数量: {len(result['fills'])}")
    print()
    
    # 显示组合价值曲线
    portfolio_df = result.get("portfolio_daily")
    if portfolio_df is not None and not portfolio_df.empty:
        print("组合价值曲线（前10天）:")
        print(portfolio_df.head(10))
        print()
        print("组合价值曲线（后10天）:")
        print(portfolio_df.tail(10))
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_etf_momentum_backtest())

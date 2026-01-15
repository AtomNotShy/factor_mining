#!/usr/bin/env python3
"""
测试ETF动量策略修复
"""

import asyncio
from datetime import date
from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine
from src.evaluation.backtesting.config import UnifiedConfig, TradeConfig, TimeConfig, FeatureFlag

async def test_etf_momentum():
    """测试ETF动量策略"""
    print("=== 测试ETF动量策略修复 ===")
    
    # 创建策略
    strategy = ETFMomentumJoinQuantStrategy(
        strategy_id="test_etf_momentum",
        etf_pool=["QQQ", "SPY", "GLD", "TLT"],
        lookback_days=20,
        r2_threshold=0.5,
    )
    
    # 创建配置
    config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005,
            max_position_size=0.2,
            max_positions=1,
            stake_amount=20000,
        ),
        time=TimeConfig(
            signal_timeframe="1d",
            execution_timeframe="1d",
            warmup_days=260,
            clock_mode="daily",
        ),
        features=FeatureFlag.VECTORIZED | FeatureFlag.FREQTRADE_PROTOCOL,
    )
    
    # 创建回测引擎
    engine = UnifiedBacktestEngine(config=config)
    
    # 运行回测（短期测试）
    try:
        result = await engine.run(
            strategies=[strategy],
            universe=["QQQ", "SPY", "GLD", "TLT"],
            start=date(2024, 1, 1),
            end=date(2024, 3, 31),  # 3个月测试
            auto_download=False,  # 不自动下载数据
        )
        
        print(f"回测结果:")
        print(f"  策略: {result.strategy_name}")
        print(f"  初始资金: ${result.initial_capital:,.2f}")
        print(f"  最终权益: ${result.final_equity:,.2f}")
        print(f"  总回报: {result.total_return_pct:.2f}%")
        print(f"  交易次数: {result.total_trades}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        
        # 分析信号和交易
        print(f"\n信号分析:")
        print(f"  信号总数: {len(result.signals)}")
        print(f"  订单总数: {len(result.orders)}")
        print(f"  成交总数: {len(result.fills)}")
        
        # 检查是否有过度交易
        if result.total_trades > 20:  # 3个月超过20次交易可能过度
            print(f"  ⚠️ 警告: 交易次数可能过多 ({result.total_trades}次)")
        else:
            print(f"  ✓ 交易次数正常 ({result.total_trades}次)")
            
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_etf_momentum())

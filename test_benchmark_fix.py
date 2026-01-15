#!/usr/bin/env python3
"""
测试基准曲线显示修复
运行一个简单的回测，检查基准数据是否正确生成
"""

import asyncio
import pandas as pd
from datetime import date
import json

from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine
from src.evaluation.backtesting.config import UnifiedConfig, TradeConfig, TimeConfig

async def test_benchmark():
    """测试基准数据生成"""
    print("=" * 60)
    print("测试基准曲线显示修复")
    print("=" * 60)
    
    # 创建配置 - 使用正确的类型初始化
    config = UnifiedConfig()
    # 修改交易配置
    config.trade.initial_capital = 100000
    config.trade.commission_rate = 0.001
    # 修改时间配置
    config.time.signal_timeframe = "1d"
    config.time.warmup_days = 260
    
    # 创建策略
    strategy = ETFMomentumJoinQuantStrategy()
    
    # 创建回测引擎
    engine = UnifiedBacktestEngine(config=config)
    
    # 运行回测（使用少量ETF，缩短时间范围以加快测试）
    print("运行回测...")
    try:
        result = await engine.run(
            strategies=[strategy],
            universe=["QQQ", "SPY"],
            start=date(2024, 1, 1),
            end=date(2024, 3, 31),  # 仅测试3个月，加快速度
            ctx=None,
            auto_download=False  # 假设已有数据
        )
        
        print(f"回测完成，ID: {result.run_id}")
        
        # 检查结果中的基准数据
        print("\n检查基准数据:")
        print(f"基准符号: {result.benchmark_symbol}")
        print(f"基准收益率: {result.benchmark_return:.4%}")
        print(f"基准净值曲线长度: {len(result.benchmark_equity) if result.benchmark_equity else 0}")
        
        if result.benchmark_equity:
            # 检查基准净值曲线是否变化
            unique_values = len(set(result.benchmark_equity))
            print(f"基准净值曲线唯一值数量: {unique_values}")
            
            if unique_values > 1:
                print("✅ 基准净值曲线有变化，不是水平线")
                print(f"  最小值: ${min(result.benchmark_equity):.2f}")
                print(f"  最大值: ${max(result.benchmark_equity):.2f}")
                print(f"  第一个值: ${result.benchmark_equity[0]:.2f}")
                print(f"  最后一个值: ${result.benchmark_equity[-1]:.2f}")
            else:
                print("❌ 基准净值曲线是水平线（所有值相同）")
                print(f"  所有值均为: ${result.benchmark_equity[0] if result.benchmark_equity else 0:.2f}")
        else:
            print("❌ 基准净值曲线为空")
        
        # 检查策略净值曲线
        print(f"\n策略净值曲线长度: {len(result.portfolio_daily)}")
        if result.portfolio_daily:
            strategy_equity = [day['equity'] for day in result.portfolio_daily]
            unique_strategy = len(set(strategy_equity))
            print(f"策略净值曲线唯一值数量: {unique_strategy}")
            
            if unique_strategy > 1:
                print("✅ 策略净值曲线有变化")
            else:
                print("❌ 策略净值曲线是水平线")
        
        # 检查时间戳对齐
        print(f"\n时间戳数量: {len(result.portfolio_daily)}")
        print(f"基准净值曲线数量: {len(result.benchmark_equity) if result.benchmark_equity else 0}")
        
        if result.benchmark_equity and len(result.portfolio_daily) == len(result.benchmark_equity):
            print("✅ 策略和基准数据长度一致")
        else:
            print("❌ 策略和基准数据长度不一致")
            
        # 输出部分数据供检查
        print("\n前5个数据点:")
        for i in range(min(5, len(result.portfolio_daily))):
            day = result.portfolio_daily[i]
            benchmark_val = result.benchmark_equity[i] if result.benchmark_equity and i < len(result.benchmark_equity) else None
            print(f"  Day {i}: 策略=${day.get('equity', 0):.2f}, 基准=${benchmark_val if benchmark_val else 'N/A':.2f}")
        
        return True
        
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_benchmark())
    if success:
        print("\n✅ 测试完成，请检查前端图表是否正常显示基准曲线")
    else:
        print("\n❌ 测试失败")

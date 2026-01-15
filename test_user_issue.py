#!/usr/bin/env python3
"""
测试用户报告的问题：Total Return 18%，但Equity Curve显示100,000变成了1,919,381
"""

import asyncio
import pandas as pd
from datetime import date, datetime
import json
import yaml

from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine
from src.evaluation.backtesting.config import UnifiedConfig, TradeConfig, TimeConfig, FeatureFlag

async def test_user_issue():
    """测试用户报告的问题"""
    print("=" * 80)
    print("测试用户报告的问题：Total Return vs Equity Curve不一致")
    print("=" * 80)
    
    # 读取用户配置文件
    try:
        with open("etf_momentum_joinquant_backtest.yaml", "r") as f:
            user_config = yaml.safe_load(f)
        print("读取用户配置文件成功")
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        user_config = {}
    
    # 创建策略 - 使用用户配置
    strategy = ETFMomentumJoinQuantStrategy(
        strategy_id="etf_momentum",
        etf_pool=user_config.get("etf_pool", ["QQQ", "SPY", "IWM", "TLT", "GLD"]),
        lookback_days=user_config.get("lookback_days", 20),
        r2_threshold=user_config.get("r2_threshold", 0.5),
    )
    
    # 创建配置 - 使用用户配置
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
    
    # 运行回测（使用用户可能使用的日期范围）
    print("开始回测 (2023-01-01 到 2024-12-31)")
    
    try:
        result = await engine.run(
            strategies=[strategy],
            universe=["QQQ", "SPY", "IWM", "TLT", "GLD"],
            start=date(2023, 1, 1),  # 2年测试
            end=date(2024, 12, 31),
            auto_download=False,
        )
        
        # 分析结果
        print("\n" + "=" * 80)
        print("回测结果分析")
        print("=" * 80)
        
        print(f"初始资金: ${result.initial_capital:,.2f}")
        print(f"最终权益: ${result.final_equity:,.2f}")
        print(f"总回报: {result.total_return_pct:.2f}%")
        print(f"交易次数: {result.total_trades}")
        print(f"胜率: {result.win_rate:.2%}")
        print(f"夏普比率: {result.sharpe_ratio:.2f}")
        
        # 检查资金计算
        calculated_return = ((result.final_equity / result.initial_capital - 1) * 100)
        print(f"\n资金计算检查:")
        print(f"  基于final_equity计算的总回报: {calculated_return:.2f}%")
        print(f"  result.total_return_pct: {result.total_return_pct:.2f}%")
        
        diff = abs(calculated_return - result.total_return_pct)
        if diff > 0.01:
            print(f"  ⚠️ 资金计算不一致! 差异: {diff:.2f}%")
        
        # 分析净值曲线
        if result.portfolio_daily:
            print(f"\n净值曲线分析:")
            print(f"  净值记录数: {len(result.portfolio_daily)}")
            
            # 检查净值变化
            first_day = result.portfolio_daily[0]
            last_day = result.portfolio_daily[-1]
            
            print(f"  第一天净值: ${first_day.get('equity', 0):,.2f}")
            print(f"  最后一天净值: ${last_day.get('equity', 0):,.2f}")
            print(f"  第一天现金: ${first_day.get('cash', 0):,.2f}")
            print(f"  最后一天现金: ${last_day.get('cash', 0):,.2f}")
            
            # 检查现金和权益的关系
            if last_day.get('cash', 0) < 0:
                print(f"  ⚠️ 最后一天现金为负: ${last_day.get('cash', 0):,.2f}")
            
            # 计算持仓价值
            positions_value = last_day.get('equity', 0) - last_day.get('cash', 0)
            print(f"  最后一天持仓价值: ${positions_value:,.2f}")
            
            # 检查净值曲线是否异常
            max_equity = max(day.get('equity', 0) for day in result.portfolio_daily)
            min_equity = min(day.get('equity', 0) for day in result.portfolio_daily)
            print(f"  最高净值: ${max_equity:,.2f}")
            print(f"  最低净值: ${min_equity:,.2f}")
            
            # 检查是否有异常高的净值
            if max_equity > result.initial_capital * 10:  # 超过10倍
                print(f"  ⚠️ 净值异常高！最高净值是初始资金的{max_equity/result.initial_capital:.1f}倍")
        
        # 分析交易详情
        if result.fills:
            print(f"\n交易详情分析:")
            
            # 按交易对分组
            trades_by_symbol = {}
            for fill in result.fills:
                symbol = fill.get('symbol', 'unknown')
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(fill)
            
            print(f"  交易对分布:")
            for symbol, trades in trades_by_symbol.items():
                print(f"    {symbol}: {len(trades)} 次交易")
            
            # 检查交易价格
            print(f"\n  交易价格检查:")
            unusual_prices = []
            for i, fill in enumerate(result.fills[:10]):  # 只显示前10个
                price = fill.get('price', 0)
                symbol = fill.get('symbol', 'N/A')
                side = fill.get('side', 'N/A')
                qty = fill.get('qty', 0)
                
                print(f"    交易 #{i+1}: {symbol} {side} {qty:.2f} @ ${price:.2f}")
                
                # 检查价格是否异常
                if price > 1000:  # 价格超过1000
                    unusual_prices.append((symbol, price))
            
            if unusual_prices:
                print(f"\n  ⚠️ 发现异常价格:")
                for symbol, price in unusual_prices:
                    print(f"    {symbol}: ${price:.2f}")
        
        # 保存详细结果
        with open("user_issue_result.json", "w") as f:
            result_data = {
                "initial_capital": result.initial_capital,
                "final_equity": result.final_equity,
                "total_return_pct": result.total_return_pct,
                "calculated_return": calculated_return,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio,
                "signals_count": len(result.signals),
                "orders_count": len(result.orders),
                "fills_count": len(result.fills),
                "portfolio_daily_summary": {
                    "first_day": result.portfolio_daily[0] if result.portfolio_daily else None,
                    "last_day": result.portfolio_daily[-1] if result.portfolio_daily else None,
                    "count": len(result.portfolio_daily),
                    "max_equity": max_equity if result.portfolio_daily else 0,
                    "min_equity": min_equity if result.portfolio_daily else 0,
                },
                "trades_by_symbol": {symbol: len(trades) for symbol, trades in trades_by_symbol.items()} if result.fills else {},
            }
            json.dump(result_data, f, indent=2, default=str)
        
        print(f"\n详细结果已保存到: user_issue_result.json")
        
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_user_issue())

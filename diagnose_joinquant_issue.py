#!/usr/bin/env python3
"""
诊断JoinQuant ETF轮动策略问题

分析回测结果，找出为什么现金会变成负数
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    # 读取回测结果
    results_dir = Path("reports/joinquant_etf_rotation")
    result_files = list(results_dir.glob("backtest_result_*.json"))
    
    if not result_files:
        print("未找到回测结果文件")
        return
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"分析文件: {latest_file}")
    
    with open(latest_file, 'r') as f:
        result = json.load(f)
    
    # 分析关键指标
    print("\n=== 关键指标分析 ===")
    print(f"初始资金: ${result['initial_capital']:,.2f}")
    print(f"最终权益: ${result['final_equity']:,.2f}")
    print(f"总收益率: {result['total_return_pct']:.2f}%")
    print(f"总交易次数: {result['total_trades']}")
    print(f"胜率: {result['win_rate_pct']:.2f}%")
    print(f"盈亏比: {result['profit_factor']:.2f}")
    
    # 分析成交记录
    fills = result['fills']
    print(f"\n=== 成交记录分析 ===")
    print(f"总成交数: {len(fills)}")
    
    if fills:
        # 转换为DataFrame
        fills_df = pd.DataFrame(fills)
        
        # 分析成交类型
        buy_fills = fills_df[fills_df['side'] == 'BUY']
        sell_fills = fills_df[fills_df['side'] == 'SELL']
        
        print(f"买入成交: {len(buy_fills)}")
        print(f"卖出成交: {len(sell_fills)}")
        
        # 分析成交金额
        if 'price' in fills_df.columns and 'qty' in fills_df.columns:
            fills_df['value'] = fills_df['price'] * fills_df['qty']
            print(f"\n成交金额统计:")
            print(f"  总成交金额: ${fills_df['value'].sum():,.2f}")
            print(f"  平均成交金额: ${fills_df['value'].mean():,.2f}")
            print(f"  最大成交金额: ${fills_df['value'].max():,.2f}")
            print(f"  最小成交金额: ${fills_df['value'].min():,.2f}")
            
            # 检查异常成交
            small_fills = fills_df[fills_df['value'] < 100]
            print(f"\n小额成交(<$100): {len(small_fills)}")
            if len(small_fills) > 0:
                print("前10个小额成交:")
                for _, row in small_fills.head(10).iterrows():
                    print(f"  {row['timestamp']} {row['symbol']} {row['side']} {row['qty']:.4f} @ ${row['price']:.2f} = ${row['value']:.2f}")
    
    # 分析权益曲线
    equity_files = list(results_dir.glob("equity_curve_*.csv"))
    if equity_files:
        latest_equity = max(equity_files, key=lambda x: x.stat().st_mtime)
        equity_df = pd.read_csv(latest_equity)
        
        print(f"\n=== 权益曲线分析 ===")
        print(f"数据点数: {len(equity_df)}")
        
        # 检查现金和权益
        print(f"\n现金统计:")
        print(f"  初始现金: ${equity_df['cash'].iloc[0]:,.2f}")
        print(f"  最终现金: ${equity_df['cash'].iloc[-1]:,.2f}")
        print(f"  最小现金: ${equity_df['cash'].min():,.2f}")
        print(f"  最大现金: ${equity_df['cash'].max():,.2f}")
        
        print(f"\n权益统计:")
        print(f"  初始权益: ${equity_df['equity'].iloc[0]:,.2f}")
        print(f"  最终权益: ${equity_df['equity'].iloc[-1]:,.2f}")
        print(f"  最小权益: ${equity_df['equity'].min():,.2f}")
        print(f"  最大权益: ${equity_df['equity'].max():,.2f}")
        
        # 找出现金变为负数的点
        negative_cash = equity_df[equity_df['cash'] < 0]
        if len(negative_cash) > 0:
            print(f"\n⚠️  现金变为负数的时间点:")
            for _, row in negative_cash.head(5).iterrows():
                print(f"  {row['date']}: 现金=${row['cash']:.2f}, 权益=${row['equity']:.2f}")
        
        # 计算每日变化
        equity_df['equity_change'] = equity_df['equity'].diff()
        equity_df['cash_change'] = equity_df['cash'].diff()
        
        # 找出权益大幅下降的点
        large_drops = equity_df[equity_df['equity_change'] < -1000]
        if len(large_drops) > 0:
            print(f"\n⚠️  权益大幅下降的时间点:")
            for _, row in large_drops.head(5).iterrows():
                print(f"  {row['date']}: 权益变化=${row['equity_change']:.2f}, 现金变化=${row['cash_change']:.2f}")
    
    # 分析策略参数
    print(f"\n=== 策略参数分析 ===")
    print("问题可能原因:")
    print("1. 仓位计算错误 - stake_amount设置不当")
    print("2. 手续费和滑点过高")
    print("3. 频繁交易导致成本累积")
    print("4. 组合选择逻辑有问题")
    
    # 建议修复
    print(f"\n=== 建议修复 ===")
    print("1. 检查仓位计算逻辑，确保stake_amount合理")
    print("2. 降低交易频率或增加过滤条件")
    print("3. 调整手续费和滑点设置")
    print("4. 添加仓位大小限制")
    print("5. 检查组合选择逻辑，避免过度交易")

if __name__ == "__main__":
    main()
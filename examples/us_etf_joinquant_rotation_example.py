#!/usr/bin/env python3
"""
USETFJoinQuantRotationStrategy 使用示例

这个脚本演示了如何使用命令行工具回测 USETFJoinQuantRotationStrategy 策略。

策略特点：
- 美股ETF轮动策略（聚宽风格）
- 加权线性回归计算动量（近期数据权重更高）
- R²过滤不稳定趋势
- 基于ATR动态调整回溯期
- 多重风控过滤（跌幅过滤、连续下跌过滤）
- 每次只持有一只ETF，轮动到动量最强的ETF

使用方法：
1. 直接运行此脚本查看示例命令
2. 复制命令到终端执行回测
3. 或修改参数进行自定义回测
"""

import subprocess
import sys
from datetime import datetime, timedelta

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n📋 {description}")
    print(f"   $ {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 命令执行成功")
            # 显示部分输出
            lines = result.stdout.split('\n')
            for line in lines[-20:]:  # 显示最后20行
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"❌ 命令执行失败 (返回码: {result.returncode})")
            print(f"   错误输出: {result.stderr[:200]}...")
    except Exception as e:
        print(f"❌ 执行异常: {e}")

def main():
    """主函数"""
    print_header("USETFJoinQuantRotationStrategy 回测示例")
    
    # 计算日期范围
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)  # 一年
    
    print(f"📅 回测日期范围: {start_date} ~ {end_date}")
    print(f"💡 提示: 确保已安装所需依赖并下载了数据")
    
    # 示例1: 基本回测（使用默认参数）
    print_header("示例1: 基本回测（默认参数）")
    cmd1 = f"python src/main.py backtest --strategy us_etf_joinquant_rotation --start {start_date} --end {end_date}"
    run_command(cmd1, "使用默认ETF池和参数进行回测")
    
    # 示例2: 自定义ETF池
    print_header("示例2: 自定义ETF池")
    cmd2 = f"python src/main.py backtest --strategy us_etf_joinquant_rotation --symbols SPY,QQQ,IWM,TLT,GLD --start {start_date} --end {end_date}"
    run_command(cmd2, "使用自定义ETF池（SPY, QQQ, IWM, TLT, GLD）")
    
    # 示例3: 自定义策略参数
    print_header("示例3: 自定义策略参数")
    params_json = '{"min_lookback_days": 25, "max_lookback_days": 50, "r2_threshold": 0.6, "stoploss": -0.08}'
    cmd3 = f'python src/main.py backtest --strategy us_etf_joinquant_rotation --params \'{params_json}\' --start {start_date} --end {end_date}'
    run_command(cmd3, "使用自定义参数：回溯期25-50天，R²阈值0.6，止损8%")
    
    # 示例4: 使用单个参数设置
    print_header("示例4: 使用单个参数设置")
    cmd4 = f"python src/main.py backtest --strategy us_etf_joinquant_rotation --param min_lookback_days=30 --param max_lookback_days=60 --param r2_threshold=0.55 --start {start_date} --end {end_date}"
    run_command(cmd4, "使用--param参数逐个设置")
    
    # 示例5: 调整资金和手续费
    print_header("示例5: 调整资金和手续费")
    cmd5 = f"python src/main.py backtest --strategy us_etf_joinquant_rotation --initial-capital 50000 --commission 0.001 --slippage 0.0005 --start {start_date} --end {end_date}"
    run_command(cmd5, "初始资金5万，手续费0.1%，滑点0.05%")
    
    # 示例6: 列出所有可用策略
    print_header("示例6: 查看可用策略")
    cmd6 = "python src/main.py backtest --list-strategies"
    run_command(cmd6, "列出系统中所有可用策略")
    
    # 策略参数说明
    print_header("策略参数说明")
    print("""
USETFJoinQuantRotationStrategy 支持以下参数：

必需参数：
- etf_pool: ETF标的池列表，如 ["SPY", "QQQ", "IWM"]

动态回溯参数：
- min_lookback_days: 最小回溯天数 (默认: 20)
- max_lookback_days: 最大回溯天数 (默认: 60)
- use_dynamic_lookback: 是否使用动态回溯期 (默认: True)
- atr_period: ATR计算周期 (默认: 20)

过滤参数：
- r2_threshold: R²最小阈值 (默认: 0.5)
- decline_filter_days: 跌幅过滤天数 (默认: 3)
- decline_filter_threshold: 跌幅过滤阈值 (默认: 0.95, 即5%)
- consecutive_decline_threshold: 连续下跌过滤阈值 (默认: 0.96, 即4%)

持仓参数：
- target_positions: 目标持仓数量 (默认: 1，每次只持有一只ETF)
- max_weight: 单标最大权重 (默认: 1.0)
- min_weight: 单标最小权重 (默认: 0.0)

风控参数：
- stoploss: 止损比例 (默认: -0.10, 即-10%)
- trailing_stop: 是否启用追踪止损 (默认: False)
- trailing_stop_positive: 追踪止损正偏差 (默认: 0.02)

使用示例：
python src/main.py backtest --strategy us_etf_joinquant_rotation \\
    --param etf_pool='["SPY","QQQ","IWM"]' \\
    --param min_lookback_days=25 \\
    --param max_lookback_days=50 \\
    --param r2_threshold=0.6 \\
    --param stoploss=-0.08 \\
    --start 2023-01-01 --end 2024-01-01
    """)
    
    print_header("数据准备提示")
    print("""
如果回测时提示数据不足，请先下载数据：

1. 下载ETF数据：
   python src/main.py download --symbols SPY,QQQ,IWM,TLT,GLD --days 730

2. 或使用批量下载脚本：
   python download_etf_data.py

3. 检查数据文件：
   ls -la data/ib/ohlcv/1d/*.parquet
    """)
    
    print_header("完成")
    print("✅ 示例命令已生成")
    print("💡 提示：复制上述命令到终端执行即可开始回测")
    print("📊 回测结果将显示在终端，详细报告可访问前端界面")

if __name__ == "__main__":
    main()
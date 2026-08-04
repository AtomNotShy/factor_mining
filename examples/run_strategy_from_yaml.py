#!/usr/bin/env python3
"""
使用YAML配置运行ETF动量轮动策略

本示例演示如何从YAML配置文件加载策略参数并运行回测。

运行方式：
    python3 examples/run_strategy_from_yaml.py --config etf_momentum_rotation.yaml

Author: Factor Mining System
Date: 2024-01-15
"""

import sys
import yaml
from datetime import date
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.user_strategies.etf_momentum_rotation import ETFMomentumRotationStrategy
from src.evaluation.backtesting.engine import UnifiedBacktestEngine
from src.evaluation.backtesting.config import (
    UnifiedConfig,
    TradeConfig,
    TimeConfig,
    FeatureFlag,
)


def load_config(yaml_path: str) -> dict:
    """
    从YAML文件加载配置
    
    Args:
        yaml_path: YAML文件路径
        
    Returns:
        配置字典
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    主函数
    """
    print("=" * 70)
    print("使用YAML配置运行ETF动量轮动策略")
    print("=" * 70)
    
    # 1. 解析命令行参数
    # =========================================================================
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python3 run_strategy_from_yaml.py --config <yaml文件路径>")
        print("\n示例:")
        print("  python3 run_strategy_from_yaml.py --config etf_momentum_rotation.yaml")
        sys.exit(1)
    
    yaml_path = sys.argv[sys.argv.index('--config') + 1]
    
    if not Path(yaml_path).exists():
        print(f"错误: 配置文件不存在: {yaml_path}")
        sys.exit(1)
    
    # 2. 加载配置
    # =========================================================================
    print(f"\n📂 加载配置: {yaml_path}")
    config_dict = load_config(yaml_path)
    print("✅ 配置加载成功")
    
    # 3. 创建策略实例
    # =========================================================================
    print("\n📊 创建策略...")
    
    strategy_config = config_dict.get('strategy', {})
    parameters = config_dict.get('parameters', {})
    
    strategy = ETFMomentumRotationStrategy(
        strategy_id=strategy_config.get('id', 'etf_momentum_rotation'),
        etf_pool=config_dict.get('universe', {}).get('symbols', []),
        min_days=parameters.get('min_days', 20),
        max_days=parameters.get('max_days', 60),
        drop_threshold=parameters.get('drop_threshold', 0.95),
        consecutive_drop_threshold=parameters.get('consecutive_drop_threshold', 0.95),
        premium_rate_threshold=parameters.get('premium_rate_threshold', 5.0),
        premium_penalty=parameters.get('premium_penalty', 1.0),
        target_positions=parameters.get('target_positions', 1),
        stoploss=parameters.get('stoploss', -0.10),
    )
    
    print(f"✅ 策略创建成功: {strategy.strategy_name}")
    
    # 4. 创建回测配置
    # =========================================================================
    print("\n⚙️ 创建回测配置...")
    
    backtest_config = config_dict.get('backtest', {})
    
    unified_config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=backtest_config.get('initial_capital', 100000),
            commission_rate=backtest_config.get('commission_rate', 0.001),
            slippage_rate=backtest_config.get('slippage_rate', 0.0005),
            max_position_size=backtest_config.get('max_position_size', 1.0),
            max_positions=backtest_config.get('max_positions', 1),
            stake_amount=backtest_config.get('stake_amount', None),
        ),
        time=TimeConfig(
            signal_timeframe=backtest_config.get('signal_timeframe', '1d'),
            execution_timeframe=backtest_config.get('execution_timeframe', '1d'),
            warmup_days=backtest_config.get('warmup_days', 70),
            clock_mode=backtest_config.get('clock_mode', 'daily'),
        ),
        features=(
            FeatureFlag.VECTORIZED | 
            FeatureFlag.FREQTRADE_PROTOCOL
        ),
    )
    
    print(f"✅ 回测配置创建成功")
    print(f"   初始资金: ${unified_config.trade.initial_capital:,.2f}")
    print(f"   佣金费率: {unified_config.trade.commission_rate:.2%}")
    print(f"   滑点: {unified_config.trade.slippage_rate:.2%}")
    
    # 5. 创建回测引擎
    # =========================================================================
    print("\n🚀 初始化回测引擎...")
    
    engine = UnifiedBacktestEngine(config=unified_config)
    print("✅ 回测引擎创建成功")
    
    # 6. 获取回测周期
    # =========================================================================
    period_config = config_dict.get('period', {})
    
    if 'start_date' in period_config and 'end_date' in period_config:
        # 使用指定的日期范围
        start_date = date.fromisoformat(period_config['start_date'])
        end_date = date.fromisoformat(period_config['end_date'])
    elif 'days' in period_config:
        # 使用天数
        from datetime import timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=period_config['days'])
    else:
        # 默认值
        start_date = date(2023, 1, 1)
        end_date = date(2024, 12, 31)
    
    print(f"\n📅 回测周期: {start_date} ~ {end_date}")
    
    # 7. 运行回测
    # =========================================================================
    print("\n📈 运行回测...")
    print("-" * 50)
    
    import asyncio
    
    async def run_backtest():
        try:
            result = await engine.run(
                strategies=[strategy],
                universe=strategy.etf_pool,
                start=start_date,
                end=end_date,
                auto_download=False,
            )
            
            # 输出结果
            print("\n" + "=" * 70)
            print("📊 回测结果:")
            print("=" * 70)
            print(f"策略名称: {result.strategy_name}")
            print(f"初始资金: ${result.initial_capital:,.2f}")
            print(f"最终权益: ${result.final_equity:,.2f}")
            print(f"总收益率: {result.total_return_pct:.2f}%")
            print(f"年化收益率: {result.annual_return_pct:.2f}%")
            print(f"夏普比率: {result.sharpe_ratio:.4f}")
            print(f"最大回撤: {result.max_drawdown_pct:.2f}%")
            print(f"波动率: {result.volatility_pct:.2f}%")
            print(f"交易次数: {result.total_trades}")
            print(f"胜率: {result.win_rate:.2%}")
            
            # 交易统计
            print("\n📈 交易统计:")
            print("-" * 50)
            print(f"总交易次数: {result.total_trades}")
            print(f"做多次数: {result.long_trades}")
            print(f"做空次数: {result.short_trades}")
            print(f"平仓次数: {result.close_trades}")
            print(f"胜率: {result.win_rate:.2%}")
            print(f"平均盈利: ${result.avg_trade_pnl:,.2f}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    result = asyncio.run(run_backtest())
    
    if result:
        print("\n" + "=" * 70)
        print("✅ 回测完成！")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ 回测失败！")
        print("=" * 70)


if __name__ == "__main__":
    main()

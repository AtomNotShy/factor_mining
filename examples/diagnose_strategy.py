"""
策略诊断脚本：分析为什么策略没有产生交易信号
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone, timedelta

# 添加项目根目录到路径
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.strategies.vwap.vwap_pullback_v2 import (
    VWAPPullbackStrategyV2,
    VWAPPullbackParams,
    in_regular_trading_hours_utc,
)
from src.data.collectors.polygon import PolygonCollector


async def diagnose_strategy(symbol: str = "SPY", timeframe: str = "1d", days: int = 300):
    """
    诊断策略为什么没有产生交易信号
    
    Args:
        symbol: 标的代码
        timeframe: 时间周期
        days: 回测天数
    """
    print("=" * 80)
    print(f"策略诊断：{symbol} {timeframe} ({days}天)")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    collector = PolygonCollector()
    try:
        if not collector.connect():
            print("❌ 无法连接到数据源")
            return
        
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        bars = await collector.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_date,
            limit=None
        )
        
        if bars.empty:
            print("❌ 没有获取到数据")
            return
        
        print(f"✅ 获取到 {len(bars)} 根K线")
        print(f"   时间范围: {bars.index.min()} 至 {bars.index.max()}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 检查数据质量
    print("\n[2] 检查数据质量...")
    price_data = bars[["open", "high", "low", "close", "volume"]].copy()
    
    missing = price_data.isnull().sum()
    if missing.any():
        print(f"⚠️  缺失值: {missing[missing > 0].to_dict()}")
    else:
        print("✅ 无缺失值")
    
    zero_volume = (price_data["volume"] == 0).sum()
    if zero_volume > 0:
        print(f"⚠️  零成交量K线: {zero_volume} 根 ({zero_volume/len(price_data)*100:.1f}%)")
    else:
        print("✅ 无零成交量K线")
    
    # 3. 准备策略数据
    print("\n[3] 准备策略数据...")
    strategy = VWAPPullbackStrategyV2()
    try:
        aligned_data = strategy._prepare_data(price_data)
        print(f"✅ 数据准备完成: {len(aligned_data)} 根K线")
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return
    
    if aligned_data.empty:
        print("❌ 准备后的数据为空")
        return
    
    # 4. 检查策略参数
    print("\n[4] 检查策略参数...")
    params: VWAPPullbackParams = strategy.params or VWAPPullbackParams()
    
    print(f"   时间周期: {timeframe}")
    print(f"   策略设计周期: {strategy.timeframe}")
    if timeframe != strategy.timeframe:
        print(f"   ⚠️  警告: 数据周期({timeframe})与策略设计周期({strategy.timeframe})不匹配！")
        print(f"      策略是为分钟线设计的，使用日线数据可能导致条件无法满足")
    
    print(f"\n   过滤条件:")
    print(f"   - 最小ATR (bps): {params.min_atr_bps}")
    print(f"   - 最小成交量比率: {params.min_volume_ratio}")
    print(f"   - 上冲幅度 (bps): {params.impulse_bps}")
    print(f"   - 最大回踩 (bps): {params.max_pullback_bps}")
    print(f"   - 收复幅度 (bps): {params.reclaim_bps}")
    print(f"   - 冷却期 (bars): {params.cooldown_bars}")
    print(f"   - 日内最大交易数: {params.max_trades_per_day}")
    
    # 5. 分析过滤条件
    print("\n[5] 分析过滤条件通过率...")
    
    # ATR过滤
    atr_bps = aligned_data["atr_bps"]
    atr_pass = (atr_bps >= params.min_atr_bps).sum()
    atr_pass_rate = atr_pass / len(aligned_data) * 100
    print(f"   ATR过滤: {atr_pass}/{len(aligned_data)} ({atr_pass_rate:.1f}%) 通过")
    if atr_pass_rate < 10:
        print(f"      ⚠️  ATR过滤太严格，只有{atr_pass_rate:.1f}%的K线通过")
    
    # 成交量过滤
    vol_ratio = aligned_data["vol_ratio"]
    vol_pass = (vol_ratio >= params.min_volume_ratio).sum()
    vol_pass_rate = vol_pass / len(aligned_data) * 100
    print(f"   成交量过滤: {vol_pass}/{len(aligned_data)} ({vol_pass_rate:.1f}%) 通过")
    if vol_pass_rate < 10:
        print(f"      ⚠️  成交量过滤太严格，只有{vol_pass_rate:.1f}%的K线通过")
    
    # 趋势过滤
    ema_fast = aligned_data["ema_fast"]
    ema_slow = aligned_data["ema_slow"]
    close_price = aligned_data["close"]
    trend_ok = (ema_fast > ema_slow) & (close_price > ema_slow)
    trend_pass = trend_ok.sum()
    trend_pass_rate = trend_pass / len(aligned_data) * 100
    print(f"   趋势过滤: {trend_pass}/{len(aligned_data)} ({trend_pass_rate:.1f}%) 通过")
    if trend_pass_rate < 10:
        print(f"      ⚠️  趋势过滤太严格，只有{trend_pass_rate:.1f}%的K线通过")
    
    # 交易时间过滤（仅对分钟线有效）
    if timeframe.endswith("m"):
        rth_mask = in_regular_trading_hours_utc(
            aligned_data.index,
            start_et=params.entry_start_et,
            end_et=params.entry_end_et,
        )
        rth_pass = rth_mask.sum()
        rth_pass_rate = rth_pass / len(aligned_data) * 100
        print(f"   交易时间过滤: {rth_pass}/{len(aligned_data)} ({rth_pass_rate:.1f}%) 通过")
    else:
        print(f"   交易时间过滤: 日线数据，跳过此过滤")
        rth_pass_rate = 100
    
    # 6. 分析形态条件
    print("\n[6] 分析形态条件（上冲-回踩-收复）...")
    
    # 计算上冲
    impulse_lookback = params.impulse_lookback
    impulse_scores = []
    pullback_scores = []
    reclaim_scores = []
    bullish_bar_scores = []
    
    for i in range(impulse_lookback, len(aligned_data)):
        recent = aligned_data.iloc[i - impulse_lookback:i + 1]
        vwap_current = aligned_data.iloc[i]["vwap"]
        
        # 上冲
        impulse = ((recent["high"] - recent["vwap"]) / recent["vwap"] * 10000.0).max()
        impulse_ok = impulse >= params.impulse_bps
        impulse_scores.append(impulse_ok)
        
        # 回踩
        low_price = aligned_data.iloc[i]["low"]
        pullback_bps = (low_price - vwap_current) / vwap_current * 10000.0
        pullback_ok = pullback_bps >= -params.max_pullback_bps
        pullback_scores.append(pullback_ok)
        
        # 收复
        close_price = aligned_data.iloc[i]["close"]
        reclaim_ok = close_price >= vwap_current * (1.0 + params.reclaim_bps / 10000.0)
        reclaim_scores.append(reclaim_ok)
        
        # 看涨K线
        if i > 0:
            open_price = aligned_data.iloc[i]["open"]
            prev_close = aligned_data.iloc[i - 1]["close"]
            bullish_bar = (close_price > open_price) and (close_price > prev_close)
            bullish_bar_scores.append(bullish_bar)
        else:
            bullish_bar_scores.append(False)
    
    if impulse_scores:
        impulse_pass = sum(impulse_scores)
        impulse_rate = impulse_pass / len(impulse_scores) * 100
        print(f"   上冲条件: {impulse_pass}/{len(impulse_scores)} ({impulse_rate:.1f}%) 通过")
        if impulse_rate < 5:
            print(f"      ⚠️  上冲条件太严格，只有{impulse_rate:.1f}%的K线通过")
        
        pullback_pass = sum(pullback_scores)
        pullback_rate = pullback_pass / len(pullback_scores) * 100
        print(f"   回踩条件: {pullback_pass}/{len(pullback_scores)} ({pullback_rate:.1f}%) 通过")
        
        reclaim_pass = sum(reclaim_scores)
        reclaim_rate = reclaim_pass / len(reclaim_scores) * 100
        print(f"   收复条件: {reclaim_pass}/{len(reclaim_scores)} ({reclaim_rate:.1f}%) 通过")
        if reclaim_rate < 5:
            print(f"      ⚠️  收复条件太严格，只有{reclaim_rate:.1f}%的K线通过")
        
        bullish_pass = sum(bullish_bar_scores)
        bullish_rate = bullish_pass / len(bullish_bar_scores) * 100
        print(f"   看涨K线: {bullish_pass}/{len(bullish_bar_scores)} ({bullish_rate:.1f}%) 通过")
        
        # 所有条件同时满足
        all_conditions = [
            impulse_scores[i] and pullback_scores[i] and reclaim_scores[i] and bullish_bar_scores[i]
            for i in range(len(impulse_scores))
        ]
        all_pass = sum(all_conditions)
        all_rate = all_pass / len(all_conditions) * 100
        print(f"\n   所有形态条件同时满足: {all_pass}/{len(all_conditions)} ({all_rate:.1f}%)")
        if all_pass == 0:
            print(f"      ❌ 没有任何K线同时满足所有形态条件！")
    
    # 7. 综合诊断
    print("\n[7] 综合诊断...")
    issues = []
    
    if timeframe != "1m":
        issues.append(f"⚠️  数据周期({timeframe})与策略设计周期(1m)不匹配")
        issues.append("   建议：使用分钟线数据，或修改策略适配日线")
    
    if atr_pass_rate < 10:
        issues.append(f"⚠️  ATR过滤太严格 ({atr_pass_rate:.1f}%通过率)")
        issues.append(f"   建议：降低 min_atr_bps (当前: {params.min_atr_bps})")
    
    if vol_pass_rate < 10:
        issues.append(f"⚠️  成交量过滤太严格 ({vol_pass_rate:.1f}%通过率)")
        issues.append(f"   建议：降低 min_volume_ratio (当前: {params.min_volume_ratio})")
    
    if trend_pass_rate < 10:
        issues.append(f"⚠️  趋势过滤太严格 ({trend_pass_rate:.1f}%通过率)")
        issues.append("   建议：检查EMA参数或放宽趋势条件")
    
    if impulse_scores and sum(impulse_scores) == 0:
        issues.append(f"⚠️  上冲条件从未满足")
        issues.append(f"   建议：降低 impulse_bps (当前: {params.impulse_bps})")
    
    if reclaim_scores and sum(reclaim_scores) == 0:
        issues.append(f"⚠️  收复条件从未满足")
        issues.append(f"   建议：降低 reclaim_bps (当前: {params.reclaim_bps})")
    
    if all_conditions and sum(all_conditions) == 0:
        issues.append("❌ 所有形态条件从未同时满足")
        issues.append("   这是导致没有交易信号的主要原因")
    
    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ 未发现明显问题，建议检查日志查看具体过滤原因")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    import asyncio
    
    # 默认参数
    symbol = "SPY"
    timeframe = "1d"  # 改为 "1m" 如果使用分钟线
    days = 300
    
    # 可以从命令行参数读取
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    if len(sys.argv) > 2:
        timeframe = sys.argv[2]
    if len(sys.argv) > 3:
        days = int(sys.argv[3])
    
    # 运行诊断
    asyncio.run(diagnose_strategy(symbol, timeframe, days))

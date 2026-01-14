"""
回测结果处理器
负责将 BacktestEngine 的原始输出转换为前端可用的增强结果格式。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from src.evaluation.metrics.comprehensive import EnhancedAnalyzer
from src.utils.logger import get_logger

logger = get_logger("result_processor")

class BacktestResultProcessor:
    """回测结果处理器"""
    
    @staticmethod
    def process(
        engine_result: Dict[str, Any],
        strategy: Any,
        universe: List[str],
        initial_capital: float,
        commission_rate: float,
        slippage_rate: float,
        benchmark_symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理回测引擎原始结果
        
        Args:
            engine_result: BacktestEngine.run() 的返回值
            strategy: 策略实例
            universe: 标的池
            initial_capital: 初始资产
            commission_rate: 手续费率
            slippage_rate: 滑点率
            benchmark_symbol: 基准标的
            
        Returns:
            满足前端请求的增强结果字典
        """
        # 1. 提取基础数据
        fills = engine_result.get("fills", [])
        portfolio_df = engine_result.get("portfolio_daily", pd.DataFrame())
        bars = engine_result.get("bars", pd.DataFrame())
        run_id = engine_result.get("run_id", "unknown")
        
        if portfolio_df.empty:
            logger.warning("回测净值曲线为空")
            return {"error": "No equity data generated"}

        # Ensure index is datetime
        if 'timestamp' in portfolio_df.columns:
            portfolio_df = portfolio_df.set_index('timestamp')
        
        if not isinstance(portfolio_df.index, pd.DatetimeIndex):
             logger.warning("Portfolio dataframe index is not DatetimeIndex, attempting conversion...")
             try:
                 portfolio_df.index = pd.to_datetime(portfolio_df.index)
             except Exception as e:
                 logger.error(f"Failed to convert index to datetime: {e}")
                 return {"error": "Invalid portfolio data format"}

        # 2. 确定有效起始日期 (从第一笔交易开始，避免前面的平点)
        effective_start_dt = portfolio_df.index[0]
        first_fill_ts = None
        for fill in fills:
            ts_fill = getattr(fill, "ts_fill_utc", None)
            if ts_fill is not None:
                if first_fill_ts is None or ts_fill < first_fill_ts:
                    first_fill_ts = ts_fill
        
        if first_fill_ts is not None:
            first_fill_date = first_fill_ts.date()
            if first_fill_date > effective_start_dt.date():
                effective_start_dt = pd.Timestamp(first_fill_date).tz_localize("UTC")

        # 裁剪净值曲线
        portfolio_df = portfolio_df[portfolio_df.index >= effective_start_dt]
        if portfolio_df.empty:
            # 如果裁剪后为空（不应该发生），恢复原始数据
            portfolio_df = engine_result.get("portfolio_daily")
            effective_start_dt = portfolio_df.index[0]

        equity_series = portfolio_df['equity']
        returns = equity_series.pct_change().dropna()

        # 3. 使用 EnhancedAnalyzer 计算核心指标
        analyzer = EnhancedAnalyzer(benchmark_symbol=benchmark_symbol or "SPY")
        
        # 准备基准数据
        if not hasattr(effective_start_dt, "strftime"):
             logger.error(f"无效的起始日期类型: {type(effective_start_dt)} (value: {effective_start_dt})")
             return {"error": "Invalid date in result data"}

        benchmark_data = analyzer.get_benchmark_data(
            start_date=effective_start_dt.strftime("%Y-%m-%d"),
            end_date=portfolio_df.index[-1].strftime("%Y-%m-%d")
        )
        
        benchmark_returns = None
        benchmark_equity = None
        
        if benchmark_data:
            # 这里的 benchmark_data.equity_curve 是累计净值 (起始为1.0)，需要换算回资产金额
            benchmark_equity = [initial_capital * x for x in benchmark_data.equity_curve]
            
            # 使用 BenchmarkAnalyzer 获取对齐的收益率以进行精确计算
            from src.evaluation.metrics.benchmark import BenchmarkAnalyzer
            ba = BenchmarkAnalyzer(benchmark_symbol or "SPY")
            
            # 尝试本地
            b_prices = ba.get_benchmark_data(
                effective_start_dt.strftime("%Y-%m-%d"),
                portfolio_df.index[-1].strftime("%Y-%m-%d"),
                data_source="local"
            )
            # 尝试 IB
            if b_prices is None or b_prices.empty:
                b_prices = ba.get_benchmark_data(
                    effective_start_dt.strftime("%Y-%m-%d"),
                    portfolio_df.index[-1].strftime("%Y-%m-%d"),
                    data_source="ib"
                )
            
            if b_prices is not None and not b_prices.empty:
                try:
                    if isinstance(b_prices, pd.DataFrame):
                        benchmark_returns = b_prices['close'].pct_change().fillna(0)
                    else:
                        benchmark_returns = b_prices.pct_change().fillna(0)
                except Exception as e:
                    logger.warning(f"从基准价格计算收益率失败: {e}")
            else:
                logger.warning(f"未能加载对齐的基准收益率数据: {benchmark_symbol or 'SPY'}")
        else:
            logger.warning(f"未能获得基准数据对象 (benchmark_data is None)")
        
        # 如果没有基准数据，创建一个线性增长的虚拟基准
        if benchmark_equity is None or len(benchmark_equity) == 0:
            benchmark_equity = [
                initial_capital * (1 + i * (equity_series.iloc[-1]/equity_series.iloc[0]-1) / max(len(portfolio_df) - 1, 1))
                for i in range(len(portfolio_df))
            ]

        # 综合分析
        processed_trades = BacktestResultProcessor._format_trades(fills)
        
        comp_results = analyzer.comprehensive_analysis(
            returns=returns,
            portfolio_value=equity_series,
            trades=processed_trades,
            benchmark_returns=benchmark_returns,
            benchmark_equity=benchmark_equity,
            periods_per_year=252 # 假定美股日线
        )

        # 4. 准备价格数据 (用于前端图表)
        price_data = {"timestamps": [], "close": [], "volume": []}
        if not bars.empty and 'symbol' in bars.columns:
            # 使用第一个标的作为参考
            ref_symbol = universe[0] if universe else bars['symbol'].unique()[0]
            symbol_bars = bars[bars['symbol'] == ref_symbol]
            symbol_bars = symbol_bars[symbol_bars.index >= effective_start_dt]
            if not symbol_bars.empty:
                price_data["timestamps"] = [str(d)[:10] for d in symbol_bars.index]
                price_data["close"] = symbol_bars['close'].tolist()
                price_data["volume"] = symbol_bars['volume'].tolist() if 'volume' in symbol_bars.columns else []

        # 5. 组装最终结果
        strategy_name = "unknown"
        if hasattr(strategy, "strategy_id"):
            strategy_name = strategy.strategy_id
        elif hasattr(strategy, "name"):
            strategy_name = strategy.name
        elif isinstance(strategy, type):
            strategy_name = strategy.__name__
        elif strategy is not None:
            strategy_name = strategy.__class__.__name__

        final_result = {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "symbol": ",".join(universe) if len(universe) <= 5 else f"{universe[0]} (+{len(universe)-1})",
            "universe": universe,
            "backtest_period": {
                "start_date": effective_start_dt.isoformat(),
                "end_date": portfolio_df.index[-1].isoformat(),
                "days": (portfolio_df.index[-1] - effective_start_dt).days
            },
            "config": {
                "initial_capital": initial_capital,
                "commission_rate": commission_rate,
                "slippage_rate": slippage_rate,
            },
            "price_data": price_data,
            "trades": processed_trades,
            "results": {
                "final_value": float(equity_series.iloc[-1]),
                "total_return": float(comp_results.get("total_return", 0)),
            },
            "enhanced_metrics": BacktestResultProcessor._convert_numpy(comp_results),
            "equity_comparison": BacktestResultProcessor._convert_numpy(
                analyzer.generate_equity_comparison(
                    strategy_equity=equity_series.tolist(),
                    benchmark_equity=benchmark_equity,
                    timestamps=portfolio_df.index.tolist()
                )
            ),
            "benchmark_data": {
                "symbol": benchmark_data.symbol if benchmark_data else (benchmark_symbol or "SPY"),
                "total_return": benchmark_data.total_return if benchmark_data else 0.0,
                "alpha": comp_results.get("alpha"),
                "beta": comp_results.get("beta"),
                "sharpe_ratio": benchmark_data.sharpe_ratio if benchmark_data else 0.0,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return final_result

    @staticmethod
    def _format_trades(fills: List[Any]) -> List[Dict[str, Any]]:
        """将引擎生成的 Fill 对象转换为格式化的字典列表"""
        trades = []
        for i, fill in enumerate(fills):
            ts = getattr(fill, "ts_fill_utc", None)
            side = getattr(fill, "side", None)
            side_str = side.value.lower() if side else ""
            price = float(getattr(fill, "price", 0))
            qty = float(getattr(fill, "qty", 0))
            
            trades.append({
                "id": str(i + 1),
                "timestamp": ts.isoformat() if ts else None,
                "symbol": getattr(fill, "symbol", ""),
                "order_type": side_str,
                "price": price,
                "size": qty,
                "amount": price * qty,
                "commission": float(getattr(fill, "fee", 0)),
            })
        trades.sort(key=lambda t: t.get("timestamp") or "")
        return trades

    @staticmethod
    def _convert_numpy(obj: Any) -> Any:
        """递归转换 NumPy 类型为标准 Python 类型，并处理 NaN/Inf"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, float)):
            # 处理 NaN 和 Inf
            if pd.isna(obj) or (isinstance(obj, float) and (np.isinf(obj))):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [BacktestResultProcessor._convert_numpy(x) for x in obj.tolist()]
        elif isinstance(obj, dict):
            return {k: BacktestResultProcessor._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BacktestResultProcessor._convert_numpy(x) for x in obj]
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

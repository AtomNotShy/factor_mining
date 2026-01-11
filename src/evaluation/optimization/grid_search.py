"""
网格搜索优化器
"""

import itertools
import time
from typing import Dict, List, Any
import pandas as pd
from .optimizer_base import OptimizerBase, OptimizationResult
from src.evaluation.backtesting.engine import BacktestEngine
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from datetime import datetime

class GridSearchOptimizer(OptimizerBase):
    """网格搜索优化器"""
    
    async def optimize(
        self,
        strategy_cls: Any,
        param_space: Dict[str, List[Any]],
        backtest_config: Dict[str, Any],
    ) -> OptimizationResult:
        # 生成所有参数组合
        keys = list(param_space.keys())
        values = list(param_space.values())
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        engine = BacktestEngine(
            initial_capital=backtest_config.get("initial_capital", 100000.0),
            commission_rate=backtest_config.get("commission_rate", 0.001),
            slippage_rate=backtest_config.get("slippage_rate", 0.0005)
        )
        
        # 预加载数据以提速
        start_date = backtest_config["start_date"]
        end_date = backtest_config["end_date"]
        universe = backtest_config["universe"]
        
        ctx = RunContext(
            env=Environment.RESEARCH,
            code_version="opt_v1",
            data_version="latest",
            config_hash="N/A",
            now_utc=datetime.now(),
            trading_calendar=TradingCalendar()
        )
        
        bars = await engine._load_bars(universe, start_date, end_date, ctx)
        
        results = []
        best_metric = -float("inf")
        best_params = None
        best_result_details = None
        
        start_time = time.time()
        
        for i, raw_params in enumerate(combinations):
            # 虽然有些参数可能由策略配置控制，这里根据策略类创建实例
            from src.strategies.base.strategy import StrategyConfig
            
            config = StrategyConfig(
                strategy_id=f"opt_trial_{i}",
                timeframe=backtest_config.get("timeframe", "1d"),
                params=raw_params
            )
            strategy = strategy_cls(config)
            
            # 运行回测
            res = await engine.run(
                strategies=[strategy],
                universe=universe,
                start=start_date,
                end=end_date,
                ctx=None, # 优化过程中不保存中间结果到磁盘
                bars=bars
            )
            
            # 计算目标指标 (这里简化处理，获取计算后的夏普比率等，目前引擎结果里只有 total_return 和 final_equity)
            # 我们这里计算一个简单的 Sharpe (如果可能) 或直接用 total_return
            current_metric = res.get("total_return", 0.0)
            
            # 记录试验
            trial_record = raw_params.copy()
            trial_record["metric"] = current_metric
            results.append(trial_record)
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_params = raw_params
                best_result_details = res
                
        duration = time.time() - start_time
        
        return OptimizationResult(
            params=best_params,
            metric_value=best_metric,
            all_results=pd.DataFrame(results),
            best_backtest_result=best_result_details,
            duration=duration
        )

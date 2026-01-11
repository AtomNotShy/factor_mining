"""
贝叶斯优化器
使用 Optuna 作为优化框架
"""

import time
from typing import Dict, List, Any, Optional
import pandas as pd
from .optimizer_base import OptimizerBase, OptimizationResult
from src.evaluation.backtesting.engine import BacktestEngine
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from datetime import datetime

class BayesianOptimizer(OptimizerBase):
    """贝叶斯优化器 (基于 Optuna)"""
    
    async def optimize(
        self,
        strategy_cls: Any,
        param_space: Dict[str, Any],
        backtest_config: Dict[str, Any],
        n_trials: int = 50,
    ) -> OptimizationResult:
        """
        运行贝叶斯优化
        
        Args:
            param_space: 参数空间定义，格式如:
                {
                    "lookback": ("int", 10, 100),
                    "threshold": ("float", 0.0, 0.05),
                    "p_type": ("categorical", ["A", "B"])
                }
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("请安装 optuna 以使用贝叶斯优化: pip install optuna")

        engine = BacktestEngine(
            initial_capital=backtest_config.get("initial_capital", 100000.0),
            commission_rate=backtest_config.get("commission_rate", 0.001),
            slippage_rate=backtest_config.get("slippage_rate", 0.0005)
        )
        
        # 预加载数据
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

        def objective(trial):
            import asyncio
            # 构建试验参数
            params = {}
            for name, space in param_space.items():
                p_type = space[0]
                if p_type == "int":
                    params[name] = trial.suggest_int(name, space[1], space[2])
                elif p_type == "float":
                    params[name] = trial.suggest_float(name, space[1], space[2])
                elif p_type == "categorical":
                    params[name] = trial.suggest_categorical(name, space[1])
            
            from src.strategies.base.strategy import StrategyConfig
            config = StrategyConfig(
                strategy_id=f"opt_trial_{trial.number}",
                timeframe=backtest_config.get("timeframe", "1d"),
                params=params
            )
            strategy = strategy_cls(config)
            
            res = asyncio.run(engine.run(
                strategies=[strategy],
                universe=universe,
                start=start_date,
                end=end_date,
                ctx=None,
                bars=bars
            ))
            
            # 返回目标指标
            return res.get(self.objective_metric, res.get("total_return", 0.0))

        start_time = time.time()
        
        # 创建 Optuna Study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        duration = time.time() - start_time
        
        # 转换结果
        all_results = study.trials_dataframe()
        
        return OptimizationResult(
            params=study.best_params,
            metric_value=study.best_value,
            all_results=all_results,
            best_backtest_result={}, # 可以在此处二次运行获取详情
            duration=duration
        )

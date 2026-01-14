"""
贝叶斯优化器 (现代版)
集成 HistoryLoader 和 UnifiedBacktestEngine
"""

import asyncio
import time
import optuna
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from src.evaluation.optimization.optimizer_base import OptimizerBase, OptimizationResult
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine, UnifiedConfig, FeatureFlag
from src.evaluation.backtesting.config import TradeConfig, TimeConfig
from src.data.loader import HistoryLoader
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.strategies.base.strategy import Strategy, StrategyConfig
from src.utils.logger import get_logger

class BayesianOptimizer(OptimizerBase):
    """
    基于 Optuna 的贝叶斯优化器
    """

    def __init__(self, objective_metric: str = "sharpe_ratio"):
        super().__init__(objective_metric=objective_metric)
        self.logger = get_logger("optimizer.bayesian")

    async def optimize(
        self,
        strategy_cls: Type[Strategy],
        param_space: Dict[str, Any],
        backtest_config: Dict[str, Any],
        n_trials: int = 50,
    ) -> OptimizationResult:
        """
        运行优化
        """
        self.logger.info(f"开始贝叶斯优化: {strategy_cls.__name__}, 目标: {self.objective_metric}, 试验次数: {n_trials}")
        
        # 1. 设置数据加载
        loader = HistoryLoader(
            datadir=backtest_config.get("datadir", "./data"),
            startup_candle_count=backtest_config.get("startup_candle_count", 200)
        )
        
        start_date = backtest_config["start_date"]
        end_date = backtest_config["end_date"]
        universe = backtest_config["universe"]
        timeframe = backtest_config.get("timeframe", "1d")

        # 2. 预预热/加载数据 (避免在 trial 中重复加载)
        bars = await loader.load_data(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        if bars.empty:
            raise ValueError("优化器加载数据失败，数据为空")

        # 3. 初始化引擎 (使用 UnifiedBacktestEngine)
        config = UnifiedConfig(
            trade=TradeConfig(
                initial_capital=backtest_config.get("stake_amount", 100000.0),
                commission_rate=backtest_config.get("commission", 0.001),
                slippage_rate=backtest_config.get("slippage", 0.0005),
            ),
            time=TimeConfig(signal_timeframe=timeframe),
            features=FeatureFlag.ALL,
        )
        engine = UnifiedBacktestEngine(config=config)

        def objective(trial: optuna.Trial):
            # 构造本次试验的参数
            params = {}
            for name, space in param_space.items():
                p_type, p_min, p_max = space
                if p_type == "int":
                    params[name] = trial.suggest_int(name, p_min, p_max)
                elif p_type == "float":
                    params[name] = trial.suggest_float(name, p_min, p_max)
                elif p_type == "categorical":
                    params[name] = trial.suggest_categorical(name, p_min) # p_min 为 list
            
            # 创建策略实例
            s_config = StrategyConfig(
                strategy_id=f"trial_{trial.number}",
                timeframe=timeframe,
                params=params
            )
            strategy = strategy_cls(s_config)

            # 设置上下文
            ctx = RunContext(
                env=Environment.RESEARCH,
                code_version="hyperopt",
                data_version="latest",
                config_hash=str(trial.number),
                now_utc=datetime.now(),
                trading_calendar=TradingCalendar()
            )

            # 运行回测 (同步包装)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(engine.run(
                    strategies=[strategy],
                    universe=universe,
                    start=start_date.date(),
                    end=end_date.date(),
                    ctx=ctx,
                    bars=bars
                ))
            finally:
                loop.close()

            # 获取目标指标
            val = results.get(self.objective_metric, 0.0)
            if pd.isna(val):
                return -1.0 # 惩罚 NaN 结果
            return val

        # 4. 运行优化
        start_time = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        duration = time.time() - start_time

        self.logger.info(f"优化完成! 最佳参数: {study.best_params}, 最佳指标值: {study.best_value:.4f}")

        return OptimizationResult(
            params=study.best_params,
            metric_value=study.best_value,
            all_results=study.trials_dataframe(),
            best_backtest_result={}, # 可选：保存详情
            duration=duration
        )

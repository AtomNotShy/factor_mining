"""
Walk-Forward Analysis (WFA)
滚动窗口训练/验证/测试

⚠️ 重要说明：
- 本模块实现了真正的Walk-Forward分析，包括训练期的参数优化
- 参数优化方法：网格搜索 (Grid Search)
- 过拟合检测：比较训练期和测试期的性能衰减
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import date, timedelta
from itertools import product
import pandas as pd

from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.evaluation.backtesting.engine import BacktestEngine
from src.strategies.base.strategy import Strategy
from src.utils.logger import get_logger


class WalkForwardAnalyzer:
    """Walk-Forward分析器"""
    
    def __init__(self, engine: Optional[BacktestEngine] = None):
        self.engine = engine or BacktestEngine()
        self.logger = get_logger("walk_forward")
    
    async def run(
        self,
        strategies: List[Strategy],
        universe: List[str],
        start: date,
        end: date,
        train_window_days: int = 252,  # 1年
        test_window_days: int = 63,  # 1季度
        step_days: int = 21,  # 1个月步进
        param_grid: Optional[Dict[str, List[Any]]] = None,  # 参数网格
        optimization_metric: str = "sharpe_ratio",  # 优化指标
        ctx: Optional[RunContext] = None,
    ) -> Dict[str, Any]:
        """
        运行真正的Walk-Forward分析（带参数优化）
        
        Args:
            strategies: 策略列表
            universe: 股票池
            start: 起始日期
            end: 结束日期
            train_window_days: 训练窗口天数
            test_window_days: 测试窗口天数
            step_days: 步进天数
            param_grid: 参数网格（可选，如果不提供则不做参数优化）
                示例: {"param1": [0.1, 0.2, 0.3], "param2": [10, 20, 30]}
            optimization_metric: 优化指标 (sharpe_ratio, total_return, max_drawdown)
            ctx: 运行上下文
            
        Returns:
            WFA结果字典，包含：
            - wfa_results: 每个窗口的结果
            - total_windows: 窗口数量
            - pass_rate: 通过率
            - optimization_results: 参数优化结果（如果有）
            - degradation_analysis: 过拟合分析
        """
        results = []
        optimization_results = []
        current_start = start
        
        while current_start < end:
            # 计算窗口
            train_start = current_start
            train_end = train_start + timedelta(days=train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_window_days)
            
            if test_end > end:
                break
            
            self.logger.info(
                f"WFA窗口: 训练 {train_start} - {train_end}, "
                f"测试 {test_start} - {test_end}"
            )
            
            train_ctx = RunContext(
                env=ctx.env if ctx else Environment.RESEARCH,
                code_version=ctx.code_version if ctx else "1.0.0",
                data_version=ctx.data_version if ctx else "latest",
                config_hash=ctx.config_hash if ctx else "",
                now_utc=ctx.now_utc if ctx else pd.Timestamp.utcnow(),
                trading_calendar=ctx.trading_calendar if ctx else TradingCalendar(),
            )
            
            test_ctx = RunContext(
                env=ctx.env if ctx else Environment.RESEARCH,
                code_version=ctx.code_version if ctx else "1.0.0",
                data_version=ctx.data_version if ctx else "latest",
                config_hash=ctx.config_hash if ctx else "",
                now_utc=ctx.now_utc if ctx else pd.Timestamp.utcnow(),
                trading_calendar=ctx.trading_calendar if ctx else TradingCalendar(),
            )
            
            # =================================================================
            # 训练阶段：参数优化（如果提供了param_grid）
            # =================================================================
            best_params = None
            if param_grid:
                self.logger.info(f"开始参数优化，参数网格: {param_grid}")
                
                # 网格搜索
                best_params, opt_result = await self._grid_search(
                    strategies=strategies,
                    universe=universe,
                    start=train_start,
                    end=train_end,
                    param_grid=param_grid,
                    metric=optimization_metric,
                    ctx=train_ctx,
                )
                
                self.logger.info(f"最优参数: {best_params}, 训练期指标: {opt_result}")
                
                optimization_results.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'best_params': best_params,
                    'best_metric': opt_result,
                })
                
                # 应用最优参数到策略
                for strategy in strategies:
                    if best_params:
                        strategy.set_params(best_params)
            
            # =================================================================
            # 测试阶段：使用最优参数（或默认参数）回测
            # =================================================================
            test_result = await self.engine.run(
                strategies=strategies,
                universe=universe,
                start=test_start,
                end=test_end,
                ctx=test_ctx,
            )
            
            results.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'test_result': test_result,
                'used_params': best_params,  # 记录测试时使用的参数
            })
            
            # 步进
            current_start += timedelta(days=step_days)
        
        # =================================================================
        # 过拟合分析
        # =================================================================
        degradation_analysis = self._analyze_degradation(results, optimization_results)
        
        return {
            'wfa_results': results,
            'total_windows': len(results),
            'pass_rate': self._calculate_pass_rate(results),
            'optimization_results': optimization_results,
            'degradation_analysis': degradation_analysis,
            'param_grid': param_grid,
            'optimization_metric': optimization_metric,
        }
    
    async def _grid_search(
        self,
        strategies: List[Strategy],
        universe: List[str],
        start: date,
        end: date,
        param_grid: Dict[str, List[Any]],
        metric: str,
        ctx: RunContext,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        网格搜索最优参数
        
        Args:
            strategies: 策略列表
            universe: 股票池
            start: 开始日期
            end: 结束日期
            param_grid: 参数网格
            metric: 优化指标
            ctx: 运行上下文
            
        Returns:
            (最优参数, 最优指标值)
        """
        if not param_grid:
            return None, 0.0
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))
        
        self.logger.info(f"网格搜索: {len(all_combinations)} 个参数组合")
        
        best_params = None
        best_metric_value = float('-inf') if metric != "max_drawdown" else float('inf')
        
        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))
            
            # 应用参数
            for strategy in strategies:
                strategy.set_params(params)
            
            # 回测
            result = await self.engine.run(
                strategies=strategies,
                universe=universe,
                start=start,
                end=end,
                ctx=ctx,
            )
            
            # 获取指标值
            metric_value = result.get(metric, 0.0)
            if metric == "max_drawdown":
                # 对于max_drawdown，越小越好
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
            else:
                # 对于其他指标，越大越好
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
            
            self.logger.debug(
                f"  组合 {i+1}/{len(all_combinations)}: {params} -> {metric}={metric_value:.4f}"
            )
        
        return best_params, best_metric_value
    
    def _analyze_degradation(
        self,
        results: List[Dict],
        optimization_results: List[Dict],
    ) -> Dict[str, Any]:
        """
        分析过拟合程度（训练期 vs 测试期性能衰减）
        
        Returns:
            过拟合分析结果
        """
        if not results or not optimization_results:
            return {'degradation': None, 'note': '无足够数据进行过拟合分析'}
        
        train_metrics = [r['best_metric'] for r in optimization_results]
        test_metrics = [r['test_result'].get('sharpe_ratio', 0) for r in results]
        
        if len(train_metrics) != len(test_metrics):
            return {'degradation': None, 'note': '训练/测试期数量不匹配'}
        
        # 计算衰减
        degradations = []
        for train_m, test_m in zip(train_metrics, test_metrics):
            if train_m != 0:
                degradation = (train_m - test_m) / abs(train_m)
                degradations.append(degradation)
        
        if not degradations:
            return {'degradation': None, 'note': '无法计算衰减'}
        
        avg_degradation = sum(degradations) / len(degradations)
        
        # 过拟合风险评估
        if avg_degradation < 0.1:
            risk_level = "低"
            risk_description = "性能衰减较小，过拟合风险低"
        elif avg_degradation < 0.3:
            risk_level = "中"
            risk_description = "存在一定衰减，建议进一步验证"
        else:
            risk_level = "高"
            risk_description = "衰减严重，可能存在过拟合"
        
        return {
            'degradation': avg_degradation,
            'risk_level': risk_level,
            'risk_description': risk_description,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'degradations': degradations,
        }
    
    def _calculate_pass_rate(self, results: List[Dict]) -> float:
        """计算通过率（基于测试期收益 > 0）"""
        if not results:
            return 0.0
        
        passed = sum(
            1 for r in results
            if r['test_result'].get('total_return', 0) > 0
        )
        return passed / len(results)


class ParameterPerturbation:
    """参数扰动测试"""
    
    def __init__(self, engine: Optional[BacktestEngine] = None):
        self.engine = engine or BacktestEngine()
        self.logger = get_logger("parameter_perturbation")
    
    async def run(
        self,
        strategy: Strategy,
        universe: List[str],
        start: date,
        end: date,
        base_params: Dict[str, Any],
        perturbation_pct: float = 0.1,  # ±10%
        ctx: Optional[RunContext] = None,
    ) -> Dict[str, Any]:
        """
        运行参数扰动测试
        
        Args:
            strategy: 策略
            universe: 股票池
            start: 起始日期
            end: 结束日期
            base_params: 基准参数
            perturbation_pct: 扰动百分比
            ctx: 运行上下文
            
        Returns:
            参数扰动结果
        """
        results = []
        
        # 基准参数
        base_result = await self.engine.run(
            strategies=[strategy],
            universe=universe,
            start=start,
            end=end,
            ctx=ctx,
        )
        results.append({
            'params': base_params,
            'perturbation': 0.0,
            'result': base_result,
        })
        
        # 参数扰动（±10%, ±20%）
        for pct in [perturbation_pct, -perturbation_pct, 2 * perturbation_pct, -2 * perturbation_pct]:
            perturbed_params = {}
            for key, value in base_params.items():
                if isinstance(value, (int, float)):
                    perturbed_params[key] = value * (1 + pct)
                else:
                    perturbed_params[key] = value
            
            # 创建新策略实例（简化：直接修改配置）
            strategy.config.params = perturbed_params
            
            perturbed_result = await self.engine.run(
                strategies=[strategy],
                universe=universe,
                start=start,
                end=end,
                ctx=ctx,
            )
            
            results.append({
                'params': perturbed_params,
                'perturbation': pct,
                'result': perturbed_result,
            })
        
        # 计算稳定性
        base_return = base_result.get('total_return', 0)
        perturbed_returns = [r['result'].get('total_return', 0) for r in results[1:]]
        
        stability = self._calculate_stability(base_return, perturbed_returns)
        
        return {
            'results': results,
            'base_return': base_return,
            'perturbed_returns': perturbed_returns,
            'stability': stability,
        }
    
    def _calculate_stability(self, base_return: float, perturbed_returns: List[float]) -> float:
        """计算稳定性（简化：基于收益差异）"""
        if not perturbed_returns:
            return 1.0
        
        differences = [abs(r - base_return) for r in perturbed_returns]
        avg_diff = sum(differences) / len(differences)
        
        # 稳定性 = 1 - 归一化的平均差异
        max_diff = max(abs(base_return), max(abs(r) for r in perturbed_returns))
        if max_diff == 0:
            return 1.0
        
        stability = 1.0 - (avg_diff / max_diff)
        return max(0.0, min(1.0, stability))


class Gate:
    """策略准入Gate"""
    
    def __init__(self):
        self.logger = get_logger("gate")
    
    def pass_gate(
        self,
        metrics: Dict[str, Any],
        robustness: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        判断是否通过准入Gate
        
        Args:
            metrics: 回测指标
            robustness: 稳健性测试结果（WFA/参数扰动等）
            
        Returns:
            (是否通过, 失败原因列表)
        """
        failures = []
        
        # 1. 成本后为正
        total_return = metrics.get('total_return', 0)
        if total_return <= 0:
            failures.append(f"成本后收益非正: {total_return:.2%}")
        
        # 2. Max DD 在阈值内
        max_drawdown = metrics.get('max_drawdown', 1.0)
        if max_drawdown > 0.20:  # 20%阈值
            failures.append(f"最大回撤超过阈值: {max_drawdown:.2%}")
        
        # 3. WFA通过率
        if robustness and 'wfa_pass_rate' in robustness:
            wfa_pass_rate = robustness['wfa_pass_rate']
            if wfa_pass_rate < 0.6:  # 60%阈值
                failures.append(f"WFA通过率不足: {wfa_pass_rate:.2%}")
        
        # 4. 参数扰动稳定性
        if robustness and 'param_stability' in robustness:
            param_stability = robustness['param_stability']
            if param_stability < 0.7:  # 70%阈值
                failures.append(f"参数扰动稳定性不足: {param_stability:.2%}")
        
        passed = len(failures) == 0
        
        if passed:
            self.logger.info("策略通过Gate准入")
        else:
            self.logger.warning(f"策略未通过Gate准入: {failures}")
        
        return passed, failures

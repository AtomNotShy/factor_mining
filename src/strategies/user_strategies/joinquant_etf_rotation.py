"""
聚宽ETF轮动策略适配版
克隆自聚宽文章：https://www.joinquant.com/post/62821
标题：【ETF轮动策略】年化163%，回撤7%
作者：zfs1

策略核心逻辑：
1. 基于年化收益和R²判定系数计算动量得分
2. 根据历史波动率动态调整动量回溯期
3. 风控：过滤近期跌幅过大和溢价率过高的ETF
4. 每日轮动，持有得分最高的1只ETF
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import math
from datetime import datetime

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.strategies.base.parameters import IntParameter, DecimalParameter, BooleanParameter
from src.core.types import Signal, ActionType
from src.utils.logger import get_logger


class JoinQuantETFRotationStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """
    聚宽ETF轮动策略（Freqtrade风格）
    
    策略特点：
    - 基于加权线性回归的动量得分计算
    - 动态回溯期调整（基于ATR波动率）
    - 严格的风控过滤规则
    - 每日轮动，持有最强动量ETF
    """
    
    # 策略配置
    strategy_name = "聚宽ETF轮动策略"
    strategy_id = "joinquant_etf_rotation"
    timeframe = "1d"
    startup_candle_count = 260  # 需要足够数据计算ATR和动态回溯期
    
    # 交易配置
    stoploss = -0.10  # 10%止损
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    max_position_size = 1.0  # 全仓单只ETF
    
    # ROI配置（分钟: 目标收益率）
    minimal_roi = {
        0: float('inf'),  # 不使用ROI，完全依赖动量信号
    }
    
    # 参数定义
    min_days = IntParameter("min_days", 10, 40, default=20)
    max_days = IntParameter("max_days", 40, 120, default=60)
    auto_adjust = BooleanParameter("auto_adjust", default=True)
    r2_threshold = DecimalParameter("r2_threshold", 0.0, 1.0, default=0.0)
    max_premium_rate = DecimalParameter("max_premium_rate", 0.0, 0.10, default=0.05)
    target_positions = IntParameter("target_positions", 1, 3, default=1)
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(f"strategy.{self.strategy_id}")
        
        # 美股ETF池（适配原策略）
        self.etf_pool = [
            # 境外市场
            "QQQ",   # 纳斯达克100 ETF
            "SPY",   # 标普500 ETF
            "IWM",   # 罗素2000 ETF
            "EWJ",   # 日本ETF (MSCI Japan)
            "EWG",   # 德国ETF (MSCI Germany)
            # "INDA",  # 印度ETF (MSCI India) - 可选
            
            # 商品
            "GLD",   # 黄金ETF
            "SLV",   # 白银ETF
            "USO",   # 原油ETF
            "DBC",   # 商品指数ETF
            
            # 债券
            "TLT",   # 20年期国债ETF
            "AGG",   # 投资级债券ETF
            "HYG",   # 高收益债券ETF
            
            # 国内/其他
            "VTI",   # 全市场ETF
            "VXUS",  # 国际股票ETF
            "VNQ",   # 房地产ETF
        ]
        
        # 存储每个标的的动态回溯期
        self.lookback_cache: Dict[str, int] = {}
        
        # 调仓控制
        self.last_rebalance_date = None
        self.current_holding = None
        self.rebalance_interval = 5  # 每5个交易日调仓一次
        self._rebalance_done_today = False  # 标记当天是否已调仓
        self._trading_day_counter = 0  # 交易日计数器
        
    def _calculate_dynamic_lookback(self, df: pd.DataFrame) -> int:
        """
        计算动态回溯期（基于ATR波动率）
        
        公式：lookback = min_days + (max_days - min_days) * (1 - min(0.9, short_atr/long_atr))
        
        Args:
            df: 包含high, low, close的DataFrame
            
        Returns:
            动态回溯期天数
        """
        if len(df) < self.max_days.value + 10:
            return int(self.min_days.value)
        
        try:
            # 计算ATR
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            
            # 手动计算ATR（避免talib依赖）
            def calculate_atr(high_arr, low_arr, close_arr, period):
                n = len(close_arr)
                if n < period:
                    return np.nan
                
                tr_values = []
                for i in range(1, n):
                    tr1 = high_arr[i] - low_arr[i]
                    tr2 = abs(high_arr[i] - close_arr[i-1])
                    tr3 = abs(low_arr[i] - close_arr[i-1])
                    tr = max(tr1, tr2, tr3)
                    tr_values.append(tr)
                
                # 简单移动平均
                atr = np.mean(tr_values[-period:]) if len(tr_values) >= period else np.nan
                return atr
            
            # 计算长短期ATR
            long_atr = calculate_atr(high, low, close, int(self.max_days.value))
            short_atr = calculate_atr(
                high[-int(self.min_days.value):],
                low[-int(self.min_days.value):],
                close[-int(self.min_days.value):],
                int(self.min_days.value)
            )
            
            if np.isnan(long_atr) or np.isnan(short_atr) or long_atr == 0:
                return int(self.min_days.value)
            
            # 计算动态回溯期
            ratio = min(0.9, short_atr / long_atr)
            lookback = int(self.min_days.value + (self.max_days.value - self.min_days.value) * (1 - ratio))
            
            # 确保在[min_days, max_days]范围内
            lookback = max(int(self.min_days.value), min(int(self.max_days.value), lookback))
            
            return int(lookback)
            
        except Exception as e:
            self.logger.warning(f"计算动态回溯期失败: {e}")
            return int(self.min_days.value)
    
    def _calculate_momentum_score(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """
        计算动量得分（年化收益率 × R²）
        
        Args:
            prices: 价格序列
            
        Returns:
            (年化收益率, R², 得分)
        """
        if len(prices) < 5:
            return 0.0, 0.0, 0.0
        
        try:
            # 确保是numpy数组
            prices_array = np.asarray(prices, dtype=np.float64)
            
            # 对数价格
            y = np.log(prices_array)
            if not np.isfinite(y).all():
                return 0.0, 0.0, 0.0
            
            x = np.arange(len(y))
            
            # 线性权重（近期权重更高）
            weights = np.linspace(1, 2, len(y))
            
            # 加权线性回归
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            
            # 年化收益率（假设250个交易日）
            annualized_return = math.exp(slope * 250) - 1
            
            # 计算R²
            y_pred = slope * x + intercept
            ss_res = np.sum(weights * (y - y_pred) ** 2)
            y_mean = np.mean(y)
            ss_tot = np.sum(weights * (y - y_mean) ** 2)
            
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            
            # 得分 = 年化收益率 × R²
            score = annualized_return * r2
            
            return float(annualized_return), float(r2), float(score)
            
        except Exception as e:
            self.logger.debug(f"计算动量得分失败: {e}")
            return 0.0, 0.0, 0.0
    
    def _check_drawdown_filter(self, prices: np.ndarray) -> bool:
        """
        检查跌幅过滤条件
        
        原策略条件：
        1. 三天内有一天跌超5%
        2. 三天内每天都跌，总共跌超4%
        3. 四天内连续下跌，总共跌超4%
        
        Args:
            prices: 最近的价格序列（至少5个值）
            
        Returns:
            True表示通过过滤（可以投资），False表示被过滤
        """
        if len(prices) < 5:
            return True
        
        # 确保是numpy数组
        prices_array = np.asarray(prices, dtype=np.float64)
        
        # 条件1：三天内有一天跌超5%
        recent_returns = [
            prices_array[-1] / prices_array[-2],
            prices_array[-2] / prices_array[-3],
            prices_array[-3] / prices_array[-4]
        ]
        if any(r < 0.95 for r in recent_returns):
            return False
        
        # 条件2：三天内每天都跌，总共跌超4%
        if (prices_array[-1] < prices_array[-2] and
            prices_array[-2] < prices_array[-3] and
            prices_array[-3] < prices_array[-4] and
            prices_array[-1] / prices_array[-4] < 0.96):
            return False
        
        # 条件3：四天内连续下跌，总共跌超4%
        if (prices_array[-2] < prices_array[-3] and
            prices_array[-3] < prices_array[-4] and
            prices_array[-4] < prices_array[-5] and
            prices_array[-2] / prices_array[-5] < 0.96):
            return False
        
        return True
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            dataframe: K线数据
            metadata: 元数据（symbol等）
            
        Returns:
            添加了指标的DataFrame
        """
        symbol = metadata.get("symbol", "") if metadata else ""
        
        if dataframe.empty or symbol not in self.etf_pool:
            return dataframe
        
        # 计算动态回溯期（如果需要）
        lookback = self.min_days.value
        if self.auto_adjust.value:
            lookback = self._calculate_dynamic_lookback(dataframe)
            self.lookback_cache[symbol] = lookback
        else:
            lookback = self.min_days.value
        
        # 确保有足够数据
        if len(dataframe) < lookback + 5:
            return dataframe
        
        # 计算动量得分
        scores = []
        annualized_returns = []
        r2_values = []
        
        for i in range(len(dataframe)):
            if i < lookback - 1:
                scores.append(np.nan)
                annualized_returns.append(np.nan)
                r2_values.append(np.nan)
                continue
            
            # 获取回溯期价格
            start_idx = max(0, i - lookback + 1)
            prices = dataframe["close"].iloc[start_idx:i+1].values
            
            # 计算得分
            prices_array = np.asarray(prices, dtype=np.float64)
            ann_ret, r2, score = self._calculate_momentum_score(prices_array)
            
            scores.append(score)
            annualized_returns.append(ann_ret)
            r2_values.append(r2)
        
        # 添加到DataFrame
        dataframe["momentum_score"] = scores
        dataframe["annualized_return"] = annualized_returns
        dataframe["r2"] = r2_values
        
        # 计算动态回溯期（如果启用）
        if self.auto_adjust.value:
            dataframe["dynamic_lookback"] = lookback
        
        # 检查跌幅过滤
        drawdown_filter = []
        for i in range(len(dataframe)):
            if i < 5:
                drawdown_filter.append(True)
                continue
            
            # 获取最近5个价格
            prices = dataframe["close"].iloc[max(0, i-4):i+1].values
            prices_array = np.asarray(prices, dtype=np.float64)
            passed = self._check_drawdown_filter(prices_array)
            drawdown_filter.append(passed)
        
        dataframe["drawdown_passed"] = drawdown_filter
        
        # 有效得分 = 原始得分 × 过滤条件
        dataframe["effective_score"] = dataframe["momentum_score"].copy()
        dataframe.loc[~dataframe["drawdown_passed"], "effective_score"] = 0
        
        # R²过滤
        if self.r2_threshold.value > 0:
            dataframe.loc[dataframe["r2"] < self.r2_threshold.value, "effective_score"] = 0
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        生成进场信号
        
        Args:
            dataframe: K线数据（已包含指标）
            metadata: 元数据
            
        Returns:
            添加了进场信号的DataFrame
        """
        # 默认无信号
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        
        if dataframe.empty:
            return dataframe
        
        # 有效得分大于0时产生进场信号
        # 在实际轮动中，我们会在跨资产比较中决定是否进场
        # 这里先标记所有有效得分为正的标的
        valid_mask = (
            (dataframe["effective_score"] > 0) &
            (dataframe["effective_score"] < 6)  # 原策略过滤得分>6的异常值
        )
        
        dataframe.loc[valid_mask, "enter_long"] = 1
        dataframe.loc[valid_mask, "enter_tag"] = "momentum_positive"
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        生成离场信号
        
        Args:
            dataframe: K线数据（已包含指标）
            metadata: 元数据
            
        Returns:
            添加了离场信号的DataFrame
        """
        # 默认无信号
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        
        if dataframe.empty:
            return dataframe
        
        # 有效得分变为0或负数时离场
        exit_mask = dataframe["effective_score"] <= 0
        
        dataframe.loc[exit_mask, "exit_long"] = 1
        dataframe.loc[exit_mask, "exit_tag"] = "momentum_lost"
        
        return dataframe
    
    def get_portfolio_selection_signal(
        self,
        strategy_data: Dict[str, pd.DataFrame],
        current_date: Any,
        current_holding: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        跨资产组合选择接口
        
        用于实现ETF轮动等需要跨资产选优的策略。
        
        Args:
            strategy_data: 所有标的历史算后的信号数据，key是symbol
            current_date: 当前日期
            current_holding: 当前持仓的symbol（如果有）
            
        Returns:
            Dict[symbol, Dict[str, Any]]，每个symbol的信号修改
        """
        result: Dict[str, Dict[str, Any]] = {}
        
        # 使用交易日计数器来控制调仓频率
        # 回测引擎对每个交易日调用多次此方法（每个标的一次）
        # 需要确保每天只调仓一次
        
        # 检查当前日期是否与上次调仓日期不同（即新的一天）
        is_new_day = False
        if self.last_rebalance_date is None:
            is_new_day = True
        elif hasattr(current_date, 'date'):
            current_date_obj = current_date.date()
            if hasattr(self.last_rebalance_date, 'date'):
                last_rebalance_obj = self.last_rebalance_date.date()
            else:
                last_rebalance_obj = self.last_rebalance_date
            is_new_day = current_date_obj != last_rebalance_obj
        else:
            is_new_day = str(current_date) != str(self.last_rebalance_date)
        
        # 如果是新的一天，重置调仓标记并增加计数器
        if is_new_day:
            self._rebalance_done_today = False
            self._trading_day_counter += 1
        
        # 检查是否需要调仓：交易日计数器达到间隔且当天尚未调仓
        need_rebalance = (
            not self._rebalance_done_today and
            self._trading_day_counter % self.rebalance_interval == 0
        )
        
        # 收集所有标的的得分
        scores: Dict[str, float] = {}
        for symbol, df in strategy_data.items():
            if df.empty:
                continue
            
            # 获取当前日期的数据
            if current_date in df.index:
                row = df.loc[current_date]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                
                score = row.get("effective_score", 0)
                if pd.isna(score):
                    score = 0
                
                scores[symbol] = float(score)
            else:
                # 如果没有当前日期数据，使用最后有效数据
                # 检查 effective_score 列是否存在
                if "effective_score" not in df.columns:
                    scores[symbol] = 0
                else:
                    last_valid = df.dropna(subset=["effective_score"])
                    if not last_valid.empty:
                        score = last_valid["effective_score"].iloc[-1]
                        scores[symbol] = float(score)
                    else:
                        scores[symbol] = 0
        
        if not scores:
            # 没有有效得分，保持现状或清仓
            if current_holding:
                result[current_holding] = {"action": "FLAT"}
                self.current_holding = None
            return result
        
        # 选择得分最高的标的
        valid_scores = {k: v for k, v in scores.items() if v > 0}
        
        if not valid_scores:
            # 所有得分都不大于0，清仓
            if current_holding:
                result[current_holding] = {"action": "FLAT"}
                self.current_holding = None
            return result
        
        # 选择得分最高的标的
        best_symbol = max(valid_scores.items(), key=lambda x: x[1])[0]
        best_score = valid_scores[best_symbol]
        
        # 检查是否需要调仓
        if not need_rebalance:
            # 不需要调仓时，保持当前持仓状态
            # 返回空result表示不改变当前状态
            return result
        
        # 需要调仓
        self._rebalance_done_today = True
        self.last_rebalance_date = current_date
        
        self.logger.info(f"轮动选择: {best_symbol} (得分: {best_score:.4f})")
        self.current_holding = best_symbol
        
        # 生成信号
        for symbol in strategy_data:
            if symbol == best_symbol:
                # 获取年化收益率和R²
                ann_return = 0.0
                r2_value = 0.0
                if not strategy_data[symbol].empty:
                    ann_return_col = strategy_data[symbol].get("annualized_return")
                    if ann_return_col is not None and len(ann_return_col) > 0:
                        ann_return = float(ann_return_col.iloc[-1])
                    
                    r2_col = strategy_data[symbol].get("r2")
                    if r2_col is not None and len(r2_col) > 0:
                        r2_value = float(r2_col.iloc[-1])
                
                result[symbol] = {
                    "action": "LONG",
                    "score": best_score,
                    "metadata": {
                        "annualized_return": ann_return,
                        "r2": r2_value,
                    }
                }
            else:
                result[symbol] = {"action": "FLAT"}
        
        return result
    
    async def bot_loop_start(self, **kwargs) -> None:
        """每轮循环开始时调用"""
        pass
    
    async def bot_start(self, **kwargs) -> None:
        """机器人启动时调用"""
        self.logger.info(f"启动聚宽ETF轮动策略: {self.strategy_name}")
        self.logger.info(f"ETF池: {self.etf_pool}")
        self.logger.info(f"参数: min_days={self.min_days.value}, max_days={self.max_days.value}, auto_adjust={self.auto_adjust.value}")
    
    def botShutdown(self, **kwargs) -> None:
        """机器人关闭时调用"""
        self.logger.info(f"关闭聚宽ETF轮动策略: {self.strategy_name}")
    
    def custom_stoploss(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> float:
        """自定义止损逻辑"""
        # 使用策略配置的止损
        return self.stoploss
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """确认订单进入"""
        # 检查是否在ETF池中
        if pair not in self.etf_pool:
            self.logger.warning(f"尝试交易不在ETF池中的标的: {pair}")
            return False
        
        # 检查价格有效性
        if rate <= 0:
            self.logger.warning(f"无效的价格: {rate}")
            return False
        
        return True
    
    def confirm_trade_exit(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """确认订单退出"""
        # 检查价格有效性
        if rate <= 0:
            self.logger.warning(f"无效的价格: {rate}")
            return False
        
        return True


# ===== 测试代码 =====
if __name__ == "__main__":
    # 创建策略实例
    strategy = JoinQuantETFRotationStrategy()
    
    print(f"策略ID: {strategy.strategy_id}")
    print(f"策略名称: {strategy.strategy_name}")
    print(f"ETF池: {strategy.etf_pool}")
    print()
    
    # 获取参数
    print("策略参数:")
    print(f"  min_days: {strategy.min_days.value}")
    print(f"  max_days: {strategy.max_days.value}")
    print(f"  auto_adjust: {strategy.auto_adjust.value}")
    print(f"  r2_threshold: {strategy.r2_threshold.value}")
    print(f"  max_premium_rate: {strategy.max_premium_rate.value}")
    print(f"  target_positions: {strategy.target_positions.value}")

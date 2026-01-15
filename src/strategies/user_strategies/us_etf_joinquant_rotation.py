"""
美股ETF轮动策略（聚宽风格）

克隆自聚宽文章：
1. https://www.joinquant.com/post/62821 - 【ETF轮动策略】年化163%，回撤7%
2. https://www.joinquant.com/post/60824 - 【思路分享】动量ETF轮动之基于历史波动率动态调整历史回溯期
3. https://www.joinquant.com/post/42673 - 【回顾3】ETF策略之核心资产轮动

核心逻辑：
1. 基于加权线性回归计算年化收益率（近N天）
2. R²判定系数过滤不稳定趋势
3. 基于历史波动率（ATR或收益率标准差）动态调整回溯期
4. 多重风控过滤（跌幅过滤、连续下跌过滤）
5. 每次只持有一只ETF，轮动到动量最强的ETF

重写于 Freqtrade 框架：
- 使用 FreqtradeStrategy 协议
- 实现 populate_indicators/entry_trend/exit_trend
- 支持 ROI 表和止损配置
- 完整的生命周期回调
"""

from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from datetime import datetime
import math

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.strategies.base.indicators import sma, ema, atr
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.core.types import Signal, PortfolioState, RiskState, OrderIntent
    from src.core.context import RunContext


logger = get_logger("strategy.us_etf_joinquant_rotation")


class USETFJoinQuantRotationStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """
    美股ETF轮动策略（聚宽风格）

    特点：
    - 加权线性回归计算动量（近期数据权重更高）
    - R²过滤不稳定趋势
    - 基于ATR动态调整回溯期
    - 多重风控过滤（跌幅过滤、连续下跌过滤）
    - 每次只持有一只ETF，轮动到动量最强的ETF

    Example:
        >>> from src.strategies.user_strategies.us_etf_joinquant_rotation import (
        ...     USETFJoinQuantRotationStrategy
        ... )
        >>> strategy = USETFJoinQuantRotationStrategy()
        >>> # 或自定义参数
        >>> strategy = USETFJoinQuantRotationStrategy(
        ...     strategy_id="custom_us_etf_rotation",
        ...     etf_pool=["SPY", "QQQ", "IWM"],
        ...     min_lookback_days=20,
        ...     max_lookback_days=60
        ... )
    """

    # ============================================================================
    # 策略配置 
    # ============================================================================

    strategy_name = "US ETF JoinQuant Rotation"
    strategy_id: str = "us_etf_joinquant_rotation"
    timeframe = "1d"
    startup_candle_count = 70  # 最大回溯期+10天用于ATR计算

    # ROI 配置 (分钟: 目标收益率)
    # 不使用自动 ROI，使用退出信号
    minimal_roi: Dict[int, float] = {
        0: float('inf'),
    }

    # 止损配置
    stoploss = -0.10  # -10% 止损

    # 追踪止损
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = False

    # 仓位配置
    position_adjustment_enable = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ============================================================================
    # 策略参数
    # ============================================================================

    # 美股ETF标的池
    etf_pool: List[str] = [
        # 大盘指数
        "SPY",    # S&P 500
        "QQQ",    # Nasdaq 100
        "DIA",    # Dow Jones
        "IWM",    # Russell 2000 (小盘股)
        # 国际/新兴市场
        "EFA",    # EAFE (欧洲、澳洲、远东)
        "EEM",    # 新兴市场
        "VWO",    # 新兴市场 (Vanguard)
        # 债券
        "TLT",    # 20+年国债
        "IEF",    # 7-10年国债
        "LQD",    # 投资级公司债
        # 商品
        "GLD",    # 黄金
        "SLV",    # 白银
        "USO",    # 原油
        # 行业/主题
        "XLK",    # 科技
        "XLV",    # 医疗保健
        "XLF",    # 金融
        "XLE",    # 能源
    ]

    # 回溯参数
    min_lookback_days: int = 20  # 最小回溯天数
    max_lookback_days: int = 60  # 最大回溯天数
    use_dynamic_lookback: bool = True  # 是否使用动态回溯期
    atr_period: int = 20  # ATR计算周期

    # 过滤参数
    r2_threshold: float = 0.5  # R²最小阈值
    decline_filter_days: int = 3  # 跌幅过滤天数
    decline_filter_threshold: float = 0.95  # 跌幅过滤阈值 (5%)
    consecutive_decline_threshold: float = 0.96  # 连续下跌过滤阈值 (4%)

    # 持仓参数
    target_positions: int = 1  # 目标持仓数量（每次只持有一只ETF）
    max_weight: float = 1.0  # 单标最大权重
    min_weight: float = 0.0  # 单标最小权重

    # 常量
    WEIGHT_START: float = 1.0
    WEIGHT_END: float = 2.0
    TRADING_DAYS: int = 252

    # ============================================================================
    # 构造函数
    # ============================================================================

    def __init__(
        self,
        strategy_id: Optional[str] = None,
        etf_pool: Optional[List[str]] = None,
        min_lookback_days: Optional[int] = None,
        max_lookback_days: Optional[int] = None,
        use_dynamic_lookback: Optional[bool] = None,
        atr_period: Optional[int] = None,
        r2_threshold: Optional[float] = None,
        decline_filter_days: Optional[int] = None,
        decline_filter_threshold: Optional[float] = None,
        consecutive_decline_threshold: Optional[float] = None,
        target_positions: Optional[int] = None,
        max_weight: Optional[float] = None,
        min_weight: Optional[float] = None,
        stoploss: Optional[float] = None,
        trailing_stop: Optional[bool] = None,
        trailing_stop_positive: Optional[float] = None,
    ):
        """
        初始化策略实例

        Args:
            strategy_id: 策略ID
            etf_pool: ETF标的池
            min_lookback_days: 最小回溯天数
            max_lookback_days: 最大回溯天数
            use_dynamic_lookback: 是否使用动态回溯期
            atr_period: ATR计算周期
            r2_threshold: R²最小阈值
            decline_filter_days: 跌幅过滤天数
            decline_filter_threshold: 跌幅过滤阈值
            consecutive_decline_threshold: 连续下跌过滤阈值
            target_positions: 目标持仓数量
            max_weight: 单标最大权重
            min_weight: 单标最小权重
            stoploss: 止损比例
            trailing_stop: 是否启用追踪止损
            trailing_stop_positive: 追踪止损正偏差
        """
        # 调用父类构造函数
        super().__init__()

        # 覆盖默认值（如果提供了参数）
        if strategy_id is not None:
            self.strategy_id = strategy_id
        if etf_pool is not None:
            self.etf_pool = etf_pool
        if min_lookback_days is not None:
            self.min_lookback_days = min_lookback_days
        if max_lookback_days is not None:
            self.max_lookback_days = max_lookback_days
        if use_dynamic_lookback is not None:
            self.use_dynamic_lookback = use_dynamic_lookback
        if atr_period is not None:
            self.atr_period = atr_period
        if r2_threshold is not None:
            self.r2_threshold = r2_threshold
        if decline_filter_days is not None:
            self.decline_filter_days = decline_filter_days
        if decline_filter_threshold is not None:
            self.decline_filter_threshold = decline_filter_threshold
        if consecutive_decline_threshold is not None:
            self.consecutive_decline_threshold = consecutive_decline_threshold
        if target_positions is not None:
            self.target_positions = target_positions
        if max_weight is not None:
            self.max_weight = max_weight
        if min_weight is not None:
            self.min_weight = min_weight
        if stoploss is not None:
            self.stoploss = stoploss
        if trailing_stop is not None:
            self.trailing_stop = trailing_stop
        if trailing_stop_positive is not None:
            self.trailing_stop_positive = trailing_stop_positive

        self.logger = get_logger(f"strategy.{self.strategy_id}")

    # ============================================================================
    # 生命周期回调
    # ============================================================================

    async def bot_start(self, **kwargs) -> None:
        """机器人启动时调用"""
        self.logger.info(f"🤖 [BOT START] {self.strategy_name}")
        self.logger.info(f"   ETF Pool: {self.etf_pool}")
        self.logger.info(f"   Lookback Range: {self.min_lookback_days}-{self.max_lookback_days} days")
        self.logger.info(f"   Dynamic Lookback: {self.use_dynamic_lookback}")
        self.logger.info(f"   R² Threshold: {self.r2_threshold}")

    async def bot_loop_start(self, **kwargs) -> None:
        """每轮循环开始时调用"""
        pass

    # ============================================================================
    # 核心指标计算
    # ============================================================================

    def _calculate_dynamic_lookback(self, high: pd.Series, low: pd.Series, close: pd.Series) -> int:
        """
        基于ATR计算动态回溯期

        根据聚宽策略逻辑：
        lookback = min_days + (max_days - min_days) * (1 - min(0.9, short_atr/long_atr))

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            动态回溯期（整数）
        """
        if len(close) < self.max_lookback_days + 10:
            return self.min_lookback_days  # 数据不足时使用最小回溯期

        try:
            # 计算长期和短期ATR
            long_atr_result = atr(high, low, close, self.max_lookback_days)
            short_atr_result = atr(high, low, close, self.min_lookback_days)
            
            # 获取最新的ATR值
            long_atr_series = long_atr_result.atr
            short_atr_series = short_atr_result.atr
            
            if len(long_atr_series) == 0 or len(short_atr_series) == 0:
                return self.min_lookback_days
            
            if pd.isna(long_atr_series.iloc[-1]) or pd.isna(short_atr_series.iloc[-1]):
                return self.min_lookback_days
            
            long_atr_val = float(long_atr_series.iloc[-1])
            short_atr_val = float(short_atr_series.iloc[-1])
            
            if long_atr_val == 0:
                return self.min_lookback_days
            
            # 计算ATR比率
            atr_ratio = short_atr_val / long_atr_val
            
            # 限制比率在[0, 0.9]范围内
            atr_ratio = min(0.9, max(0, atr_ratio))
            
            # 计算动态回溯期
            lookback = int(self.min_lookback_days +
                          (self.max_lookback_days - self.min_lookback_days) *
                          (1 - atr_ratio))
            
            # 确保在[min_days, max_days]范围内
            lookback = max(self.min_lookback_days, min(self.max_lookback_days, lookback))
            
            self.logger.debug(f"Dynamic lookback: {lookback} days (ATR ratio: {atr_ratio:.4f})")
            return lookback
            
        except Exception as e:
            self.logger.warning(f"动态回溯期计算失败: {e}")
            return self.min_lookback_days

    def _calculate_weighted_regression_momentum(self, prices: pd.Series, lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        计算加权线性回归动量（聚宽风格）

        使用加权最小二乘法，近期数据权重更高

        Args:
            prices: 价格序列
            lookback: 回溯期（如果为None，使用动态计算或默认值）

        Returns:
            Dict包含 momentum, r2, annual_volatility, valid, lookback_used
        """
        if lookback is None:
            lookback = self.min_lookback_days
        
        if len(prices) < lookback:
            return {
                "momentum": 0.0,
                "r2": 0.0,
                "annual_volatility": 0.0,
                "valid": False,
                "lookback_used": lookback,
                "error": f"数据不足 {lookback} 天"
            }

        # 使用最近lookback天的数据
        recent_prices = prices.iloc[-lookback:] if len(prices) > lookback else prices
        n = len(recent_prices)

        # 创建加权向量（线性递增权重）
        # 近期数据权重更高，权重范围 [1, 2]
        weights = np.linspace(self.WEIGHT_START, self.WEIGHT_END, n)

        try:
            # 对数价格
            y = np.log(recent_prices.values)
            x = np.arange(n)
            
            # 加权线性回归
            sum_w = np.sum(weights)
            sum_xw = np.sum(x * weights)
            sum_yw = np.sum(y * weights)
            sum_x2w = np.sum(x**2 * weights)
            sum_xyw = np.sum(x * y * weights)

            denominator = sum_w * sum_x2w - sum_xw**2
            if abs(denominator) < 1e-10:
                return {
                    "momentum": 0.0,
                    "r2": 0.0,
                    "annual_volatility": 0.0,
                    "valid": False,
                    "lookback_used": lookback,
                    "error": "线性回归分母接近零"
                }

            slope = (sum_w * sum_xyw - sum_xw * sum_yw) / denominator
            intercept = (sum_yw - slope * sum_xw) / sum_w

            # 年化收益率
            annual_return = math.exp(slope * self.TRADING_DAYS) - 1

            # 计算R²
            y_pred = slope * x + intercept
            ss_res = np.sum(weights * (y - y_pred)**2)
            ss_tot = np.sum(weights * (y - np.mean(y))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # 波动率（年化）
            returns = np.diff(np.log(recent_prices.values))
            if len(returns) >= 2:
                annual_volatility = float(np.std(returns) * np.sqrt(self.TRADING_DAYS))
            else:
                annual_volatility = 0.0

            return {
                "momentum": float(annual_return),
                "r2": float(r2),
                "annual_volatility": annual_volatility,
                "valid": True,
                "lookback_used": lookback,
                "error": None
            }

        except Exception as e:
            self.logger.exception("加权回归动量计算发生错误")
            return {
                "momentum": 0.0,
                "r2": 0.0,
                "annual_volatility": 0.0,
                "valid": False,
                "lookback_used": lookback,
                "error": str(e)
            }

    def _apply_risk_filters(self, prices: pd.Series) -> bool:
        """
        应用风控过滤（聚宽风格）

        过滤条件：
        1. 三天内有一天跌超5%
        2. 三天内每天都跌，总共跌超4%
        3. 四天内连续下跌等

        Args:
            prices: 价格序列（最近几天的价格）

        Returns:
            True: 通过风控过滤
            False: 未通过风控过滤
        """
        if len(prices) < self.decline_filter_days + 1:
            return True  # 数据不足时通过过滤

        # 获取最近几天的价格
        recent_prices = prices.iloc[-self.decline_filter_days-1:]
        
        # 计算每日收益率
        returns = []
        for i in range(1, len(recent_prices)):
            prev_price = recent_prices.iloc[i-1]
            curr_price = recent_prices.iloc[i]
            if prev_price > 0:
                daily_return = curr_price / prev_price
                returns.append(daily_return)
        
        if len(returns) < self.decline_filter_days:
            return True  # 数据不足时通过过滤
        
        # 条件1：三天内有一天跌超5%
        large_decline = any(r < self.decline_filter_threshold for r in returns)
        
        # 条件2：三天内每天都跌，总共跌超4%
        all_decline = all(r < 1.0 for r in returns)
        total_decline = np.prod(returns) if len(returns) > 0 else 1.0
        consecutive_decline = all_decline and (total_decline < self.consecutive_decline_threshold)
        
        # 条件3：四天内连续下跌（扩展检查）
        if len(returns) >= 4:
            # 检查是否有连续4天下跌
            consecutive_count = 0
            for r in returns:
                if r < 1.0:
                    consecutive_count += 1
                    if consecutive_count >= 4:
                        return False  # 连续4天下跌，过滤
                else:
                    consecutive_count = 0
        
        # 如果满足任一过滤条件，返回False
        if large_decline or consecutive_decline:
            self.logger.debug(f"风控过滤触发: large_decline={large_decline}, consecutive_decline={consecutive_decline}")
            return False
        
        return True

    def _calculate_momentum_score(
        self,
        momentum_result: Dict[str, Any],
    ) -> float:
        """
        综合动量评分

        考虑R²过滤和风险过滤

        Args:
            momentum_result: 动量计算结果

        Returns:
            float: 调整后的动量评分
        """
        # 基本验证
        if not momentum_result.get("valid", False):
            return 0.0

        # R²过滤
        r2 = momentum_result.get("r2", 0)
        if r2 < self.r2_threshold:
            self.logger.debug(
                f"R² ({r2:.4f}) < 阈值 ({self.r2_threshold})"
            )
            return 0.0

        # 获取动量值
        momentum = momentum_result.get("momentum", 0.0)
        
        # 波动率调整（可选）
        volatility = momentum_result.get("annual_volatility", 0)
        if volatility > 0.5:  # 高波动率惩罚
            adjusted_momentum = momentum * 0.7
            self.logger.debug(
                f"高波动率触发降权: vol={volatility:.4f}"
            )
        else:
            adjusted_momentum = momentum

        return float(adjusted_momentum)

    # ============================================================================
    # Freqtrade 协议方法
    # ============================================================================

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            dataframe: K线数据
            metadata: 元数据（包含 symbol）

        Returns:
            添加了指标的 DataFrame
        """
        symbol = metadata.get("symbol", "unknown") if metadata else "unknown"

        # 初始化列
        dataframe['momentum'] = 0.0
        dataframe['momentum_r2'] = 0.0
        dataframe['momentum_volatility'] = 0.0
        dataframe['momentum_valid'] = 0
        dataframe['momentum_score'] = 0.0
        dataframe['dynamic_lookback'] = self.min_lookback_days

        # 使用滚动窗口计算动量
        close_series = pd.Series(dataframe['close'].values, index=dataframe.index)
        high_series = pd.Series(dataframe['high'].values, index=dataframe.index)
        low_series = pd.Series(dataframe['low'].values, index=dataframe.index)
        
        # 为每一行计算滚动动量
        for i in range(len(dataframe)):
            if i < self.max_lookback_days:
                continue
                
            # 获取滚动窗口数据
            window_start = max(0, i - self.max_lookback_days)
            window_prices = close_series.iloc[window_start:i+1]
            window_high = high_series.iloc[window_start:i+1]
            window_low = low_series.iloc[window_start:i+1]
            
            # 计算动态回溯期
            if self.use_dynamic_lookback:
                lookback = self._calculate_dynamic_lookback(
                    window_high, window_low, window_prices
                )
            else:
                lookback = self.min_lookback_days
            
            # 计算动量
            momentum_result = self._calculate_weighted_regression_momentum(window_prices, lookback)
            
            # 填充结果
            dataframe.at[dataframe.index[i], 'momentum'] = momentum_result['momentum']
            dataframe.at[dataframe.index[i], 'momentum_r2'] = momentum_result['r2']
            dataframe.at[dataframe.index[i], 'momentum_volatility'] = momentum_result['annual_volatility']
            dataframe.at[dataframe.index[i], 'momentum_valid'] = 1 if momentum_result['valid'] else 0
            dataframe.at[dataframe.index[i], 'dynamic_lookback'] = lookback
            
            # 计算动量评分
            dataframe.at[dataframe.index[i], 'momentum_score'] = self._calculate_momentum_score(momentum_result)

        # 添加均线用于参考
        dataframe['sma_20'] = sma(close_series, 20)
        dataframe['sma_50'] = sma(close_series, 50)

        # 价格相对均线位置
        dataframe['price_vs_sma20'] = dataframe['close'] / dataframe['sma_20'] - 1
        dataframe['price_vs_sma50'] = dataframe['close'] / dataframe['sma_50'] - 1

        # 记录最后一天的指标
        if len(dataframe) > 0:
            last_idx = len(dataframe) - 1
            last_score = dataframe.iloc[last_idx]['momentum_score']
            last_r2 = dataframe.iloc[last_idx]['momentum_r2']
            last_lookback = dataframe.iloc[last_idx]['dynamic_lookback']
            self.logger.debug(
                f"{symbol} indicators: momentum_score={last_score:.4f}, "
                f"r2={last_r2:.4f}, lookback={last_lookback}"
            )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        生成进场信号

        在 dataframe 中添加 'enter_long' 列
        enter_long=1 表示买入信号

        Args:
            dataframe: K线数据（已包含指标）
            metadata: 元数据

        Returns:
            添加了进场信号的 DataFrame
        """
        # 初始化信号列
        dataframe['enter_long'] = 0
        dataframe['enter_tag'] = ""

        if len(dataframe) < 2:
            return dataframe

        # 计算动量评分变化
        dataframe['momentum_score_prev'] = dataframe['momentum_score'].shift(1)
        
        # 条件：动量评分为正、R²超过阈值、且动量评分从负变正
        entry_condition = (
            (dataframe['momentum_score'] > 0) &
            (dataframe['momentum_r2'] >= self.r2_threshold) &
            (dataframe['momentum_score_prev'] <= 0)
        )
        
        # 确保我们有足够的数据
        if 'momentum_valid' in dataframe.columns:
            entry_condition = entry_condition & (dataframe['momentum_valid'] == 1)

        # 应用风险过滤
        def apply_risk_filter(row_idx):
            if row_idx < self.decline_filter_days:
                return True
            window_prices = dataframe['close'].iloc[row_idx-self.decline_filter_days:row_idx+1]
            return self._apply_risk_filters(window_prices)
        
        # 为满足条件的行应用风险过滤
        risk_filter_mask = pd.Series([apply_risk_filter(i) for i in range(len(dataframe))], index=dataframe.index)
        entry_condition = entry_condition & risk_filter_mask

        # 应用条件
        dataframe.loc[entry_condition, 'enter_long'] = 1
        dataframe.loc[entry_condition, 'enter_tag'] = "momentum_positive_risk_passed"

        # 记录信号统计
        enter_count = int(dataframe['enter_long'].sum())
        if enter_count > 0:
            self.logger.info(
                f"{metadata.get('symbol', 'unknown') if metadata else 'unknown'}: "
                f"生成 {enter_count} 个进场信号"
            )

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        生成离场信号

        在 dataframe 中添加 'exit_long' 列
        exit_long=1 表示卖出信号

        Args:
            dataframe: K线数据（已包含指标）
            metadata: 元数据

        Returns:
            添加了离场信号的 DataFrame
        """
        # 初始化信号列
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ""

        if len(dataframe) < 2:
            return dataframe

        # 离场条件：
        # 1. 动量变为负值
        # 2. R²低于阈值
        # 3. 触发风险过滤
        exit_condition = (
            (dataframe['momentum_score'] < 0) |
            (dataframe['momentum_r2'] < self.r2_threshold)
        )

        # 应用风险过滤（如果风险过滤失败，也需要离场）
        def check_risk_exit(row_idx):
            if row_idx < self.decline_filter_days:
                return False
            window_prices = dataframe['close'].iloc[row_idx-self.decline_filter_days:row_idx+1]
            return not self._apply_risk_filters(window_prices)
        
        risk_exit_mask = pd.Series([check_risk_exit(i) for i in range(len(dataframe))], index=dataframe.index)
        exit_condition = exit_condition | risk_exit_mask

        # 处理布尔值情况 - 避免直接检查Series的布尔值
        # 检查是否是pandas Series
        if isinstance(exit_condition, pd.Series):
            # 这是Series对象
            dataframe.loc[exit_condition, 'exit_long'] = 1
            dataframe.loc[exit_condition, 'exit_tag'] = "momentum_reversal_or_risk"
        else:
            # 这是标量布尔值
            try:
                # 尝试转换为布尔值
                if bool(exit_condition):
                    dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                    dataframe.loc[dataframe.index[-1], 'exit_tag'] = "momentum_reversal_or_risk"
            except (ValueError, TypeError):
                # 如果转换失败，跳过
                pass

        # 记录信号统计
        exit_long_sum = dataframe['exit_long'].sum()
        if hasattr(exit_long_sum, '__len__'):
            exit_count = int(exit_long_sum)
        else:
            exit_count = 1 if exit_long_sum else 0
            
        if exit_count > 0:
            self.logger.info(
                f"{metadata.get('symbol', 'unknown') if metadata else 'unknown'}: "
                f"生成 {exit_count} 个离场信号"
            )

        return dataframe

    # ============================================================================
    # 自定义回调方法
    # ============================================================================

    def custom_stoploss(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> float:
        """
        自定义止损逻辑

        Args:
            pair: 交易对
            current_profit: 当前盈亏比例
            current_rate: 当前价格
            current_time: 当前时间

        Returns:
            止损价格或比例
        """
        # 动态止损：盈利时移动止损到成本价
        if current_profit > 0.02:  # 盈利超过2%
            return 0.0  # 移动到成本价
        elif current_profit > 0.05:  # 盈利超过5%
            return -0.02  # 锁定2%利润

        return self.stoploss  # 使用默认止损

    def custom_sell(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> Optional[str]:
        """
        自定义卖出逻辑

        Args:
            pair: 交易对
            current_profit: 当前盈亏比例
            current_rate: 当前价格
            current_time: 当前时间

        Returns:
            卖出原因或 None（使用默认逻辑）
        """
        # 盈利超过10%且动量转负时主动止盈
        if current_profit > 0.10:
            return "profit_target_momentum_reversal"

        return None

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
        """
        确认订单进入

        Args:
            pair: 交易对
            order_type: 订单类型
            amount: 数量
            rate: 价格
            time_in_force: 有效期
            current_time: 当前时间

        Returns:
            True: 确认下单
            False: 取消订单
        """
        self.logger.info(
            f"[CONFIRM ENTRY] {pair} {order_type} {amount:.4f} @ {rate:.4f}"
        )
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
        """
        确认订单退出

        Args:
            pair: 交易对
            order_type: 订单类型
            amount: 数量
            rate: 价格
            time_in_force: 有效期
            current_time: 当前时间

        Returns:
            True: 确认平仓
            False: 取消平仓
        """
        self.logger.info(
            f"[CONFIRM EXIT] {pair} {order_type} {amount:.4f} @ {rate:.4f}"
        )
        return True

    def order_filled(
        self,
        pair: str,
        order: Any,
        current_time: datetime,
        **kwargs
    ) -> None:
        """
        订单成交后调用

        Args:
            pair: 交易对
            order: 订单成交对象
            current_time: 当前时间
        """
        self.logger.info(
            f"[ORDER FILLED] {pair} | "
            f"price={getattr(order, 'price', 'N/A')} | "
            f"qty={getattr(order, 'qty', 'N/A')}"
        )

    def botShutdown(self, **kwargs) -> None:
        """机器人关闭时调用"""
        self.logger.info(f"[BOT SHUTDOWN] {self.strategy_name}")

    # ============================================================================
    # 实用方法
    # ============================================================================

    def get_etf_pool_info(self) -> Dict[str, Dict[str, float]]:
        """
        获取ETF池的当前动量信息

        Returns:
            Dict包含每个ETF的动量数据
        """
        # 如果有存储的 vectorized_data，从中获取信息
        result = {}
        if hasattr(self, '_vectorized_data') and self._vectorized_data:
            for symbol, df in self._vectorized_data.items():
                if not df.empty and 'momentum_score' in df.columns:
                    last_score = df['momentum_score'].iloc[-1]
                    if 'momentum_r2' in df.columns:
                        r2_series = df['momentum_r2']
                        last_r2 = r2_series.iloc[-1] if len(r2_series) > 0 else 0.0
                    else:
                        last_r2 = 0.0
                    result[symbol] = {
                        'momentum_score': float(last_score) if not pd.isna(last_score) else 0.0,
                        'r2': float(last_r2) if not pd.isna(last_r2) else 0.0,
                    }
        return result

    @classmethod
    def get_strategy_config_schema(cls) -> Dict[str, Any]:
        """
        获取策略配置Schema

        Returns:
            配置Schema字典
        """
        return {
            "strategy_name": {
                "type": "string",
                "description": "策略名称",
                "default": cls.strategy_name
            },
            "strategy_id": {
                "type": "string",
                "description": "策略ID",
                "default": cls.strategy_id
            },
            "timeframe": {
                "type": "string",
                "description": "时间框架",
                "default": cls.timeframe
            },
            "etf_pool": {
                "type": "list",
                "description": "ETF标的池",
                "default": cls.etf_pool,
                "items": {"type": "string"}
            },
            "min_lookback_days": {
                "type": "integer",
                "description": "最小回溯天数",
                "default": cls.min_lookback_days,
                "range": [5, 252]
            },
            "max_lookback_days": {
                "type": "integer",
                "description": "最大回溯天数",
                "default": cls.max_lookback_days,
                "range": [10, 252]
            },
            "use_dynamic_lookback": {
                "type": "boolean",
                "description": "是否使用动态回溯期",
                "default": cls.use_dynamic_lookback
            },
            "r2_threshold": {
                "type": "number",
                "description": "R²最小阈值",
                "default": cls.r2_threshold,
                "range": [0.0, 1.0]
            },
            "stoploss": {
                "type": "number",
                "description": "止损比例",
                "default": cls.stoploss,
                "range": [-1.0, 0.0]
            },
            "target_positions": {
                "type": "integer",
                "description": "目标持仓数量",
                "default": cls.target_positions,
                "range": [1, 10]
            }
        }

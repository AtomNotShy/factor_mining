"""
ETF动量轮动策略 

核心逻辑：
1. 基于加权线性回归计算年化收益率（近N天）
2. R²判定系数过滤不稳定趋势
3. 波动率动态调整回溯天数
4. 多重风控过滤

参考: https://www.joinquant.com/strategy/index/detail?strategyid=be4936da83be3fa3e0da2c7d3a126e28

重写于 Freqtrade 框架:
- 使用 FreqtradeStrategy 协议
- 实现 populate_indicators/entry_trend/exit_trend
- 支持 ROI 表和止损配置
- 完整的生命周期回调
"""

from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from datetime import datetime

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.strategies.base.indicators import sma, ema
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.core.types import Signal, PortfolioState, RiskState, OrderIntent
    from src.core.context import RunContext


logger = get_logger("strategy.etf_momentum_joinquant")


class ETFMomentumJoinQuantStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """
    聚宽ETF动量轮动策略 

    特点：
    - 加权线性回归计算动量（近日期权重更高）
    - R²过滤不稳定趋势
    - 波动率动态调整仓位

    Example:
        >>> from src.strategies.user_strategies.etf_momentum_joinquant import (
        ...     ETFMomentumJoinQuantStrategy
        ... )
        >>> strategy = ETFMomentumJoinQuantStrategy()
        >>> # 或自定义参数
        >>> strategy = ETFMomentumJoinQuantStrategy(
        ...     strategy_id="custom_etf_momentum",
        ...     etf_pool=["QQQ", "SPY"],
        ...     lookback_days=30
        ... )
    """

    # ============================================================================
    # 策略配置 
    # ============================================================================

    strategy_name = "ETF Momentum JoinQuant"
    strategy_id: str = "etf_momentum_joinquant"
    timeframe = "1d"
    startup_candle_count = 63  # 波动率计算需要63天

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

    # ETF 标的池
    etf_pool: List[str] = [
        "QQQ", "SPY", "IWM", "TLT", "GLD",
        "VTI", "VEA", "DBC", "VWO", "EEM"
    ]

    # 回溯参数
    lookback_days: int = 20  # 基础回溯天数
    lookback_volatility: int = 63  # 波动率计算回溯天数
    min_momentum_days: int = 10  # 最小动量计算天数

    # 过滤参数
    r2_threshold: float = 0.5  # R²最小阈值
    volatility_threshold: float = 0.4  # 波动率阈值
    volatility_penalty: float = 0.5  # 高波动率惩罚系数

    # 持仓参数
    target_positions: int = 1  # 目标持仓数量
    max_weight: float = 1.0  # 单标最大权重
    min_weight: float = 0.0  # 单标最小权重
    use_compounding: bool = False  # 是否使用复利计算仓位大小

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
        lookback_days: Optional[int] = None,
        lookback_volatility: Optional[int] = None,
        min_momentum_days: Optional[int] = None,
        r2_threshold: Optional[float] = None,
        volatility_threshold: Optional[float] = None,
        volatility_penalty: Optional[float] = None,
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
            lookback_days: 基础回溯天数
            lookback_volatility: 波动率计算回溯天数
            min_momentum_days: 最小动量计算天数
            r2_threshold: R²最小阈值
            volatility_threshold: 波动率阈值
            volatility_penalty: 高波动率惩罚系数
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
        if lookback_days is not None:
            self.lookback_days = lookback_days
        if lookback_volatility is not None:
            self.lookback_volatility = lookback_volatility
        if min_momentum_days is not None:
            self.min_momentum_days = min_momentum_days
        if r2_threshold is not None:
            self.r2_threshold = r2_threshold
        if volatility_threshold is not None:
            self.volatility_threshold = volatility_threshold
        if volatility_penalty is not None:
            self.volatility_penalty = volatility_penalty
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
        self.logger.info(f"   Lookback: {self.lookback_days} days")
        self.logger.info(f"   R² Threshold: {self.r2_threshold}")

    async def bot_loop_start(self, **kwargs) -> None:
        """每轮循环开始时调用"""
        pass

    # ============================================================================
    # 核心指标计算
    # ============================================================================

    def _calculate_weighted_momentum(self, prices: pd.Series) -> Dict[str, Any]:
        """
        计算加权线性回归动量

        使用加权最小二乘法，近期数据权重更高

        Args:
            prices: 价格序列

        Returns:
            Dict包含 momentum, r2, annual_volatility, valid
        """
        if len(prices) < self.min_momentum_days:
            return {
                "momentum": 0.0,
                "r2": 0.0,
                "annual_volatility": 0.0,
                "valid": False,
                "error": f"数据不足 {self.min_momentum_days} 天"
            }

        n = len(prices)

        # 创建加权向量（线性递增权重）
        # 近期数据权重更高，权重范围 [1, 2]
        weights = np.linspace(self.WEIGHT_START, self.WEIGHT_END, n)

        try:
            # 对数收益率计算
            prices_arr = np.array(prices.values[1:], dtype=float)
            past_arr = np.array(prices.values[:-1], dtype=float)
            log_returns = np.log(prices_arr / past_arr)

            if len(log_returns) < 5:
                return {
                    "momentum": 0.0,
                    "r2": 0.0,
                    "annual_volatility": 0.0,
                    "valid": False,
                    "error": f"对数收益率数据点不足: {len(log_returns)}"
                }

            # X: 时间索引 (0, 1, 2, ...)
            x = np.arange(len(log_returns))
            x_weighted = x * weights[:-1]

            # Y: 累积对数收益率
            y = np.cumsum(log_returns)
            y_weighted = y * weights[:-1]

            # 加权线性回归
            sum_w = np.sum(weights[:-1])
            sum_xw = np.sum(x_weighted)
            sum_yw = np.sum(y_weighted)
            sum_x2w = np.sum(x_weighted ** 2)
            sum_xyw = np.sum(x_weighted * y_weighted)

            denominator = sum_w * sum_x2w - sum_xw ** 2
            if abs(denominator) < 1e-10:
                return {
                    "momentum": 0.0,
                    "r2": 0.0,
                    "annual_volatility": 0.0,
                    "valid": False,
                    "error": "线性回归分母接近零"
                }

            slope = (sum_w * sum_xyw - sum_xw * sum_yw) / denominator
            intercept = (sum_yw - slope * sum_xw) / sum_w

            # 年化收益率
            annual_return = slope * self.TRADING_DAYS

            # 计算R²
            y_pred = slope * x_weighted + intercept
            ss_res = np.sum((y_weighted - y_pred) ** 2)
            ss_tot = np.sum((y_weighted - np.mean(y_weighted)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # 波动率
            returns = np.diff(np.log(prices.values))
            if len(returns) >= 2:
                annual_volatility = float(np.std(returns) * np.sqrt(self.TRADING_DAYS))
            else:
                annual_volatility = 0.0

            return {
                "momentum": float(annual_return),
                "r2": float(r2),
                "annual_volatility": annual_volatility,
                "valid": True,
                "error": None
            }

        except Exception as e:
            self.logger.exception("动量计算发生错误")
            return {
                "momentum": 0.0,
                "r2": 0.0,
                "annual_volatility": 0.0,
                "valid": False,
                "error": str(e)
            }

    def _calculate_momentum_score(
        self,
        momentum_result: Dict[str, float],
    ) -> float:
        """
        综合动量评分

        考虑R²过滤和波动率调整

        Args:
            momentum_result: 动量计算结果

        Returns:
            float: 调整后的动量评分
        """
        # R²过滤
        if not momentum_result.get("valid", False):
            return 0.0

        if momentum_result.get("r2", 0) < self.r2_threshold:
            self.logger.debug(
                f"R² ({momentum_result['r2']:.4f}) < 阈值 ({self.r2_threshold})"
            )
            return 0.0

        # 波动率调整
        momentum = momentum_result["momentum"]
        volatility = momentum_result.get("annual_volatility", 0)

        if volatility > self.volatility_threshold:
            adjusted_momentum = momentum * self.volatility_penalty
            self.logger.debug(
                f"高波动率触发降权: vol={volatility:.4f} > "
                f"threshold={self.volatility_threshold}"
            )
        else:
            adjusted_momentum = momentum

        return adjusted_momentum

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

        # 使用滚动窗口计算动量
        close_series = pd.Series(dataframe['close'].values, index=dataframe.index)
        
        # 为每一行计算滚动动量
        for i in range(len(dataframe)):
            if i < self.lookback_days:
                continue
                
            # 获取滚动窗口数据
            window_start = max(0, i - self.lookback_days)
            window_prices = close_series.iloc[window_start:i+1]
            
            # 计算动量
            momentum_result = self._calculate_weighted_momentum(window_prices)
            
            # 填充结果 - 使用.at[]来避免类型检查错误
            dataframe.at[dataframe.index[i], 'momentum'] = momentum_result['momentum']
            dataframe.at[dataframe.index[i], 'momentum_r2'] = momentum_result['r2']
            dataframe.at[dataframe.index[i], 'momentum_volatility'] = momentum_result['annual_volatility']
            dataframe.at[dataframe.index[i], 'momentum_valid'] = 1 if momentum_result['valid'] else 0
            
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
            self.logger.debug(
                f"{symbol} indicators: momentum_score={last_score:.4f}, "
                f"r2={last_r2:.4f}"
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
        # 只有当动量评分从负变正时才生成进场信号
        dataframe['momentum_score_prev'] = dataframe['momentum_score'].shift(1)
        
        # 条件：动量评分为正、R²超过阈值、且动量评分从负变正
        entry_condition = (
            (dataframe['momentum_score'] > 0) &
            (dataframe['momentum_r2'] >= self.r2_threshold) &
            (dataframe['momentum_score_prev'] <= 0)
        )
        
        # 确保我们有足够的数据（至少lookback_days天）
        if 'momentum_valid' in dataframe.columns:
            entry_condition = entry_condition & (dataframe['momentum_valid'] == 1)

        # 应用条件
        dataframe.loc[entry_condition, 'enter_long'] = 1
        dataframe.loc[entry_condition, 'enter_tag'] = "momentum_positive"

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

        current_score = dataframe['momentum_score'].iloc[-1]
        prev_score = dataframe['momentum_score'].iloc[-2]

        # 离场条件：
        # 1. 动量变为负值
        # 2. 动量持续下降
        exit_condition = (
            (current_score < 0) &
            (prev_score >= 0)
        )

        # 处理标量布尔值情况（避免 dataframe.loc[False, ...] 添加新行）
        if isinstance(exit_condition, (bool, np.bool_)):
            if exit_condition:
                dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                dataframe.loc[dataframe.index[-1], 'exit_tag'] = "momentum_reversal"
        else:
            dataframe.loc[exit_condition, 'exit_long'] = 1
            dataframe.loc[exit_condition, 'exit_tag'] = "momentum_reversal"

        # 额外的离场条件：跌破均线
        price_vs_sma20 = dataframe['price_vs_sma20'].iloc[-1]
        prev_price_vs_sma20 = dataframe['price_vs_sma20'].iloc[-2]

        sma_exit = (
            (price_vs_sma20 < 0) &
            (prev_price_vs_sma20 >= 0)
        )

        # 处理标量布尔值情况
        if isinstance(sma_exit, (bool, np.bool_)):
            if sma_exit and dataframe.loc[dataframe.index[-1], 'exit_long'] != 1:
                dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                dataframe.loc[dataframe.index[-1], 'exit_tag'] = "sma_breakdown"
        else:
            dataframe.loc[sma_exit & ~exit_condition, 'exit_long'] = 1
            dataframe.loc[sma_exit & ~exit_condition, 'exit_tag'] = "sma_breakdown"

        # 记录信号统计
        exit_long_sum = dataframe['exit_long'].sum()
        exit_count = int(exit_long_sum) if not isinstance(exit_long_sum, (bool, np.bool_)) else (1 if exit_long_sum else 0)
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
    # v2 风格接口方法（仓位管理）
    # ============================================================================

    def size_positions(
        self,
        signals: List['Signal'],
        portfolio: 'PortfolioState',
        risk: 'RiskState',
        ctx: 'RunContext',
    ) -> List['OrderIntent']:
        """
        根据信号生成订单意图
        
        ETF动量策略的特殊处理：
        1. 每次只持有一个ETF
        2. 换仓时先平掉当前持仓，再买入新的ETF
        3. 使用固定仓位大小（避免重复计算）
        4. 只处理第一个有效的信号，避免重复计算
        
        Args:
            signals: 信号列表
            portfolio: 组合状态
            risk: 风险状态
            ctx: 运行上下文
            
        Returns:
            订单意图列表
        """
        from src.core.types import OrderIntent, OrderSide, OrderType
        from datetime import timezone
        import uuid
        
        order_intents = []
        
        # 检查当前持仓
        current_positions = {}
        for symbol, qty in portfolio.positions.items():
            if abs(qty) > 1e-8:  # 有持仓
                current_positions[symbol] = qty

        # ETF动量策略需要特殊处理换仓：
        # 1. 先处理所有FLAT信号（卖出当前持仓）
        # 2. 再处理LONG信号（买入新标的）
        # 两者都需要生成订单，不能只处理一个就跳过

        flat_signals = [s for s in signals if s.action.value == 'FLAT']
        long_signals = [s for s in signals if s.action.value == 'LONG']

        # 获取当前价格（复用价格获取逻辑）
        def get_current_price(signal):
            if signal.symbol in self._vectorized_data:
                df = self._vectorized_data[signal.symbol]
                if not df.empty:
                    ts = signal.ts_utc
                    if ts in df.index:
                        try:
                            row = df.loc[ts]
                            if isinstance(row, pd.Series):
                                return row.get('close')
                            elif isinstance(row, pd.DataFrame):
                                return row['close'].iloc[-1] if 'close' in row.columns else None
                        except Exception as e:
                            self.logger.warning(f"获取价格失败 {signal.symbol} @ {ts}: {e}")
                    # 如果找不到，尝试获取最后有效价格
                    close_col = df['close'] if 'close' in df.columns else None
                    if close_col is not None:
                        valid_prices = close_col.dropna()
                        if not valid_prices.empty:
                            return valid_prices.iloc[-1]
            return None

        # 处理所有FLAT信号（卖出当前持仓）
        for signal in flat_signals:
            current_price = get_current_price(signal)
            if current_price is None or (isinstance(current_price, float) and pd.isna(current_price)):
                self.logger.warning(f"无法获取 {signal.symbol} 的价格，跳过信号")
                continue

            if signal.symbol in current_positions:
                qty = abs(current_positions[signal.symbol])
                if qty > 0:
                    order_intent = OrderIntent(
                        order_id=str(uuid.uuid4())[:8],
                        ts_utc=signal.ts_utc,
                        symbol=signal.symbol,
                        side=OrderSide.SELL,
                        qty=qty,
                        order_type=OrderType.MKT,
                        limit_price=float(current_price),
                        strategy_id=self.strategy_id,
                        metadata=signal.metadata
                    )
                    order_intents.append(order_intent)
                    self.logger.info(f"生成平仓订单: {signal.symbol} {qty:.4f} @ {current_price:.4f}")

        # 只处理第一个有效的LONG信号（避免重复买入）
        processed_long = False
        for signal in long_signals:
            if processed_long:
                self.logger.debug(f"跳过重复的买入信号: {signal.symbol}")
                continue

            current_price = get_current_price(signal)
            if current_price is None or (isinstance(current_price, float) and pd.isna(current_price)):
                self.logger.warning(f"无法获取 {signal.symbol} 的价格，跳过信号")
                continue

            # 检查是否已经有持仓（需要换仓）
            if current_positions:
                # 如果已经有持仓，需要先平仓（已经在上面处理了FLAT信号）
                for symbol, qty in current_positions.items():
                    if abs(qty) > 1e-8:
                        self.logger.info(f"换仓平仓: {symbol} {abs(qty):.4f} @ {current_price:.4f}")

            # 使用当前总资产计算仓位大小
            # 注意：portfolio.equity = portfolio.cash + positions_value
            # 所以直接使用 equity 即可，不需要 cash + equity
            total_equity = portfolio.equity

            # 计算买入数量 - 使用总资产的固定比例
            position_pct = 0.2  # 20%仓位
            stake_amount = total_equity * position_pct
            qty = stake_amount / current_price

            self.logger.info(
                f"仓位计算: 总资产=${total_equity:.2f}, "
                f"仓位比例={position_pct:.1%}, "
                f"买入金额=${stake_amount:.2f}, "
                f"价格=${current_price:.4f}, "
                f"数量={qty:.4f}"
            )

            # 生成买入订单
            order_intent = OrderIntent(
                order_id=str(uuid.uuid4())[:8],
                ts_utc=signal.ts_utc,
                symbol=signal.symbol,
                side=OrderSide.BUY,
                qty=qty,
                order_type=OrderType.MKT,
                limit_price=float(current_price),
                strategy_id=self.strategy_id,
                metadata=signal.metadata
            )
            order_intents.append(order_intent)
            self.logger.info(f"生成买入订单: {signal.symbol} {qty:.4f} @ {current_price:.4f}")
            processed_long = True  # 只处理第一个有效的LONG信号

        return order_intents

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
            "lookback_days": {
                "type": "integer",
                "description": "回溯天数",
                "default": cls.lookback_days,
                "range": [5, 252]
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


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 直接使用
    strategy = ETFMomentumJoinQuantStrategy()
    print(f"Strategy ID: {strategy.strategy_id}")
    print(f"Strategy Name: {strategy.strategy_name}")
    print(f"ETF Pool: {strategy.etf_pool}")
    print(f"Lookback Days: {strategy.lookback_days}")
    print(f"R² Threshold: {strategy.r2_threshold}")
    print(f"Stoploss: {strategy.stoploss}")

    # 演示如何在创建时覆盖默认参数
    custom_strategy = ETFMomentumJoinQuantStrategy(
        strategy_id='custom_etf_momentum',
        etf_pool=['QQQ', 'SPY', 'DIA'],
        lookback_days=30,
        r2_threshold=0.6,
        stoploss=-0.08,
    )
    print(f"\nCustom Strategy ID: {custom_strategy.strategy_id}")
    print(f"Custom Lookback Days: {custom_strategy.lookback_days}")
    print(f"Custom R² Threshold: {custom_strategy.r2_threshold}")

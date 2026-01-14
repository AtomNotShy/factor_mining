"""
Freqtrade 风格的策略接口（IStrategy Protocol）

完整实现 Freqtrade 的生命周期回调：
1. bot_start() - 机器人启动时调用
2. bot_loop_start() - 每轮开始时调用
3. populate_indicators() - 批量计算指标
4. populate_entry_trend() - 生成进场信号
5. populate_exit_trend() - 生成离场信号
6. custom_stoploss() - 自定义止损
7. custom_sell() - 自定义卖出
8. custom_buy() - 自定义买入
9. confirm_trade_entry() - 确认订单进入
10. confirm_trade_exit() - 确认订单退出
11. adjust_trade_position() - 调整仓位
12. order_filled() - 订单成交后
13. botShutdown() - 关闭时调用
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd

from src.core.types import Fill, OrderSide, OrderType
from src.utils.logger import get_logger


logger = get_logger("strategy.interface")


class FreqtradeStrategy(ABC):
    """
    Freqtrade 风格策略基类
    
    完整实现 Freqtrade 的策略生命周期回调
    """
    
    # 策略配置（子类应覆盖）
    strategy_name: str = ""
    strategy_id: str = ""
    timeframe: str = "1d"
    startup_candle_count: int = 30
    
    # ROI 配置（分钟: 目标收益率）
    minimal_roi: Dict[int, float] = {
        0: float('inf'),
    }
    
    # 止损配置
    stoploss: float = -0.10
    trailing_stop: bool = False
    trailing_stop_positive: float = 0.0
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False
    
    # 订单类型
    order_types: Dict[str, str] = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    
    # 仓位配置
    position_adjustment_enable: bool = False
    use_exit_signal: bool = True
    exit_profit_only: bool = False
    ignore_roi_if_entry_signal: bool = False
    max_position_size: float = 1.0  # 默认允许全仓 (100%)
    
    # 保护机制
    protections: List[Dict] = []
    
    def __init__(self):
        self.logger = get_logger(f"strategy.{self.strategy_id or self.__class__.__name__.lower()}")
        self._vectorized_data: Dict[str, pd.DataFrame] = {}  # 存储处理后的数据
        self._processed_symbols: set = set()  # 记录已处理的标的
    
    async def bot_start(self, **kwargs) -> None:
        """
        机器人启动时调用一次
        
        Args:
            **kwargs: 上下文参数
        """
        self.logger.info(f"Bot started: {self.strategy_name}")
        pass
    
    async def bot_loop_start(self, **kwargs) -> None:
        """
        每轮循环开始时调用
        
        Args:
            **kwargs: 上下文参数
        """
        pass
    
    @abstractmethod
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            dataframe: K线数据
            metadata: 元数据（symbol, timeframe 等）
            
        Returns:
            添加了指标的 DataFrame
        """
        pass
    
    @abstractmethod
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        生成进场信号
        
        在 dataframe 中添加 'enter_long', 'enter_short' 列
        enter_long=1 表示买入信号
        
        Args:
            dataframe: K线数据（已包含指标）
            metadata: 元数据
            
        Returns:
            添加了进场信号的 DataFrame
        """
        pass
    
    @abstractmethod
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        生成离场信号
        
        在 dataframe 中添加 'exit_long', 'exit_short' 列
        exit_long=1 表示卖出信号
        
        Args:
            dataframe: K线数据（已包含指标）
            metadata: 元数据
            
        Returns:
            添加了离场信号的 DataFrame
        """
        pass
    
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
            **kwargs: 额外参数
            
        Returns:
            止损价格或比例
        """
        return self.stoploss
    
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
            **kwargs: 额外参数
            
        Returns:
            卖出原因或 None（使用默认逻辑）
        """
        return None
    
    def custom_buy(
        self,
        pair: str,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> Optional[str]:
        """
        自定义买入逻辑
        
        Args:
            pair: 交易对
            current_rate: 当前价格
            current_time: 当前时间
            **kwargs: 额外参数
            
        Returns:
            买入原因或 None（使用默认逻辑）
        """
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
            **kwargs: 额外参数
            
        Returns:
            True: 确认下单
            False: 取消订单
        """
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
            **kwargs: 额外参数
            
        Returns:
            True: 确认平仓
            False: 取消平仓
        """
        return True
    
    def adjust_trade_position(
        self,
        trade: Dict,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        current_stake_amount: float,
        **kwargs
    ) -> Optional[float]:
        """
        调整仓位（加仓/减仓）
        
        Args:
            trade: 交易信息字典
            current_time: 当前时间
            current_rate: 当前价格
            current_profit: 当前盈亏
            min_stake: 最小仓位
            max_stake: 最大仓位
            current_stake_amount: 当前仓位金额
            **kwargs: 额外参数
            
        Returns:
            调整金额（正数加仓，负数减仓，None 不调整）
        """
        return None
    
    def order_filled(
        self,
        pair: str,
        order: Fill,
        current_time: datetime,
        **kwargs
    ) -> None:
        """
        订单成交后调用
        
        Args:
            pair: 交易对
            order: 订单成交对象
            current_time: 当前时间
            **kwargs: 额外参数
        """
        self.logger.info(
            f"Order filled: {pair} {order.side.value} {order.qty} @ {order.price}"
        )
    
    def botShutdown(self, **kwargs) -> None:
        """
        机器人关闭时调用
        
        Args:
            **kwargs: 上下文参数
        """
        self.logger.info(f"Bot shutdown: {self.strategy_name}")
    
    def version(self) -> int:
        """
        策略版本号
        
        Returns:
            版本号
        """
        return 1
    
    def enable_lookahead_check(self) -> bool:
        """
        是否开启未来数据检查（默认关闭）
        
        Returns:
            是否启用
        """
        return False
    
    def fee(
        self,
        pair: str,
        order_type: str,
        is_maker: bool,
        amount: float,
        price: float,
    ) -> float:
        """
        计算手续费
        
        Args:
            pair: 交易对
            order_type: 订单类型
            is_maker: 是否是 maker 单
            amount: 数量
            price: 价格
            
        Returns:
            手续费金额
        """
        return 0.001  # 默认费率

    # ===== v2 风格接口方法（用于与 engine 兼容）=====

    def generate_signals(
        self,
        md: 'MarketData',
        ctx: 'RunContext',
    ) -> List['Signal']:
        """
        从处理后的 DataFrame 生成信号（v2 风格接口）
        
        从 populate_entry_trend/populate_exit_trend 处理后的数据中提取信号。
        引擎会在调用此方法前先调用 populate_indicators 等方法。
        
        Args:
            md: 市场数据（bars 包含处理后的数据）
            ctx: 运行上下文
            
        Returns:
            信号列表
        """
        from src.core.types import Signal, ActionType
        from datetime import timezone
        import pandas as pd
        
        signals = []
        
        # 尝试从 md.bars 获取信号（当前 tick 的数据）
        if hasattr(md, 'bars') and not md.bars.empty:
            bars = md.bars
            # 检查是否有信号列
            if 'enter_long' in bars.columns or 'exit_long' in bars.columns:
                # 获取最后一行（当前 tick）
                if isinstance(bars, pd.DataFrame) and not bars.empty:
                    row = bars.iloc[-1]
                    current_ts = row.name
                    
                    # 检查进场信号
                    enter_long = row.get('enter_long', 0)
                    if hasattr(enter_long, 'item'):
                        enter_long = enter_long.item()
                    if isinstance(enter_long, (int, float)) and enter_long == 1:
                        signal = Signal(
                            ts_utc=current_ts.to_pydatetime() if hasattr(current_ts, 'to_pydatetime') else datetime.now(timezone.utc),
                            symbol=row.get('symbol', ''),
                            strategy_id=self.strategy_id,
                            action=ActionType.LONG,
                            strength=1.0,
                            metadata={
                                'enter_tag': row.get('enter_tag', ''),
                                'current_price': row.get('close')
                            }
                        )
                        signals.append(signal)
                    
                    # 检查离场信号
                    exit_long = row.get('exit_long', 0)
                    if hasattr(exit_long, 'item'):
                        exit_long = exit_long.item()
                    if isinstance(exit_long, (int, float)) and exit_long == 1:
                        signal = Signal(
                            ts_utc=current_ts.to_pydatetime() if hasattr(current_ts, 'to_pydatetime') else datetime.now(timezone.utc),
                            symbol=row.get('symbol', ''),
                            strategy_id=self.strategy_id,
                            action=ActionType.FLAT,
                            strength=1.0,
                            metadata={
                                'exit_tag': row.get('exit_tag', ''),
                                'current_price': row.get('close')
                            }
                        )
                        signals.append(signal)
        
        # 如果 md.bars 没有信号列，尝试从 _vectorized_data 获取
        if not signals and hasattr(self, '_vectorized_data'):
            for symbol, dataframe in self._vectorized_data.items():
                if dataframe.empty:
                    continue

                # 尝试找到当前时间对应的行
                # 优先使用 ctx.dt（回测时由引擎设置），而不是 md.bars.index[-1]
                current_ts = None

                # 首先尝试从 ctx 获取（优先，因为是当前处理的 tick）
                if hasattr(ctx, 'current_ts') and ctx.current_ts is not None:
                    current_ts = ctx.current_ts
                elif hasattr(ctx, 'dt') and ctx.dt is not None:
                    current_ts = ctx.dt
                # 其次尝试从 md.bars 获取当前时间
                elif hasattr(md, 'bars') and not md.bars.empty:
                    bars = md.bars
                    if hasattr(bars.index, '__getitem__') and len(bars) > 0:
                        last_idx = bars.index[-1]
                        current_ts = last_idx
                else:
                    continue

                # 处理时区：确保与 dataframe.index 类型一致
                df_idx = dataframe.index
                ts_to_use = current_ts
                
                # 如果 dataframe.index 是 tz-naive，确保 current_ts 也是 naive
                if isinstance(df_idx, pd.DatetimeIndex):
                    if df_idx.tz is None and current_ts.tzinfo is not None:
                        ts_to_use = current_ts.tz_localize(None)
                    elif df_idx.tz is not None and current_ts.tzinfo is None:
                        ts_to_use = current_ts.tz_localize('UTC')

                if ts_to_use in dataframe.index:
                    row = dataframe.loc[ts_to_use]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]

                    # 检查进场信号
                    enter_long = row.get('enter_long', 0)
                    if hasattr(enter_long, 'item'):
                        enter_long = enter_long.item()
                    if isinstance(enter_long, (int, float)) and enter_long == 1:
                        signal = Signal(
                            ts_utc=current_ts.to_pydatetime() if hasattr(current_ts, 'to_pydatetime') else datetime.now(timezone.utc),
                            symbol=symbol,
                            strategy_id=self.strategy_id,
                            action=ActionType.LONG,
                            strength=1.0,
                            metadata={
                                'enter_tag': row.get('enter_tag', ''),
                                'current_price': row.get('close')
                            }
                        )
                        signals.append(signal)
                    
                    # 检查离场信号
                    exit_long = row.get('exit_long', 0)
                    if hasattr(exit_long, 'item'):
                        exit_long = exit_long.item()
                    if isinstance(exit_long, (int, float)) and exit_long == 1:
                        signal = Signal(
                            ts_utc=current_ts.to_pydatetime() if hasattr(current_ts, 'to_pydatetime') else datetime.now(timezone.utc),
                            symbol=symbol,
                            strategy_id=self.strategy_id,
                            action=ActionType.FLAT,
                            strength=1.0,
                            metadata={
                                'exit_tag': row.get('exit_tag', ''),
                                'current_price': row.get('close')
                            }
                        )
                        signals.append(signal)
        
        return signals

    def size_positions(
        self,
        signals: List['Signal'],
        portfolio: 'PortfolioState',
        risk: 'RiskState',
        ctx: 'RunContext',
    ) -> List['OrderIntent']:
        """
        根据信号生成订单意图（v2 风格接口）
        
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

        for signal in signals:
            # 默认使用可用资金的 100%
            # 使用 equity 作为总资产，cash 可能为 0
            available_capital = portfolio.equity if portfolio.equity > 0 else portfolio.cash

            if available_capital <= 0:
                continue

            # 计算仓位大小（简单实现：全仓买入）
            stake_amount = available_capital
            # 获取当前价格，优先从 metadata 获取，否则从信号中推断
            current_price = signal.metadata.get('current_price')
            if current_price is None or (isinstance(current_price, float) and pd.isna(current_price)):
                # 如果 metadata 中没有价格，尝试从 bars 中获取
                if signal.symbol in self._vectorized_data:
                    df = self._vectorized_data[signal.symbol]
                    if not df.empty:
                        # 尝试获取信号时间点的价格
                        ts = signal.ts_utc
                        if ts in df.index:
                            current_price = df.loc[ts, 'close'] if 'close' in df.columns else None
                        # 如果找不到，尝试获取最后有效价格
                        if current_price is None or (isinstance(current_price, float) and pd.isna(current_price)):
                            # 获取最后一个有效的 close 值
                            close_col = df['close'] if 'close' in df.columns else None
                            if close_col is not None:
                                # 找到最后一个非 nan 值
                                valid_prices = close_col.dropna()
                                if not valid_prices.empty:
                                    current_price = valid_prices.iloc[-1]

            if current_price is None or (isinstance(current_price, float) and pd.isna(current_price)):
                continue

            qty = stake_amount / current_price

            # 创建订单意图
            if signal.action.value in ['LONG', 'BUY']:
                side = OrderSide.BUY
            elif signal.action.value in ['SHORT', 'SELL']:
                side = OrderSide.SELL
            else:  # FLAT
                # 平仓：卖出全部持仓
                if signal.symbol in portfolio.positions and portfolio.positions[signal.symbol] > 0:
                    qty = abs(portfolio.positions[signal.symbol])
                    side = OrderSide.SELL
                else:
                    continue

            order_intent = OrderIntent(
                order_id=str(uuid.uuid4())[:8],
                ts_utc=signal.ts_utc,
                symbol=signal.symbol,
                side=side,
                qty=qty,
                order_type=OrderType.MKT,
                limit_price=float(current_price),
                strategy_id=self.strategy_id,
                metadata=signal.metadata
            )
            order_intents.append(order_intent)

        return order_intents

    def risk_checks(
        self,
        order_intents: List['OrderIntent'],
        portfolio: 'PortfolioState',
        risk: 'RiskState',
        ctx: 'RunContext',
    ) -> tuple[List['OrderIntent'], List[Dict[str, Any]]]:
        """
        风险检查（v2 风格接口）
        
        Args:
            order_intents: 订单意图列表
            portfolio: 组合状态
            risk: 风险状态
            ctx: 运行上下文
            
        Returns:
            (approved_intents, blocks) 元组
        """
        approved = []
        blocks = []

        for intent in order_intents:
            # 基本风险检查
            if intent.qty <= 0:
                blocks.append({
                    'order_id': intent.order_id,
                    'reason': 'qty_zero_or_negative',
                    'symbol': intent.symbol
                })
                continue

            # 检查仓位大小限制（仅对买入订单）
            max_position = getattr(self, 'max_position_size', 1.0)  # 默认 100%
            if intent.side == OrderSide.BUY and portfolio.equity > 0:
                position_value = abs(intent.qty * (intent.limit_price or 0))
                position_pct = position_value / portfolio.equity
                # 使用 >= 而不是 >，允许等于最大仓位的情况
                if position_pct >= max_position and max_position < 1.0:
                    blocks.append({
                        'order_id': intent.order_id,
                        'reason': 'position_too_large',
                        'symbol': intent.symbol,
                        'position_pct': position_pct,
                        'max_pct': max_position
                    })
                    continue

            approved.append(intent)

        return approved, blocks


class StrategyLifecycleMixin:
    """
    策略生命周期混入类
    
    为现有 Strategy 类添加 Freqtrade 风格的回调支持
    """
    
    async def bot_start(self, **kwargs) -> None:
        """机器人启动时调用"""
        pass
    
    async def bot_loop_start(self, **kwargs) -> None:
        """每轮开始时调用"""
        pass
    
    def custom_stoploss(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> float:
        """自定义止损"""
        pass
    
    def custom_sell(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> Optional[str]:
        """自定义卖出"""
        pass
    
    def custom_buy(
        self,
        pair: str,
        current_rate: float,
        current_time: datetime,
        **kwargs
    ) -> Optional[str]:
        """自定义买入"""
        pass
    
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
        pass
    
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
        pass
    
    def adjust_trade_position(
        self,
        trade: Dict,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        current_stake_amount: float,
        **kwargs
    ) -> Optional[float]:
        """调整仓位"""
        pass
    
    def order_filled(
        self,
        pair: str,
        order: Fill,
        current_time: datetime,
        **kwargs
    ) -> None:
        """订单成交后"""
        pass
    
    def botShutdown(self, **kwargs) -> None:
        """机器人关闭"""
        pass

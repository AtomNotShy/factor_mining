"""
美股 ETF 动量轮动策略
改编自聚宽策略，适配美股市场

核心特性：
1. 动态动量周期：基于 ATR 波动率自动调整回溯天数（20-60天）
2. 加权线性回归：计算年化收益率和 R² 拟合度
3. 风控机制：过滤连续下跌标的
4. 美股 ETF 池：涵盖科技、商品、债券等多元化资产

性能优化：
- 使用 TA-Lib 进行高性能技术指标计算
- C 语言实现，比纯 Python 快 10-100 倍
"""

import math
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import talib

from src.strategies.base.strategy import Strategy, StrategyConfig
from src.core.types import Signal, OrderIntent, PortfolioState, RiskState, MarketData, ActionType, OrderSide, OrderType
from src.core.context import RunContext
from src.utils.logger import get_logger


class USETFMomentumStrategy(Strategy):
    """美股 ETF 动量轮动策略"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                strategy_id="us_etf_momentum",
                timeframe="1d",
                params={
                    # ETF 池配置
                    "etf_pool": [
                        # 科技板块
                        "QQQ",   # 纳斯达克100
                        "XLK",   # 科技精选
                        "SOXX",  # 半导体
                        # 大盘指数
                        "SPY",   # 标普500
                        "DIA",   # 道琼斯
                        "IWM",   # 罗素2000
                        # 商品
                        "GLD",   # 黄金
                        "SLV",   # 白银
                        "USO",   # 原油
                        # 债券
                        "TLT",   # 20年期国债
                        "IEF",   # 7-10年期国债
                        # 新兴市场
                        "EEM",   # 新兴市场
                        "FXI",   # 中国大盘
                    ],
                    # 动量参数
                    "min_lookback_days": 20,
                    "max_lookback_days": 60,
                    "default_lookback_days": 25,
                    "auto_adjust_lookback": True,  # 是否自动调整回溯期
                    # 基准配置
                    "benchmark_symbol": "SPY",
                    # 持仓参数
                    "target_positions": 1,  # 持有标的数量
                    "rebalance_frequency": "weekly",  # daily/weekly
                    "exit_rank": 0,  # 跌出前N名触发风控卖出（0=跟随target_positions）
                    # 风控参数
                    "max_single_day_drop": 0.05,  # 单日最大跌幅阈值
                    "max_consecutive_drop": 0.05,  # 连续跌幅阈值
                    "min_score": 0.0,  # 最小得分
                    "max_score": 6.0,  # 最大得分
                }
            )
        
        super().__init__(config)
        self.logger = get_logger("us_etf_momentum")
        self._last_rank: Dict[str, int] = {}
        self._last_scores: Dict[str, float] = {}

        if int(self.config.params.get("exit_rank", 0) or 0) <= 0:
            self.config.params["exit_rank"] = int(self.config.params.get("target_positions", 1))

    def _is_rebalance_day(self, ctx: RunContext) -> bool:
        frequency = str(self.config.params.get("rebalance_frequency", "weekly")).lower()
        if frequency == "daily":
            return True
        if frequency != "weekly":
            return True

        current_date = ctx.now_utc.date()
        week_start = current_date - timedelta(days=current_date.weekday())
        try:
            trading_days = ctx.trading_calendar.trading_days_between(week_start, current_date)
            if trading_days:
                return current_date == trading_days[0]
        except Exception as e:
            self.logger.debug(f"周度再平衡日期判断失败，回退到周一: {e}")
        return current_date.weekday() == 0
    
    def _calculate_dynamic_lookback(self, df: pd.DataFrame) -> int:
        """基于 ATR 动态计算回溯天数"""
        min_days = self.config.params["min_lookback_days"]
        max_days = self.config.params["max_lookback_days"]
        
        try:
            # 使用 TA-Lib 计算 ATR
            long_atr = talib.ATR(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), timeperiod=max_days)
            short_atr = talib.ATR(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), timeperiod=min_days)
            
            long_atr_val = float(long_atr[-1])
            short_atr_val = float(short_atr[-1])
            
            if long_atr_val > 0:
                ratio = min(0.9, short_atr_val / long_atr_val)
                lookback = int(min_days + (max_days - min_days) * (1 - ratio))
            else:
                lookback = self.config.params["default_lookback_days"]
                
        except Exception as e:
            self.logger.warning(f"动态回溯计算失败: {e}，使用默认值")
            lookback = self.config.params["default_lookback_days"]
        
        return lookback
    
    def _calculate_momentum_score(self, symbol: str, df: pd.DataFrame, current_price: float) -> Dict:
        """计算单个标的的动量得分"""
        # 动态调整回溯期
        if self.config.params["auto_adjust_lookback"]:
            lookback = self._calculate_dynamic_lookback(df)
        else:
            lookback = self.config.params["default_lookback_days"]
        
        # 构建价格序列
        prices = np.append(df["close"].to_numpy()[-lookback:], current_price)
        
        # 加权线性回归
        y = np.log(prices)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        
        # 计算年化收益率
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_return = math.exp(slope * 252) - 1
        
        # 计算 R²
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # 计算得分
        score = annualized_return * r2
        
        # 风控检查
        # 1. 单日跌幅超过阈值
        max_drop = self.config.params["max_single_day_drop"]
        con1 = min(
            prices[-1] / prices[-2],
            prices[-2] / prices[-3],
            prices[-3] / prices[-4]
        ) < (1 - max_drop)
        
        # 2. 连续三天下跌且总跌幅超过阈值
        con2 = (
            (prices[-1] < prices[-2]) and
            (prices[-2] < prices[-3]) and
            (prices[-3] < prices[-4]) and
            (prices[-1] / prices[-4] < (1 - max_drop))
        )
        
        # 3. 前三天连续下跌
        con3 = (
            (prices[-2] < prices[-3]) and
            (prices[-3] < prices[-4]) and
            (prices[-4] < prices[-5]) and
            (prices[-2] / prices[-5] < (1 - max_drop))
        )
        
        if con1 or con2 or con3:
            score = 0
        
        return {
            "symbol": symbol,
            "score": score,
            "annualized_return": annualized_return,
            "r2": r2,
            "lookback": lookback,
            "current_price": current_price,
        }
    
    def generate_signals(self, md: MarketData, ctx: RunContext) -> List[Signal]:
        """生成交易信号"""
        self.logger.debug(f"==> 执行策略: {self.strategy_id} [{ctx.now_utc}]")
        signals = []
        
        try:
            # Get available symbols from data, fall back to etf_pool config
            available_symbols = set(md.bars_all['symbol'].unique()) if md.bars_all is not None and 'symbol' in md.bars_all.columns else set()
            if available_symbols:
                # Only consider symbols that exist in the loaded data
                etf_pool = [s for s in self.config.params["etf_pool"] if s in available_symbols]
                self.logger.debug(f"使用已加载的数据符号: {etf_pool}")
            else:
                etf_pool = self.config.params["etf_pool"]
            
            scores = []
            
            # 计算每个 ETF 的得分
            for symbol in etf_pool:
                try:
                    if md.bars_all is None:
                        self.logger.warning(f"MarketData.bars_all 未设置，无法计算 {symbol}")
                        continue

                    symbol_bars = md.bars_all[md.bars_all['symbol'] == symbol] if 'symbol' in md.bars_all.columns else md.bars_all

                    if len(symbol_bars) < self.config.params["max_lookback_days"] + 10:
                        self.logger.warning(f"{symbol} 历史数据不足，跳过")
                        continue

                    current_price = float(symbol_bars.iloc[-1]['close'])

                    result = self._calculate_momentum_score(symbol, symbol_bars, current_price)
                    scores.append(result)
                    
                except Exception as e:
                    self.logger.error(f"计算 {symbol} 得分失败: {e}")
                    continue
            
            if scores:
                ranked_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
                self._last_rank = {s["symbol"]: i + 1 for i, s in enumerate(ranked_scores)}
                self._last_scores = {s["symbol"]: float(s["score"]) for s in ranked_scores}
            else:
                self._last_rank = {}
                self._last_scores = {}

            # 过滤并排序
            min_score = self.config.params["min_score"]
            max_score = self.config.params["max_score"]
            
            valid_scores = [s for s in scores if min_score < s["score"] < max_score]
            valid_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # 选择前 N 个
            target_num = self.config.params["target_positions"]
            top_etfs = valid_scores[:target_num]
            
            # 生成信号
            current_time = ctx.now_utc
            
            for etf in top_etfs:
                current_price = float(etf.get("current_price", 0) or 0)
                signal = Signal(
                    ts_utc=current_time,
                    symbol=etf["symbol"],
                    strategy_id=self.strategy_id,
                    action=ActionType.LONG,
                    strength=float(etf["score"]),
                    stop_price=None,
                    take_profit=None,
                    ttl_bars=None,
                    metadata={
                        "current_price": current_price,
                        "annualized_return": etf["annualized_return"],
                        "r2": etf["r2"],
                        "lookback": etf["lookback"],
                    }
                )
                signals.append(signal)
            
            if signals:
                self.logger.info(f"生成 {len(signals)} 个信号: {[s.symbol for s in signals]}")
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {e}")
            self._last_rank = {}
            self._last_scores = {}
        
        return signals
    
    def size_positions(
        self,
        signals: List[Signal],
        portfolio: PortfolioState,
        risk: RiskState,
        ctx: RunContext
    ) -> List[OrderIntent]:
        """计算目标仓位（轮动：卖出非目标标的，买入目标标的）"""
        orders = []

        exit_rank = int(self.config.params.get("exit_rank", 0) or 0)
        if exit_rank <= 0:
            exit_rank = int(self.config.params.get("target_positions", 1))
        target_positions = int(self.config.params.get("target_positions", 1))
        rank_map = self._last_rank or {}
        score_map = self._last_scores or {}

        # 日内风险退出：动量分数 < 0 或排名跌出前N
        exit_symbols = set()
        exit_reasons = {}
        for symbol, qty in portfolio.positions.items():
            if abs(qty) < 1e-8:
                continue
            score = score_map.get(symbol)
            rank = rank_map.get(symbol)
            if score is None or rank is None:
                continue
            if score < 0:
                exit_symbols.add(symbol)
                exit_reasons[symbol] = "risk_score"
            elif rank > exit_rank:
                exit_symbols.add(symbol)
                exit_reasons[symbol] = "risk_rank"

        for symbol in sorted(exit_symbols):
            qty = abs(portfolio.positions.get(symbol, 0.0))
            if qty <= 0:
                continue
            orders.append(OrderIntent(
                ts_utc=ctx.now_utc,
                symbol=symbol,
                side=OrderSide.SELL,
                qty=float(qty),
                order_type=OrderType.MKT,
                limit_price=None,
                stop_price=None,
                strategy_id=self.strategy_id,
                metadata={"exit_reason": exit_reasons.get(symbol, "risk_exit")},
            ))

        remaining_symbols = {
            symbol
            for symbol, qty in portfolio.positions.items()
            if abs(qty) >= 1e-8 and symbol not in exit_symbols
        }

        if not self._is_rebalance_day(ctx):
            if signals:
                desired = min(len(signals), target_positions)
                if desired > 0:
                    target_weight = 1.0 / desired
                    for signal in signals:
                        if len(remaining_symbols) >= desired:
                            break
                        if signal.symbol in remaining_symbols or signal.symbol in exit_symbols:
                            continue
                        price = float(signal.metadata.get("current_price", 0) or 0)
                        if price <= 0:
                            continue
                        qty = (portfolio.equity * target_weight) / price
                        orders.append(OrderIntent(
                            ts_utc=ctx.now_utc,
                            symbol=signal.symbol,
                            side=OrderSide.BUY,
                            qty=float(qty),
                            order_type=OrderType.MKT,
                            limit_price=None,
                            stop_price=None,
                            strategy_id=self.strategy_id,
                            metadata={
                                **(signal.metadata or {}),
                                "entry_tag": "risk_replace",
                            },
                        ))
                        remaining_symbols.add(signal.symbol)
            return orders

        if not signals:
            # 无信号时清空持仓
            for symbol, qty in portfolio.positions.items():
                if abs(qty) < 1e-8:
                    continue
                if symbol in exit_symbols:
                    continue
                orders.append(OrderIntent(
                    ts_utc=ctx.now_utc,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=float(abs(qty)),
                    order_type=OrderType.MKT,
                    limit_price=None,
                    stop_price=None,
                    strategy_id=self.strategy_id,
                    metadata={"exit_reason": "no_signal"},
                ))
            return orders
        
        # 计算每个标的的目标权重（等权）
        target_weight = 1.0 / len(signals)
        target_value = portfolio.equity * target_weight

        target_symbols = {s.symbol for s in signals}

        # 先卖出不在目标列表中的持仓
        for symbol, qty in portfolio.positions.items():
            if abs(qty) < 1e-8:
                continue
            if symbol not in target_symbols:
                if symbol in exit_symbols:
                    continue
                orders.append(OrderIntent(
                    ts_utc=ctx.now_utc,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=float(abs(qty)),
                    order_type=OrderType.MKT,
                    limit_price=None,
                    stop_price=None,
                    strategy_id=self.strategy_id,
                    metadata={"exit_reason": "rebalance_remove"},
                ))
        
        signal_map = {signal.symbol: signal for signal in signals}
        for signal in signals:
            # 获取当前价格
            price = signal.metadata.get("current_price", 100.0)
            target_value = portfolio.equity * target_weight
            
            # 检查当前持仓
            current_qty = portfolio.positions.get(signal.symbol, 0.0)
            
            if price <= 0:
                continue

            target_qty = target_value / price
            delta = target_qty - current_qty

            if abs(delta) < 1e-8:
                continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            qty = abs(delta)
            entry_tag = "rebalance_add" if side == OrderSide.BUY else "rebalance_trim"
            exit_reason = "rebalance_trim" if side == OrderSide.SELL else None
            metadata = {**(signal.metadata or {}), "entry_tag": entry_tag}
            if exit_reason:
                metadata["exit_reason"] = exit_reason
            order = OrderIntent(
                ts_utc=ctx.now_utc,
                symbol=signal.symbol,
                side=side,
                qty=float(qty),
                order_type=OrderType.MKT,
                limit_price=None,
                stop_price=None,
                strategy_id=self.strategy_id,
                metadata=metadata
            )
            orders.append(order)
        
        return orders

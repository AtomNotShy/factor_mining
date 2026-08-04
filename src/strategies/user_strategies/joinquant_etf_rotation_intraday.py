"""
聚宽ETF轮动策略（盘中价格版本）

策略核心逻辑：
1. 基于年化收益和R²判定系数计算动量得分
2. 使用"25个历史日收盘价 + 当日9:50盘中价"计算动量
3. 每日9:50轮动，持有得分最高的1只ETF
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import math
from datetime import datetime

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.strategies.base.parameters import (
    IntParameter,
    DecimalParameter,
    BooleanParameter,
    create_time_parameter_str,
)
from src.utils.logger import get_logger


class JoinQuantETFRotationIntradayStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """聚宽ETF轮动策略（盘中价格版本）"""

    strategy_name = "聚宽ETF轮动策略（盘中）"
    strategy_id = "joinquant_etf_rotation_intraday"
    timeframe = "1m"
    informative_timeframes = ["1d"]
    startup_candle_count = 260

    stoploss = -0.10
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    max_position_size = 1.0

    minimal_roi = {0: float("inf")}

    # 轮动参数
    min_days = IntParameter("min_days", 10, 40, default=20)
    max_days = IntParameter("max_days", 40, 120, default=60)
    auto_adjust = BooleanParameter("auto_adjust", default=True)
    r2_threshold = DecimalParameter("r2_threshold", 0.0, 1.0, default=0.0)

    # 动量计算参数
    lookback_days = IntParameter("lookback_days", 20, 30, default=25)
    intraday_time = create_time_parameter_str(
        name="intraday_time",
        start_time="9:00",
        end_time="16:00",
        default_time="9:50",
    )

    def __init__(self):
        super().__init__()
        self.logger = get_logger(f"strategy.{self.strategy_id}")

        self.etf_pool = [
            "SPY",
            "QQQ",
            "IWM",
            "TLT",
            "GLD",
            "USO",
            "VNQ",
            "XLF",
            "XLE",
            "XLK",
            "XLV",
            "XLI",
            "XLP",
            "XLY",
            "XLU",
        ]

        # 缓存
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._intraday_price_cache: Dict[str, pd.Series] = {}

        # 调仓控制
        self.last_rebalance_date = None
        self.current_holding: Optional[str] = None
        self.rebalance_interval = 5
        self._rebalance_done_today = False
        self._trading_day_counter = 0

        # 数据源
        self.data_feed: Optional[Any] = None

    def _get_param_value(self, param: Any) -> Any:
        """获取参数值"""
        return param.value if hasattr(param, "value") else param

    def _decimal_time_to_str(self, value: Any) -> str:
        """将小数时间转换为HH:MM字符串"""
        if isinstance(value, (int, float)):
            hours = int(value)
            minutes = int(round((value - hours) * 60))
            if minutes >= 60:
                hours += 1
                minutes = 0
            return f"{hours:02d}:{minutes:02d}"
        return str(value)

    def _build_daily_from_informative(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """从合并后的informative列构造日线数据"""
        daily_cols = {
            "open_1d": "open",
            "high_1d": "high",
            "low_1d": "low",
            "close_1d": "close",
            "volume_1d": "volume",
        }

        if any(col not in minute_df.columns for col in daily_cols):
            return pd.DataFrame()

        df = minute_df.copy()
        df["date"] = [d.date() for d in df.index]
        daily = (
            df[["date"] + list(daily_cols.keys())]
            .dropna(subset=["close_1d"])
            .groupby("date", as_index=True)
            .last()
            .rename(columns=daily_cols)
        )
        daily.index.name = "date"
        return daily

    def _get_intraday_price(self, minute_df: pd.DataFrame, time_str: str = "09:50") -> pd.Series:
        """获取指定时刻的盘中价格，返回以日期为索引的Series"""
        if minute_df.empty:
            return pd.Series(dtype=float)

        hour, minute = map(int, time_str.split(":"))
        # 确保索引是 DatetimeIndex
        dt_index = pd.DatetimeIndex(minute_df.index)
        mask = (dt_index.hour == hour) & (dt_index.minute == minute)
        price_at_time = minute_df.loc[mask, "close"]

        if price_at_time.empty:
            return pd.Series(dtype=float)

        # 转换为以日期为索引
        daily_price = pd.Series(price_at_time.values, index=[d.date() for d in price_at_time.index])
        return daily_price

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> float:
        """计算ATR"""
        n = len(close)
        if n < period + 1:
            return np.nan

        tr_values = []
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr_values.append(max(tr1, tr2, tr3))

        if len(tr_values) < period:
            return np.nan

        return float(np.mean(tr_values[-period:]))

    def _calculate_dynamic_lookback(self, df: pd.DataFrame) -> int:
        """计算动态回溯期"""
        min_days = self._get_param_value(self.min_days)
        max_days = self._get_param_value(self.max_days)

        if len(df) < max_days + 10:
            return int(min_days)

        high = np.asarray(df["high"].values, dtype=np.float64)
        low = np.asarray(df["low"].values, dtype=np.float64)
        close = np.asarray(df["close"].values, dtype=np.float64)

        long_atr = self._calculate_atr(high, low, close, int(max_days))
        short_atr = self._calculate_atr(
            np.asarray(high[-int(min_days) :], dtype=np.float64),
            np.asarray(low[-int(min_days) :], dtype=np.float64),
            np.asarray(close[-int(min_days) :], dtype=np.float64),
            int(min_days),
        )

        if np.isnan(long_atr) or np.isnan(short_atr) or long_atr == 0:
            return int(min_days)

        ratio = min(0.9, short_atr / long_atr)
        lookback = int(min_days + (max_days - min_days) * (1 - ratio))
        lookback = max(int(min_days), min(int(max_days), lookback))

        return lookback

    def _calculate_momentum_score(self, prices: np.ndarray) -> tuple[float, float, float]:
        """计算动量得分：年化收益率 × R²"""
        if len(prices) < 5:
            return 0.0, 0.0, 0.0

        prices_array = np.asarray(prices, dtype=np.float64)
        y = np.log(prices_array)

        if not np.isfinite(y).all():
            return 0.0, 0.0, 0.0

        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))

        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_return = math.exp(slope * 250) - 1

        y_pred = slope * x + intercept
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        y_mean = np.mean(y)
        ss_tot = np.sum(weights * (y - y_mean) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        score = annualized_return * r2

        return float(annualized_return), float(r2), float(score)

    def _check_drawdown_filter(self, prices: np.ndarray) -> bool:
        """检查跌幅过滤条件"""
        if len(prices) < 5:
            return True

        prices_array = np.asarray(prices, dtype=np.float64)

        # 条件1：三天内有一天跌超5%
        recent_returns = [
            prices_array[-1] / prices_array[-2],
            prices_array[-2] / prices_array[-3],
            prices_array[-3] / prices_array[-4],
        ]
        if any(r < 0.95 for r in recent_returns):
            return False

        # 条件2：三天内每天都跌，总共跌超4%
        if (
            prices_array[-1] < prices_array[-2]
            and prices_array[-2] < prices_array[-3]
            and prices_array[-3] < prices_array[-4]
            and prices_array[-1] / prices_array[-4] < 0.96
        ):
            return False

        # 条件3：四天内连续下跌，总共跌超4%
        if (
            prices_array[-2] < prices_array[-3]
            and prices_array[-3] < prices_array[-4]
            and prices_array[-4] < prices_array[-5]
            and prices_array[-2] / prices_array[-5] < 0.96
        ):
            return False

        return True

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None
    ) -> pd.DataFrame:
        """计算技术指标"""
        symbol = metadata.get("symbol", "") if metadata else ""

        if dataframe.empty or symbol not in self.etf_pool:
            return dataframe

        daily_df = self._daily_cache.get(symbol, pd.DataFrame())
        if daily_df.empty:
            daily_df = self._build_daily_from_informative(dataframe)
            if not daily_df.empty:
                self._daily_cache[symbol] = daily_df

        if daily_df.empty:
            return dataframe

        # 获取9:50价格
        if symbol not in self._intraday_price_cache:
            intraday_time = self._decimal_time_to_str(self._get_param_value(self.intraday_time))
            self._intraday_price_cache[symbol] = self._get_intraday_price(dataframe, intraday_time)

        intraday_950 = self._intraday_price_cache[symbol]

        if intraday_950.empty:
            return dataframe

        # 合并数据：以日期为索引
        combined = pd.DataFrame(
            {
                "close_prev": daily_df["close"],
                "price_950": intraday_950.reindex(daily_df.index),
            }
        )

        # 计算动态回溯期
        lookback = self._get_param_value(self.lookback_days)
        if self._get_param_value(self.auto_adjust):
            lookback = self._calculate_dynamic_lookback(daily_df)

        # 计算动量得分
        scores = []
        annualized_returns = []
        r2_values = []
        drawdown_passed = []

        for i in range(len(combined)):
            if i < lookback - 1:
                scores.append(np.nan)
                annualized_returns.append(np.nan)
                r2_values.append(np.nan)
                drawdown_passed.append(True)
                continue

            # 历史lookback天收盘价
            historical = np.asarray(
                combined["close_prev"].iloc[i - lookback + 1 : i + 1].values, dtype=np.float64
            )

            # 今日9:50价
            current_950 = combined["price_950"].iloc[i]

            if pd.isna(current_950):
                current_950 = combined["close_prev"].iloc[i]

            # 拼接数据：lookback收盘 + 1盘中
            prices = np.append(historical, float(current_950))
            ann_ret, r2, score = self._calculate_momentum_score(prices)

            scores.append(float(score))
            annualized_returns.append(float(ann_ret))
            r2_values.append(float(r2))

            # 跌幅过滤
            prices_for_check = np.asarray(
                combined["close_prev"].iloc[max(0, i - 4) : i + 1].values, dtype=np.float64
            )
            drawdown_passed.append(self._check_drawdown_filter(prices_for_check))

        # 添加到日线数据
        daily_df = daily_df.copy()
        daily_df["momentum_score"] = scores
        daily_df["annualized_return"] = annualized_returns
        daily_df["r2"] = r2_values
        daily_df["drawdown_passed"] = drawdown_passed
        daily_df["effective_score"] = [s if p else 0 for s, p in zip(scores, drawdown_passed)]

        # R²过滤
        r2_thresh = self._get_param_value(self.r2_threshold)
        if r2_thresh > 0:
            daily_df["effective_score"] = [
                s if r >= r2_thresh else 0 for s, r in zip(daily_df["effective_score"], r2_values)
            ]

        # 更新缓存
        self._daily_cache[symbol] = daily_df

        # 将日频数据合并到分钟级dataframe
        dataframe = dataframe.copy()
        dataframe["date"] = [d.date() for d in dataframe.index]

        daily_df_indexed = daily_df.reset_index()
        daily_df_indexed["date"] = pd.to_datetime(daily_df_indexed["date"]).dt.date

        merged = dataframe.merge(
            daily_df_indexed[
                [
                    "date",
                    "momentum_score",
                    "annualized_return",
                    "r2",
                    "effective_score",
                    "drawdown_passed",
                ]
            ],
            on="date",
            how="left",
        )

        dataframe["momentum_score"] = merged["momentum_score"].ffill().fillna(0)
        dataframe["annualized_return"] = merged["annualized_return"].ffill().fillna(0)
        dataframe["r2"] = merged["r2"].ffill().fillna(0)
        dataframe["effective_score"] = merged["effective_score"].ffill().fillna(0)
        dataframe["drawdown_passed"] = merged["drawdown_passed"].ffill().fillna(True)

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None
    ) -> pd.DataFrame:
        """生成进场信号"""
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        if dataframe.empty or "effective_score" not in dataframe.columns:
            return dataframe

        valid_mask = (dataframe["effective_score"] > 0) & (dataframe["effective_score"] < 6)

        dataframe.loc[valid_mask, "enter_long"] = 1
        dataframe.loc[valid_mask, "enter_tag"] = "momentum_positive"

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None
    ) -> pd.DataFrame:
        """生成离场信号"""
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        if dataframe.empty or "effective_score" not in dataframe.columns:
            return dataframe

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
        """跨资产组合选择"""
        result: Dict[str, Dict[str, Any]] = {}

        # 检查是否需要调仓
        is_new_day = self.last_rebalance_date is None or str(current_date) != str(
            self.last_rebalance_date
        )

        if is_new_day:
            self._rebalance_done_today = False
            self._trading_day_counter += 1

        need_rebalance = (
            not self._rebalance_done_today
            and self._trading_day_counter % self.rebalance_interval == 0
        )

        # 获取当天日期
        if hasattr(current_date, "date"):
            lookup_date = current_date.date()
        elif hasattr(current_date, "year"):
            lookup_date = current_date.date()
        else:
            lookup_date = current_date

        # 收集得分
        scores: Dict[str, float] = {}
        ann_returns: Dict[str, float] = {}
        r2_values: Dict[str, float] = {}

        for symbol, df in strategy_data.items():
            if df.empty:
                continue

            if hasattr(df.index, "date"):
                matching_rows = df[df.index.date == lookup_date]
            else:
                matching_rows = df[df.index == current_date]

            if not matching_rows.empty:
                row = matching_rows.iloc[-1]
                score = row.get("effective_score", 0)
                ann_ret = row.get("annualized_return", 0)
                r2 = row.get("r2", 0)

                scores[symbol] = float(score) if pd.notna(score) else 0
                ann_returns[symbol] = float(ann_ret) if pd.notna(ann_ret) else 0
                r2_values[symbol] = float(r2) if pd.notna(r2) else 0
            else:
                scores[symbol] = 0
                ann_returns[symbol] = 0
                r2_values[symbol] = 0

        if not scores:
            if current_holding:
                result[current_holding] = {"action": "FLAT"}
                self.current_holding = None
            return result

        # 选择得分最高的
        valid_scores = {k: v for k, v in scores.items() if v > 0}

        if not valid_scores:
            if current_holding:
                result[current_holding] = {"action": "FLAT"}
                self.current_holding = None
            return result

        best_symbol = max(valid_scores.items(), key=lambda x: x[1])[0]
        best_score = valid_scores[best_symbol]

        if not need_rebalance:
            return result

        self._rebalance_done_today = True
        self.last_rebalance_date = current_date

        self.logger.info(f"轮动选择: {best_symbol} (得分: {best_score:.4f})")
        self.current_holding = best_symbol

        for symbol in strategy_data:
            if symbol == best_symbol:
                result[symbol] = {
                    "action": "LONG",
                    "score": best_score,
                    "metadata": {
                        "annualized_return": ann_returns.get(symbol, 0),
                        "r2": r2_values.get(symbol, 0),
                    },
                }
            else:
                result[symbol] = {"action": "FLAT"}

        return result

    async def bot_start(self, **kwargs) -> None:
        """启动时调用"""
        self.logger.info(f"启动 {self.strategy_name}")
        self.logger.info(f"ETF池: {self.etf_pool}")
        intraday_time = self._decimal_time_to_str(self._get_param_value(self.intraday_time))
        self.logger.info(
            f"参数: lookback_days={self.lookback_days.value}, intraday_time={intraday_time}"
        )

    def botShutdown(self, **kwargs) -> None:
        """关闭时调用"""
        self.logger.info(f"关闭 {self.strategy_name}")

    def custom_stoploss(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time: datetime,
        **kwargs,
    ) -> float:
        """自定义止损"""
        return self.stoploss

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """确认订单进入"""
        if pair not in self.etf_pool:
            return False
        if rate <= 0:
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
        **kwargs,
    ) -> bool:
        """确认订单退出"""
        if rate <= 0:
            return False
        return True


if __name__ == "__main__":
    strategy = JoinQuantETFRotationIntradayStrategy()
    print(f"策略ID: {strategy.strategy_id}")
    print(f"策略名称: {strategy.strategy_name}")
    print(f"时间框架: {strategy.timeframe}")
    print(f"ETF池: {strategy.etf_pool}")
    print(f"lookback_days: {strategy.lookback_days.value}")
    print(f"intraday_time: {strategy.intraday_time.value}")

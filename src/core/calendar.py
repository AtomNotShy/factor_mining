"""
交易日历
美股交易日历抽象（最小实现）
"""

from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Union
import pandas as pd
from src.utils.logger import get_logger


class TradingCalendar:
    """交易日历（美股）"""

    def __init__(self):
        self.logger = get_logger("trading_calendar")
        self._calendar = self._load_calendar()
        # 美股主要节假日（简化版，优先使用 pandas_market_calendars）
        self.holidays = self._load_holidays()
        # 交易分钟数据缓存
        self._trading_minutes_cache: Dict[str, pd.DatetimeIndex] = {}

    def _load_calendar(self):
        """加载 pandas_market_calendars 的交易日历（优先）"""
        try:
            import pandas_market_calendars as mcal

            nyse = mcal.get_calendar("NYSE")
            self.logger.info("使用 pandas_market_calendars NYSE 日历")
            return nyse
        except Exception as e:
            self.logger.debug(f"加载 pandas_market_calendars 失败，回退简化日历: {e}")
            return None

    def _load_holidays(self) -> List[date]:
        """加载节假日列表（简化实现）"""
        start = pd.Timestamp("1990-01-01")
        end = pd.Timestamp("2100-12-31")

        if self._calendar is not None:
            try:
                holidays_obj = self._calendar.holidays
                if callable(holidays_obj):
                    holidays_obj = holidays_obj()
                if hasattr(holidays_obj, "holidays"):
                    holiday_index = holidays_obj.holidays(start=start, end=end)
                    return [d.date() for d in holiday_index]
                if isinstance(holidays_obj, tuple):
                    holiday_dates = []
                    for part in holidays_obj:
                        if hasattr(part, "holidays"):
                            part_index = part.holidays(start=start, end=end)
                            holiday_dates.extend(list(part_index))
                        else:
                            try:
                                part_index = pd.to_datetime(part)
                                holiday_dates.extend(list(part_index))
                            except Exception:
                                continue
                    if holiday_dates:
                        return [d.date() for d in pd.DatetimeIndex(holiday_dates)]
            except Exception as e:
                self.logger.debug(f"加载NYSE节假日失败，将使用交易日推断: {e}")

            try:
                weekdays = pd.date_range(start, end, freq="B")
                valid_days = self._calendar.valid_days(start, end)
                holiday_index = weekdays.difference(valid_days)
                return [d.date() for d in holiday_index]
            except Exception as e:
                self.logger.debug(f"通过交易日推断节假日失败，回退USFederalHolidayCalendar: {e}")

        try:
            from pandas.tseries.holiday import USFederalHolidayCalendar
            from pandas.tseries.offsets import Easter

            calendar = USFederalHolidayCalendar()
            holiday_index = calendar.holidays(start=start, end=end)

            # 向量化计算好的星期五（耶稣受难日前两天）
            years = pd.date_range(start, end, freq="YS")
            # 使用向量化方式计算
            good_fridays = pd.DatetimeIndex([
                (pd.Timestamp(y) + Easter() - pd.Timedelta(days=2)).date()
                for y in years
            ])
            holiday_index = holiday_index.union(pd.DatetimeIndex(good_fridays))

            return [d.date() for d in holiday_index]
        except Exception as e:
            self.logger.warning(f"加载节假日失败，回退为空: {e}")
            return []
    
    def is_trading_day(self, dt: date) -> bool:
        """
        判断是否为交易日
        
        Args:
            dt: 日期
            
        Returns:
            是否为交易日
        """
        if self._calendar is not None:
            dt_ts = pd.Timestamp(dt)
            return len(self._calendar.valid_days(dt_ts, dt_ts)) > 0

        # 周末不是交易日
        if dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        
        # 节假日不是交易日
        if dt in self.holidays:
            return False
        
        return True
    
    def next_trading_day(self, dt: date, n: int = 1) -> date:
        """
        获取下一个交易日
        
        Args:
            dt: 起始日期
            n: 向前推n个交易日（默认1）
            
        Returns:
            下一个交易日
        """
        current = dt
        count = 0
        
        while count < n:
            current += timedelta(days=1)
            if self.is_trading_day(current):
                count += 1
        
        return current
    
    def prev_trading_day(self, dt: date, n: int = 1) -> date:
        """
        获取上一个交易日
        
        Args:
            dt: 起始日期
            n: 向后推n个交易日（默认1）
            
        Returns:
            上一个交易日
        """
        current = dt
        count = 0
        
        while count < n:
            current -= timedelta(days=1)
            if current < date(1900, 1, 1):  # 防止无限循环
                raise ValueError(f"无法找到 {n} 个交易日前的日期")
            if self.is_trading_day(current):
                count += 1
        
        return current
    
    def trading_days_between(self, start: date, end: date) -> List[date]:
        """
        获取两个日期之间的所有交易日
        
        Args:
            start: 起始日期
            end: 结束日期
            
        Returns:
            交易日列表
        """
        if self._calendar is not None:
            valid_days = self._calendar.valid_days(pd.Timestamp(start), pd.Timestamp(end))
            return [d.date() for d in valid_days]

        trading_days = []
        current = start
        
        while current <= end:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def get_trading_days(self, start: date, end: date) -> pd.DatetimeIndex:
        """
        获取交易日范围（返回DatetimeIndex）

        Args:
            start: 起始日期
            end: 结束日期

        Returns:
            交易日DatetimeIndex
        """
        if self._calendar is not None:
            return self._calendar.valid_days(pd.Timestamp(start), pd.Timestamp(end))

        days = self.trading_days_between(start, end)
        return pd.DatetimeIndex([pd.Timestamp(d) for d in days])

    def get_schedule(
        self,
        start: Union[datetime, date],
        end: Union[datetime, date],
        tz: str = "America/New_York",
    ) -> pd.DataFrame:
        """
        返回交易时段信息表

        Args:
            start: 起始日期
            end: 结束日期
            tz: 时区，默认"America/New_York"

        Returns:
            DataFrame with columns:
            - market_open: NYSE开市时间 (ET)
            - market_close: NYSE收市时间 (ET)
        """
        if self._calendar is None:
            self.logger.warning("pandas_market_calendars 未可用，返回空schedule")
            empty_series = pd.Series([], dtype="datetime64[ns, America/New_York]")
            return pd.DataFrame({"market_open": empty_series, "market_close": empty_series})

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        schedule = self._calendar.schedule(start_date=start_ts, end_date=end_ts)
        return schedule

    def get_trading_minutes(
        self,
        start: Union[datetime, date],
        end: Union[datetime, date],
        tz: str = "America/New_York",
        interval: str = "1min",
    ) -> pd.DatetimeIndex:
        """
        返回会话内分钟级时间轴（建议缓存）

        Args:
            start: 起始日期
            end: 结束日期
            tz: 时区，默认"America/New_York"
            interval: 时间间隔，默认"1min"

        Returns:
            UTC时间戳的DatetimeIndex
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        # 处理 NaT 情况
        if pd.isna(start_ts) or pd.isna(end_ts):
            self.logger.warning("start 或 end 日期无效，返回空分钟时间轴")
            return pd.DatetimeIndex([])
        # 转换为 datetime 以避免类型问题
        start_dt = start_ts.to_pydatetime()
        end_dt = end_ts.to_pydatetime()
        start_str = start_dt.strftime("%Y%m%d")  # type: ignore[union-attr]
        end_str = end_dt.strftime("%Y%m%d")  # type: ignore[union-attr]
        cache_key = f"{start_str}_{end_str}_{tz}_{interval}"

        if cache_key in self._trading_minutes_cache:
            return self._trading_minutes_cache[cache_key]

        if self._calendar is None:
            self.logger.warning("pandas_market_calendars 未可用，返回空分钟时间轴")
            return pd.DatetimeIndex([])

        # 使用类型断言绕过类型检查
        schedule = self.get_schedule(start_dt, end_dt, tz)  # type: ignore[arg-type]

        minutes_list = []
        for idx, row in schedule.iterrows():
            market_open = row["market_open"]
            market_close = row["market_close"]

            session_minutes = pd.date_range(
                start=market_open, end=market_close, freq=interval
            )
            minutes_list.append(session_minutes)

        if minutes_list:
            all_minutes = minutes_list[0]
            for idx in minutes_list[1:]:
                all_minutes = all_minutes.union(idx)
        else:
            all_minutes = pd.DatetimeIndex([])

        self._trading_minutes_cache[cache_key] = all_minutes
        return all_minutes

    def session_time(
        self, date_obj: date, hhmm: str = "09:50", tz: str = "America/New_York"
    ) -> pd.Timestamp:
        """
        将"指定时刻"映射到UTC时间戳

        Args:
            date_obj: 日期
            hhmm: 时间字符串，格式为"HH:MM"，默认"09:50"
            tz: 时区，默认"America/New_York"

        Returns:
            UTC时间戳

        Example:
            >>> calendar.session_time(date(2024, 1, 2), "09:50")
            Timestamp('2024-01-02 14:50:00+00:00')  # UTC
        """
        time_parts = hhmm.split(":")
        if len(time_parts) != 2:
            raise ValueError(f"hhmm 格式错误，应为 'HH:MM'，实际为 '{hhmm}'")

        hour, minute = int(time_parts[0]), int(time_parts[1])

        local_time = pd.Timestamp(
            year=date_obj.year,
            month=date_obj.month,
            day=date_obj.day,
            hour=hour,
            minute=minute,
            tz=tz,
        )

        utc_time = local_time.tz_convert("UTC")
        # 处理 NaT 情况
        if pd.isna(utc_time):  # type: ignore[comparison-overlap]
            raise ValueError(f"无法将 {date_obj} {hhmm} {tz} 转换为有效的 UTC 时间")
        return utc_time  # type: ignore[return-value]

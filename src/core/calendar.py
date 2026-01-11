"""
交易日历
美股交易日历抽象（最小实现）
"""

from datetime import datetime, date, timedelta
from typing import List, Optional
import pandas as pd
from src.utils.logger import get_logger


class TradingCalendar:
    """交易日历（美股）"""
    
    def __init__(self):
        self.logger = get_logger("trading_calendar")
        self._calendar = self._load_calendar()
        # 美股主要节假日（简化版，优先使用 pandas_market_calendars）
        self.holidays = self._load_holidays()

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
            from pandas.tseries.holiday import USFederalHolidayCalendar, Easter
            from pandas.tseries.offsets import Day

            calendar = USFederalHolidayCalendar()
            holiday_index = calendar.holidays(start=start, end=end)

            years = pd.date_range(start, end, freq="YS")
            good_fridays = (years + Easter()) - Day(2)
            holiday_index = holiday_index.union(good_fridays)

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

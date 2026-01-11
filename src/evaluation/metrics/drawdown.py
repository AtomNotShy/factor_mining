"""
回撤分析模块
提供回撤区间识别和详细分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from src.utils.logger import get_logger


@dataclass
class DrawdownWindow:
    """回撤区间"""
    start_date: str
    end_date: str
    peak_date: str
    trough_date: str
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'peak_date': self.peak_date,
            'trough_date': self.trough_date,
            'peak_value': self.peak_value if self.peak_value is not None else 0.0,
            'trough_value': self.trough_value if self.trough_value is not None else 0.0,
            'drawdown_pct': self.drawdown_pct,
            'duration_days': self.duration_days,
            'recovery_days': self.recovery_days
        }


@dataclass
class DrawdownStatistics:
    """回撤统计"""
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    avg_drawdown_duration: float
    total_drawdown_periods: int
    drawdown_windows: List[DrawdownWindow]
    time_in_drawdown_pct: float
    ulcer_index: float
    burke_ratio: float


class DrawdownAnalyzer:
    """回撤分析器"""
    
    def __init__(self):
        self.logger = get_logger("drawdown_analyzer")
    
    def calculate_drawdown_series(
        self,
        equity_curve: pd.Series
    ) -> pd.Series:
        """计算回撤序列"""
        try:
            if len(equity_curve) == 0:
                return pd.Series(dtype=float)
            
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            return drawdown
            
        except Exception as e:
            self.logger.error(f"计算回撤序列失败: {e}")
            return pd.Series(dtype=float)
    
    def _date_to_str(self, dt) -> str:
        """将日期转换为字符串"""
        try:
            if isinstance(dt, str):
                return dt
            return str(pd.to_datetime(dt))
        except:
            return ""
    
    def _get_date_index(self, dates: List, target_dt) -> Optional[int]:
        """获取日期在列表中的索引"""
        try:
            target_ts = pd.to_datetime(target_dt)
            for i, d in enumerate(dates):
                if pd.to_datetime(d) == target_ts:
                    return i
            return None
        except:
            return None
    
    def find_drawdown_windows(
        self,
        equity_curve: pd.Series,
        min_drawdown_threshold: float = 0.02,
        min_duration_days: int = 1
    ) -> List[DrawdownWindow]:
        """识别所有显著回撤区间"""
        try:
            if len(equity_curve) < 2:
                return []
            
            drawdown = self.calculate_drawdown_series(equity_curve)
            dates_list = list(drawdown.index)
            n = len(dates_list)
            
            in_drawdown = False
            drawdown_start = None
            peak_date = None
            peak_value = 0.0
            
            windows = []
            
            for i, (date, dd) in enumerate(drawdown.items()):
                if dd < -min_drawdown_threshold and not in_drawdown:
                    in_drawdown = True
                    drawdown_start = date
                    
                    peak_idx = i
                    for j in range(i - 1, -1, -1):
                        if drawdown.iloc[j] >= 0:
                            peak_idx = j
                            break
                    if peak_idx < n:
                        peak_date = dates_list[peak_idx]
                        peak_value = float(equity_curve.iloc[peak_idx])
                
                elif in_drawdown and dd >= 0:
                    trough_idx = i
                    
                    start_idx = self._get_date_index(dates_list, drawdown_start)
                    if start_idx is None:
                        start_idx = i
                    
                    for j in range(i, start_idx - 1, -1):
                        if j >= 0 and drawdown.iloc[j] < dd:
                            trough_idx = j
                    
                    if trough_idx < n:
                        trough_date = dates_list[trough_idx]
                        trough_value = float(equity_curve.iloc[trough_idx])
                        
                        duration = i - start_idx
                        
                        if duration >= min_duration_days:
                            windows.append(DrawdownWindow(
                                start_date=self._date_to_str(drawdown_start),
                                end_date=self._date_to_str(trough_date),
                                peak_date=self._date_to_str(peak_date),
                                trough_date=self._date_to_str(trough_date),
                                peak_value=peak_value if peak_value is not None else 0.0,
                                trough_value=trough_value if trough_value is not None else 0.0,
                                drawdown_pct=float(drawdown.iloc[trough_idx]) if drawdown.iloc[trough_idx] is not None else 0.0,
                                duration_days=duration,
                                recovery_days=None
                            ))
                    
                    in_drawdown = False
            
            return self._add_recovery_days(windows, equity_curve, dates_list)
            
        except Exception as e:
            self.logger.error(f"识别回撤区间失败: {e}")
            return []
    
    def _add_recovery_days(
        self,
        windows: List[DrawdownWindow],
        equity_curve: pd.Series,
        dates_list: List
    ) -> List[DrawdownWindow]:
        """计算恢复天数"""
        try:
            if not windows:
                return windows
            
            equity_array = equity_curve.values
            n = len(dates_list)
            
            for window in windows:
                if window.recovery_days is not None:
                    continue
                
                trough_idx = self._get_date_index(dates_list, window.trough_date)
                if trough_idx is None:
                    window.recovery_days = None
                    continue
                
                trough_value = window.trough_value
                recovery_found = False
                
                for i in range(trough_idx + 1, n):
                    if equity_array[i] >= trough_value * (1 + abs(window.drawdown_pct) * 0.5):
                        window.recovery_days = i - trough_idx
                        recovery_found = True
                        break
                
                if not recovery_found:
                    window.recovery_days = None
            
            return windows
            
        except Exception as e:
            self.logger.error(f"计算恢复天数失败: {e}")
            return windows
    
    def find_max_drawdown_window(
        self,
        equity_curve: pd.Series
    ) -> DrawdownWindow:
        """找到最大回撤区间"""
        try:
            if len(equity_curve) == 0:
                return DrawdownWindow(
                    start_date="", end_date="", peak_date="", trough_date="",
                    peak_value=np.nan, trough_value=np.nan,
                    drawdown_pct=0, duration_days=0
                )
            
            drawdown = self.calculate_drawdown_series(equity_curve)
            dates_list = list(drawdown.index)
            
            min_dd_idx = int(drawdown.values.argmin())
            min_dd_value = float(drawdown.iloc[min_dd_idx])
            trough_date = dates_list[min_dd_idx]
            trough_value = float(equity_curve.iloc[min_dd_idx])
            
            peak_idx = 0
            for j in range(min_dd_idx, -1, -1):
                if drawdown.iloc[j] >= 0:
                    peak_idx = j
                    break
            peak_date = dates_list[peak_idx]
            peak_value = float(equity_curve.iloc[peak_idx])
            
            return DrawdownWindow(
                start_date=self._date_to_str(peak_date),
                end_date=self._date_to_str(trough_date),
                peak_date=self._date_to_str(peak_date),
                trough_date=self._date_to_str(trough_date),
                peak_value=peak_value,
                trough_value=trough_value,
                drawdown_pct=min_dd_value,
                duration_days=min_dd_idx - peak_idx
            )
            
        except Exception as e:
            self.logger.error(f"查找最大回撤区间失败: {e}")
            return DrawdownWindow(
                start_date="", end_date="", peak_date="", trough_date="",
                peak_value=np.nan, trough_value=np.nan,
                drawdown_pct=0, duration_days=0
            )
    
    def calculate_ulcer_index(self, equity_curve: pd.Series) -> float:
        """计算溃疡指数 (Ulcer Index)"""
        try:
            if len(equity_curve) < 2:
                return np.nan
            
            drawdown = self.calculate_drawdown_series(equity_curve)
            squared_drawdown = (drawdown * 100) ** 2
            ui = float(np.sqrt(squared_drawdown.mean()))
            
            return ui
            
        except Exception as e:
            self.logger.error(f"计算溃疡指数失败: {e}")
            return np.nan
    
    def calculate_burke_ratio(
        self,
        equity_curve: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """计算 Burke 比率"""
        try:
            if len(equity_curve) < 2:
                return np.nan
            
            cagr = self._calculate_cagr(equity_curve)
            drawdown = self.calculate_drawdown_series(equity_curve)
            mdd_std = float((drawdown * 100).std())
            
            if mdd_std == 0:
                return np.nan
            
            return (cagr - risk_free_rate) / mdd_std
            
        except Exception as e:
            self.logger.error(f"计算Burke比率失败: {e}")
            return np.nan
    
    def _calculate_cagr(self, equity_curve: pd.Series) -> float:
        """计算CAGR"""
        try:
            if len(equity_curve) < 2:
                return np.nan
            
            total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
            years = len(equity_curve) / 252
            
            if years <= 0:
                return np.nan
            
            return (1 + total_return) ** (1 / years) - 1
            
        except:
            return np.nan
    
    def calculate_avg_drawdown(self, equity_curve: pd.Series) -> float:
        """计算平均回撤"""
        try:
            drawdown = self.calculate_drawdown_series(equity_curve)
            return float(drawdown.mean())
        except:
            return 0.0
    
    def calculate_time_in_drawdown(
        self,
        equity_curve: pd.Series
    ) -> float:
        """计算回撤时间占比"""
        try:
            if len(equity_curve) == 0:
                return 0.0
            
            drawdown = self.calculate_drawdown_series(equity_curve)
            time_in_dd = (drawdown < 0).sum() / len(drawdown)
            
            return float(time_in_dd)
            
        except Exception as e:
            self.logger.error(f"计算回撤时间占比失败: {e}")
            return 0.0
    
    def comprehensive_drawdown_analysis(
        self,
        equity_curve: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """完整回撤分析"""
        try:
            if len(equity_curve) < 2:
                return {}
            
            drawdown = self.calculate_drawdown_series(equity_curve)
            max_dd_window = self.find_max_drawdown_window(equity_curve)
            drawdown_windows = self.find_drawdown_windows(equity_curve)
            
            avg_drawdown = self.calculate_avg_drawdown(equity_curve)
            time_in_drawdown = self.calculate_time_in_drawdown(equity_curve)
            ulcer_index = self.calculate_ulcer_index(equity_curve)
            burke_ratio = self.calculate_burke_ratio(equity_curve, risk_free_rate)
            
            durations = [w.duration_days for w in drawdown_windows]
            avg_duration = float(np.mean(durations)) if durations else 0
            
            return {
                'max_drawdown': float(drawdown.min()),
                'max_drawdown_duration': max_dd_window.duration_days,
                'max_drawdown_window': max_dd_window.to_dict(),
                'avg_drawdown': avg_drawdown,
                'avg_drawdown_duration': avg_duration,
                'total_drawdown_periods': len(drawdown_windows),
                'drawdown_windows': [w.to_dict() for w in drawdown_windows],
                'time_in_drawdown_pct': time_in_drawdown,
                'ulcer_index': ulcer_index,
                'burke_ratio': burke_ratio,
                'drawdown_series': drawdown.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"完整回撤分析失败: {e}")
            return {}
    
    def get_drawdown_periods_for_chart(
        self,
        equity_curve: pd.Series
    ) -> List[Dict]:
        """获取用于图表展示的回撤区间"""
        try:
            windows = self.find_drawdown_windows(equity_curve)
            
            periods = []
            for window in windows:
                periods.append({
                    'x0': window.start_date,
                    'x1': window.end_date,
                    'y0': window.peak_value,
                    'y1': window.trough_value,
                    'fillcolor': 'rgba(255, 0, 0, 0.1)',
                    'line': {'color': 'rgba(255, 0, 0, 0.5)'}
                })
            
            return periods
            
        except Exception as e:
            self.logger.error(f"获取回撤区间失败: {e}")
            return []

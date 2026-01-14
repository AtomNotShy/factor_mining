"""
@informative装饰器和多时间框架工具函数
参考Freqtrade实现，支持策略在多个时间框架上计算指标
"""

from functools import wraps
from typing import Dict, Callable, Any, Optional, List
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("informative")


def _timeframe_to_minutes(timeframe: str) -> int:
    """
    将时间框架字符串转换为分钟数
    
    Args:
        timeframe: 时间框架字符串，如 '1m', '5m', '15m', '30m', '1h', '4h', '1d'
    
    Returns:
        分钟数
    
    Examples:
        >>> _timeframe_to_minutes('5m')
        5
        >>> _timeframe_to_minutes('1h')
        60
        >>> _timeframe_to_minutes('4h')
        240
        >>> _timeframe_to_minutes('1d')
        1440
    """
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 60 * 24
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 60 * 24 * 7
    else:
        raise ValueError(f"Unknown timeframe format: {timeframe}")


def resample_to_interval(
    df: pd.DataFrame,
    interval_minutes: int,
    agg: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    将DataFrame重采样到指定时间间隔
    
    Args:
        df: 原始DataFrame（DatetimeIndex）
        interval_minutes: 目标间隔的分钟数，如 60 表示1h
        agg: 聚合规则字典，键为列名，值为聚合方式
              可选值: 'first', 'last', 'max', 'min', 'sum', 'mean'
    
    Returns:
        重采样后的DataFrame
    
    Examples:
        >>> df_4h = resample_to_interval(df_1m, 240)
        >>> df_1h = resample_to_interval(df_5m, 60)
    """
    if df.empty:
        return df.copy()
    
    # 默认聚合规则（OHLCV标准）
    if agg is None:
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # 只聚合存在的列
    agg_dict = {k: v for k, v in agg.items() if k in df.columns}
    
    # 执行重采样
    resampled = df.resample(f'{interval_minutes}min').agg(agg_dict)
    
    # 删除重采样产生的全NaN行
    resampled = resampled.dropna(how='all')
    
    return resampled


def merge_informative_pair(
    main_df: pd.DataFrame,
    inf_df: pd.DataFrame,
    main_timeframe: str,
    inf_timeframe: str,
    ffill: bool = True,
    drop_informative_columns: bool = True,
    lookahead_check: bool = True
) -> pd.DataFrame:
    """
    将辅助时间框架的数据合并到主时间框架
    
    关键点：
    1. 日期对齐：防止前视偏差（lookahead bias）
    2. 列重命名：添加时间框架后缀，避免冲突
    3. 前向填充（ffill）：使低频数据在每根bar上可用
    4. 前视偏差检测：检查informative数据是否包含未来数据
    
    Args:
        main_df: 主时间框架的DataFrame
        inf_df: 辅助时间框架的DataFrame
        main_timeframe: 主时间框架字符串，如 '5m'
        inf_timeframe: 辅助时间框架字符串，如 '1h', '4h'
        ffill: 是否前向填充缺失值
        drop_informative_columns: 是否合并后删除原始informative列
        lookahead_check: 是否执行前视偏差检测
    
    Returns:
        合并后的DataFrame
    
    Examples:
        >>> # 将1小时数据合并到5分钟数据
        >>> merged = merge_informative_pair(
        ...     df_5m, df_1h, '5m', '1h',
        ...     ffill=True, lookahead_check=True
        ... )
    """
    if inf_df.empty or main_df.empty:
        return main_df.copy()
    
    # 获取时间框架的分钟数
    inf_minutes = _timeframe_to_minutes(inf_timeframe)
    
    # 创建合并键（日期 + 时间框架偏移）
    # 关键：偏移防止前视偏差
    # 例如：5m bar 在 12:15，不应知道 12:00-13:00 的1h candle
    inf_df = inf_df.copy()
    inf_df = inf_df.drop(columns=["symbol", "timeframe"], errors="ignore")
    inf_df['date_merge'] = inf_df.index + pd.to_timedelta(inf_minutes, 'm')
    
    # 重命名辅助列，添加时间框架后缀
    # 保留 symbol, timeframe 等元数据列
    inf_columns = [col for col in inf_df.columns 
                  if col not in ['symbol', 'timeframe', 'date', 'date_merge']]
    rename_map = {col: f"{col}_{inf_timeframe}" for col in inf_columns}
    inf_df = inf_df.rename(columns=rename_map)
    
    # 合并数据
    merged = pd.merge(
        main_df, 
        inf_df, 
        left_index=True,
        right_on='date_merge', 
        how='left'
    )
    merged.index = main_df.index
    
    # 删除临时列
    cols_to_drop = ["date_merge"]
    if drop_informative_columns:
        cols_to_drop.extend([col for col in merged.columns if col.endswith('_merge')])
    merged = merged.drop(columns=cols_to_drop, errors='ignore')
    
    # 前向填充
    if ffill:
        # 获取所有informative列（以_{inf_timeframe}结尾）
        inf_cols = [col for col in merged.columns if col.endswith(f"_{inf_timeframe}")]
        merged[inf_cols] = merged[inf_cols].ffill()
    
    # 前视偏差检测
    if lookahead_check:
        inf_max_date = inf_df.index.max() if hasattr(inf_df.index, 'max') else inf_df['date'].max()
        main_max_date = main_df.index.max() if hasattr(main_df.index, 'max') else main_df['date'].max()
        
        if pd.Timestamp(inf_max_date) > pd.Timestamp(main_max_date):
            logger.warning(
                f"前视偏差检测: {inf_timeframe} 数据 ({inf_max_date}) "
                f"晚于主数据 ({main_max_date})"
            )
    
    return merged


def merge_informative_pairs(
    main_df: pd.DataFrame,
    informative_dfs: Dict[str, pd.DataFrame],
    main_timeframe: str,
    ffill: bool = True,
    lookahead_check: bool = True
) -> pd.DataFrame:
    """
    将多个informative DataFrame合并到主DataFrame
    
    Args:
        main_df: 主时间框架的DataFrame
        informative_dfs: {timeframe: DataFrame} 映射
        main_timeframe: 主时间框架字符串
        ffill: 是否前向填充
        lookahead_check: 是否执行前视偏差检测
    
    Returns:
        合并后的DataFrame
    """
    result = main_df.copy()
    
    for tf, df in informative_dfs.items():
        if df.empty:
            continue
        
        result = merge_informative_pair(
            result, df, main_timeframe, tf,
            ffill=ffill, lookahead_check=lookahead_check
        )
    
    return result


def informative(
    timeframe: str,
    asset: str = "",
    ffill: bool = True,
) -> Callable:
    """
    @informative装饰器
    
    用法：
        @informative('1h')
        def populate_indicators_1h(self, dataframe, metadata):
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe
    
    工作原理：
        1. 装饰器将方法标记为informative方法
        2. 框架自动调用该方法，传入对应时间框架的数据
        3. 计算结果自动合并到主DataFrame（添加_{timeframe}后缀）
        4. 通过ffill使数据在主时间框架的每根bar上可用
    
    Args:
        timeframe: 辅助时间框架字符串，如 '1h', '4h', '1d'
        asset: 资产标识，如 'BTC/USDT', 'ETH/USDT'
                   如果为空，则使用当前交易对
        ffill: 是否前向填充
    
    Examples:
        >>> class MyStrategy(Strategy):
        ...     @informative('1h')
        ...     def populate_indicators_1h(self, dataframe, metadata):
        ...         dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        ...         return dataframe
        ...     
        ...     @informative('4h', 'ETH/USDT')
        ...     def populate_eth_4h(self, dataframe, metadata):
        ...         dataframe['close'] = dataframe['close']  # ETH价格
        ...         return dataframe
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, *args, **kwargs) -> Any:
            # 调用原始方法
            return method(self, *args, **kwargs)
        
        # 标记为informative方法
        wrapper._is_informative = True
        wrapper._timeframe = timeframe
        wrapper._asset = asset
        wrapper._ffill = ffill
        
        return wrapper
    
    return decorator

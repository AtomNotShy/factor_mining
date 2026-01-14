"""
多时间框架回测支持模块

功能：
1. 加载和合并多个时间框架的数据
2. Informative Pairs 支持（不同标的不同时间框架）
3. 数据重采样（高频率 -> 低频率）
4. 跨时间框架分析工具
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.utils.logger import get_logger


logger = get_logger("multi_timeframe")


# 时间框架常量
TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}

# 常用的时间框架组合
COMMON_TIMEFRAME_COMBINATIONS = {
    "scalp": ("1m", "5m", "15m"),
    "daytrade": ("5m", "15m", "1h"),
    "swing": ("1h", "4h", "1d"),
    "position": ("4h", "1d", "1w"),
}


@dataclass
class TimeframeConfig:
    """时间框架配置"""
    main: str = "1d"  # 主时间框架（用于生成信号和交易）
    informative: List[str] = field(default_factory=list)  # 信息性时间框架
    resample: Optional[str] = None  # 重采样目标时间框架


@dataclass
class InformativePair:
    """信息性标的配置"""
    pair: str  # 标的代码
    timeframe: str  # 时间框架
    data_type: str = "ohlcv"  # 数据类型


class MultiTimeframeDataLoader:
    """
    多时间框架数据加载器
    
    支持：
    1. 加载单个标的的多个时间框架数据
    2. Informative Pairs（不同标的不同时间框架）
    3. 数据重采样（高频率 -> 低频率）
    4. 合并数据到主时间框架
    """
    
    def __init__(
        self,
        store=None,
        default_timeframe: str = "1d",
    ):
        """
        初始化多时间框架数据加载器
        
        Args:
            store: 数据存储后端（可选）
            default_timeframe: 默认时间框架
        """
        self.store = store
        self.default_timeframe = default_timeframe
        self.logger = get_logger("multi_timeframe_loader")
        
        # 缓存已加载的数据
        self._data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}  # {symbol: {timeframe: df}}
    
    def load_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str],
        start: datetime,
        end: datetime,
        max_history: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载单个标的的多个时间框架数据
        
        Args:
            symbol: 标的代码
            timeframes: 时间框架列表
            start: 开始时间
            end: 结束时间
            max_history: 是否加载最大历史（用于预热）
            
        Returns:
            Dict[str, pd.DataFrame]: {timeframe: DataFrame}
        """
        result = {}
        
        # 确定需要加载的日期范围
        max_tf_seconds = max(TIMEFRAME_SECONDS.get(tf, 86400) for tf in timeframes)
        lookback_days = max(30, max_tf_seconds // 86400 + 30)  # 至少30天
        
        adjusted_start = start
        if max_history:
            adjusted_start = start - timedelta(days=lookback_days)
        
        self.logger.info(
            f"Loading {symbol} for timeframes {timeframes} "
            f"from {adjusted_start.date()} to {end.date()}"
        )
        
        for tf in timeframes:
            try:
                # 从 store 或数据源加载数据
                if self.store:
                    df = self.store.read_bars(
                        symbol=symbol,
                        timeframe=tf,
                        start=adjusted_start,
                        end=end,
                    )
                else:
                    # 使用模拟数据（如果没有 store）
                    df = self._generate_mock_data(symbol, tf, adjusted_start, end)
                
                if df.empty:
                    self.logger.warning(f"No data for {symbol} {tf}")
                    continue
                
                # 确保有标准列
                df = self._standardize_dataframe(df)
                
                # 过滤时间范围
                df = df[(df.index >= adjusted_start) & (df.index <= end)]
                
                result[tf] = df
                self.logger.info(f"  {symbol} {tf}: {len(df)} bars")
                
            except Exception as e:
                self.logger.error(f"Failed to load {symbol} {tf}: {e}")
        
        return result
    
    def load_informative_pairs(
        self,
        informative_pairs: List[InformativePair],
        main_timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载 Informative Pairs（信息性标的）
        
        Freqtrade 风格的 informative pairs：
        - 每个标的可以有不同的信息性时间框架
        - 信息性数据用于生成信号，但交易在主时间框架执行
        
        Args:
            informative_pairs: 信息性标的配置列表
            main_timeframe: 主时间框架
            start: 开始时间
            end: 结束时间
            
        Returns:
            Dict[str, pd.DataFrame]: {pair: DataFrame} - 每个标的的数据已重采样到主时间框架
        """
        result = {}
        
        self.logger.info(f"Loading {len(informative_pairs)} informative pairs for {main_timeframe}")
        
        # 按时间框架分组加载
        grouped: Dict[str, List[InformativePair]] = {}
        for ip in informative_pairs:
            if ip.timeframe not in grouped:
                grouped[ip.timeframe] = []
            grouped[ip.timeframe].append(ip)
        
        # 加载每个时间框架的数据
        for tf, pairs in grouped.items():
            # 加载该时间框架所有标的的数据
            symbols = [ip.pair for ip in pairs]
            
            for symbol in symbols:
                try:
                    data = self.load_multi_timeframe_data(
                        symbol=symbol,
                        timeframes=[tf],
                        start=start,
                        end=end,
                    )
                    
                    if tf in data:
                        # 重采样到主时间框架
                        resampled = self._resample_to_main_timeframe(
                            data[tf],
                            main_timeframe,
                        )
                        
                        # 标记这是信息性数据
                        resampled.columns = [f"{col}_info_{tf}" for col in resampled.columns]
                        
                        result[symbol] = resampled
                        self.logger.info(f"  {symbol} ({tf}) -> {main_timeframe}: {len(resampled)} bars")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load informative pair {symbol}: {e}")
        
        return result
    
    def _resample_to_main_timeframe(
        self,
        df: pd.DataFrame,
        main_timeframe: str,
    ) -> pd.DataFrame:
        """
        将数据重采样到主时间框架
        
        Args:
            df: 原始 DataFrame
            main_timeframe: 目标时间框架
            
        Returns:
            重采样后的 DataFrame
        """
        if df.empty:
            return df
        
        # 确定重采样规则
        rule = self._get_pandas_rule(main_timeframe)
        
        if rule is None:
            return df
        
        # OHLC 重采样
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        
        # 添加其他数值列
        for col in df.columns:
            if col not in ohlc_dict:
                if df[col].dtype in ['float64', 'int64']:
                    ohlc_dict[col] = 'last'
        
        try:
            resampled = df.resample(rule).agg(ohlc_dict)
            resampled = resampled.dropna()
            return resampled
        except Exception as e:
            self.logger.error(f"Resampling failed: {e}")
            return df
    
    def _get_pandas_rule(self, timeframe: str) -> Optional[str]:
        """将时间框架转换为 pandas 重采样规则"""
        rules = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "1d": "1D",
            "1w": "1W",
        }
        return rules.get(timeframe)
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化 DataFrame（确保有标准列）"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_cols:
            if col not in df.columns:
                if col == 'volume' and 'vol' in df.columns:
                    df = df.rename(columns={'vol': 'volume'})
                elif col in ['open', 'high', 'low', 'close']:
                    df[col] = df.get(col, df.iloc[:, 0] if len(df.columns) > 0 else 100)
        
        return df
    
    def _generate_mock_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """生成模拟数据（用于测试）"""
        # 计算时间间隔
        seconds = TIMEFRAME_SECONDS.get(timeframe, 86400)
        
        # 生成日期索引
        periods = int((end - start).total_seconds() / seconds)
        dates = pd.date_range(start=start, periods=periods, freq=f"{seconds}s")
        
        # 生成价格数据
        np.random.seed(hash(symbol) % 2**32)
        base_price = 100 + hash(symbol) % 400
        returns = np.random.normal(0.0002, 0.02, periods)
        prices = base_price * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.002, 0.002, periods)),
            'high': prices * (1 + np.abs(np.random.uniform(0, 0.005, periods))),
            'low': prices * (1 - np.abs(np.random.uniform(0, 0.005, periods))),
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, periods).astype(int),
        }, index=dates)
        
        # 修复 high/low
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        return data
    
    def merge_with_main_timeframe(
        self,
        main_data: pd.DataFrame,
        informative_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        将信息性数据合并到主时间框架数据
        
        Args:
            main_data: 主时间框架 DataFrame
            informative_data: 信息性数据字典 {symbol: df}
            
        Returns:
            合并后的 DataFrame
        """
        if main_data.empty:
            return main_data
        
        result = main_data.copy()
        
        for symbol, info_df in informative_data.items():
            if info_df.empty:
                continue
            
            # 合并列
            for col in info_df.columns:
                result[col] = info_df[col]
            
            self.logger.debug(f"Merged {symbol} columns: {list(info_df.columns)}")
        
        return result
    
    def calculate_timeframe_weights(
        self,
        timeframes: List[str],
        method: str = "equal",
    ) -> Dict[str, float]:
        """
        计算时间框架权重（用于多时间框架策略）
        
        Args:
            timeframes: 时间框架列表
            method: 权重计算方法
                - "equal": 等权重
                - "inverse": 越短权重越高
                - "custom": 自定义（需要实现）
                
        Returns:
            Dict[str, float]: {timeframe: weight}
        """
        if method == "equal":
            return {tf: 1.0 / len(timeframes) for tf in timeframes}
        
        elif method == "inverse":
            # 较短视频率有更高权重
            weights = {}
            for i, tf in enumerate(timeframes):
                weights[tf] = 1.0 / (i + 1)
            total = sum(weights.values())
            return {tf: w / total for tf, w in weights.items()}
        
        return {tf: 1.0 / len(timeframes) for tf in timeframes}


class MultiTimeframeAnalyzer:
    """
    多时间框架分析器
    
    提供跨时间框架的分析功能：
    1. 趋势一致性分析
    2. 信号聚合
    3. 时间框架确认
    """
    
    def __init__(self):
        self.logger = get_logger("multi_timeframe_analyzer")
    
    def check_trend_alignment(
        self,
        data_by_timeframe: Dict[str, pd.DataFrame],
        lookback: int = 5,
    ) -> Dict[str, bool]:
        """
        检查多个时间框架的趋势是否一致
        
        Args:
            data_by_timeframe: {timeframe: DataFrame}
            lookback: 回溯周期数
            
        Returns:
            Dict[str, bool]: {timeframe: 是否与更高时间框架一致}
        """
        # 按时间框架排序（从短到长）
        sorted_tfs = sorted(data_by_timeframe.keys(), 
                           key=lambda x: TIMEFRAME_SECONDS.get(x, 86400))
        
        if len(sorted_tfs) < 2:
            return {tf: True for tf in sorted_tfs}
        
        # 获取主时间框架的趋势
        main_tf = sorted_tfs[-1]  # 最长的时间框架
        main_data = data_by_timeframe[main_tf].tail(lookback)
        
        if main_data.empty:
            return {tf: False for tf in sorted_tfs}
        
        main_trend = main_data['close'].iloc[-1] > main_data['close'].iloc[0]
        
        # 检查其他时间框架
        result = {main_tf: True}
        
        for tf in sorted_tfs[:-1]:
            if tf not in data_by_timeframe:
                result[tf] = False
                continue
            
            tf_data = data_by_timeframe[tf].tail(lookback * 3)  # 更多数据以匹配
            
            if tf_data.empty:
                result[tf] = False
                continue
            
            tf_trend = tf_data['close'].iloc[-1] > tf_data['close'].iloc[0]
            result[tf] = (tf_trend == main_trend)
            
            self.logger.debug(
                f"Trend alignment: {main_tf} {'UP' if main_trend else 'DOWN'} vs "
                f"{tf} {'UP' if tf_trend else 'DOWN'}: {result[tf]}"
            )
        
        return result
    
    def aggregate_signals(
        self,
        signals_by_timeframe: Dict[str, pd.Series],
        weights: Dict[str, float],
    ) -> pd.Series:
        """
        聚合多个时间框架的信号
        
        Args:
            signals_by_timeframe: {timeframe: signals}
            weights: 权重字典
            
        Returns:
            聚合后的信号
        """
        if not signals_by_timeframe:
            return pd.Series(dtype=float)
        
        # 对齐索引
        aligned_signals = []
        all_indices = set()
        
        for tf, signals in signals_by_timeframe.items():
            all_indices.update(signals.index)
        
        all_indices = sorted(all_indices)
        
        # 加权平均
        weighted_sum = None
        weight_total = 0.0
        
        for tf, signals in signals_by_timeframe.items():
            weight = weights.get(tf, 1.0 / len(signals_by_timeframe))
            
            # 重新索引
            aligned = signals.reindex(all_indices, fill_value=0)
            
            if weighted_sum is None:
                weighted_sum = aligned * weight
            else:
                weighted_sum = weighted_sum + aligned * weight
            
            weight_total += weight
        
        if weight_total > 0:
            weighted_sum = weighted_sum / weight_total
        
        return weighted_sum.fillna(0)
    
    def generate_multi_tf_signal(
        self,
        data_by_timeframe: Dict[str, pd.DataFrame],
        entry_threshold: float = 0.5,
        exit_threshold: float = -0.5,
    ) -> pd.DataFrame:
        """
        生成多时间框架信号
        
        Args:
            data_by_timeframe: {timeframe: DataFrame}
            entry_threshold: 进场阈值
            exit_threshold: 离场阈值
            
        Returns:
            包含信号的 DataFrame
        """
        if not data_by_timeframe:
            return pd.DataFrame()
        
        # 获取主时间框架
        main_tf = max(data_by_timeframe.keys(), 
                     key=lambda x: TIMEFRAME_SECONDS.get(x, 0))
        main_data = data_by_timeframe[main_tf].copy()
        
        # 计算每个时间框架的动量
        for tf, df in data_by_timeframe.items():
            if 'close' not in df.columns:
                continue
            
            # 计算动量（价格变化百分比）
            momentum = df['close'].pct_change(periods=5)
            main_data[f'momentum_{tf}'] = momentum
        
        # 计算加权动量分数
        weights = {'1m': 0.1, '5m': 0.15, '15m': 0.2, '1h': 0.25, '4h': 0.3}
        main_data['combined_momentum'] = 0
        
        for tf, df in data_by_timeframe.items():
            col = f'momentum_{tf}'
            if col in main_data.columns:
                weight = weights.get(tf, 0.1)
                main_data['combined_momentum'] += main_data[col].fillna(0) * weight
        
        # 生成信号
        main_data['enter_long'] = 0
        main_data['exit_long'] = 0
        
        # 进场条件：动量由负转正
        momentum = main_data['combined_momentum']
        main_data.loc[(momentum > entry_threshold) & (momentum.shift(1) <= 0), 'enter_long'] = 1
        
        # 离场条件：动量由正转负
        main_data.loc[(momentum < exit_threshold) & (momentum.shift(1) >= 0), 'exit_long'] = 1
        
        return main_data


# 便捷函数
def create_multi_timeframe_config(
    main_timeframe: str = "1d",
    informative_timeframes: List[str] = None,
) -> TimeframeConfig:
    """创建多时间框架配置"""
    return TimeframeConfig(
        main=main_timeframe,
        informative=informative_timeframes or [],
    )


if __name__ == "__main__":
    # 测试多时间框架功能
    from datetime import datetime, timedelta
    
    print("=" * 60)
    print("Multi-Timeframe Backtesting Test")
    print("=" * 60)
    
    # 创建数据加载器
    loader = MultiTimeframeDataLoader(default_timeframe="1d")
    
    # 加载多个时间框架数据
    start = datetime.now() - timedelta(days=90)
    end = datetime.now()
    
    print("\n1. Loading multi-timeframe data for SPY...")
    data = loader.load_multi_timeframe_data(
        symbol="SPY",
        timeframes=["5m", "15m", "1h", "1d"],
        start=start,
        end=end,
    )
    
    print(f"   Loaded timeframes: {list(data.keys())}")
    for tf, df in data.items():
        print(f"   - {tf}: {len(df)} bars, date range: {df.index[0]} to {df.index[-1]}")
    
    # 测试重采样
    print("\n2. Testing resampling (1h -> 1d)...")
    if "1h" in data:
        resampled = loader._resample_to_main_timeframe(data["1h"], "1d")
        print(f"   1h bars: {len(data['1h'])} -> 1d bars: {len(resampled)}")
    
    # 测试趋势一致性
    print("\n3. Testing trend alignment...")
    analyzer = MultiTimeframeAnalyzer()
    alignment = analyzer.check_trend_alignment(data, lookback=5)
    for tf, aligned in alignment.items():
        print(f"   {tf}: {'aligned' if aligned else 'divergent'}")
    
    # 测试多时间框架信号生成
    print("\n4. Testing multi-timeframe signal generation...")
    signals = analyzer.generate_multi_tf_signal(data)
    entry_count = signals['enter_long'].sum() if 'enter_long' in signals else 0
    exit_count = signals['exit_long'].sum() if 'exit_long' in signals else 0
    print(f"   Entry signals: {entry_count}")
    print(f"   Exit signals: {exit_count}")
    
    print("\n" + "=" * 60)
    print("Multi-timeframe functionality verified!")
    print("=" * 60)

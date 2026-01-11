"""
特征计算引擎
统一入口：compute_features(symbols, start, end, timeframe, feature_set) -> FeatureFrame
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.data.storage.parquet_store import ParquetDataFrameStore
from src.data.storage.versioning import generate_data_version, get_code_version
from src.config.settings import get_settings
from .registry import FeatureRegistry, get_feature_registry


class FeatureEngine:
    """特征计算引擎"""
    
    def __init__(
        self,
        store: Optional[ParquetDataFrameStore] = None,
        registry: Optional[FeatureRegistry] = None,
    ):
        settings = get_settings()
        self.store = store or ParquetDataFrameStore(settings.storage.data_dir)
        self.registry = registry or get_feature_registry()
        self.logger = get_logger("feature_engine")
    
    def compute_features(
        self,
        symbols: List[str],
        start: date,
        end: date,
        timeframe: str,
        feature_set: str = "core_v1",
        data_version: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        计算特征
        
        Args:
            symbols: 标的列表
            start: 起始日期
            end: 结束日期
            timeframe: 时间周期
            feature_set: 特征集名称（core_v1/intraday_v1等）
            data_version: 数据版本（如果为None，自动生成）
            
        Returns:
            FeatureFrame（index: (ts_utc, symbol), columns: feature_*）
        """
        # 生成数据版本
        if data_version is None:
            data_version = generate_data_version(
                source="polygon_api",
                timeframe=timeframe,
                symbols=symbols,
            )
        
        # 加载bars数据
        bars_list = []
        for symbol in symbols:
            bars = self.store.read_dataset(
                dataset="bars",
                partition={"symbol": symbol, "timeframe": timeframe},
                data_version=data_version,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
            )
            if not bars.empty:
                bars['symbol'] = symbol
                bars_list.append(bars)
        
        if not bars_list:
            self.logger.warning("没有可用的bars数据")
            return pd.DataFrame()
        
        # 合并所有标的的数据
        all_bars = pd.concat(bars_list)
        
        # 根据feature_set选择特征计算函数
        feature_funcs = self._get_feature_functions(feature_set)
        
        # 计算特征
        features_list = []
        for symbol in symbols:
            symbol_bars = all_bars[all_bars['symbol'] == symbol].copy()
            if symbol_bars.empty:
                continue
            
            # 计算每个特征
            symbol_features = pd.DataFrame(index=symbol_bars.index)
            symbol_features['symbol'] = symbol
            
            for feature_name, func in feature_funcs.items():
                try:
                    feature_values = func(symbol_bars)
                    if isinstance(feature_values, pd.Series):
                        symbol_features[f'f_{feature_name}'] = feature_values
                    else:
                        self.logger.warning(f"特征 {feature_name} 返回的不是Series")
                except Exception as e:
                    self.logger.error(f"计算特征 {feature_name} 失败: {e}")
                    symbol_features[f'f_{feature_name}'] = np.nan
            
            features_list.append(symbol_features)
        
        if not features_list:
            return pd.DataFrame()
        
        # 合并所有特征
        features_df = pd.concat(features_list)
        
        # 设置多级索引 (ts_utc, symbol)
        if 'symbol' in features_df.columns:
            features_df = features_df.set_index(['symbol'], append=True)
        
        # 保存特征
        code_version = get_code_version()
        config_hash = self._hash_feature_config(feature_set, feature_funcs)
        
        # 保存到Parquet
        for symbol in symbols:
            symbol_features = features_df.xs(symbol, level='symbol', drop_level=False)
            if not symbol_features.empty:
                self.store.write_dataset(
                    dataset="features",
                    df=symbol_features.reset_index(level='symbol'),
                    partition={"symbol": symbol, "timeframe": timeframe, "feature_set": feature_set},
                    data_version=data_version,
                )
        
        return features_df
    
    def _get_feature_functions(self, feature_set: str) -> Dict[str, callable]:
        """
        获取特征计算函数字典
        
        Args:
            feature_set: 特征集名称
            
        Returns:
            {feature_name: function} 字典
        """
        if feature_set == "core_v1":
            return self._get_core_features()
        elif feature_set == "intraday_v1":
            return self._get_intraday_features()
        else:
            self.logger.warning(f"未知的特征集: {feature_set}，使用core_v1")
            return self._get_core_features()
    
    def _get_core_features(self) -> Dict[str, callable]:
        """获取核心特征函数"""
        from src.factors.technical.momentum import MomentumFactor
        from src.factors.technical.volatility import VolatilityFactor
        
        features = {}
        
        # 收益率
        def returns(bars: pd.DataFrame) -> pd.Series:
            return bars['close'].pct_change()
        features['returns'] = returns
        
        # MA
        def ma_20(bars: pd.DataFrame) -> pd.Series:
            return bars['close'].rolling(20).mean()
        features['ma_20'] = ma_20
        
        def ma_50(bars: pd.DataFrame) -> pd.Series:
            return bars['close'].rolling(50).mean()
        features['ma_50'] = ma_50
        
        # ATR
        def atr(bars: pd.DataFrame) -> pd.Series:
            high_low = bars['high'] - bars['low']
            high_close = np.abs(bars['high'] - bars['close'].shift())
            low_close = np.abs(bars['low'] - bars['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(14).mean()
        features['atr'] = atr
        
        # 相对强弱（vs SPY，简化实现）
        def relative_strength(bars: pd.DataFrame) -> pd.Series:
            # 简化：使用自身收益率的滚动均值
            return bars['close'].pct_change().rolling(20).mean()
        features['relative_strength'] = relative_strength
        
        return features
    
    def _get_intraday_features(self) -> Dict[str, callable]:
        """获取日内特征函数"""
        features = self._get_core_features()
        
        # 添加日内特征
        def gap(bars: pd.DataFrame) -> pd.Series:
            return (bars['open'] - bars['close'].shift(1)) / bars['close'].shift(1)
        features['gap'] = gap
        
        return features
    
    def _hash_feature_config(self, feature_set: str, feature_funcs: Dict) -> str:
        """生成特征配置hash"""
        import hashlib
        import json
        
        config = {
            "feature_set": feature_set,
            "features": sorted(feature_funcs.keys()),
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

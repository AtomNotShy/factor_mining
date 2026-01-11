"""
因子计算基类
定义统一的因子计算接口和规范
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from src.utils.logger import get_logger


@dataclass
class FactorMetadata:
    """因子元数据"""
    name: str  # 因子名称
    description: str  # 因子描述
    category: str  # 因子分类（technical, fundamental, alternative）
    sub_category: str  # 子分类
    calculation_window: int  # 计算窗口
    update_frequency: str  # 更新频率
    data_requirements: List[str]  # 数据需求
    version: str = "1.0.0"  # 版本号
    author: str = ""  # 作者
    created_date: datetime = None  # 创建日期


class BaseFactor(ABC):
    """因子计算基类"""
    
    def __init__(self, metadata: FactorMetadata):
        self.metadata = metadata
        self.logger = get_logger(f"factor.{metadata.name}")
        self.cache = {}  # 简单缓存
        
    @property
    def name(self) -> str:
        """因子名称"""
        return self.metadata.name
    
    @property
    def category(self) -> str:
        """因子分类"""
        return self.metadata.category
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算因子值
        
        Args:
            data: 输入数据，包含OHLCV等
            **kwargs: 其他参数
            
        Returns:
            因子值序列，索引为时间戳
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            是否有效
        """
        if data.empty:
            self.logger.warning("输入数据为空")
            return False
        
        # 检查必需的列
        required_cols = self.metadata.data_requirements
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.warning(f"缺少必需的列: {missing_cols}")
            return False
        
        # 检查数据长度
        if len(data) < self.metadata.calculation_window:
            self.logger.warning(f"数据长度不足，需要至少 {self.metadata.calculation_window} 条")
            return False
        
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        # 排序
        if not processed_data.index.is_monotonic_increasing:
            processed_data = processed_data.sort_index()
        
        # 处理缺失值
        processed_data = self._handle_missing_values(processed_data)
        
        # 异常值处理
        processed_data = self._handle_outliers(processed_data)
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 前向填充（使用新API避免弃用警告）
        data = data.ffill()
        
        # 如果仍有缺失值，使用后向填充
        data = data.bfill()
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """处理异常值（使用Z-score方法）"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > threshold
                
                if outliers.any():
                    # 使用中位数替换异常值
                    data.loc[outliers, col] = data[col].median()
                    self.logger.debug(f"处理了 {outliers.sum()} 个异常值在列 {col}")
        
        return data
    
    def calculate_with_validation(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        带验证的因子计算
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            因子值序列
        """
        try:
            # 验证数据
            if not self.validate_data(data):
                return pd.Series(dtype=float)
            
            # 预处理数据
            processed_data = self.preprocess_data(data)
            
            # 计算因子
            factor_values = self.calculate(processed_data, **kwargs)
            
            # 后处理
            factor_values = self.postprocess_factor(factor_values)
            
            self.logger.debug(f"成功计算因子 {self.name}, 得到 {len(factor_values)} 个值")
            return factor_values
            
        except Exception as e:
            self.logger.error(f"计算因子 {self.name} 失败: {e}")
            return pd.Series(dtype=float)
    
    def postprocess_factor(self, factor_values: pd.Series) -> pd.Series:
        """
        因子后处理
        
        Args:
            factor_values: 原始因子值
            
        Returns:
            处理后的因子值
        """
        # 去除无穷大值
        factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
        
        # 标准化（可选）
        if hasattr(self, 'standardize') and self.standardize:
            factor_values = self._standardize(factor_values)
        
        return factor_values
    
    def _standardize(self, series: pd.Series, method: str = 'zscore') -> pd.Series:
        """标准化因子值"""
        if method == 'zscore':
            return (series - series.mean()) / series.std()
        elif method == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif method == 'rank':
            return series.rank(pct=True)
        else:
            return series
    
    def get_factor_info(self) -> Dict[str, Any]:
        """获取因子信息"""
        return {
            'name': self.metadata.name,
            'description': self.metadata.description,
            'category': self.metadata.category,
            'sub_category': self.metadata.sub_category,
            'calculation_window': self.metadata.calculation_window,
            'update_frequency': self.metadata.update_frequency,
            'data_requirements': self.metadata.data_requirements,
            'version': self.metadata.version,
            'author': self.metadata.author,
            'created_date': self.metadata.created_date
        }
    
    def __str__(self) -> str:
        return f"Factor({self.metadata.name})"
    
    def __repr__(self) -> str:
        return f"Factor(name='{self.metadata.name}', category='{self.metadata.category}')"


class TechnicalFactor(BaseFactor):
    """技术因子基类"""
    
    def __init__(self, metadata: FactorMetadata):
        metadata.category = "technical"
        super().__init__(metadata)


class FundamentalFactor(BaseFactor):
    """基本面因子基类"""
    
    def __init__(self, metadata: FactorMetadata):
        metadata.category = "fundamental"
        super().__init__(metadata)


class AlternativeFactor(BaseFactor):
    """另类因子基类"""
    
    def __init__(self, metadata: FactorMetadata):
        metadata.category = "alternative"
        super().__init__(metadata)


class FactorRegistry:
    """因子注册表"""
    
    def __init__(self):
        self._factors = {}
        self.logger = get_logger("factor_registry")
    
    def register(self, factor: BaseFactor):
        """注册因子"""
        self._factors[factor.name] = factor
        self.logger.info(f"注册因子: {factor.name}")
    
    def get_factor(self, name: str) -> Optional[BaseFactor]:
        """获取因子"""
        return self._factors.get(name)
    
    def list_factors(self, category: str = None) -> List[str]:
        """列出因子"""
        if category:
            return [name for name, factor in self._factors.items() 
                   if factor.category == category]
        return list(self._factors.keys())
    
    def get_factors_by_category(self, category: str) -> Dict[str, BaseFactor]:
        """按分类获取因子"""
        return {name: factor for name, factor in self._factors.items() 
                if factor.category == category}


# 全局因子注册表
factor_registry = FactorRegistry() 
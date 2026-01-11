"""
特征注册表
管理特征的元数据和可追溯性
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from src.utils.logger import get_logger
from src.data.storage.db_store import DatabaseStore


@dataclass
class FeatureMetadata:
    """特征元数据"""
    feature_name: str
    description: str
    inputs: Dict[str, Any]  # 依赖字段
    params: Dict[str, Any]  # 参数
    code_ref: str  # 函数/模块路径
    code_version: str


class FeatureRegistry:
    """特征注册表管理器"""
    
    def __init__(self, db_store: Optional[DatabaseStore] = None):
        self.db_store = db_store or DatabaseStore()
        self.logger = get_logger("feature_registry")
        self._local_registry: Dict[str, FeatureMetadata] = {}
    
    def register(
        self,
        feature_name: str,
        description: str,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        code_ref: str,
        code_version: Optional[str] = None,
    ) -> FeatureMetadata:
        """
        注册特征
        
        Args:
            feature_name: 特征名称
            description: 描述
            inputs: 依赖字段
            params: 参数
            code_ref: 代码引用路径
            code_version: 代码版本（如果为None，自动获取）
            
        Returns:
            FeatureMetadata
        """
        from src.data.storage.versioning import get_code_version
        
        if code_version is None:
            code_version = get_code_version()
        
        metadata = FeatureMetadata(
            feature_name=feature_name,
            description=description,
            inputs=inputs,
            params=params,
            code_ref=code_ref,
            code_version=code_version,
        )
        
        # 保存到本地缓存
        self._local_registry[feature_name] = metadata
        
        # 保存到数据库
        try:
            self.db_store.register_feature(
                feature_name=feature_name,
                description=description,
                inputs=inputs,
                params=params,
                code_ref=code_ref,
                code_version=code_version,
            )
        except Exception as e:
            self.logger.warning(f"保存特征注册信息到数据库失败: {e}")
        
        self.logger.info(f"注册特征: {feature_name}")
        return metadata
    
    def get(self, feature_name: str) -> Optional[FeatureMetadata]:
        """获取特征元数据"""
        # 先查本地缓存
        if feature_name in self._local_registry:
            return self._local_registry[feature_name]
        
        # 查数据库
        try:
            db_feature = self.db_store.get_feature(feature_name)
            if db_feature:
                metadata = FeatureMetadata(
                    feature_name=db_feature.feature_name,
                    description=db_feature.description or "",
                    inputs=db_feature.inputs or {},
                    params=db_feature.params or {},
                    code_ref=db_feature.code_ref or "",
                    code_version=db_feature.code_version,
                )
                self._local_registry[feature_name] = metadata
                return metadata
        except Exception as e:
            self.logger.warning(f"从数据库获取特征信息失败: {e}")
        
        return None
    
    def list_features(self) -> List[str]:
        """列出所有已注册的特征"""
        return list(self._local_registry.keys())


# 全局特征注册表
_global_registry: Optional[FeatureRegistry] = None


def get_feature_registry() -> FeatureRegistry:
    """获取全局特征注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = FeatureRegistry()
    return _global_registry

"""
数据适配器工厂
统一创建和管理数据适配器实例
"""

from typing import Dict, Optional, Type
import logging
from datetime import datetime

from src.data.adapter import DataAdapter, HistoricalDataAdapter, LiveDataAdapter, DataRequest
from src.data.storage.parquet_store import ParquetDataFrameStore
from src.data.adapter.ib_adapter import IBLiveAdapter
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("adapter_factory")


class AdapterFactory:
    """数据适配器工厂"""
    
    _adapters: Dict[str, DataAdapter] = {}
    _config = None
    
    @classmethod
    def get_config(cls):
        """获取配置"""
        if cls._config is None:
            cls._config = get_settings()
        return cls._config
    
    @classmethod
    def create_historical_adapter(cls, config: Optional[Dict] = None) -> HistoricalDataAdapter:
        """创建历史数据适配器"""
        config = config or {}
        settings = cls.get_config()
        
        # 创建存储后端
        storage = ParquetDataFrameStore(settings.storage.data_dir)
        
        # 创建适配器
        adapter_config = {
            "cache_size": config.get("cache_size", 1000),
            "cache_ttl": config.get("cache_ttl", 3600),
        }
        
        adapter = HistoricalDataAdapter(storage, adapter_config)
        logger.info("创建 HistoricalDataAdapter")
        
        return adapter
    
    @classmethod
    def create_live_adapter(cls, adapter_type: str = "ib", config: Optional[Dict] = None) -> LiveDataAdapter:
        """创建实时数据适配器"""
        config = config or {}
        settings = cls.get_config()
        
        if adapter_type.lower() == "ib":
            # IB 适配器配置
            ib_config = {
                "host": settings.ib.host,
                "port": settings.ib.port,
                "client_id": settings.ib.client_id,
                "reconnect_delay": config.get("reconnect_delay", 5),
                "heartbeat_interval": config.get("heartbeat_interval", 30),
                "cache_ttl": config.get("cache_ttl", 1),
            }
            adapter = IBLiveAdapter(ib_config)
            logger.info("创建 IBLiveAdapter")
        else:
            raise ValueError(f"不支持的实时适配器类型: {adapter_type}")
        
        return adapter
    
    @classmethod
    def get_adapter(cls, mode: str = "backtest", adapter_type: str = "ib") -> DataAdapter:
        """获取适配器（单例模式）"""
        key = f"{mode}_{adapter_type}"
        
        if key not in cls._adapters:
            if mode == "backtest":
                adapter = cls.create_historical_adapter()
            elif mode == "live":
                adapter = cls.create_live_adapter(adapter_type)
            else:
                raise ValueError(f"不支持的适配器模式: {mode}")
            
            cls._adapters[key] = adapter
        
        return cls._adapters[key]
    
    @classmethod
    async def get_data(
        cls,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1d",
        mode: str = "backtest",
        adapter_type: str = "ib",
        include_previous: int = 0,
    ):
        """统一数据获取接口"""
        adapter = cls.get_adapter(mode, adapter_type)
        
        request = DataRequest(
            symbol=symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            mode=mode,
            include_previous=include_previous,
        )
        
        return await adapter.get_data(request)
    
    @classmethod
    def clear_adapters(cls):
        """清理所有适配器"""
        for adapter in cls._adapters.values():
            if hasattr(adapter, 'disconnect'):
                # 异步方法，这里只记录
                logger.info(f"清理适配器: {type(adapter).__name__}")
        cls._adapters.clear()
    
    @classmethod
    def get_adapter_stats(cls) -> Dict[str, Dict]:
        """获取适配器统计信息"""
        stats = {}
        for key, adapter in cls._adapters.items():
            adapter_stats = {
                "type": type(adapter).__name__,
                "config": adapter.config if hasattr(adapter, 'config') else {},
            }
            
            # 添加指标信息
            if hasattr(adapter, 'metrics'):
                adapter_stats["metrics"] = {
                    "cache_hits": adapter.metrics.cache_hits,
                    "cache_misses": adapter.metrics.cache_misses,
                    "cache_hit_rate": adapter.metrics.cache_hit_rate,
                    "errors": adapter.metrics.errors,
                    "total_requests": adapter.metrics.total_requests,
                }
            
            stats[key] = adapter_stats
        
        return stats


# 全局工厂实例
adapter_factory = AdapterFactory()
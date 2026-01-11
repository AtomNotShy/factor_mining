"""
数据存储模型
定义数据库表结构（用于元数据和运行索引）
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


class BacktestRun(Base):
    """回测运行记录表"""
    __tablename__ = "backtest_runs"
    
    run_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at_utc = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # 策略和配置
    strategy_id = Column(String(100), nullable=False, index=True)
    universe_id = Column(String(100))  # 关联 universe 快照规则
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # 成本模型配置（JSON）
    cost_model = Column(JSON, nullable=False)
    
    # 结果路径和指标
    results_path = Column(String(500))  # 指向 Parquet/报告路径
    metrics = Column(JSON)  # 汇总指标
    pass_gate = Column(Boolean, default=False, index=True)
    
    # 版本化字段
    data_version = Column(String(100), nullable=False, index=True)
    code_version = Column(String(100), nullable=False)
    config_hash = Column(String(64), nullable=False, index=True)
    
    # 索引
    __table_args__ = (
        Index("idx_backtest_runs_strategy_date", "strategy_id", "start_date", "end_date"),
        Index("idx_backtest_runs_config_hash", "config_hash"),
    )


class DataIngestion(Base):
    """数据拉取批次记录"""
    __tablename__ = "data_ingestions"
    
    ingestion_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at_utc = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # 数据源信息
    source = Column(String(50), nullable=False)  # polygon_api, etc.
    data_version = Column(String(100), nullable=False, unique=True, index=True)
    symbols = Column(JSON)  # 拉取的标的列表
    timeframes = Column(JSON)  # 时间周期列表
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    
    # 元数据（使用meta_data避免与SQLAlchemy的metadata属性冲突）
    meta_data = Column(JSON, name='metadata')  # 数据库列名仍为metadata
    
    # 索引
    __table_args__ = (
        Index("idx_data_ingestions_source_date", "source", "start_date", "end_date"),
    )


class FeatureRegistry(Base):
    """特征注册表"""
    __tablename__ = "feature_registry"
    
    feature_name = Column(String(200), primary_key=True)
    description = Column(Text)
    inputs = Column(JSON)  # 依赖字段
    params = Column(JSON)  # 参数
    code_ref = Column(String(500))  # 函数/模块路径
    created_at_utc = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    code_version = Column(String(100), nullable=False)
    
    # 索引
    __table_args__ = (
        Index("idx_feature_registry_code_version", "code_version"),
    )

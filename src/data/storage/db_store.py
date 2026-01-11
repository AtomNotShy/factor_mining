"""
数据库存储层
用于存储元数据和运行索引（backtest_runs, data_ingestions等）
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager

from src.config.settings import get_settings
from src.utils.logger import get_logger
from .models import Base, BacktestRun, DataIngestion, FeatureRegistry


class DatabaseStore:
    """数据库存储管理器"""
    
    def __init__(self, database_url: Optional[str] = None, init_tables: bool = True):
        settings = get_settings()
        if database_url is None:
            database_url = settings.database.url
        
        self.database_url = database_url
        self.logger = get_logger("db_store")
        self.available: bool = True
        self._init_error: Optional[Exception] = None
        
        # 创建引擎（使用NullPool避免连接池问题）
        self.engine = create_engine(
            database_url,
            poolclass=NullPool,
            echo=False,
        )
        
        # 创建session工厂
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # 初始化表结构
        if init_tables:
            self._init_tables()
    
    def _init_tables(self):
        """初始化数据库表"""
        try:
            Base.metadata.create_all(self.engine)
            self.logger.info("数据库表初始化完成")
        except Exception as e:
            # 不要让整个 API 因为 DB 不可用直接崩溃；保留可观测错误，按需在调用时再报错。
            self.available = False
            self._init_error = e
            self.logger.error(f"数据库表初始化失败（将以降级模式继续运行）: {e}")
    
    @contextmanager
    def get_session(self):
        """获取数据库session（上下文管理器）"""
        if not self.available:
            raise RuntimeError(
                f"Database is unavailable (database_url={self.database_url}). "
                f"Original error: {self._init_error}"
            )
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def save_backtest_run(
        self,
        run_id: str,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        cost_model: Dict[str, Any],
        data_version: str,
        code_version: str,
        config_hash: str,
        results_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        pass_gate: bool = False,
        universe_id: Optional[str] = None,
    ) -> BacktestRun:
        """保存回测运行记录"""
        with self.get_session() as session:
            run = BacktestRun(
                run_id=run_id,
                strategy_id=strategy_id,
                universe_id=universe_id,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                cost_model=cost_model,
                results_path=results_path,
                metrics=metrics,
                pass_gate=pass_gate,
                data_version=data_version,
                code_version=code_version,
                config_hash=config_hash,
            )
            session.add(run)
            session.flush()
            self.logger.info(f"保存回测运行记录: {run_id}")
            return run
    
    def get_backtest_run(self, run_id: str) -> Optional[BacktestRun]:
        """获取回测运行记录"""
        with self.get_session() as session:
            return session.query(BacktestRun).filter_by(run_id=run_id).first()
    
    def list_backtest_runs(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BacktestRun]:
        """列出回测运行记录"""
        with self.get_session() as session:
            query = session.query(BacktestRun)
            if strategy_id:
                query = query.filter_by(strategy_id=strategy_id)
            query = query.order_by(BacktestRun.created_at_utc.desc())
            return query.limit(limit).offset(offset).all()
    
    def save_data_ingestion(
        self,
        data_version: str,
        source: str,
        symbols: List[str],
        timeframes: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataIngestion:
        """保存数据拉取批次记录"""
        with self.get_session() as session:
            ingestion = DataIngestion(
                data_version=data_version,
                source=source,
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                meta_data=metadata or {},  # 使用meta_data避免与SQLAlchemy的metadata冲突
            )
            session.add(ingestion)
            session.flush()
            self.logger.info(f"保存数据拉取批次: {data_version}")
            return ingestion
    
    def get_data_ingestion(self, data_version: str) -> Optional[DataIngestion]:
        """获取数据拉取批次记录"""
        with self.get_session() as session:
            return session.query(DataIngestion).filter_by(data_version=data_version).first()
    
    def register_feature(
        self,
        feature_name: str,
        description: str,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        code_ref: str,
        code_version: str,
    ) -> FeatureRegistry:
        """注册特征"""
        with self.get_session() as session:
            # 检查是否已存在
            existing = session.query(FeatureRegistry).filter_by(feature_name=feature_name).first()
            if existing:
                # 更新
                existing.description = description
                existing.inputs = inputs
                existing.params = params
                existing.code_ref = code_ref
                existing.code_version = code_version
                feature = existing
            else:
                # 新建
                feature = FeatureRegistry(
                    feature_name=feature_name,
                    description=description,
                    inputs=inputs,
                    params=params,
                    code_ref=code_ref,
                    code_version=code_version,
                )
                session.add(feature)
            session.flush()
            self.logger.info(f"注册特征: {feature_name}")
            return feature
    
    def get_feature(self, feature_name: str) -> Optional[FeatureRegistry]:
        """获取特征注册信息"""
        with self.get_session() as session:
            return session.query(FeatureRegistry).filter_by(feature_name=feature_name).first()

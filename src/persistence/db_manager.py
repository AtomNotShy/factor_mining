"""
数据库管理类
负责初始化连接并在任务结束后保存结果
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.persistence.models import Base, BacktestRunModel, TradeModel, OrderFillModel
from src.utils.logger import get_logger

class DatabaseManager:
    """
    持久化管理器
    """

    def __init__(self, db_url: str = "sqlite:///./data/trades.db"):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = get_logger("persistence.db")
        self._init_db()

    def _init_db(self):
        """初始化表结构"""
        Base.metadata.create_all(self.engine)
        self.logger.info("数据库初始化完成")

    def save_backtest_run(self, results: Dict[str, Any], config: Any):
        """
        保存回测结果
        """
        session = self.Session()
        try:
            run_id = results.get("run_id", "unnamed")
            
            # 1. 创建 Run 记录
            run_model = BacktestRunModel(
                id=run_id,
                strategy_id=config.strategy.name,
                start_date=datetime.combine(results.get("start_date", datetime.now().date()), datetime.min.time()),
                end_date=datetime.combine(results.get("end_date", datetime.now().date()), datetime.min.time()),
                initial_capital=results.get("initial_capital"),
                final_equity=results.get("final_equity"),
                total_return=results.get("total_return"),
                sharpe_ratio=results.get("sharpe_ratio"),
                max_drawdown=results.get("max_drawdown"),
                config_json=config.dict() # 保存 Pydantic 模型
            )
            session.add(run_model)

            # 2. 保存成交记录 (Fills)
            fills = results.get("fills", [])
            for fill in fills:
                fill_model = OrderFillModel(
                    id=fill.fill_id,
                    run_id=run_id,
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    side=fill.side.value,
                    price=fill.price,
                    qty=fill.qty,
                    fee=fill.fee,
                    ts_fill=fill.ts_fill_utc
                )
                session.add(fill_model)
            
            # 3. 可以在此处增加简单的 Trade 聚合逻辑 (开仓/平仓匹配)
            # 为了简化，此版本先存 Fills

            session.commit()
            self.logger.info(f"回测记录已持久化: {run_id}")
        except Exception as e:
            session.rollback()
            self.logger.error(f"持久化失败: {e}")
        finally:
            session.close()

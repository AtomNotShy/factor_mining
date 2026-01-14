"""
数据库模型定义
使用 SQLAlchemy ORM
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class BacktestRunModel(Base):
    """回测运行记录"""
    __tablename__ = 'backtest_runs'

    id = Column(String(36), primary_key=True) # run_id (UUID)
    strategy_id = Column(String(100), nullable=False)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    final_equity = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    config_json = Column(JSON) # 完整的配置快照
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联
    trades = relationship("TradeModel", back_populates="run")

class TradeModel(Base):
    """交易记录 (开仓到平仓的完整周期)"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey('backtest_runs.id'))
    symbol = Column(String(20), nullable=False)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    entry_price = Column(Float)
    exit_price = Column(Float)
    qty = Column(Float)
    pnl = Column(Float)
    pnl_ratio = Column(Float)
    
    # 关联
    run = relationship("BacktestRunModel", back_populates="trades")
    # 一个交易可能包含多个成交记录 (Fills)
    fills = relationship("OrderFillModel", back_populates="trade")

class OrderFillModel(Base):
    """订单成交细节 (Fills)"""
    __tablename__ = 'order_fills'

    id = Column(String(36), primary_key=True) # fill_id
    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=True)
    run_id = Column(String(36), ForeignKey('backtest_runs.id'))
    order_id = Column(String(36))
    symbol = Column(String(20))
    side = Column(String(10)) # BUY/SELL
    price = Column(Float)
    qty = Column(Float)
    fee = Column(Float)
    ts_fill = Column(DateTime)

    trade = relationship("TradeModel", back_populates="fills")

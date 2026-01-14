"""
配置架构定义
使用 Pydantic 进行强类型验证
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class TradingConfig(BaseModel):
    """交易相关配置"""
    stake_amount: float = Field(..., description="初始投入金额/单笔投入金额")
    max_open_trades: int = Field(5, description="最大同时持仓数")
    timeframe: str = Field("1d", description="基准时间框架")
    dry_run: bool = Field(True, description="是否为离线/模拟运行")

class BrokerConfig(BaseModel):
    """券商相关配置"""
    name: str = Field("simulated", description="券商名称 (simulated, ib, etc.)")
    commission: float = Field(0.001, description="手续费率")
    slippage: float = Field(0.0005, description="滑点率")
    account_id: Optional[str] = None

class RiskConfig(BaseModel):
    """风控相关配置"""
    stoploss: Optional[float] = Field(None, description="固定止损百分比 (如 -0.1 表示 10%)")
    trailing_stop: bool = Field(False, description="是否开启移动止损")
    trailing_stop_positive: Optional[float] = Field(None, description="盈利超过 offset 后的移动止损点")
    trailing_stop_positive_offset: Optional[float] = Field(0.0, description="触发 trailing_stop_positive 的收益阈值")
    roi_table: Dict[int, float] = Field(default_factory=dict, description="ROI (时间(分钟)->收益) 对照表")
    max_open_trades: Optional[int] = Field(None, description="最大同时持仓数 (覆盖 trading 里的定义)")

class DataConfig(BaseModel):
    """数据相关配置"""
    datadir: str = Field("./data", description="数据存储目录")
    startup_candle_count: int = Field(200, description="预热数据量 (用于计算 MA 等指标)")
    universe: List[str] = Field(default_factory=list, description="股票池")
    benchmark_symbol: str = Field("SPY", description="基准指标标的")

class StrategyConfigSchema(BaseModel):
    """策略相关配置"""
    name: str = Field(..., description="策略类名")
    params: Dict[str, Any] = Field(default_factory=dict, description="针对策略的自定义参数")

class ConfigSchema(BaseModel):
    """全局配置架构"""
    trading: TradingConfig
    broker: BrokerConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data: DataConfig
    strategy: StrategyConfigSchema
    
    @classmethod
    def from_yaml(cls, path: str):
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

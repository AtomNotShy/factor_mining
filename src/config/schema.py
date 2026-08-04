"""
配置架构定义
使用 Pydantic 进行强类型验证
"""

from typing import List, Dict, Optional, Any, Literal
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
    timeframes: List[str] = Field(default_factory=list, description="时间框架列表(主时间框架在前)")
    auto_download: bool = Field(True, description="缺数据时是否自动下载")
    provider: str = Field("auto", description="数据源偏好 (auto/ib/polygon/alpaca)")


class TimeConfigSchema(BaseModel):
    """回测时间框架配置"""
    signal_timeframe: Optional[str] = Field(default=None, description="信号时间框架")
    execution_timeframe: Optional[str] = Field(default=None, description="执行时间框架")
    informative_timeframes: List[str] = Field(default_factory=list, description="辅助时间框架")
    detail_timeframe: Optional[str] = Field(default=None, description="明细时间框架")
    clock_mode: Optional[Literal["daily", "bar", "hybrid"]] = Field(default=None, description="时钟模式")
    execution_time: Optional[str] = Field(default=None, description="执行时间(HH:MM)")

class StrategyConfigSchema(BaseModel):
    """策略相关配置"""
    name: str = Field(..., description="策略类名")
    params: Dict[str, Any] = Field(default_factory=dict, description="针对策略的自定义参数")

class TimeRangeConfig(BaseModel):
    """回测时间范围配置"""
    start: str = Field("", description="开始日期 YYYY-MM-DD")
    end: str = Field("", description="结束日期 YYYY-MM-DD")

class ConfigSchema(BaseModel):
    """全局配置架构"""
    trading: TradingConfig
    broker: BrokerConfig
    risk: RiskConfig
    data: DataConfig
    strategy: StrategyConfigSchema
    time: Optional[TimeConfigSchema] = Field(default=None, description="时间框架配置")
    time_range: Optional[TimeRangeConfig] = Field(default=None, description="回测时间范围")
    
    @classmethod
    def from_yaml(cls, path: str):
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

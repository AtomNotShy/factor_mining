"""
配置管理模块
使用Pydantic进行配置验证和类型检查
"""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import List, Dict, Any, Optional
import os


def _running_in_docker() -> bool:
    """
    Best-effort Docker detection.

    - In docker containers, /.dockerenv usually exists.
    - In k8s/other container runtimes it may not; allow explicit override.
    """
    if os.getenv("RUNNING_IN_DOCKER", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return os.path.exists("/.dockerenv")


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    host: str = Field(default="localhost", description="数据库主机")
    port: int = Field(default=5432, description="数据库端口")
    username: str = Field(default="postgres", description="数据库用户名")
    password: str = Field(default="password", description="数据库密码")
    database: str = Field(default="factor_mining", description="数据库名称")
    
    @property
    def url(self) -> str:
        # docker-compose 通常使用服务名 `postgres` 作为主机名；
        # 但在本机直接运行（uvicorn/python）时，这个主机名无法解析，会导致启动阶段报错。
        host = self.host
        if host == "postgres" and not _running_in_docker():
            host = "localhost"
        return f"postgresql://{self.username}:{self.password}@{host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis配置"""
    host: str = Field(default="localhost", description="Redis主机")
    port: int = Field(default=6379, description="Redis端口")
    password: Optional[str] = Field(default=None, description="Redis密码")
    db: int = Field(default=0, description="Redis数据库索引")
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"


class InfluxDBSettings(BaseSettings):
    """InfluxDB配置"""
    url: str = Field(default="http://localhost:8086", description="InfluxDB URL")
    token: Optional[str] = Field(default=None, description="认证Token")
    org: str = Field(default="factor-mining", description="组织名称")
    bucket: str = Field(default="market-data", description="存储桶名称")
    
    class Config:
        env_prefix = "INFLUXDB_"


class StorageSettings(BaseSettings):
    """本地存储配置"""
    data_dir: str = Field(default="data", description="本地数据目录（相对或绝对路径）")

    class Config:
        env_prefix = "STORAGE_"


class ExchangeSettings(BaseSettings):
    """交易所API配置"""
    binance_api_key: Optional[str] = Field(default=None, description="币安API密钥")
    binance_secret: Optional[str] = Field(default=None, description="币安API密钥")
    okx_api_key: Optional[str] = Field(default=None, description="OKX API密钥")
    okx_secret: Optional[str] = Field(default=None, description="OKX API密钥")
    okx_passphrase: Optional[str] = Field(default=None, description="OKX API密码")
    
    class Config:
        env_prefix = "EXCHANGE_"


class PolygonSettings(BaseSettings):
    """Polygon.io API配置（美股/ETF等）"""
    api_key: Optional[str] = Field(default=None, description="Polygon API Key")
    base_url: str = Field(default="https://api.polygon.io", description="Polygon API Base URL")
    adjusted: bool = Field(default=True, description="是否使用复权/调整后数据")
    ssl_verify: bool = Field(default=True, description="是否校验证书（默认开启）")
    ssl_ca_bundle: Optional[str] = Field(default=None, description="自定义CA证书路径（可选）")

    class Config:
        env_prefix = "POLYGON_"


class IBSettings(BaseSettings):
    """Interactive Brokers API配置"""
    host: str = Field(default="127.0.0.1", description="TWS/IB Gateway 主机地址")
    port: int = Field(default=7497, description="TWS/IB Gateway 端口（7497=模拟账户，7496=实盘账户）")
    client_id: int = Field(default=1, description="基准客户端ID（不建议直接使用）")
    collector_client_id: int = Field(default=2, description="数据采集器专用客户端ID")
    broker_client_id: int = Field(default=1, description="交易执行/回测专用客户端ID")
    account: Optional[str] = Field(default=None, description="账户ID（如果为None，使用默认账户）")
    timeout: float = Field(default=10.0, description="连接超时时间（秒）")
    readonly: bool = Field(default=False, description="是否只读模式（不执行交易）")

    class Config:
        env_prefix = "IB_"


class DataSettings(BaseSettings):
    """数据配置"""
    symbols: List[str] = Field(default=["BTC/USDT", "ETH/USDT"], description="交易对列表")
    timeframes: List[str] = Field(default=["1m", "5m", "15m", "1h", "4h", "1d"], description="时间周期")
    max_history_days: int = Field(default=365, description="最大历史数据天数")
    update_interval: int = Field(default=60, description="数据更新间隔(秒)")
    
    class Config:
        env_prefix = "DATA_"


class FactorSettings(BaseSettings):
    """因子配置"""
    calculation_window: int = Field(default=20, description="因子计算窗口")
    min_history_periods: int = Field(default=100, description="最小历史周期数")
    outlier_threshold: float = Field(default=3.0, description="异常值阈值(标准差倍数)")
    missing_threshold: float = Field(default=0.1, description="缺失值阈值")
    
    class Config:
        env_prefix = "FACTOR_"


class BacktestSettings(BaseSettings):
    """回测配置"""
    initial_capital: float = Field(default=100000.0, description="初始资金")
    commission: float = Field(default=0.001, description="手续费率")
    slippage: float = Field(default=0.0005, description="滑点")
    benchmark: str = Field(default="BTC/USDT", description="基准资产")
    
    class Config:
        env_prefix = "BACKTEST_"


class APISettings(BaseSettings):
    """API配置"""
    host: str = Field(default="0.0.0.0", description="API主机")
    port: int = Field(default=8000, description="API端口")
    debug: bool = Field(default=False, description="调试模式")
    cors_origins: List[str] = Field(default=["*"], description="CORS允许的源")
    secret_key: str = Field(default="your-secret-key", description="JWT密钥")
    
    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="日志格式"
    )
    file_path: str = Field(default="logs/app.log", description="日志文件路径")
    rotation: str = Field(default="10 MB", description="日志轮转大小")
    retention: str = Field(default="30 days", description="日志保留时间")
    
    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """主配置类"""
    # 基本信息
    app_name: str = Field(default="Factor Mining System", description="应用名称")
    version: str = Field(default="1.0.0", description="版本号")
    description: str = Field(default="系统化的因子挖掘系统", description="应用描述")
    
    # 子配置
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    influxdb: InfluxDBSettings = InfluxDBSettings()
    storage: StorageSettings = StorageSettings()
    exchange: ExchangeSettings = ExchangeSettings()
    polygon: PolygonSettings = PolygonSettings()
    ib: IBSettings = IBSettings()
    data: DataSettings = DataSettings()
    factor: FactorSettings = FactorSettings()
    backtest: BacktestSettings = BacktestSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    
    @model_validator(mode="before")
    @classmethod
    def _map_flat_env_to_nested(cls, values):
        """
        兼容 .env 中使用的扁平变量命名（如 DB_HOST、API_HOST、POLYGON_API_KEY）。

        pydantic-settings v2 默认更偏向用嵌套分隔符（如 DATABASE__HOST），
        但本项目历史上使用了 *_HOST 这类写法，因此在此做一次映射。
        """
        if not isinstance(values, dict):
            return values

        def set_nested(parent: str, child: str, legacy_key: str):
            if legacy_key not in values:
                return
            parent_val = values.get(parent)
            if not isinstance(parent_val, dict):
                parent_val = {}
                values[parent] = parent_val
            if child not in parent_val:
                parent_val[child] = values[legacy_key]
            values.pop(legacy_key, None)

        # Database (DB_*)
        set_nested("database", "host", "db_host")
        set_nested("database", "port", "db_port")
        set_nested("database", "username", "db_username")
        set_nested("database", "password", "db_password")
        set_nested("database", "database", "db_database")

        # Redis (REDIS_*)
        set_nested("redis", "host", "redis_host")
        set_nested("redis", "port", "redis_port")
        set_nested("redis", "password", "redis_password")
        set_nested("redis", "db", "redis_db")

        # InfluxDB (INFLUXDB_*)
        set_nested("influxdb", "url", "influxdb_url")
        set_nested("influxdb", "token", "influxdb_token")
        set_nested("influxdb", "org", "influxdb_org")
        set_nested("influxdb", "bucket", "influxdb_bucket")

        # API (API_*)
        set_nested("api", "host", "api_host")
        set_nested("api", "port", "api_port")
        set_nested("api", "debug", "api_debug")
        set_nested("api", "cors_origins", "api_cors_origins")
        set_nested("api", "secret_key", "api_secret_key")

        # Logging (LOG_*)
        set_nested("logging", "level", "log_level")
        set_nested("logging", "format", "log_format")
        set_nested("logging", "file_path", "log_file_path")
        set_nested("logging", "rotation", "log_rotation")
        set_nested("logging", "retention", "log_retention")

        # Storage (STORAGE_*)
        set_nested("storage", "data_dir", "storage_data_dir")

        # Polygon (POLYGON_*)
        set_nested("polygon", "api_key", "polygon_api_key")
        set_nested("polygon", "base_url", "polygon_base_url")
        set_nested("polygon", "adjusted", "polygon_adjusted")

        # Interactive Brokers (IB_*)
        set_nested("ib", "host", "ib_host")
        set_nested("ib", "port", "ib_port")
        set_nested("ib", "client_id", "ib_client_id")
        set_nested("ib", "collector_client_id", "ib_collector_client_id")
        set_nested("ib", "broker_client_id", "ib_broker_client_id")
        set_nested("ib", "account", "ib_account")
        set_nested("ib", "timeout", "ib_timeout")
        set_nested("ib", "readonly", "ib_readonly")

        return values

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings 

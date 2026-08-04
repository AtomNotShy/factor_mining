"""
数据采集器基类
定义统一的数据采集接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime


class Logger:
    """简单的日志记录器"""
    def __init__(self, name: str):
        self.name = name
    
    def info(self, message: str):
        print(f"INFO [{self.name}]: {message}")
    
    def error(self, message: str):
        print(f"ERROR [{self.name}]: {message}")
    
    def warning(self, message: str):
        print(f"WARNING [{self.name}]: {message}")
    
    def debug(self, message: str):
        print(f"DEBUG [{self.name}]: {message}")


class BaseDataCollector(ABC):
    """数据采集器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger(f"collector.{name}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接到数据源"""
        pass
    
    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """获取OHLCV数据"""
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证交易对格式"""
        return symbol and "/" in symbol
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """验证时间周期格式"""
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        return timeframe in valid_timeframes
    
    async def health_check(self) -> Dict:
        """健康检查"""
        try:
            result = await self.connect()
            return {
                "status": "healthy" if result else "unhealthy",
                "name": self.name,
                "connected": result
            }
        except Exception as e:
            return {
                "status": "error",
                "name": self.name,
                "error": str(e),
                "connected": False
            }
    
    def format_ohlcv_data(self, ohlcv_data: List, symbol: str, timeframe: str) -> pd.DataFrame:
        """格式化OHLCV数据"""
        if not ohlcv_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if df.empty:
            return df
        
        # 添加一些基础技术指标
        try:
            # 简单移动平均
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # 收益率
            df['returns'] = df['close'].pct_change()
            
            # 波动率
            df['volatility'] = df['returns'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
        
        return df
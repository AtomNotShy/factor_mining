"""
交易所数据采集器
支持主流交易所的数据采集
"""

import ccxt.async_support as ccxt
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from .base import BaseDataCollector, Logger
from src.config.settings import get_settings


# CCXT 交易所类型的类型注解
_ExchangeType = Any  # ccxt.async_support.binance/okx 实例类型


class ExchangeCollector(BaseDataCollector):
    """交易所数据采集器"""
    
    def __init__(self, exchange_name: str):
        super().__init__(f"exchange_{exchange_name}")
        self.exchange_name = exchange_name
        self.exchange: _ExchangeType = None  # type: ignore
        self.settings = get_settings().exchange
        self._setup_exchange()
    
    def _setup_exchange(self):
        """设置交易所"""
        try:
            if self.exchange_name.lower() == "binance":
                self.exchange = ccxt.binance({
                    'apiKey': self.settings.binance_api_key,
                    'secret': self.settings.binance_secret,
                    'sandbox': False,
                    'rateLimit': 1200,
                    'enableRateLimit': True,
                })
            elif self.exchange_name.lower() == "okx":
                self.exchange = ccxt.okx({
                    'apiKey': self.settings.okx_api_key,
                    'secret': self.settings.okx_secret,
                    'password': self.settings.okx_passphrase,
                    'sandbox': False,
                    'rateLimit': 100,
                    'enableRateLimit': True,
                })
            else:
                raise ValueError(f"不支持的交易所: {self.exchange_name}")
                
        except Exception as e:
            self.logger.error(f"交易所设置失败: {e}")
            raise
    
    async def connect(self) -> bool:
        """连接到交易所"""
        try:
            if not self.exchange:
                self._setup_exchange()
            
            # 测试连接
            await self.exchange.load_markets()
            self.logger.info(f"成功连接到 {self.exchange_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self.exchange:
            try:
                if hasattr(self.exchange, 'close'):
                    await self.exchange.close()
                self.logger.info(f"已断开与 {self.exchange_name} 的连接")
            except Exception as e:
                self.logger.warning(f"断开连接时出现警告: {e}")
    
    async def get_symbols(self) -> List[str]:
        """获取可用的交易对列表"""
        try:
            if not self.exchange:
                await self.connect()
            
            markets = self.exchange.markets
            symbols = [symbol for symbol in markets.keys() if markets[symbol]['active']]
            
            self.logger.info(f"获取到 {len(symbols)} 个交易对")
            return symbols
            
        except Exception as e:
            self.logger.error(f"获取交易对列表失败: {e}")
            return []
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = 1000
    ) -> pd.DataFrame:
        """获取OHLCV数据"""
        try:
            if not self.validate_symbol(symbol) or not self.validate_timeframe(timeframe):
                raise ValueError(f"无效的参数: symbol={symbol}, timeframe={timeframe}")
            
            if not self.exchange:
                await self.connect()
            
            # 转换时间戳
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)
            
            # 获取数据
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, timeframe, since_ts, limit
            )
            
            # 格式化数据
            df = self.format_ohlcv_data(ohlcv, symbol, timeframe)
            
            # 计算技术指标
            df = self.calculate_technical_indicators(df)
            
            self.logger.info(f"获取 {symbol} {timeframe} 数据 {len(df)} 条")
            return df
            
        except Exception as e:
            self.logger.error(f"获取OHLCV数据失败: {e}")
            return pd.DataFrame()
    
    async def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        """获取订单簿数据"""
        try:
            if not self.exchange:
                await self.connect()
            
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'timestamp': orderbook.get('timestamp'),
                'datetime': orderbook.get('datetime'),
                'bids': orderbook.get('bids', []),
                'asks': orderbook.get('asks', []),
                'bid': orderbook['bids'][0][0] if orderbook.get('bids') else None,
                'ask': orderbook['asks'][0][0] if orderbook.get('asks') else None,
                'spread': None
            }
            
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}")
            return {}
    
    async def get_trades(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = 1000
    ) -> pd.DataFrame:
        """获取交易数据"""
        try:
            if not self.exchange:
                await self.connect()
            
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)
            
            trades = await self.exchange.fetch_trades(symbol, since_ts, limit)
            
            if not trades:
                return pd.DataFrame()
            
            df = pd.DataFrame(trades)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取交易数据失败: {e}")
            return pd.DataFrame()
    
    async def get_ticker(self, symbol: str) -> Dict:
        """获取行情数据"""
        try:
            if not self.exchange:
                await self.connect()
            
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            self.logger.error(f"获取行情数据失败: {e}")
            return {}
    
    async def get_24h_stats(self, symbol: str) -> Dict:
        """获取24小时统计数据"""
        try:
            ticker = await self.get_ticker(symbol)
            
            if not ticker:
                return {}
            
            return {
                'symbol': symbol,
                'price': ticker.get('last'),
                'change': ticker.get('change'),
                'percentage': ticker.get('percentage'),
                'volume': ticker.get('baseVolume'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'open': ticker.get('open'),
                'close': ticker.get('close'),
                'timestamp': ticker.get('timestamp')
            }
            
        except Exception as e:
            self.logger.error(f"获取24小时统计失败: {e}")
            return {}


class BinanceCollector(ExchangeCollector):
    """币安数据采集器"""
    
    def __init__(self):
        super().__init__("binance")


class OKXCollector(ExchangeCollector):
    """OKX数据采集器"""
    
    def __init__(self):
        super().__init__("okx")


class MultiExchangeCollector:
    """多交易所数据采集器"""
    
    def __init__(self, exchanges: List[str] = []):
        if exchanges is None:
            exchanges = ["binance"]
        
        self.collectors = {}
        self.logger = Logger("multi_exchange_collector")
        
        for exchange in exchanges:
            try:
                if exchange.lower() == "binance":
                    self.collectors[exchange] = BinanceCollector()
                elif exchange.lower() == "okx":
                    self.collectors[exchange] = OKXCollector()
                else:
                    self.logger.warning(f"不支持的交易所: {exchange}")
            except Exception as e:
                self.logger.error(f"初始化 {exchange} 采集器失败: {e}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """连接所有交易所"""
        results = {}
        for name, collector in self.collectors.items():
            results[name] = await collector.connect()
        return results
    
    async def disconnect_all(self):
        """断开所有连接"""
        for collector in self.collectors.values():
            await collector.disconnect()
    
    async def get_ohlcv_from_best_source(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = 1000
    ) -> pd.DataFrame:
        """从最佳数据源获取OHLCV数据"""
        
        # 优先级顺序
        priority = ["binance", "okx"]
        
        for exchange in priority:
            if exchange in self.collectors:
                try:
                    df = await self.collectors[exchange].get_ohlcv(
                        symbol, timeframe, since, limit
                    )
                    if not df.empty:
                        df['exchange'] = exchange
                        return df
                except Exception as e:
                    self.logger.warning(f"{exchange} 获取数据失败: {e}")
                    continue
        
        self.logger.error(f"所有交易所都无法获取 {symbol} 数据")
        return pd.DataFrame()
    
    async def health_check_all(self) -> Dict:
        """所有交易所健康检查"""
        results = {}
        for name, collector in self.collectors.items():
            results[name] = await collector.health_check()
        return results 
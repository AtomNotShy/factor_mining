"""
批量下载股票日线数据脚本
功能：
- 从 tickers.txt 读取股票代码（每行一个）
- 限速控制（避免API频率限制）
- 自动重试机制（指数退避）
- 保存为 Parquet 格式
- 下载2年日线数据
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode, urljoin
import time
import ssl
import certifi
import sys
from loguru import logger

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.settings import get_settings


class RateLimiter:
    """限速器：控制请求频率"""
    
    def __init__(self, max_calls: int = 5, period: float = 1.0):
        """
        Args:
            max_calls: 每个周期内的最大请求数
            period: 周期长度（秒）
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """获取许可，如果超过限制则等待"""
        async with self._lock:
            now = time.time()
            # 移除过期的调用记录
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) >= self.max_calls:
                # 计算需要等待的时间
                oldest_call = min(self.calls)
                wait_time = self.period - (now - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # 重新清理
                    now = time.time()
                    self.calls = [t for t in self.calls if now - t < self.period]
            
            self.calls.append(time.time())


class DailyDataDownloader:
    """日线数据下载器"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polygon.io",
        output_dir: str = "data/daily",
        max_retries: int = 5,
        rate_limit_calls: int = 5,
        rate_limit_period: float = 1.0,
        years: int = 2,
    ):
        """
        Args:
            api_key: Polygon API Key
            base_url: API基础URL
            output_dir: 输出目录
            max_retries: 最大重试次数
            rate_limit_calls: 限速：每个周期内的最大请求数
            rate_limit_period: 限速：周期长度（秒）
            years: 下载年数
        """
        self.api_key = api_key
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)
        self.years = years
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 配置日志
        logger.add(
            "logs/download_daily.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self._session
    
    async def close(self):
        """关闭会话"""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
    
    async def _fetch_with_retry(
        self,
        url: str,
        symbol: str,
        retry_count: int = 0
    ) -> Optional[dict]:
        """
        带重试的HTTP请求
        
        Args:
            url: 请求URL
            symbol: 股票代码（用于日志）
            retry_count: 当前重试次数
            
        Returns:
            JSON响应数据，失败返回None
        """
        # 限速控制
        await self.rate_limiter.acquire()
        
        session = await self._get_session()
        
        try:
            async with session.get(url) as resp:
                # 处理429（频率限制）
                if resp.status == 429:
                    if retry_count < self.max_retries:
                        wait_time = 2 ** retry_count  # 指数退避：1s, 2s, 4s, 8s, 16s
                        logger.warning(
                            f"{symbol}: 频率限制(429)，等待 {wait_time} 秒后重试 "
                            f"({retry_count + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        return await self._fetch_with_retry(url, symbol, retry_count + 1)
                    else:
                        logger.error(f"{symbol}: 频率限制(429)，重试 {self.max_retries} 次后失败")
                        return None
                
                # 处理其他错误
                if resp.status != 200:
                    if retry_count < self.max_retries and resp.status >= 500:
                        # 服务器错误，重试
                        wait_time = 2 ** retry_count
                        logger.warning(
                            f"{symbol}: HTTP {resp.status}，等待 {wait_time} 秒后重试 "
                            f"({retry_count + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        return await self._fetch_with_retry(url, symbol, retry_count + 1)
                    else:
                        error_text = await resp.text()
                        logger.error(f"{symbol}: HTTP {resp.status} - {error_text[:200]}")
                        return None
                
                return await resp.json()
                
        except asyncio.TimeoutError:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.warning(
                    f"{symbol}: 请求超时，等待 {wait_time} 秒后重试 "
                    f"({retry_count + 1}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)
                return await self._fetch_with_retry(url, symbol, retry_count + 1)
            else:
                logger.error(f"{symbol}: 请求超时，重试 {self.max_retries} 次后失败")
                return None
                
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.warning(
                    f"{symbol}: 请求异常 {type(e).__name__}: {e}，等待 {wait_time} 秒后重试 "
                    f"({retry_count + 1}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)
                return await self._fetch_with_retry(url, symbol, retry_count + 1)
            else:
                logger.error(f"{symbol}: 请求异常 {type(e).__name__}: {e}，重试 {self.max_retries} 次后失败")
                return None
    
    async def download_symbol(self, symbol: str) -> bool:
        """
        下载单个股票的数据
        
        Args:
            symbol: 股票代码（如 AAPL）
            
        Returns:
            是否成功
        """
        symbol = symbol.strip().upper()
        if not symbol:
            return False
        
        logger.info(f"开始下载 {symbol} 的 {self.years} 年日线数据...")
        
        # 计算日期范围
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.years * 365)
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # 构建API URL
        path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date_str}/{end_date_str}"
        url = urljoin(self.base_url, path)
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        
        full_url = url + "?" + urlencode(params)
        
        # 获取数据（处理分页）
        all_results = []
        next_url = full_url
        
        page_count = 0
        max_pages = 100  # 防止无限循环
        
        while next_url and page_count < max_pages:
            page_count += 1
            payload = await self._fetch_with_retry(next_url, symbol)
            
            if payload is None:
                logger.error(f"{symbol}: 获取数据失败")
                return False
            
            results = payload.get("results", [])
            if not results:
                break
            
            all_results.extend(results)
            
            # 检查是否有下一页
            next_url = payload.get("next_url")
            if next_url:
                # next_url 通常不包含 apiKey，需要补上
                join_char = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{join_char}apiKey={self.api_key}"
            else:
                break
        
        if not all_results:
            logger.warning(f"{symbol}: 未获取到数据")
            return False
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 转换时间戳（Polygon返回的是毫秒级Unix时间戳，UTC）
        # 转换为UTC时间后去掉时区信息，保持与项目其他部分一致
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.set_index("datetime").sort_index()
        
        # 重命名列
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })
        
        # 选择需要的列
        keep_cols = ["open", "high", "low", "close", "volume"]
        if "vw" in df.columns:
            df = df.rename(columns={"vw": "vwap"})
            keep_cols.append("vwap")
        if "n" in df.columns:
            df = df.rename(columns={"n": "trades"})
            keep_cols.append("trades")
        
        df = df[keep_cols]
        
        # 保存为Parquet
        output_file = self.output_dir / f"{symbol}_1d.parquet"
        
        # 确保索引是DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"{symbol}: 索引类型错误")
            return False
        
        # 写入Parquet（保留索引信息）
        df_output = df.copy()
        df_output.insert(0, "datetime", df_output.index)
        
        try:
            # 使用临时文件，原子性写入
            tmp_file = output_file.with_suffix(".parquet.tmp")
            df_output.to_parquet(tmp_file, index=False, engine="pyarrow")
            tmp_file.replace(output_file)
            
            logger.info(
                f"{symbol}: 成功下载 {len(df)} 条数据 "
                f"({start_date_str} 至 {end_date_str})，已保存至 {output_file}"
            )
            return True
            
        except Exception as e:
            logger.error(f"{symbol}: 保存Parquet文件失败: {e}")
            return False
    
    async def download_batch(self, symbols: List[str], max_concurrent: int = 3):
        """
        批量下载多个股票
        
        Args:
            symbols: 股票代码列表
            max_concurrent: 最大并发数
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(symbol: str):
            async with semaphore:
                return await self.download_symbol(symbol)
        
        tasks = [download_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        success_count = sum(1 for r in results if r is True)
        fail_count = len(results) - success_count
        
        logger.info(f"\n下载完成: 成功 {success_count} 个，失败 {fail_count} 个")


def load_tickers(tickers_file: str) -> List[str]:
    """
    从文件加载股票代码列表
    
    Args:
        tickers_file: tickers.txt文件路径
        
    Returns:
        股票代码列表
    """
    tickers_path = Path(tickers_file)
    if not tickers_path.exists():
        raise FileNotFoundError(f"文件不存在: {tickers_file}")
    
    with open(tickers_path, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    # 过滤空行和注释
    tickers = [t for t in tickers if t and not t.startswith("#")]
    
    return tickers


async def main():
    """主函数"""
    # 从配置获取API Key
    settings = get_settings()
    api_key = settings.polygon.api_key
    
    if not api_key:
        logger.error("未配置 Polygon API Key，请设置环境变量 POLYGON_API_KEY")
        return
    
    # 配置参数
    tickers_file = "tickers.txt"
    output_dir = "data/daily"
    years = 2
    max_concurrent = 3  # 并发下载数
    rate_limit_calls = 5  # 每秒最多5个请求
    rate_limit_period = 1.0
    
    # 加载股票代码
    try:
        tickers = load_tickers(tickers_file)
        logger.info(f"从 {tickers_file} 加载了 {len(tickers)} 个股票代码")
    except FileNotFoundError as e:
        logger.error(f"加载股票代码失败: {e}")
        logger.info(f"请创建 {tickers_file} 文件，每行一个股票代码，例如：")
        logger.info("AAPL")
        logger.info("MSFT")
        logger.info("GOOGL")
        return
    
    if not tickers:
        logger.warning("未找到有效的股票代码")
        return
    
    # 创建下载器
    downloader = DailyDataDownloader(
        api_key=api_key,
        base_url=settings.polygon.base_url,
        output_dir=output_dir,
        max_retries=5,
        rate_limit_calls=rate_limit_calls,
        rate_limit_period=rate_limit_period,
        years=years,
    )
    
    try:
        # 批量下载
        await downloader.download_batch(tickers, max_concurrent=max_concurrent)
    finally:
        await downloader.close()
    
    logger.info("所有任务完成")


if __name__ == "__main__":
    asyncio.run(main())

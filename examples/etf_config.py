"""
ETF回测配置

14只ETF（价格<=100，适合1000刀初始资金）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


ETF_LIST: List[str] = [
    "EFA",   # MSCI EAFE ETF (国际发达市场)
    "EEM",   # MSCI Emerging Markets ETF (新兴市场)
    "USMV",  # MSCI USA Minimum Volatility ETF (低波动)
    "XLE",   # Energy Select Sector SPDR ETF (能源)
    "VNQ",   # Vanguard Real Estate ETF (房地产)
    "TLT",   # iShares 20+ Year Treasury Bond ETF (长期国债)
    "IEF",   # iShares 7-10 Year Treasury Bond ETF (中期国债)
    "SHY",   # iShares 1-3 Year Treasury Bond ETF (短期国债)
    "LQD",   # iShares iBoxx $ Investment Grade Corporate Bond ETF (投资级债)
    "HYG",   # iShares iBoxx $ High Yield Corporate Bond ETF (高收益债)
    "DBC",   # Invesco DB Commodity Index Tracking ETF (商品)
    "USO",   # United States Oil Fund ETF (原油)
    "MTUM",  # MSCI USA Momentum Factor ETF (动量因子)
]

ETF_CATEGORIES: Dict[str, List[str]] = {
    "international": ["EFA", "EEM"],
    "us_factor": ["USMV", "MTUM"],
    "sector": ["XLE", "VNQ"],
    "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG"],
    "commodity": ["DBC", "USO"],
}

ETF_NAMES: Dict[str, str] = {
    "EFA": "iShares MSCI EAFE ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "USMV": "iShares MSCI USA Minimum Volatility ETF",
    "XLE": "Energy Select Sector SPDR Fund",
    "VNQ": "Vanguard Real Estate ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "DBC": "Invesco DB Commodity Index Tracking Fund",
    "USO": "United States Oil Fund LP",
    "MTUM": "iShares MSCI USA Momentum Factor ETF",
}


@dataclass
class ETFBacktestConfig:
    """ETF回测配置"""
    etfs: List[str] = field(default_factory=lambda: ETF_LIST)
    start_date: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime.now())
    initial_capital: float = 1000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    rebalance_freq: str = "monthly"
    max_position_pct: float = 0.2
    min_position_pct: float = 0.05
    data_source: str = "ib"
    timeframe: str = "1d"
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 1


DEFAULT_CONFIG = ETFBacktestConfig()


def get_etf_by_category(category: str) -> List[str]:
    """按类别获取ETF列表"""
    return ETF_CATEGORIES.get(category, [])


def get_all_categories() -> List[str]:
    """获取所有类别"""
    return list(ETF_CATEGORIES.keys())


if __name__ == "__main__":
    print("ETF List:")
    for etf in ETF_LIST:
        name = ETF_NAMES.get(etf, etf)
        print(f"  {etf}: {name}")
    
    print("\nCategories:")
    for cat, etfs in ETF_CATEGORIES.items():
        print(f"  {cat}: {etfs}")

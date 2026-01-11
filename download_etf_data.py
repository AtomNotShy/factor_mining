"""
使用IB下载美股ETF历史数据
"""

import asyncio
from datetime import datetime, timedelta, timezone
from src.data.collectors.ib_history import IBHistoryCollector

# 美股ETF列表（常见的宽基和行业ETF）
US_ETFS = [
    # 宽基指数ETF
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "VTI",   # Total Stock Market
    "VOO",   # S&P 500
    "VXX",   # Volatility
    
    # 国际市场ETF
    "VEA",   # Developed Markets
    "VWO",   # Emerging Markets
    "EFA",   # EAFE
    "EEM",   # Emerging Markets
    "IEFA",  # Developed ex US
    "IEMG",  # Emerging ex China
    
    # 债券ETF
    "TLT",   # 20+ Year Treasury
    "IEF",   # 7-10 Year Treasury
    "SHY",   # 1-3 Year Treasury
    "AGG",   # Aggregate Bond
    "LQD",   # Investment Grade Corporate
    "HYG",   # High Yield Corporate
    "SJNK",  # High Yield Short
    
    # 行业ETF
    "XLF",   # Financials
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLK",   # Technology
    "XLV",   # Health Care
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLC",   # Communication Services
    
    # 其他热门ETF
    "DIA",   # Dow Jones
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Oil
    "UNG",   # Natural Gas
    "VNQ",   # Real Estate
    "DBA",   # Agriculture
]


async def download_etf_data():
    """下载美股ETF数据"""
    print("=" * 60)
    print("IB美股ETF数据下载")
    print("=" * 60)
    
    # 创建IB数据收集器（默认配置：127.0.0.1:4002）
    collector = IBHistoryCollector(
        host="127.0.0.1",
        port=4002,
        client_id=1,
    )
    
    try:
        # 连接到IB
        print("\n正在连接到IB Gateway/TWS...")
        ok = await collector.connect()
        if not ok:
            print("❌ 连接IB失败，请确保IB Gateway/TWS已启动")
            return False
        
        print("✅ 连接IB成功")
        
        # 设置日期范围（过去2年）
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=730)  # 2年
        
        print(f"\n📅 下载期间: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"📊 下载标的数量: {len(US_ETFS)}")
        print()
        
        # 批量下载数据
        success_count = 0
        fail_count = 0
        
        for symbol in US_ETFS:
            print(f"正在下载 {symbol}...", end=" ")
            
            try:
                df = await collector.get_ohlcv_async(
                    symbol=symbol,
                    timeframe="1d",
                    since=start_date,
                    use_cache=False,  # 强制重新下载
                )
                
                if len(df) > 0:
                    print(f"✅ {len(df)} 条数据")
                    success_count += 1
                else:
                    print("❌ 无数据")
                    fail_count += 1
                    
            except Exception as e:
                print(f"❌ 错误: {e}")
                fail_count += 1
        
        print()
        print("=" * 60)
        print(f"下载完成: 成功 {success_count}, 失败 {fail_count}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 下载过程出错: {e}")
        return False
        
    finally:
        await collector.disconnect()
        print("\n已断开IB连接")


def download_sync():
    """同步调用下载函数"""
    return asyncio.run(download_etf_data())


if __name__ == "__main__":
    print("IB美股ETF数据下载脚本")
    print("请确保IB Gateway/TWS已启动，并已启用API接口")
    print()
    
    download_sync()

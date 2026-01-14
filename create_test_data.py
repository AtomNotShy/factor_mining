"""
创建测试数据文件
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_test_data():
    """创建测试数据"""
    # 创建数据目录
    data_dir = "data/bars/symbol=AAPL/timeframe=1d"
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成测试数据
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="B")  # 工作日
    n = len(dates)
    
    # 生成价格数据
    np.random.seed(42)
    base_price = 180.0
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 创建DataFrame
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices * (1 + np.random.normal(0, 0.005, n)),
        "high": prices * (1 + np.abs(np.random.normal(0.01, 0.01, n))),
        "low": prices * (1 - np.abs(np.random.normal(0.01, 0.01, n))),
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, n),
    })
    
    # 确保价格合理性
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999
    
    # 设置索引
    df = df.set_index("datetime")
    
    # 保存为Parquet文件
    output_path = os.path.join(data_dir, "AAPL_1d.parquet")
    df.to_parquet(output_path)
    
    print(f"✅ 创建测试数据: {output_path}")
    print(f"数据形状: {df.shape}")
    print(f"数据范围: {df.index.min()} - {df.index.max()}")
    
    return df

if __name__ == "__main__":
    df = create_test_data()
    print("\n前5行数据:")
    print(df.head())
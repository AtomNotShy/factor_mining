import pandas as pd
import numpy as np
from src.evaluation.metrics.benchmark import BenchmarkAnalyzer

# Mock strategy returns
idx = pd.date_range("2025-01-13", periods=5, freq="D", tz="UTC")
s_ret = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01], index=idx)

# Load real benchmark data
ba = BenchmarkAnalyzer("SPY")
b_data = ba.get_benchmark_data("2025-01-13", "2026-01-13")
if b_data is not None:
    b_ret = b_data['close'].pct_change().fillna(0)
    print(f"Strategy index type: {type(s_ret.index)}")
    print(f"Benchmark index type: {type(b_ret.index)}")
    print(f"Strategy index sample: {s_ret.index[0]}")
    print(f"Benchmark index sample: {b_ret.index[0]}")
    
    s_align, b_align = ba.align_returns(s_ret, b_ret)
    print(f"Aligned length: {len(s_align)}")
    if len(s_align) == 0:
        print("ALIGMENT FAILED: Empty intersection")
else:
    print("Failed to load benchmark data")

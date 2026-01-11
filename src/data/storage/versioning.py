"""
版本化工具
生成和管理 data_version, code_version, config_hash
"""

from datetime import datetime
from typing import Dict, Any, Optional
import hashlib
import json
import subprocess


def generate_data_version(
    source: str,
    timeframe: str,
    symbols: Optional[list] = None,
    timestamp: Optional[datetime] = None,
) -> str:
    """
    生成数据版本标识
    
    Args:
        source: 数据源（如 polygon_api）
        timeframe: 时间周期
        symbols: 标的列表（可选）
        timestamp: 时间戳（可选，默认当前时间）
        
    Returns:
        数据版本字符串，格式: {source}_{timeframe}_{date}_{hash}
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    date_str = timestamp.strftime("%Y%m%d")
    
    # 构建唯一标识
    parts = [source, timeframe, date_str]
    if symbols:
        symbols_str = "_".join(sorted(symbols[:10]))  # 限制长度
        parts.append(symbols_str)
    
    base_str = "_".join(parts)
    
    # 生成短hash
    hash_str = hashlib.md5(base_str.encode()).hexdigest()[:8]
    
    return f"{base_str}_{hash_str}"


def get_code_version() -> str:
    """
    获取代码版本（git commit SHA）
    
    Returns:
        代码版本字符串（8位commit SHA 或 "unknown"）
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def hash_config(config: Dict[str, Any]) -> str:
    """
    生成配置的稳定hash
    
    Args:
        config: 配置字典
        
    Returns:
        16位hex字符串
    """
    # 排序键以确保一致性
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def extract_backtest_config(
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
    strategy_params: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """
    提取回测配置（用于生成config_hash）
    
    Args:
        initial_capital: 初始资金
        commission_rate: 手续费率
        slippage_rate: 滑点率
        strategy_params: 策略参数
        **kwargs: 其他配置
        
    Returns:
        配置字典
    """
    return {
        "initial_capital": initial_capital,
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
        "strategy_params": strategy_params,
        **kwargs,
    }

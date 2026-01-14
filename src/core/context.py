"""
运行上下文
包含环境、版本、配置等运行时信息
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import hashlib
import json

from .calendar import TradingCalendar


class Environment(str, Enum):
    """运行环境"""
    RESEARCH = "research"  # 研究模式
    PAPER = "paper"  # 纸交易
    LIVE = "live"  # 实盘


@dataclass
class RunContext:
    """运行上下文"""
    env: Environment  # research/paper/live
    code_version: str  # 代码版本（git commit SHA 或版本字符串）
    data_version: str  # 数据版本（拉取批次标识）
    config_hash: str  # 配置hash（用于可重放）
    now_utc: datetime  # 当前UTC时间
    trading_calendar: TradingCalendar  # 交易日历
    config: Dict[str, Any] = field(default_factory=dict)  # 源配置字典
    cross_section: Optional[Dict[str, Any]] = None  # 横截面数据（用于向量化轮动策略）
    
    @classmethod
    def create(
        cls,
        env: Environment,
        code_version: Optional[str] = None,
        data_version: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        trading_calendar: Optional[TradingCalendar] = None,
    ) -> "RunContext":
        """
        创建运行上下文
        
        Args:
            env: 运行环境
            code_version: 代码版本（如果为None，尝试从git获取）
            data_version: 数据版本（如果为None，使用时间戳）
            config: 配置字典（用于生成config_hash）
            trading_calendar: 交易日历（如果为None，创建默认的）
        """
        # 获取代码版本
        if code_version is None:
            code_version = cls._get_git_commit() or "unknown"
        
        # 获取数据版本
        if data_version is None:
            data_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # 生成配置hash
        if config is None:
            config = {}
        config_hash = cls._hash_config(config)
        
        # 交易日历
        if trading_calendar is None:
            trading_calendar = TradingCalendar()
        
        return cls(
            env=env,
            code_version=code_version,
            data_version=data_version,
            config_hash=config_hash,
            now_utc=datetime.utcnow(),
            trading_calendar=trading_calendar,
            config=config,
        )
    
    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """尝试获取git commit SHA"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]  # 取前8位
        except Exception:
            pass
        return None
    
    @staticmethod
    def _hash_config(config: Dict[str, Any]) -> str:
        """生成配置的稳定hash"""
        # 排序键以确保一致性
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "env": self.env.value,
            "code_version": self.code_version,
            "data_version": self.data_version,
            "config_hash": self.config_hash,
            "now_utc": self.now_utc.isoformat(),
            "config": self.config,
        }

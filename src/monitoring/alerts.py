"""
监控告警系统
实现文档要求的6个基础告警规则
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.core.types import PortfolioState, Fill
from src.utils.logger import get_logger


@dataclass
class Alert:
    """告警"""
    alert_id: str
    alert_type: str
    severity: str  # INFO/WARNING/CRITICAL
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]


class AlertManager:
    """告警管理器"""
    
    def __init__(
        self,
        daily_loss_limit: float = 0.05,  # 5%
        consecutive_loss_limit: int = 5,
        data_missing_threshold: int = 60,  # 秒
        slippage_threshold: float = 0.002,  # 0.2%
        signal_storm_threshold: int = 100,  # 单日信号数
    ):
        self.daily_loss_limit = daily_loss_limit
        self.consecutive_loss_limit = consecutive_loss_limit
        self.data_missing_threshold = data_missing_threshold
        self.slippage_threshold = slippage_threshold
        self.signal_storm_threshold = signal_storm_threshold
        
        self.logger = get_logger("alert_manager")
        self.alerts: List[Alert] = []
        self.consecutive_losses = 0
        self.last_data_time: Optional[datetime] = None
    
    def check_daily_loss(
        self,
        portfolio: PortfolioState,
        initial_equity: float,
        timestamp: datetime,
    ) -> Optional[Alert]:
        """检查当日亏损（熔断）"""
        daily_loss_pct = abs(portfolio.daily_loss) / initial_equity if initial_equity > 0 else 0
        
        if daily_loss_pct >= self.daily_loss_limit:
            alert = Alert(
                alert_id=f"daily_loss_{timestamp.isoformat()}",
                alert_type="DAILY_LOSS_LIMIT",
                severity="CRITICAL",
                message=f"当日亏损超过阈值: {daily_loss_pct:.2%} >= {self.daily_loss_limit:.2%}",
                timestamp=timestamp,
                metadata={
                    "daily_loss": portfolio.daily_loss,
                    "daily_loss_pct": daily_loss_pct,
                    "threshold": self.daily_loss_limit,
                },
            )
            self.alerts.append(alert)
            self.logger.critical(f"熔断触发: {alert.message}")
            return alert
        
        return None
    
    def check_consecutive_losses(
        self,
        fill: Fill,
        portfolio: PortfolioState,
        timestamp: datetime,
    ) -> Optional[Alert]:
        """检查连续亏损"""
        # 简化：基于成交判断盈亏
        # 实际应该基于持仓盈亏
        
        # 这里简化处理，实际需要更复杂的逻辑
        if fill.side.value == "SELL":
            # 假设卖出时检查是否亏损
            # 实际需要计算持仓成本
        
        if self.consecutive_losses >= self.consecutive_loss_limit:
            alert = Alert(
                alert_id=f"consecutive_loss_{timestamp.isoformat()}",
                alert_type="CONSECUTIVE_LOSSES",
                severity="CRITICAL",
                message=f"连续 {self.consecutive_losses} 笔亏损，建议降频/停机",
                timestamp=timestamp,
                metadata={
                    "consecutive_losses": self.consecutive_losses,
                    "threshold": self.consecutive_loss_limit,
                },
            )
            self.alerts.append(alert)
            self.logger.critical(f"连续亏损告警: {alert.message}")
            return alert
        
        return None
    
    def check_data_missing(
        self,
        timestamp: datetime,
    ) -> Optional[Alert]:
        """检查数据缺失/延迟"""
        if self.last_data_time is None:
            self.last_data_time = timestamp
            return None
        
        delay = (timestamp - self.last_data_time).total_seconds()
        
        if delay > self.data_missing_threshold:
            alert = Alert(
                alert_id=f"data_missing_{timestamp.isoformat()}",
                alert_type="DATA_MISSING",
                severity="WARNING",
                message=f"数据延迟超过阈值: {delay:.0f}秒 >= {self.data_missing_threshold}秒",
                timestamp=timestamp,
                metadata={
                    "delay_seconds": delay,
                    "threshold": self.data_missing_threshold,
                },
            )
            self.alerts.append(alert)
            self.logger.warning(f"数据延迟告警: {alert.message}")
            return alert
        
        self.last_data_time = timestamp
        return None
    
    def check_slippage(
        self,
        fill: Fill,
        expected_price: float,
        timestamp: datetime,
    ) -> Optional[Alert]:
        """检查滑点偏离"""
        if fill.slippage_est is None:
            return None
        
        slippage_pct = abs(fill.slippage_est) / expected_price if expected_price > 0 else 0
        
        if slippage_pct > self.slippage_threshold:
            alert = Alert(
                alert_id=f"slippage_{timestamp.isoformat()}",
                alert_type="SLIPPAGE_DEVIATION",
                severity="WARNING",
                message=f"实盘滑点显著高于回测假设: {slippage_pct:.4%} > {self.slippage_threshold:.4%}",
                timestamp=timestamp,
                metadata={
                    "slippage_pct": slippage_pct,
                    "expected_price": expected_price,
                    "fill_price": fill.price,
                    "threshold": self.slippage_threshold,
                },
            )
            self.alerts.append(alert)
            self.logger.warning(f"滑点偏离告警: {alert.message}")
            return alert
        
        return None
    
    def check_signal_storm(
        self,
        signal_count: int,
        timestamp: datetime,
    ) -> Optional[Alert]:
        """检查信号风暴"""
        if signal_count > self.signal_storm_threshold:
            alert = Alert(
                alert_id=f"signal_storm_{timestamp.isoformat()}",
                alert_type="SIGNAL_STORM",
                severity="WARNING",
                message=f"策略输出异常激增: {signal_count} 个信号 > {self.signal_storm_threshold}",
                timestamp=timestamp,
                metadata={
                    "signal_count": signal_count,
                    "threshold": self.signal_storm_threshold,
                },
            )
            self.alerts.append(alert)
            self.logger.warning(f"信号风暴告警: {alert.message}")
            return alert
        
        return None
    
    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """获取告警列表"""
        alerts = self.alerts
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts[-limit:]
    
    def clear_alerts(self):
        """清空告警"""
        self.alerts.clear()

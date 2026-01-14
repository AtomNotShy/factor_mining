"""
系统监控和指标
提供统一的监控和性能指标收集
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import asyncio

from src.utils.logger import get_logger

logger = logging.getLogger("monitoring")


@dataclass
class Metric:
    """指标"""
    name: str
    value: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


class MetricsCollector:
    """指标收集器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics: List[Metric] = []
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.max_samples = self.config.get("max_samples", 10000)

    def counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """增加计数器"""
        key = self._make_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value
        self._record_metric(name, self.counters[key], labels)

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """设置仪表值"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        self._record_metric(name, value, labels)

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录直方图值"""
        key = self._make_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
        self._record_metric(name, value, labels)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """生成唯一键"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def _record_metric(self, name: str, value: Any, labels: Optional[Dict[str, str]] = None):
        """记录指标"""
        self.metrics.append(Metric(name=name, value=value, labels=labels or {}))
        if len(self.metrics) > self.max_samples:
            self.metrics = self.metrics[-self.max_samples:]

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """获取计数器值"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """获取仪表值"""
        key = self._make_key(name, labels)
        return self.gauges.get(key, 0.0)

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """获取直方图统计"""
        key = self._make_key(name, labels)
        values = self.histograms.get(key, [])
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "sum": sum(values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return {
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": {k: self.get_histogram_stats(k) for k in self.histograms},
            "samples": [m.to_dict() for m in self.metrics[-100:]],
        }

    def reset(self):
        """重置所有指标"""
        self.metrics = []
        self.counters = {}
        self.gauges = {}
        self.histograms = {}


class SystemMonitor:
    """系统监控器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector(config)
        self.logger = get_logger("system_monitor")
        self.running = False
        self.monitor_task = None
        self.check_interval = self.config.get("check_interval", 60)

    async def start(self):
        """启动监控"""
        if self.running:
            return

        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("系统监控已启动")

    async def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("系统监控已停止")

    async def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"监控错误: {e}")

    async def _collect_system_metrics(self):
        """收集系统指标"""
        import psutil
        import os

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.gauge("system_cpu_percent", cpu_percent)

        # 内存使用率
        memory = psutil.virtual_memory()
        self.metrics_collector.gauge("system_memory_percent", memory.percent)

        # 磁盘使用率
        disk = psutil.disk_usage("/")
        self.metrics_collector.gauge("system_disk_percent", disk.percent)

        # 进程信息
        process = psutil.Process(os.getpid())
        self.metrics_collector.gauge("process_memory_mb", process.memory_info().rss / 1024 / 1024)
        self.metrics_collector.gauge("process_cpu_percent", process.cpu_percent())

    def record_event_processed(self, event_type: str):
        """记录事件处理"""
        self.metrics_collector.counter(f"events_processed_total", labels={"type": event_type})

    def record_error(self, error_type: str, source: str):
        """记录错误"""
        self.metrics_collector.counter(f"errors_total", labels={"type": error_type, "source": source})

    def record_order_latency(self, latency_ms: float):
        """记录订单延迟"""
        self.metrics_collector.histogram("order_latency_ms", latency_ms)

    def record_backtest_duration(self, duration_ms: float):
        """记录回测持续时间"""
        self.metrics_collector.histogram("backtest_duration_ms", duration_ms)

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.metrics_collector.get_all_metrics(),
        }


class PerformanceTracker:
    """性能追踪器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_logger("performance")
        self.operations: Dict[str, List[Dict]] = {}

    async def track_operation(
        self,
        operation_name: str,
        operation_func,
        *args,
        **kwargs,
    ) -> Any:
        """追踪操作性能"""
        start_time = asyncio.get_event_loop().time()
        try:
            result = await operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        # 记录操作
        if operation_name not in self.operations:
            self.operations[operation_name] = []

        self.operations[operation_name].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "success": success,
            "error": error,
        })

        # 限制历史大小
        max_history = self.config.get("max_history", 1000)
        if len(self.operations[operation_name]) > max_history:
            self.operations[operation_name] = self.operations[operation_name][-max_history:]

        # 日志记录
        if duration_ms > 1000:
            self.logger.warning(f"操作 {operation_name} 耗时过长: {duration_ms:.2f}ms")

        return result

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """获取操作统计"""
        if operation_name not in self.operations:
            return {}

        operations = self.operations[operation_name]
        if not operations:
            return {}

        durations = [op["duration_ms"] for op in operations]
        successful = [op for op in operations if op["success"]]

        return {
            "total_count": len(operations),
            "success_count": len(successful),
            "failure_count": len(operations) - len(successful),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有统计"""
        return {
            name: self.get_operation_stats(name)
            for name in self.operations
        }


class AlertManager:
    """告警管理器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_logger("alert_manager")
        self.alerts: List[Dict] = []
        self.rules: Dict[str, Dict] = {}
        self.notification_callbacks: List[Callable] = []

        # 默认告警规则
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认告警规则"""
        self.rules = {
            "cpu_high": {
                "condition": lambda m: m.get("system_cpu_percent", 0) > 90,
                "severity": "warning",
                "message": "CPU使用率过高",
            },
            "memory_high": {
                "condition": lambda m: m.get("system_memory_percent", 0) > 90,
                "severity": "warning",
                "message": "内存使用率过高",
            },
            "order_error_high": {
                "condition": lambda m: m.get("errors_total", 0) > 10,
                "severity": "critical",
                "message": "订单错误过多",
            },
            "latency_high": {
                "condition": lambda m: m.get("order_latency_p95_ms", 0) > 5000,
                "severity": "warning",
                "message": "订单延迟过高",
            },
        }

    def add_rule(self, rule_id: str, rule: Dict):
        """添加告警规则"""
        self.rules[rule_id] = rule

    def add_notification_callback(self, callback: Callable):
        """添加通知回调"""
        self.notification_callbacks.append(callback)

    async def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict]:
        """检查告警"""
        triggered_alerts = []

        for rule_id, rule in self.rules.items():
            try:
                if rule["condition"](metrics):
                    alert = {
                        "rule_id": rule_id,
                        "severity": rule["severity"],
                        "message": rule["message"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metrics": metrics,
                    }
                    triggered_alerts.append(alert)
                    self.alerts.append(alert)

                    # 发送通知
                    for callback in self.notification_callbacks:
                        try:
                            await callback(alert)
                        except Exception as e:
                            self.logger.error(f"告警通知失败: {e}")

            except Exception as e:
                self.logger.error(f"检查告警规则失败: {rule_id}: {e}")

        return triggered_alerts

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """获取最近告警"""
        return self.alerts[-limit:]

    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        summary = {}
        for alert in self.alerts:
            severity = alert.get("severity", "unknown")
            summary[severity] = summary.get(severity, 0) + 1

        return {
            "total_alerts": len(self.alerts),
            "by_severity": summary,
            "recent_alerts": self.get_recent_alerts(5),
        }


# 全局实例
metrics_collector = MetricsCollector()
system_monitor = SystemMonitor()
performance_tracker = PerformanceTracker()
alert_manager = AlertManager()

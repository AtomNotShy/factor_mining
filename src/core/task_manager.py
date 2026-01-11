"""
异步任务管理器
提供轻量级的后台任务执行和状态查询功能

⚠️ 注意：这是内存实现，不适合生产环境
生产环境应使用 Celery + Redis/RabbitMQ
"""

import uuid
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback

from src.utils.logger import get_logger


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "Cancelled"


@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0.0 - 100.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status.value,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error,
        }


class TaskManager:
    """
    异步任务管理器（内存实现）
    
    用法:
        manager = TaskManager()
        
        # 提交任务
        task_id = manager.submit(
            name="回测任务",
            func=run_backtest,
            args=(param1, param2),
            kwargs={"option": "value"},
            callback=on_complete,  # 可选回调
        )
        
        # 查询状态
        task = manager.get_task(task_id)
        
        # 取消任务
        manager.cancel(task_id)
        
        # 列出所有任务
        tasks = manager.list_tasks(status=TaskStatus.RUNNING)
    """
    
    def __init__(self, max_workers: int = 4, max_tasks: int = 1000):
        """
        初始化任务管理器
        
        Args:
            max_workers: 最大并行工作线程数
            max_tasks: 最大保存任务数（超过后自动清理）
        """
        self.logger = get_logger("task_manager")
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._task_queue: List[Callable] = []
        self._max_workers = max_workers
        self._max_tasks = max_tasks
        self._running = True
        self._callbacks: Dict[str, Callable] = {}
        
        # 启动工作线程
        self._start_worker()
        
        self.logger.info(f"TaskManager 启动，最大并行: {max_workers}, 最大任务数: {max_tasks}")
    
    def _start_worker(self):
        """启动工作线程"""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self):
        """工作线程主循环"""
        while self._running:
            task_id = None
            
            with self._lock:
                # 查找待执行的任务
                for tid, task in self._tasks.items():
                    if task.status == TaskStatus.PENDING:
                        task_id = tid
                        break
            
            if task_id:
                self._execute_task(task_id)
            else:
                time.sleep(0.1)  # 避免 busy waiting
    
    def _execute_task(self, task_id: str):
        """执行任务"""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return
        
        try:
            # 更新状态
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.progress = 0.0
            
            self.logger.info(f"开始执行任务: {task.name} (ID: {task_id})")
            
            # 执行任务函数
            if task_id in self._callbacks:
                # 有进度回调的情况
                result = self._callbacks[task_id](task, task.result)
            else:
                # 无回调，直接执行
                result = task.result
            
            # 更新结果
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100.0
            
            self.logger.info(f"任务完成: {task.name} (ID: {task_id})")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.traceback = traceback.format_exc()
            task.completed_at = datetime.now()
            
            self.logger.error(f"任务失败: {task.name} (ID: {task_id}): {e}")
        
        # 清理过旧的任务
        self._cleanup_old_tasks()
    
    def submit(
        self,
        name: str,
        func: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        progress_callback: Optional[Callable[[Task], None]] = None,
    ) -> str:
        """
        提交任务
        
        Args:
            name: 任务名称
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            progress_callback: 进度回调函数 (接收 Task 对象)
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())[:8]
        
        # 创建包装函数
        def wrapper():
            try:
                result = func(*(args or ()), **(kwargs or {}))
                return result
            except Exception as e:
                # 捕获异常并存储
                self._tasks[task_id].error = str(e)
                self._tasks[task_id].traceback = traceback.format_exc()
                raise
        
        with self._lock:
            # 检查是否需要清理
            if len(self._tasks) >= self._max_tasks:
                self._cleanup_old_tasks()
            
            # 创建任务
            task = Task(
                id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                result=wrapper,  # 存储包装函数
            )
            
            self._tasks[task_id] = task
            
            if progress_callback:
                self._callbacks[task_id] = progress_callback
        
        self.logger.info(f"任务已提交: {name} (ID: {task_id})")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    def cancel(self, task_id: str) -> bool:
        """取消任务（仅对pending任务有效）"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                self.logger.info(f"任务已取消: {task_id}")
                return True
        return False
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 50,
    ) -> List[Task]:
        """
        列出任务
        
        Args:
            status: 按状态过滤
            limit: 返回数量限制
            
        Returns:
            任务列表
        """
        with self._lock:
            tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        # 按创建时间倒序
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return tasks[:limit]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计"""
        with self._lock:
            pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
            running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
            completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        
        return {
            'total': len(self._tasks),
            'pending': pending,
            'running': running,
            'completed': completed,
            'failed': failed,
            'max_workers': self._max_workers,
        }
    
    def _cleanup_old_tasks(self, keep_count: int = 100):
        """清理旧任务"""
        with self._lock:
            if len(self._tasks) <= keep_count:
                return
            
            # 删除已完成的最旧的任务
            completed_tasks = [
                t for t in self._tasks.values()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            
            if len(completed_tasks) > keep_count:
                # 按完成时间排序，删除最旧的
                completed_tasks.sort(key=lambda t: t.completed_at or t.created_at)
                to_remove = completed_tasks[:len(completed_tasks) - keep_count]
                
                for task in to_remove:
                    del self._tasks[task.id]
                    if task.id in self._callbacks:
                        del self._callbacks[task.id]
                
                self.logger.info(f"清理了 {len(to_remove)} 个旧任务")
    
    def shutdown(self):
        """关闭任务管理器"""
        self._running = False
        
        with self._lock:
            # 取消所有待执行的任务
            for task in self._tasks.values():
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
        
        self.logger.info("TaskManager 已关闭")


# 全局任务管理器实例
task_manager = TaskManager(max_workers=4, max_tasks=500)

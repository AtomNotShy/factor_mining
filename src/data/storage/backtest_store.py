"""
回测结果存储模块
提供回测结果的持久化存储和查询功能
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

from src.config.settings import get_settings
from src.utils.logger import get_logger


@dataclass
class BacktestRecord:
    """回测记录"""
    id: str
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: int = 0
    win_rate: Optional[float] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class BacktestStore:
    """回测结果存储"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        settings = get_settings()
        if storage_dir is None:
            storage_dir = settings.storage.data_dir
        self.storage_dir = Path(storage_dir) / "backtests"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("backtest_store")
    
    def _get_metadata_file(self) -> Path:
        """获取元数据文件路径"""
        return self.storage_dir / "metadata.json"
    
    def _get_result_file(self, backtest_id: str) -> Path:
        """获取结果文件路径"""
        return self.storage_dir / f"{backtest_id}.json"
    
    def _load_metadata(self) -> Dict[str, BacktestRecord]:
        """加载元数据"""
        metadata_file = self._get_metadata_file()
        if not metadata_file.exists():
            return {}
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    record_id: BacktestRecord(**record_data)
                    for record_id, record_data in data.items()
                }
        except Exception as e:
            self.logger.error(f"加载元数据失败: {e}")
            return {}
    
    def _save_metadata(self, metadata: Dict[str, BacktestRecord]):
        """保存元数据"""
        metadata_file = self._get_metadata_file()
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {record_id: asdict(record) for record_id, record in metadata.items()},
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")
            raise
    
    def save(self, backtest_id: str, backtest_result: Dict[str, Any]) -> BacktestRecord:
        """
        保存回测结果
        
        Args:
            backtest_id: 回测ID
            backtest_result: 回测结果字典
            
        Returns:
            回测记录
        """
        try:
            # 提取关键信息创建记录
            results = backtest_result.get('results', {}) if isinstance(backtest_result, dict) else {}
            perf_stats = results.get('performance_stats', {}) or {}
            trade_stats = results.get('trade_stats', {}) or {}
            performance = backtest_result.get('performance', {}) if isinstance(backtest_result, dict) else {}
            enhanced = backtest_result.get('enhanced_metrics', {}) if isinstance(backtest_result, dict) else {}
            trading_stats = backtest_result.get('trading_stats', {}) if isinstance(backtest_result, dict) else {}
            backtest_period = backtest_result.get('backtest_period', {})
            
            # 计算胜率
            trade_stats_source = trading_stats or trade_stats
            win_rate = None
            if trade_stats_source.get('total_trades', 0) > 0:
                win_rate = trade_stats_source.get('winning_trades', 0) / trade_stats_source.get('total_trades', 1)
            
            universe = backtest_result.get('universe') if isinstance(backtest_result, dict) else None
            symbol_value = backtest_result.get('symbol', '')
            if not symbol_value and isinstance(universe, list) and universe:
                symbol_value = ",".join(universe)

            def safe_f(v):
                if v is None: return 0.0
                try:
                    import numpy as np
                    fv = float(v)
                    if np.isnan(fv) or np.isinf(fv): return 0.0
                    # 过滤掉不合理的超大值（超过 1 万亿）
                    if abs(fv) > 1e12:
                        return 0.0
                    return fv
                except: return 0.0

            # 提取最终权益 - 数据在根级别，字段名是 final_equity
            final_val = safe_f(
                backtest_result.get('final_equity')  # 根级别
                or backtest_result.get('final_value', 0)
            )
            
            # 提取总回报 - 数据在根级别，字段名是 total_return_pct（小数形式）
            # 兼容 total_return（小数）和 total_return_pct（百分比）
            total_ret_raw = backtest_result.get('total_return') or backtest_result.get('total_return_pct', 0)
            if total_ret_raw is not None:
                total_ret = safe_f(total_ret_raw)
                # 如果 total_return_pct 很大（如 88.6 表示 88.6%），需要转换为小数形式
                if abs(total_ret) > 1 and abs(total_ret) <= 10000:
                    total_ret = total_ret / 100.0
            else:
                total_ret = 0.0
            
            # 提取夏普比率 - 数据在根级别
            sharpe_val = safe_f(
                backtest_result.get('sharpe_ratio')
                or backtest_result.get('sharpe', 0)
            )
            
            # 提取最大回撤 - 数据在根级别
            max_dd_val = safe_f(
                backtest_result.get('max_drawdown')
                or backtest_result.get('max_drawdown_pct', 0)
            )
            # 如果 max_drawdown_pct 是百分比形式，转换为小数
            if max_dd_val > 1:
                max_dd_val = max_dd_val / 100.0
            
            # 提取交易次数
            total_trades_val = backtest_result.get('total_trades', 0)
            
            # 验证结果是否有效（只过滤掉明显无效的超大值）
            if final_val > 1e12 or total_ret > 1e12:
                self.logger.warning(
                    f"跳过保存无效回测结果（值过大）: {backtest_id}, "
                    f"final_value={final_val}, total_return={total_ret}"
                )
                return BacktestRecord(
                    id=backtest_id,
                    strategy_name=backtest_result.get('strategy_name', 'unknown'),
                    symbol=symbol_value,
                    timeframe=backtest_result.get('timeframe', ''),
                    start_date=backtest_period.get('start_date', ''),
                    end_date=backtest_period.get('end_date', ''),
                    initial_capital=0,
                    final_value=0.0,
                    total_return=0.0,
                )

            record = BacktestRecord(
                id=backtest_id,
                strategy_name=backtest_result.get('strategy_name', ''),
                symbol=symbol_value,
                timeframe=backtest_result.get('timeframe', ''),
                start_date=backtest_period.get('start_date', ''),
                end_date=backtest_period.get('end_date', ''),
                initial_capital=safe_f(backtest_result.get('initial_capital', 0)),
                final_value=final_val,
                total_return=total_ret,
                sharpe_ratio=sharpe_val,
                max_drawdown=max_dd_val,
                total_trades=total_trades_val,
                win_rate=safe_f(backtest_result.get('win_rate', 0)),
            )
            
            # 保存完整结果
            result_file = self._get_result_file(backtest_id)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(backtest_result, f, indent=2, ensure_ascii=False, default=str)
            
            # 更新元数据
            metadata = self._load_metadata()
            metadata[backtest_id] = record
            self._save_metadata(metadata)
            
            self.logger.info(f"保存回测结果: {backtest_id}")
            return record
            
        except Exception as e:
            self.logger.error(f"保存回测结果失败: {e}", exc_info=True)
            raise
    
    def list(self, strategy_name: Optional[str] = None, symbol: Optional[str] = None, limit: int = 100) -> List[BacktestRecord]:
        """
        列出回测记录
        
        Args:
            strategy_name: 策略名称过滤
            symbol: 标的过滤
            limit: 返回数量限制
            
        Returns:
            回测记录列表（按创建时间倒序）
        """
        metadata = self._load_metadata()
        records = list(metadata.values())
        
        # 过滤
        if strategy_name:
            records = [r for r in records if r.strategy_name == strategy_name]
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        
        # 按创建时间排序（处理 None 值）
        def get_created_at(x):
            return x.created_at or ""
        records.sort(key=get_created_at, reverse=True)
        
        return records[:limit]
    
    def get(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """
        获取回测结果详情
        
        Args:
            backtest_id: 回测ID
            
        Returns:
            回测结果字典，如果不存在返回None
        """
        result_file = self._get_result_file(backtest_id)
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"读取回测结果失败: {e}")
            return None
    
    def delete(self, backtest_id: str) -> bool:
        """
        删除回测结果
        
        Args:
            backtest_id: 回测ID
            
        Returns:
            是否成功删除
        """
        try:
            # 删除结果文件
            result_file = self._get_result_file(backtest_id)
            if result_file.exists():
                result_file.unlink()
            
            # 更新元数据
            metadata = self._load_metadata()
            if backtest_id in metadata:
                del metadata[backtest_id]
                self._save_metadata(metadata)
            
            self.logger.info(f"删除回测结果: {backtest_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除回测结果失败: {e}")
            return False
    
    def generate_id(self) -> str:
        """生成回测ID"""
        return str(uuid.uuid4())
    
    def cleanup_invalid_records(self) -> int:
        """
        清理无效的回测记录（final_value=0 或 total_return=0）
        
        Returns:
            删除的记录数量
        """
        metadata = self._load_metadata()
        invalid_ids = []
        
        for record_id, record in metadata.items():
            # 检查是否是无记录
            if (record.final_value == 0 and 
                record.total_return == 0 and 
                record.initial_capital == 0):
                invalid_ids.append(record_id)
        
        # 删除无效记录
        deleted_count = 0
        for record_id in invalid_ids:
            # 删除结果文件
            result_file = self._get_result_file(record_id)
            if result_file.exists():
                result_file.unlink()
            
            # 从元数据中移除
            del metadata[record_id]
            deleted_count += 1
            self.logger.info(f"删除无效回测记录: {record_id}")
        
        # 保存更新后的元数据
        if deleted_count > 0:
            self._save_metadata(metadata)
        
        return deleted_count

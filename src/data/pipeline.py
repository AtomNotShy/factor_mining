"""
数据流水线
清洗、校验、复权等数据处理
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.logger import get_logger


class DataPipeline:
    """数据流水线"""
    
    def __init__(self):
        self.logger = get_logger("data_pipeline")
    
    def clean(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            bars: 原始bars数据
            
        Returns:
            清洗后的bars
        """
        if bars.empty:
            return bars
        
        df = bars.copy()
        
        # 1. 排序
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        # 2. 去重（保留最后一个）
        df = df[~df.index.duplicated(keep='last')]
        
        # 3. 处理缺失值（前向填充，然后后向填充，使用新API避免弃用警告）
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # 4. 处理异常值（价格不能为负，volume不能为负）
        if 'close' in df.columns:
            df = df[df['close'] > 0]
        if 'volume' in df.columns:
            df.loc[df['volume'] < 0, 'volume'] = 0
        
        # 5. 验证OHLC逻辑（high >= low, high >= open/close, low <= open/close）
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            if invalid.any():
                self.logger.warning(f"发现 {invalid.sum()} 条无效OHLC数据，已删除")
                df = df[~invalid]
        
        return df
    
    def validate(self, bars: pd.DataFrame) -> List[str]:
        """
        验证数据质量
        
        Args:
            bars: bars数据
            
        Returns:
            问题列表（空列表表示无问题）
        """
        issues = []
        
        if bars.empty:
            issues.append("数据为空")
            return issues
        
        # 检查必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in bars.columns]
        if missing_cols:
            issues.append(f"缺少必需列: {missing_cols}")
        
        # 检查缺失值率
        for col in required_cols:
            if col in bars.columns:
                missing_rate = bars[col].isna().sum() / len(bars)
                if missing_rate > 0.1:  # 10%阈值
                    issues.append(f"{col} 缺失值率过高: {missing_rate:.2%}")
        
        # 检查重复K线
        duplicates = bars.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"发现 {duplicates} 条重复K线")
        
        # 检查异常跳点（价格变化超过50%）
        if 'close' in bars.columns:
            returns = bars['close'].pct_change()
            large_jumps = (returns.abs() > 0.5).sum()
            if large_jumps > 0:
                issues.append(f"发现 {large_jumps} 个异常跳点（价格变化>50%）")
        
        return issues
    
    def adjust(
        self,
        bars: pd.DataFrame,
        actions: Optional[pd.DataFrame] = None,
        method: str = "split_adjust",
    ) -> pd.DataFrame:
        """
        复权处理
        
        Args:
            bars: bars数据
            actions: 公司行为数据（split/dividend）
            method: 复权方法（split_adjust: 仅拆股复权）
            
        Returns:
            复权后的bars
        """
        if bars.empty:
            return bars
        
        df = bars.copy()
        
        if actions is None or actions.empty:
            return df
        
        # 简化实现：仅处理拆股复权
        if method == "split_adjust":
            # 按日期排序actions
            actions_sorted = actions.sort_values('ex_date')
            
            # 从后往前应用拆股（避免未来函数）
            cumulative_factor = 1.0
            
            for _, action in actions_sorted.iterrows():
                if action.get('action_type') == 'SPLIT' and action.get('split_ratio'):
                    split_ratio = action['split_ratio']
                    ex_date = pd.Timestamp(action['ex_date'])
                    
                    # 对ex_date之前的数据应用复权因子
                    mask = df.index < ex_date
                    if mask.any():
                        # 调整价格和成交量
                        price_cols = ['open', 'high', 'low', 'close']
                        for col in price_cols:
                            if col in df.columns:
                                df.loc[mask, col] = df.loc[mask, col] / split_ratio
                        
                        if 'volume' in df.columns:
                            df.loc[mask, 'volume'] = df.loc[mask, 'volume'] * split_ratio
                        
                        cumulative_factor *= split_ratio
            
            self.logger.info(f"应用复权，累计因子: {cumulative_factor}")
        
        return df

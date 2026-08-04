"""
回测数据验证器

验证回测数据的完整性和质量，防止静默失败
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Set, Optional
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_validator")


@dataclass
class DataValidationResult:
    """数据验证结果"""
    universe: List[str] = field(default_factory=list)
    loaded_symbols: Set[str] = field(default_factory=set)
    failed_symbols: List[str] = field(default_factory=list)
    missing_symbols: List[str] = field(default_factory=list)
    incomplete_symbols: Dict[str, str] = field(default_factory=dict)  # symbol -> reason

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    warmup_days: int = 0

    total_bars: int = 0
    bars_per_symbol: Dict[str, int] = field(default_factory=dict)

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """添加错误"""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """添加警告"""
        self.warnings.append(msg)

    def get_success_rate(self) -> float:
        """获取成功率"""
        if not self.universe:
            return 0.0
        return len(self.loaded_symbols) / len(self.universe)

    def summary(self) -> str:
        """生成汇总报告"""
        lines = [
            "=" * 60,
            "数据验证报告",
            "=" * 60,
            f"配置标的数: {len(self.universe)}",
            f"成功加载: {len(self.loaded_symbols)} ({self.get_success_rate()*100:.1f}%)",
            f"加载失败: {len(self.failed_symbols)}",
            f"数据缺失: {len(self.missing_symbols)}",
            f"数据不完整: {len(self.incomplete_symbols)}",
        ]

        if self.loaded_symbols:
            lines.append(f"\n✅ 成功加载的标的:")
            for symbol in sorted(self.loaded_symbols):
                bars = self.bars_per_symbol.get(symbol, 0)
                lines.append(f"   {symbol}: {bars} bars")

        if self.failed_symbols:
            lines.append(f"\n❌ 加载失败的标的:")
            for symbol in self.failed_symbols:
                lines.append(f"   {symbol}")

        if self.missing_symbols:
            lines.append(f"\n⚠️  数据缺失的标的:")
            for symbol in self.missing_symbols:
                lines.append(f"   {symbol}")

        if self.incomplete_symbols:
            lines.append(f"\n⚠️  数据不完整的标的:")
            for symbol, reason in self.incomplete_symbols.items():
                lines.append(f"   {symbol}: {reason}")

        if self.errors:
            lines.append(f"\n❌ 错误:")
            for error in self.errors:
                lines.append(f"   {error}")

        if self.warnings:
            lines.append(f"\n⚠️  警告:")
            for warning in self.warnings:
                lines.append(f"   {warning}")

        lines.append("=" * 60)
        return "\n".join(lines)


class DataValidator:
    """数据验证器"""

    def __init__(
        self,
        min_success_rate: float = 0.7,  # 最低成功率70%
        require_all_symbols: bool = False,  # 是否要求所有标的都有数据
        min_bars_per_symbol: int = 100,  # 每个标的最少bar数
    ):
        self.min_success_rate = min_success_rate
        self.require_all_symbols = require_all_symbols
        self.min_bars_per_symbol = min_bars_per_symbol
        self.logger = logger

    def validate(
        self,
        universe: List[str],
        bars_map: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        warmup_days: int = 0,
    ) -> DataValidationResult:
        """
        验证回测数据

        Args:
            universe: 配置的标的列表
            bars_map: 加载的数据（timeframe -> DataFrame）
            start_date: 回测开始日期
            end_date: 回测结束日期
            warmup_days: 预热期天数

        Returns:
            验证结果
        """
        result = DataValidationResult(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            warmup_days=warmup_days,
        )

        # 1. 检查是否加载了任何数据
        if not bars_map:
            result.add_error("没有加载任何数据！")
            return result

        # 2. 检查每个时间框架的数据
        for tf, bars in bars_map.items():
            if bars.empty:
                result.add_error(f"时间框架 {tf} 的数据为空")
                continue

            result.total_bars += len(bars)

            # 3. 检查每个标的
            for symbol in universe:
                symbol_bars = bars[bars["symbol"] == symbol]

                if symbol_bars.empty:
                    result.failed_symbols.append(symbol)
                    self.logger.warning(f"⚠️  {symbol}: 数据为空")
                else:
                    result.loaded_symbols.add(symbol)
                    bar_count = len(symbol_bars)
                    result.bars_per_symbol[symbol] = bar_count

                    # 检查数据量是否足够
                    if bar_count < self.min_bars_per_symbol:
                        reason = f"数据量不足 ({bar_count} < {self.min_bars_per_symbol})"
                        result.incomplete_symbols[symbol] = reason
                        result.add_warning(f"{symbol}: {reason}")

                    # 检查数据时间范围
                    if "datetime" in symbol_bars.columns:
                        data_start = pd.to_datetime(symbol_bars["datetime"].min())
                        data_end = pd.to_datetime(symbol_bars["datetime"].max())

                        required_start = pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)

                        if data_start > required_start:
                            days_missing = (data_start - required_start).days
                            reason = f"缺少前 {days_missing} 天的数据（需要从 {required_start.date()}，实际从 {data_start.date()}）"
                            result.incomplete_symbols[symbol] = reason
                            result.add_warning(f"{symbol}: {reason}")

        # 4. 计算成功率并生成报告
        success_rate = result.get_success_rate()

        if success_rate < self.min_success_rate:
            result.add_error(
                f"数据加载成功率 ({success_rate*100:.1f}%) 低于最低要求 ({self.min_success_rate*100:.1f}%)"
            )

        if self.require_all_symbols and result.failed_symbols:
            result.add_error(
                f"要求所有标的都有数据，但以下标的加载失败: {result.failed_symbols}"
            )

        # 5. 生成汇总日志
        self.logger.info("\n" + result.summary())

        return result

    def validate_or_raise(
        self,
        universe: List[str],
        bars_map: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        warmup_days: int = 0,
    ) -> DataValidationResult:
        """
        验证数据，如果不通过则抛出异常

        Raises:
            ValueError: 如果验证失败
        """
        result = self.validate(universe, bars_map, start_date, end_date, warmup_days)

        if not result.is_valid:
            error_msg = "\n".join([
                "数据验证失败！",
                result.summary(),
            ])
            raise ValueError(error_msg)

        return result

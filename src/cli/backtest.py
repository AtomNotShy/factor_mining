"""
CLI 回测运行模块
处理回测执行、数据检查、结果输出等
"""

import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.cli.config import merge_config, load_config_file
from src.cli.strategies import get_strategy_instance, get_default_universe
from src.config.schema import ConfigSchema
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.data.storage.backtest_store import BacktestStore
from src.evaluation.backtesting import BacktestResult
from src.evaluation.backtesting.config import TradeConfig
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine, UnifiedConfig, FeatureFlag
from src.utils.cli_printer import CLIPrinter
from src.utils.logger import get_logger

logger = get_logger("cli.backtest")


def check_data_range(universe: List[str], start_date: date, end_date: date) -> Optional[str]:
    """检查数据范围，返回错误信息或 None 表示正常"""
    if not universe:
        return None

    today = date.today()

    # 检查日期有效性
    if end_date > today:
        return f"结束日期 {end_date} 在今天 {today} 之后"
    if start_date > today:
        return f"开始日期 {start_date} 在今天 {today} 之后"

    # 检查数据文件
    data_dir = Path("./data")
    first_symbol = universe[0]
    data_file = (
        data_dir / "polygon" / "ohlcv" / "adjusted" / "utc" / "1d" / f"{first_symbol}.parquet"
    )
    if not data_file.exists():
        data_file = data_dir / "ib" / "ohlcv" / "1d" / f"{first_symbol}.parquet"

    if data_file.exists():
        try:
            df = pd.read_parquet(data_file)
            if len(df) < 100:
                ib_file = data_dir / "ib" / "ohlcv" / "1d" / f"{first_symbol}.parquet"
                if ib_file.exists():
                    df = pd.read_parquet(ib_file)

            if not df.empty and 'datetime' in df.columns:
                data_end = pd.Timestamp(df['datetime'].max()).date()
                data_start = pd.Timestamp(df['datetime'].min()).date()

                if end_date > data_end:
                    return f"结束日期超过数据范围 ({data_end})"
                if start_date < data_start:
                    return f"开始日期早于数据范围 ({data_start})"
        except Exception:
            pass

    return None


async def run_backtest(args: argparse.Namespace) -> int:
    """运行回测"""
    # 1. 加载配置
    config: Optional[ConfigSchema] = None
    if args.config:
        config = load_config_file(args.config)

    merged_config = merge_config(args, config)

    # 2. 获取策略
    strategy_name = merged_config.strategy.name
    strategy = get_strategy_instance(strategy_name, merged_config.strategy.params)
    if not strategy:
        logger.error(f"未找到策略: {strategy_name}")
        return 1

    # 3. 确定回测参数
    end_date = date.today()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    elif config and config.time_range and config.time_range.end:
        end_date = datetime.strptime(config.time_range.end, "%Y-%m-%d").date()

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    elif config and config.time_range and config.time_range.start:
        start_date = datetime.strptime(config.time_range.start, "%Y-%m-%d").date()
    else:
        start_date = end_date - timedelta(days=args.days)

    # 4. 确定标的
    universe = merged_config.data.universe
    if not universe:
        universe = get_default_universe(type(strategy)) or ["SPY"]

    if args.symbols:
        universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # 5. 检查数据
    data_error = check_data_range(universe, start_date, end_date)
    if data_error:
        logger.error(f"数据检查失败: {data_error}")
        return 1

    # 6. 准备上下文
    ctx = RunContext.create(
        env=Environment.RESEARCH,
        config=merged_config.model_dump(),
        trading_calendar=TradingCalendar()
    )
    ctx.now_utc = datetime.now(timezone.utc)

    # 7. 运行回测
    engine_config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=merged_config.trading.stake_amount,
            commission_rate=merged_config.broker.commission,
            slippage_rate=merged_config.broker.slippage,
        ),
        features=FeatureFlag.ALL,
    )
    engine = UnifiedBacktestEngine(config=engine_config)

    logger.info(f"开始回测: {strategy_name}")
    logger.info(f"  标的池: {universe}")
    logger.info(f"  日期范围: {start_date} ~ {end_date}")
    logger.info(f"  初始资金: ${merged_config.trading.stake_amount:,.2f}")

    result = await engine.run(
        strategies=[strategy],
        universe=universe,
        start=start_date,
        end=end_date,
        ctx=ctx,
        auto_download=args.auto_download,
    )

    # 8. 处理结果
    if isinstance(result, dict) and "error" in result:
        logger.error(f"回测出错: {result['error']}")
        return 1

    logger.info("回测完成")

    # 统一结果格式
    if hasattr(result, 'model_dump'):
        summary = result.model_dump()
    elif hasattr(result, 'to_dict'):
        summary = result.to_dict()
    elif isinstance(result, dict):
        summary = result
    else:
        summary = {"result": result}

    # 保存结果
    store = BacktestStore()
    run_id = summary.get('run_id') or store.generate_id()
    summary['run_id'] = run_id
    try:
        store.save(run_id, summary)
        logger.info(f"结果已保存 (ID: {run_id})")
    except Exception as e:
        logger.warning(f"结果保存失败: {e}")

    # 打印报告
    if isinstance(result, BacktestResult):
        CLIPrinter.print_report(summary, result.to_dict())
    else:
        CLIPrinter.print_report(summary, result)

    print(f"\n详细图表请访问前端: http://localhost:3000")
    print(f"   Run ID: {summary.get('run_id', 'N/A')}")
    print("=" * 50)

    return 0

"""
框架主入口 (CLI)
集成配置、数据加载、回测引擎和策略
支持纯命令行、配置文件、混合三种模式
"""

from __future__ import annotations

import sys
import asyncio
import argparse
import json
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.schema import ConfigSchema, TradingConfig, BrokerConfig, DataConfig, StrategyConfigSchema, RiskConfig
from src.data.loader import HistoryLoader
from src.evaluation.backtesting.unified_engine import UnifiedBacktestEngine, UnifiedConfig, FeatureFlag, BacktestResult
from src.evaluation.backtesting.config import TradeConfig, TimeConfig
from src.evaluation.optimization.bayesian_optimizer import BayesianOptimizer
from src.persistence.db_manager import DatabaseManager
from src.data.storage.backtest_store import BacktestStore
from src.core.context import RunContext, Environment
from src.core.calendar import TradingCalendar
from src.strategies.base.strategy import strategy_registry
from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.utils.logger import get_logger

# 创建 logger
logger = get_logger("cli.main")

# Freqtrade 策略注册表（必须在导入前定义）
_freqtrade_strategy_registry: Dict[str, type] = {}

# 导入用户策略以触发注册
# ETFRotationSimple 等策略会在导入时自动注册到 strategy_registry
try:
    from src.strategies import user_strategies  # noqa: F401
except Exception as e:
    logger.warning(f"用户策略导入失败: {e}")

# 导入 Freqtrade 风格策略以触发注册
try:
    from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy
    from src.strategies.user_strategies.dual_ma import DualMAStrategy
    from src.strategies.user_strategies.mean_reversion import MeanReversionStrategy
    
    # 注册 Freqtrade 策略
    _freqtrade_strategy_registry['etf_momentum'] = ETFMomentumJoinQuantStrategy
    _freqtrade_strategy_registry['etf_momentum_joinquant'] = ETFMomentumJoinQuantStrategy
    _freqtrade_strategy_registry['dual_ma'] = DualMAStrategy
    _freqtrade_strategy_registry['mean_reversion'] = MeanReversionStrategy
    logger.info("Freqtrade 策略已注册")
except Exception as e:
    logger.warning(f"Freqtrade 策略导入失败: {e}")


# ============================================================================
# 配置合并工具函数
# ============================================================================

def _coerce_value(raw: str) -> Any:
    """将字符串值转换为适当的 Python 类型"""
    text = raw.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    if "," in text and not (text.startswith('"') and text.endswith('"')):
        return [item.strip() for item in text.split(",") if item.strip()]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_params(params_json: str, kv_params: List[str]) -> Dict[str, Any]:
    """解析策略参数"""
    params: Dict[str, Any] = {}
    if params_json:
        params.update(json.loads(params_json))
    for item in kv_params:
        if "=" not in item:
            logger.warning(f"参数格式应为 key=value: {item}")
            continue
        key, raw = item.split("=", 1)
        params[key.strip()] = _coerce_value(raw)
    return params


def _resolve_symbols(params: Dict[str, Any], cli_symbols: str) -> List[str]:
    """解析标的列表"""
    if cli_symbols:
        return [s.strip().upper() for s in cli_symbols.split(",") if s.strip()]
    for key in ("etf_pool", "small_cap_pool", "universe", "symbols"):
        value = params.get(key)
        if isinstance(value, list) and value:
            return [str(s).upper() for s in value if str(s).strip()]
    return []


def merge_config(
    cli_args: argparse.Namespace,
    config: Optional[ConfigSchema] = None,
) -> ConfigSchema:
    """
    合并配置：命令行参数 > 配置文件 > 默认值
    
    Args:
        cli_args: 命令行参数
        config: 已加载的配置文件（可选）
    
    Returns:
        合并后的完整配置
    """
    # 1. 如果没有配置文件，创建默认配置
    if config is None:
        config = ConfigSchema(
            trading=TradingConfig(
                stake_amount=cli_args.initial_capital or 100000,
                max_open_trades=5,
                timeframe="1d",
                dry_run=True,
            ),
            broker=BrokerConfig(
                name="simulated",
                commission=cli_args.commission if cli_args.commission is not None else 0.001,
                slippage=cli_args.slippage if cli_args.slippage is not None else 0.0005,
            ),
            data=DataConfig(
                datadir="./data",
                startup_candle_count=200,
                universe=[],
                benchmark_symbol="SPY",
            ),
            strategy=StrategyConfigSchema(
                name=cli_args.strategy or "VectorizedRSISignal",
                params={},
            ),
        )
    else:
        # 2. 用命令行参数覆盖配置文件
        if cli_args.strategy:
            config.strategy.name = cli_args.strategy
        
        if cli_args.initial_capital:
            config.trading.stake_amount = cli_args.initial_capital
        
        if cli_args.commission is not None:
            config.broker.commission = cli_args.commission
        
        if cli_args.slippage is not None:
            config.broker.slippage = cli_args.slippage
        
        if cli_args.symbols:
            symbols = [s.strip().upper() for s in cli_args.symbols.split(",") if s.strip()]
            if symbols:
                if "small_cap_pool" in config.strategy.params:
                    config.strategy.params["small_cap_pool"] = symbols
                else:
                    config.strategy.params["etf_pool"] = symbols
                config.data.universe = symbols
        
        # 合并策略参数
        if cli_args.params or cli_args.param:
            cli_params = _parse_params(cli_args.params or "{}", cli_args.param or [])
            if cli_params:
                # 深度合并策略参数
                existing_params = config.strategy.params.copy()
                existing_params.update(cli_params)
                config.strategy.params = existing_params
    
    return config


def get_strategy_class(strategy_name: str):
    """获取策略类 - 支持多种名称格式和 Freqtrade 策略"""
    if not strategy_name:
        return None
    
    # 尝试多种格式：原始、大写、小写、驼峰转下划线
    candidates = [
        strategy_name,
        strategy_name.lower(),
        strategy_name.upper(),
        strategy_name.replace("_", "").lower(),  # etf_rotation_simple -> etfrotationsimple
    ]
    
    # 驼峰转下划线
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', strategy_name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    candidates.append(s2)
    
    # 去重
    candidates = list(dict.fromkeys(candidates))
    
    # 1. 首先尝试 Freqtrade 策略注册表
    for name in candidates:
        if name in _freqtrade_strategy_registry:
            return _freqtrade_strategy_registry[name]
    
    # 2. 然后尝试 v2 策略注册表
    for name in candidates:
        strategy_class = strategy_registry.get_strategy(name)
        if strategy_class:
            return strategy_class
    
    return None


def list_strategies():
    """列出所有可用策略（包括 v2 和 Freqtrade）"""
    strategies = strategy_registry.list_strategies()
    
    # 合并 Freqtrade 策略
    freqtrade_strategies = list(_freqtrade_strategy_registry.keys())
    all_strategies = list(set(strategies + freqtrade_strategies))
    
    if not all_strategies:
        print("未发现策略")
        return
    
    print("=" * 50)
    print("可用策略列表")
    print("=" * 50)
    
    # 先显示 Freqtrade 策略
    for name in sorted(freqtrade_strategies):
        strategy_class = _freqtrade_strategy_registry.get(name)
        if strategy_class:
            desc = getattr(strategy_class, '__doc__', '') or getattr(strategy_class, 'strategy_name', name)
            desc = desc.strip().split('\n')[0] if isinstance(desc, str) else name
            print(f"  • {name} [Freqtrade]: {desc}")
    
    # 然后显示 v2 策略
    for name in sorted(strategies):
        if name not in freqtrade_strategies:
            strategy_class = strategy_registry.get_strategy(name)
            if strategy_class:
                desc = strategy_class.__doc__ or "无描述"
                desc = desc.strip().split('\n')[0] if desc else "无描述"
                print(f"  • {name}: {desc}")
    
    print("=" * 50)


async def run_backtest(args):
    """运行回测 - 支持三种模式"""
    logger = get_logger("cli.backtest")
    
    # 1. 确定配置来源
    config: Optional[ConfigSchema] = None
    config_path: Optional[Path] = None
    
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            # 尝试不同后缀
            for suffix in ['.yaml', '.yml', '.json']:
                alt_path = config_path.with_suffix(suffix)
                if alt_path.exists():
                    config_path = alt_path
                    break
        
        if config_path and config_path.exists():
            logger.info(f"加载配置文件: {config_path}")
            try:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = ConfigSchema.from_yaml(str(config_path))
                elif config_path.suffix == '.json':
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    config = ConfigSchema(**data)
            except Exception as e:
                logger.error(f"配置文件解析失败: {e}")
                return 1
        else:
            logger.warning(f"配置文件不存在，将使用命令行参数: {args.config}")
    else:
        logger.info("未指定配置文件，将使用命令行参数")
    
    # 2. 合并配置
    merged_config = merge_config(args, config)
    
    # 3. 获取策略
    strategy_name = merged_config.strategy.name
    strategy_class = get_strategy_class(strategy_name)
    if not strategy_class:
        logger.error(f"未找到策略: {strategy_name}")
        available = strategy_registry.list_strategies()
        logger.info(f"可用策略: {', '.join(available)}")
        return 1
    
    # Freqtrade 策略需要实例化，v2 策略可以直接使用类
    if strategy_name in _freqtrade_strategy_registry:
        strategy = strategy_class()
        # 设置策略参数
        if merged_config.strategy.params:
            for key, value in merged_config.strategy.params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
    else:
        # v2 策略
        strategy = strategy_class
        # 设置策略参数
        if merged_config.strategy.params:
            strategy_class.set_params(merged_config.strategy.params)
    
    # 获取默认标的池 (从类属性)
    if hasattr(strategy, 'etf_pool'):
        default_universe = list(strategy.etf_pool) if not isinstance(strategy, type) else list(strategy_class.etf_pool)
    elif hasattr(strategy, 'small_cap_pool'):
        default_universe = list(strategy.small_cap_pool) if not isinstance(strategy, type) else list(strategy_class.small_cap_pool)
    elif hasattr(strategy, 'universe'):
        default_universe = list(strategy.universe) if not isinstance(strategy, type) else list(strategy_class.universe)
    else:
        default_universe = []
    
    # 4. 确定回测参数
    end_date = date.today()
    
    # 优先使用命令行参数
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    # 回退到配置文件
    elif config and config.time_range and config.time_range.end:
        end_date = datetime.strptime(config.time_range.end, "%Y-%m-%d").date()
    
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    # 回退到配置文件
    elif config and config.time_range and config.time_range.start:
        start_date = datetime.strptime(config.time_range.start, "%Y-%m-%d").date()
    else:
        start_date = end_date - timedelta(days=args.days)
    
    # 确定交易标的
    universe = merged_config.data.universe
    if not universe:
        # 优先从策略类属性获取默认标的池
        if hasattr(strategy_class, 'etf_pool'):
            universe = list(strategy_class.etf_pool)
        elif hasattr(strategy_class, 'small_cap_pool'):
            universe = list(strategy_class.small_cap_pool)
        elif hasattr(strategy_class, 'universe'):
            universe = list(strategy_class.universe)
        elif merged_config.strategy.params:
            # 从策略参数获取
            for key in ['etf_pool', 'small_cap_pool', 'universe']:
                if key in merged_config.strategy.params:
                    universe = merged_config.strategy.params[key]
                    break
        if not universe:
            universe = ["SPY"]  # 默认
    
    # 如果用户指定了 symbols，优先使用
    if args.symbols:
        universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    
    # 确定基准标的
    benchmark_symbol = merged_config.data.benchmark_symbol
    if not benchmark_symbol:
        benchmark_symbol = universe[0] if universe else "SPY"
    
    # 5. 准备上下文
    ctx = RunContext.create(
        env=Environment.RESEARCH,
        config=merged_config.model_dump(),
        trading_calendar=TradingCalendar()
    )
    ctx.now_utc = datetime.now(timezone.utc)
    
    # 6. 运行回测引擎 (使用 UnifiedBacktestEngine)
    config = UnifiedConfig(
        trade=TradeConfig(
            initial_capital=merged_config.trading.stake_amount,
            commission_rate=merged_config.broker.commission,
            slippage_rate=merged_config.broker.slippage,
        ),
        features=FeatureFlag.ALL,  # 启用所有特性
    )
    engine = UnifiedBacktestEngine(config=config)
    
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
    
    # 7. 处理结果
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
    
    # 8. 打印报告
    from src.utils.cli_printer import CLIPrinter
    from src.evaluation.backtesting import BacktestResult
    if isinstance(result, BacktestResult):
        CLIPrinter.print_report(summary, result.to_dict())
    else:
        CLIPrinter.print_report(summary, result)
    
    print(f"\n💡 详细图表请访问前端: http://localhost:3000")
    print(f"   Run ID: {summary.get('run_id', 'N/A')}")
    print("=" * 50)
    
    return 0


async def run_download(args):
    """运行数据下载任务"""
    from src.data.downloader import Downloader
    
    symbols = args.symbols.split(",")
    days = args.days
    provider = args.provider
    timeframe = args.timeframe
    
    downloader = Downloader()
    await downloader.download_symbols(
        symbols=symbols,
        days=days,
        timeframe=timeframe,
        provider=provider
    )


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Factor Mining System CLI - 回测和数据管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 纯命令行模式
  python src/main.py backtest --strategy etf_rotation_simple --days 365
  
  # 配置文件模式
  python src/main.py backtest -c config.example.yaml
  
  # 混合模式（配置文件 + 命令行覆盖）
  python src/main.py backtest -c config.yaml --strategy etf_rotation_simple --initial-capital 200000
  
  # 指定日期范围
  python src/main.py backtest --strategy etf_rotation_simple --start 2023-01-01 --end 2024-12-31
  
  # 自定义策略参数
  python src/main.py backtest --strategy etf_rotation_simple --params '{"target_positions": 2}'
  python src/main.py backtest --strategy etf_rotation_simple --param target_positions=2
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # -------------------- Backtest 子命令 --------------------
    bt_parser = subparsers.add_parser("backtest", help="运行回测")
    bt_parser.add_argument("-c", "--config", default="", help="配置文件路径 (YAML/JSON)")
    
    # 策略参数
    bt_parser.add_argument("--strategy", default="", help="策略名称")
    bt_parser.add_argument("--params", default="", help="策略参数 JSON 字符串")
    bt_parser.add_argument(
        "--param", action="append", default=[], help="单个参数 key=value，可重复使用"
    )
    bt_parser.add_argument("--symbols", default="", help="标的列表，逗号分隔")
    
    # 回测范围参数
    bt_parser.add_argument("--start", default="", help="开始日期 YYYY-MM-DD")
    bt_parser.add_argument("--end", default="", help="结束日期 YYYY-MM-DD")
    bt_parser.add_argument("--days", type=int, default=365, help="回测天数（仅在未指定 start 时生效）")
    
    # 资金参数
    bt_parser.add_argument("--initial-capital", type=float, default=None, help="初始资金")
    bt_parser.add_argument("--commission", type=float, default=None, help="手续费率")
    bt_parser.add_argument("--slippage", type=float, default=None, help="滑点率")
    
    # 基准参数
    bt_parser.add_argument("--benchmark", default="", help="基准标的")
    
    # 数据参数
    bt_parser.add_argument(
        "--auto-download", dest="auto_download", action="store_true", default=True,
        help="自动补齐数据（默认开启）"
    )
    bt_parser.add_argument(
        "--no-auto-download", dest="auto_download", action="store_false",
        help="关闭自动补齐数据"
    )
    
    # 列表参数
    bt_parser.add_argument("--list-strategies", action="store_true", help="列出可用策略")
    
    # -------------------- Download 子命令 --------------------
    dl_parser = subparsers.add_parser("download", help="批量下载数据")
    dl_parser.add_argument("--symbols", required=True, help="标的代码，逗号分隔")
    dl_parser.add_argument("--days", type=int, default=365, help="回溯下载天数")
    dl_parser.add_argument("--timeframe", default="1d", help="时间周期 (1d, 1h, etc.)")
    dl_parser.add_argument(
        "--provider", default="auto", choices=["auto", "ib", "polygon"], help="数据源"
    )
    
    return parser


def main():
    """主入口"""
    parser = build_parser()
    args = parser.parse_args()
    
    # 兼容常见误用
    argv = sys.argv[1:]
    argv = ["--help" if arg == "-help" else arg for arg in argv]
    
    if not argv:
        parser.print_help()
        return 0
    
    args = parser.parse_args(argv)
    
    # 处理特殊命令
    if args.command == "backtest":
        if args.list_strategies:
            list_strategies()
            return 0
        asyncio.run(run_backtest(args))
    elif args.command == "download":
        asyncio.run(run_download(args))
    else:
        parser.print_help()
        return 0
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
CLI 配置管理模块
处理配置合并、参数解析等
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.schema import (
    ConfigSchema,
    TradingConfig,
    BrokerConfig,
    DataConfig,
    StrategyConfigSchema,
    RiskConfig,
)


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


def parse_params(params_json: str, kv_params: List[str]) -> Dict[str, Any]:
    """解析策略参数"""
    params: Dict[str, Any] = {}
    if params_json:
        params.update(json.loads(params_json))
    for item in kv_params:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        params[key.strip()] = _coerce_value(raw)
    return params


def resolve_symbols(params: Dict[str, Any], cli_symbols: str) -> List[str]:
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
            risk=RiskConfig(
                stoploss=None,
                trailing_stop=False,
                trailing_stop_positive=None,
                trailing_stop_positive_offset=0.0,
                roi_table={},
                max_open_trades=None,
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
            cli_params = parse_params(cli_args.params or "{}", cli_args.param or [])
            if cli_params:
                existing_params = config.strategy.params.copy()
                existing_params.update(cli_params)
                config.strategy.params = existing_params

    return config


def load_config_file(config_path: str) -> Optional[ConfigSchema]:
    """从文件加载配置"""
    path = Path(config_path)
    if not path.exists():
        # 尝试不同后缀
        for suffix in ['.yaml', '.yml', '.json']:
            alt_path = path.with_suffix(suffix)
            if alt_path.exists():
                path = alt_path
                break

    if not path.exists():
        return None

    try:
        if path.suffix in ['.yaml', '.yml']:
            return ConfigSchema.from_yaml(str(path))
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ConfigSchema(**data)
    except Exception:
        return None
    return None

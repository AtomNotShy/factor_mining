#!/usr/bin/env python3
"""
命令行回测入口 (v2)

⚠️  已合并到 src/main.py
⚠️  此文件保留用于向后兼容，建议使用新的统一入口

新用法:
  python src/main.py backtest --strategy us_etf_momentum --symbols SPY,QQQ,IWM --days 365
  python src/main.py backtest --strategy us_etf_momentum --params '{"target_positions":2}' --start 2023-01-01 --end 2024-12-31

此文件现在将调用 src/main.py 的 backtest 子命令。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List


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
            raise ValueError(f"参数格式应为 key=value: {item}")
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


def main():
    """主入口 - 解析参数并调用新的统一 CLI"""
    
    parser = argparse.ArgumentParser(
        description="策略回测 CLI (v2) - 已合并到 src/main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
提示: 此工具已合并到 src/main.py backtest 子命令
新用法示例:
  python src/main.py backtest --strategy etf_rotation_simple --days 365
  python src/main.py backtest --strategy etf_rotation_simple --symbols SPY,QQQ --start 2023-01-01

旧用法 (仍兼容):
  python backtest_cli.py --strategy etf_rotation_simple --days 365
        """,
    )
    parser.add_argument("--list-strategies", action="store_true", help="列出可用策略")
    parser.add_argument("--strategy", default="us_etf_momentum", help="策略名称")
    parser.add_argument("--params", default="", help="策略参数 JSON 字符串")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="单个参数 key=value，可重复使用",
    )
    parser.add_argument("--symbols", default="", help="标的列表，逗号分隔")
    parser.add_argument("--start", default="", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=365, help="回测天数（仅在未指定 start 时生效）")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0005, help="手续费率")
    parser.add_argument("--slippage", type=float, default=0.0002, help="滑点率")
    parser.add_argument("--benchmark", default="", help="基准标的（默认使用首个标的）")
    parser.add_argument(
        "--auto-download",
        dest="auto_download",
        action="store_true",
        default=True,
        help="自动补齐数据（默认开启）",
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="关闭自动补齐数据",
    )

    argv = sys.argv[1:]
    # 兼容常见误用：`-help`
    argv = ["--help" if arg == "-help" else arg for arg in argv]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.list_strategies:
        print("可用策略:")
        print("  运行 'python src/main.py backtest --list-strategies' 查看完整列表")
        return 0

    # 构建新 CLI 的参数
    new_argv = ["python", "src/main.py", "backtest"]
    
    # 策略参数
    if args.strategy:
        new_argv.extend(["--strategy", args.strategy])
    
    if args.params:
        new_argv.extend(["--params", args.params])
    
    for param in args.param:
        new_argv.extend(["--param", param])
    
    if args.symbols:
        new_argv.extend(["--symbols", args.symbols])
    
    # 回测范围参数
    if args.start:
        new_argv.extend(["--start", args.start])
    
    if args.end:
        new_argv.extend(["--end", args.end])
    
    if args.days != 365:
        new_argv.extend(["--days", str(args.days)])
    
    # 资金参数
    if args.initial_capital != 100000.0:
        new_argv.extend(["--initial-capital", str(args.initial_capital)])
    
    if args.commission != 0.0005:
        new_argv.extend(["--commission", str(args.commission)])
    
    if args.slippage != 0.0002:
        new_argv.extend(["--slippage", str(args.slippage)])
    
    # 基准参数
    if args.benchmark:
        new_argv.extend(["--benchmark", args.benchmark])
    
    # 数据参数
    if not args.auto_download:
        new_argv.append("--no-auto-download")
    
    # 打印提示
    print("=" * 50)
    print("⚠️  提示: 此工具已合并到 src/main.py")
    print(f"   正在调用: {' '.join(new_argv)}")
    print("=" * 50)
    
    # 执行新的 CLI
    try:
        result = subprocess.run(new_argv, capture_output=False, text=True)
        return result.returncode
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"执行失败: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

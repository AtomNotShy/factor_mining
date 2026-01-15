"""
CLI 命令行参数解析器
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
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

    # Backtest 子命令
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

    # Download 子命令
    dl_parser = subparsers.add_parser("download", help="批量下载数据")
    dl_parser.add_argument("--symbols", required=True, help="标的代码，逗号分隔")
    dl_parser.add_argument("--days", type=int, default=365, help="回溯下载天数")
    dl_parser.add_argument("--timeframe", default="1d", help="时间周期 (1d, 1h, etc.)")
    dl_parser.add_argument(
        "--provider", default="auto", choices=["auto", "ib", "polygon"], help="数据源"
    )

    return parser

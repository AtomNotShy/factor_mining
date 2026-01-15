"""
框架主入口 (CLI)
集成配置、数据加载、回测引擎和策略
支持纯命令行、配置文件、混合三种模式
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cli import build_parser, list_strategies, run_backtest, run_download


def main() -> int:
    """主入口"""
    parser = build_parser()

    # 兼容常见误用
    argv = sys.argv[1:]
    argv = ["--help" if arg == "-help" else arg for arg in argv]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.command == "backtest":
        if args.list_strategies:
            list_strategies()
            return 0
        return asyncio.run(run_backtest(args))
    elif args.command == "download":
        asyncio.run(run_download(args))
    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

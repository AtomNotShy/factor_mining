#!/usr/bin/env python3
"""
架构分析脚本：系统化梳理项目中的文件状态
- 活跃：被引用 >= 2 次
- 保留：被引用 == 1 次
- 废弃：存在替代版本（enhanced_*, *_v2, new_*）
- 死代码：被引用 == 0 次
"""

import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path("/Users/atom/Develop/factor_mining")
SRC_ROOT = PROJECT_ROOT / "src"


def extract_imports(file_path: Path) -> Set[str]:
    """从文件中提取所有导入的模块名"""
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def extract_all_imports(file_path: Path) -> Set[str]:
    """提取所有形式的导入"""
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in ("sys", "os", "typing", "pandas", "numpy", "datetime", "uuid", "json", "logging", "asyncio", "abc", "pathlib", "itertools", "functools", "re", "math", "statistics", "plotly", "pydantic", "fastapi", "aiohttp", "sqlalchemy", "ib_insync"):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def normalize_module_path(import_path: str, src_root: Path) -> Tuple[str, Path]:
    """规范化模块路径，返回模块名和文件路径"""
    parts = import_path.split(".")
    # 移除 "src" 前缀
    if parts and parts[0] == "src":
        parts = parts[1:]

    # 尝试匹配文件
    potential_paths = [
        src_root / "/".join(parts) + ".py",
        src_root / "/".join(parts) / "__init__.py",
    ]
    for p in potential_paths:
        if p.exists():
            return ".".join(parts), p

    return ".".join(parts), None


def get_module_name(file_path: Path, src_root: Path) -> str:
    """获取模块名（如 src.core.types -> core.types）"""
    rel_path = file_path.relative_to(src_root)
    parts = list(rel_path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)


def find_all_python_files(root: Path) -> List[Path]:
    """查找所有 Python 文件"""
    return list(root.rglob("*.py"))


def analyze_file_reference(file_path: Path, all_files: Dict[str, Path]) -> int:
    """分析文件被引用的次数"""
    ref_count = 0
    src_root = PROJECT_ROOT / "src"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception:
        return 0

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module_parts = node.module.split(".")
            module_name = module_parts[0]
            if len(module_parts) > 1:
                # 尝试匹配 from src.xxx import yyy
                full_module = node.module
                if full_module.startswith("src."):
                    full_module = full_module[4:]
                if full_module in all_files:
                    ref_count += 1
                    continue
            # 简单匹配模块名
            for name, path in all_files.items():
                if name.startswith(module_name):
                    ref_count += 1
                    break

    return ref_count


def check_for_alternatives(file_name: str, all_files: List[str]) -> List[str]:
    """检查是否存在替代版本"""
    alternatives = []

    base_name = file_name.replace(".py", "")
    suffixes = ["_v2", "_new", "2", "enhanced_", "new_", "base_"]

    for f in all_files:
        f_base = f.replace(".py", "").replace("/", ".")
        # 检查是否以 base_ 开头（原版保留）
        if f_base.endswith(f".{base_name}") and f_base != f"src.{base_name}":
            continue
        # 检查是否是替代版本
        for suffix in suffixes:
            if f_base.endswith(f".{suffix}{base_name}") or f_base == f"src.{suffix}{base_name}":
                alternatives.append(f)
            if f_base.startswith(f"src.{base_name}.") and base_name in ["engine", "report", "visualizer"]:
                # 增强版可能放在子目录
                if "enhanced" in f.lower() or "v2" in f.lower():
                    alternatives.append(f)

    return list(set(alternatives))


def analyze_architecture():
    """主分析函数"""
    all_files = {}
    file_list = find_all_python_files(SRC_ROOT)

    # 排除 __init__.py 和 __pycache__
    file_list = [f for f in file_list if "__pycache__" not in str(f) and f.name != "__init__.py"]

    # 建立文件名到路径的映射
    for f in file_list:
        module_name = get_module_name(f, SRC_ROOT)
        all_files[module_name] = f

    # 统计每个文件的引用次数
    ref_counts = defaultdict(int)
    import_graph = defaultdict(list)

    for file_path in file_list:
        module_name = get_module_name(file_path, SRC_ROOT)
        from_imports = extract_all_imports(file_path)

        for imp in from_imports:
            if imp.startswith("src."):
                imp = imp[4:]
            if imp in all_files:
                ref_counts[imp] += 1
                import_graph[imp].append(module_name)

    # 分类文件
    categories = {
        "active": [],    # 引用 >= 2
        "retained": [],  # 引用 == 1
        "dead": [],      # 引用 == 0
        "deprecated": [], # 有替代版本
    }

    file_names = [get_module_name(f, SRC_ROOT) for f in file_list]
    deprecated_candidates = []

    # 检测潜在的废弃文件（有替代版本）
    for f in file_list:
        module_name = get_module_name(f, SRC_ROOT)
        parts = module_name.split(".")

        # 检查文件名是否包含常见后缀
        name = parts[-1]
        if name.startswith(("enhanced_", "new_", "v2_", "v2", "base_", "old_")):
            continue  # 这些是替代版本，不是被替代的

        # 查找是否有增强版
        alternatives = []
        for other_file in file_list:
            other_name = get_module_name(other_file, SRC_ROOT)
            other_parts = other_name.split(".")
            other_last = other_parts[-1]

            # 如果存在 enhanced_X 和 X，认为 X 可能被废弃
            if other_last == f"enhanced_{name}" or other_last == f"{name}_v2":
                alternatives.append(other_name)

        if alternatives:
            deprecated_candidates.append((module_name, alternatives))

    # 统计文件列表（排除 __pycache__ 和 __init__.py）
    file_list = [f for f in file_list if "__pycache__" not in str(f) and f.name != "__init__.py"]

    for f in file_list:
        module_name = get_module_name(f, SRC_ROOT)
        ref_count = ref_counts.get(module_name, 0)

        if ref_count == 0:
            categories["dead"].append((module_name, ref_count, []))
        elif ref_count == 1:
            categories["retained"].append((module_name, ref_count, import_graph.get(module_name, [])))
        else:
            categories["active"].append((module_name, ref_count, import_graph.get(module_name, [])))

    # 处理废弃文件
    deprecated_files = []
    for original, alts in deprecated_candidates:
        for f in file_list:
            if get_module_name(f, SRC_ROOT) == original:
                ref_count = ref_counts.get(original, 0)
                deprecated_files.append((original, ref_count, alts))

    return categories, deprecated_files, all_files, import_graph


def print_report():
    """打印分析报告"""
    categories, deprecated_files, all_files, import_graph = analyze_architecture()

    print("=" * 80)
    print("架构分析报告")
    print("=" * 80)

    print("\n【1】活跃模块（被引用 >= 2 次）")
    print("-" * 60)
    for name, count, refs in sorted(categories["active"], key=lambda x: -x[1])[:30]:
        refs_str = ", ".join(refs[:5]) + ("..." if len(refs) > 5 else "")
        print(f"  [{count:3}] {name}")
        print(f"        被: {refs_str}")

    print(f"\n  共 {len(categories['active'])} 个活跃模块")

    print("\n【2】保留模块（被引用 == 1 次）")
    print("-" * 60)
    for name, count, refs in sorted(categories["retained"], key=lambda x: x[0]):
        print(f"  [{count:3}] {name}")
        print(f"        被: {refs[0] if refs else '(无)'}")

    print(f"\n  共 {len(categories['retained'])} 个保留模块")

    print("\n【3】废弃模块（存在增强版本）")
    print("-" * 60)
    if deprecated_files:
        for name, count, alts in sorted(deprecated_files, key=lambda x: -x[1]):
            print(f"  [废弃] {name}")
            print(f"        被引用: {count} 次")
            print(f"        增强版: {', '.join(alts)}")
    else:
        print("  未检测到明确的废弃模块")

    print(f"\n  共 {len(deprecated_files)} 个潜在废弃模块")

    print("\n【4】死代码模块（被引用 == 0 次）")
    print("-" * 60)
    dead_files = [f for f in categories["dead"] if "__pycache__" not in f[0]]
    if dead_files:
        for name, count, _ in sorted(dead_files, key=lambda x: x[0]):
            print(f"  ⚠️  {name}")
    else:
        print("  未检测到死代码")
    print(f"\n  共 {len(dead_files)} 个死代码模块")

    print("\n" + "=" * 80)
    print(f"总计: {len(categories['active']) + len(categories['retained']) + len(dead_files)} 个 Python 模块")
    print("=" * 80)


if __name__ == "__main__":
    print_report()

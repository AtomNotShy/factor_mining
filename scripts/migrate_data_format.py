#!/usr/bin/env python3
"""
迁移数据格式：从 symbol=XXX/timeframe=1d 格式迁移到 XXX/1d/data.parquet 格式
"""

import shutil
from pathlib import Path
import pandas as pd

def migrate_data():
    """迁移数据到统一格式"""
    data_dir = Path("data/bars")

    # 查找所有旧格式的数据
    old_format_dirs = list(data_dir.glob("symbol=*"))

    if not old_format_dirs:
        print("✅ 没有发现旧格式数据")
        return

    print(f"发现 {len(old_format_dirs)} 个旧格式数据")
    print("=" * 60)

    migrated = []
    errors = []

    for symbol_dir in old_format_dirs:
        # 解析symbol
        symbol = symbol_dir.name.replace("symbol=", "")

        # 查找timeframe目录
        timeframe_dirs = list(symbol_dir.glob("timeframe=*"))

        for tf_dir in timeframe_dirs:
            timeframe = tf_dir.name.replace("timeframe=", "")

            # 查找数据文件
            old_files = list(tf_dir.glob("*.parquet"))

            if not old_files:
                print(f"⚠️  {symbol}/{timeframe} - 没有数据文件")
                continue

            old_file = old_files[0]  # 取第一个文件

            # 新路径
            new_dir = data_dir / symbol / timeframe
            new_file = new_dir / "data.parquet"

            try:
                # 读取数据
                df = pd.read_parquet(old_file)

                # 创建新目录
                new_dir.mkdir(parents=True, exist_ok=True)

                # 保存到新路径
                df.to_parquet(new_file)

                print(f"✅ {symbol}/{timeframe} - {len(df)} rows")
                migrated.append((symbol, timeframe))

            except Exception as e:
                print(f"❌ {symbol}/{timeframe} - 错误: {e}")
                errors.append((symbol, timeframe, str(e)))

    print("\n" + "=" * 60)
    print("迁移完成")
    print("=" * 60)
    print(f"✅ 成功: {len(migrated)}")
    print(f"❌ 失败: {len(errors)}")

    if migrated:
        print("\n✅ 成功迁移:")
        for symbol, tf in migrated:
            print(f"   {symbol}/{tf}")

    if errors:
        print("\n❌ 迁移失败:")
        for symbol, tf, err in errors:
            print(f"   {symbol}/{tf}: {err}")

    # 询问是否删除旧文件
    if migrated:
        print("\n" + "=" * 60)
        response = input("是否删除旧格式文件？(y/N): ").strip().lower()

        if response == 'y':
            for symbol_dir in old_format_dirs:
                try:
                    shutil.rmtree(symbol_dir)
                    print(f"🗑️  删除: {symbol_dir}")
                except Exception as e:
                    print(f"❌ 删除失败 {symbol_dir}: {e}")
            print("✅ 旧文件已清理")
        else:
            print("⏭️  保留旧文件")

if __name__ == "__main__":
    migrate_data()

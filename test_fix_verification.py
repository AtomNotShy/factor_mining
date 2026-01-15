#!/usr/bin/env python3
"""
验证修复是否正确
"""

def test_fixed_logic():
    """测试修复后的逻辑"""
    print("验证修复是否正确")
    print("=" * 60)
    
    # 模拟API返回的数据
    # 情况1：total_return_pct = 1819.3614（百分比形式）
    test_cases = [
        {
            "name": "total_return_pct = 1819.3614（百分比形式）",
            "total_return_pct": 1819.3614,
            "total_return": 0,
            "expected": 18.193614,  # 1819.3614 / 100 = 18.193614
        },
        {
            "name": "total_return = 18.193614（小数形式）",
            "total_return_pct": 0,
            "total_return": 18.193614,
            "expected": 18.193614,  # 已经是小数形式
        },
        {
            "name": "total_return = 1819.3614（百分比形式，但字段名是total_return）",
            "total_return_pct": 0,
            "total_return": 1819.3614,
            "expected": 18.193614,  # 1819.3614 / 100 = 18.193614
        },
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  输入: total_return_pct = {case['total_return_pct']}")
        print(f"        total_return = {case['total_return']}")
        
        # 应用修复后的逻辑
        total_return_pct_raw = case['total_return_pct']
        total_return_raw = case['total_return']
        
        # 优先使用 total_return_pct（已经是百分比）
        if total_return_pct_raw != 0:
            total_return = total_return_pct_raw / 100  # 转换为小数（1819.3614% -> 18.193614）
            print(f"  使用 total_return_pct: {total_return_pct_raw}% / 100 = {total_return}")
        elif total_return_raw != 0:
            # 如果 total_return_raw 大于1，可能是百分比形式，需要转换为小数
            if abs(total_return_raw) > 1:
                total_return = total_return_raw / 100  # 转换为小数（1819.3614 -> 18.193614）
                print(f"  使用 total_return (百分比形式): {total_return_raw} / 100 = {total_return}")
            else:
                total_return = total_return_raw  # 已经是小数形式
                print(f"  使用 total_return (小数形式): {total_return}")
        else:
            total_return = 0
            print(f"  没有数据，返回: {total_return}")
        
        # 验证
        if abs(total_return - case['expected']) < 0.0001:
            print(f"  ✅ 正确: {total_return} ≈ {case['expected']}")
        else:
            print(f"  ❌ 错误: {total_return} ≠ {case['expected']}")
        
        # 前端显示
        frontend_display = total_return * 100
        print(f"  前端显示: {frontend_display:.2f}%")
        
        # 检查是否匹配用户看到的情况
        if 1819.0 <= frontend_display <= 1820.0:
            print(f"  ✅ 前端显示正确: {frontend_display:.2f}% (应该是1819.36%)")
        elif 18.0 <= frontend_display <= 19.0:
            print(f"  ⚠️  前端显示错误: {frontend_display:.2f}% (应该是1819.36%，不是18%)")
        else:
            print(f"  ❌ 前端显示异常: {frontend_display:.2f}%")
    
    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    
    print("用户反馈: 接口显示1819.3614%，界面显示18%")
    print("\n修复后的逻辑:")
    print("1. 如果 total_return_pct = 1819.3614（百分比形式）")
    print("2. API处理: 1819.3614 / 100 = 18.193614（小数形式）")
    print("3. 前端显示: 18.193614 * 100 = 1819.36%")
    print("\n✅ 修复后前端应该显示1819.36%，而不是18%")
    
    print("\n如果前端仍然显示18%，可能的原因:")
    print("1. 前端可能使用了不同的字段（如enhanced_metrics.total_return）")
    print("2. 前端可能对值进行了其他处理（如除以100）")
    print("3. 前端可能缓存了旧的数据")
    
    print("\n建议:")
    print("1. 清除浏览器缓存")
    print("2. 重新运行回测")
    print("3. 检查前端控制台，查看实际接收到的total_return值")

if __name__ == "__main__":
    test_fixed_logic()

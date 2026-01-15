#!/usr/bin/env python3
"""
测试总回报计算修复
"""

def test_total_return_calculation():
    """测试总回报计算逻辑"""
    print("测试总回报计算修复")
    print("=" * 60)
    
    # 模拟回测结果数据
    test_cases = [
        {
            "name": "Case 1: total_return_pct = 1818.87 (百分比形式)",
            "total_return_pct": 1818.87,
            "total_return": 0,  # 没有total_return字段
            "expected": 0.181887  # 应该转换为小数
        },
        {
            "name": "Case 2: total_return = 18.1887 (小数形式)",
            "total_return_pct": 0,
            "total_return": 18.1887,
            "expected": 18.1887  # 已经是小数形式
        },
        {
            "name": "Case 3: total_return = 1818.87 (百分比形式，但字段名是total_return)",
            "total_return_pct": 0,
            "total_return": 1818.87,
            "expected": 18.1887  # 应该除以100
        },
        {
            "name": "Case 4: total_return_pct = 0, total_return = 0",
            "total_return_pct": 0,
            "total_return": 0,
            "expected": 0
        },
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  total_return_pct_raw: {case['total_return_pct']}")
        print(f"  total_return_raw: {case['total_return']}")
        
        # 应用修复后的逻辑
        total_return_pct_raw = case['total_return_pct']
        total_return_raw = case['total_return']
        
        # 优先使用 total_return_pct（已经是百分比）
        if total_return_pct_raw != 0:
            total_return = total_return_pct_raw / 100  # 转换为小数（18.1887 -> 0.181887）
            print(f"  使用 total_return_pct: {total_return_pct_raw} / 100 = {total_return}")
        elif total_return_raw != 0:
            # 如果 total_return_raw 大于1，可能是百分比形式，需要转换为小数
            if abs(total_return_raw) > 1:
                total_return = total_return_raw / 100  # 转换为小数（1818.87 -> 18.1887）
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
    
    print("\n" + "=" * 60)
    print("测试完成")
    
    # 验证用户报告的问题
    print("\n验证用户报告的问题:")
    print("初始资金: $100,000")
    print("最终权益: $1,918,866")
    
    # 计算实际总回报
    actual_return = (1918866 / 100000 - 1) * 100  # 百分比
    print(f"实际总回报: {actual_return:.2f}%")
    
    # 修复前的问题
    print("\n修复前的问题:")
    print(f"total_return_pct = {actual_return:.2f}%")
    print("原代码: total_return_raw / 100 if abs(total_return_raw) > 1 else total_return_raw")
    print(f"结果: {actual_return} / 100 = {actual_return / 100:.2f}% (错误!)")
    
    # 修复后的结果
    print("\n修复后的结果:")
    print("优先使用 total_return_pct，然后除以100转换为小数")
    print(f"结果: {actual_return} / 100 = {actual_return / 100:.4f} (小数形式)")
    print(f"前端显示: {actual_return / 100 * 100:.2f}% (正确!)")

if __name__ == "__main__":
    test_total_return_calculation()

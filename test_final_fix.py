#!/usr/bin/env python3
"""
测试最终修复：验证总回报计算是否正确
"""

def simulate_backtest_result():
    """模拟回测结果"""
    initial_capital = 100000.0
    final_equity = 1918866.0
    
    # 计算总回报（小数形式）
    total_return_pct_decimal = (final_equity / initial_capital) - 1  # 18.18866
    total_return_pct_percent = total_return_pct_decimal * 100  # 1818.866
    
    print("模拟回测结果:")
    print(f"  初始资金: ${initial_capital:,.2f}")
    print(f"  最终权益: ${final_equity:,.2f}")
    print(f"  总回报（小数形式）: {total_return_pct_decimal:.6f}")
    print(f"  总回报（百分比形式）: {total_return_pct_percent:.2f}%")
    
    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct_decimal": total_return_pct_decimal,
        "total_return_pct_percent": total_return_pct_percent,
    }

def test_api_logic(data):
    """测试API逻辑"""
    print("\n测试API逻辑:")
    
    # 模拟EnhancedBacktestReport.to_dict()的输出
    # total_return_pct字段被乘以100（从小数形式变为百分比形式）
    enhanced_report_dict = {
        "total_return_pct": data["total_return_pct_decimal"] * 100,  # 18.18866 * 100 = 1818.866
    }
    
    print(f"  EnhancedBacktestReport.to_dict()输出:")
    print(f"    total_return_pct: {enhanced_report_dict['total_return_pct']:.2f} (百分比形式)")
    
    # 模拟API代码的处理
    total_return_pct_raw = enhanced_report_dict.get("total_return_pct", 0)
    
    # 修复后的API逻辑
    if total_return_pct_raw != 0:
        total_return = total_return_pct_raw / 100  # 转换为小数（1818.866 -> 18.18866）
        print(f"  API处理: {total_return_pct_raw:.2f} / 100 = {total_return:.6f} (小数形式)")
    else:
        total_return = 0
    
    # 前端显示
    frontend_display = total_return * 100  # 18.18866 * 100 = 1818.866%
    print(f"  前端显示: {frontend_display:.2f}%")
    
    return total_return, frontend_display

def test_old_bug(data):
    """测试旧的bug"""
    print("\n测试旧的bug（修复前）:")
    
    # 模拟旧的API逻辑
    total_return_raw = data["total_return_pct_percent"]  # 1818.866
    
    # 旧的逻辑：total_return_raw / 100 if abs(total_return_raw) > 1 else total_return_raw
    total_return = total_return_raw / 100 if abs(total_return_raw) > 1 else total_return_raw
    
    print(f"  旧API逻辑: {total_return_raw:.2f} / 100 = {total_return:.6f} (小数形式)")
    
    # 前端显示
    frontend_display = total_return * 100  # 18.18866 * 100 = 1818.866%
    print(f"  前端显示: {frontend_display:.2f}%")
    
    # 注意：旧的逻辑实际上也是正确的！问题可能在其他地方
    
    return total_return, frontend_display

def main():
    print("=" * 60)
    print("测试总回报计算修复")
    print("=" * 60)
    
    # 模拟数据
    data = simulate_backtest_result()
    
    # 测试修复后的逻辑
    new_total_return, new_display = test_api_logic(data)
    
    # 测试旧的bug
    old_total_return, old_display = test_old_bug(data)
    
    print("\n" + "=" * 60)
    print("分析:")
    print("=" * 60)
    
    print(f"实际总回报: {data['total_return_pct_percent']:.2f}%")
    print(f"修复后前端显示: {new_display:.2f}%")
    print(f"修复前前端显示: {old_display:.2f}%")
    
    # 检查差异
    diff = abs(new_display - old_display)
    if diff < 0.01:
        print(f"\n✅ 修复前后结果一致: {new_display:.2f}%")
        print("这意味着旧的API逻辑实际上也是正确的！")
        print("\n可能的问题:")
        print("1. 前端可能没有正确乘以100")
        print("2. 数据可能来自不同的来源（不是EnhancedBacktestReport）")
        print("3. 可能有其他代码路径导致问题")
    else:
        print(f"\n❌ 修复前后结果不一致: 差异 = {diff:.2f}%")
    
    print("\n建议:")
    print("1. 检查前端是否确实将total_return乘以100")
    print("2. 检查是否有其他代码路径返回不同的total_return值")
    print("3. 确保所有组件都使用相同的格式（小数形式或百分比形式）")

if __name__ == "__main__":
    main()

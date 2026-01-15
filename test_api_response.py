#!/usr/bin/env python3
"""
测试API返回的数据格式
"""

import json

# 模拟API返回的数据
def simulate_api_response():
    """模拟API返回的数据"""
    
    # 情况1：total_return是百分比形式（如1819.3614）
    response_percent = {
        "total_return": 1819.3614,  # 百分比形式
        "total_return_pct": 1819.3614,  # 百分比形式
        "final_equity": 1919361.4,
        "initial_capital": 100000.0,
    }
    
    # 情况2：total_return是小数形式（如18.193614）
    response_decimal = {
        "total_return": 18.193614,  # 小数形式
        "total_return_pct": 1819.3614,  # 百分比形式
        "final_equity": 1919361.4,
        "initial_capital": 100000.0,
    }
    
    # 情况3：只有total_return_pct字段
    response_only_pct = {
        "total_return_pct": 1819.3614,  # 百分比形式
        "final_equity": 1919361.4,
        "initial_capital": 100000.0,
    }
    
    return response_percent, response_decimal, response_only_pct

def test_frontend_display(response, name):
    """测试前端如何显示"""
    print(f"\n{name}:")
    print(f"  API返回: total_return = {response.get('total_return', 'N/A')}")
    print(f"           total_return_pct = {response.get('total_return_pct', 'N/A')}")
    
    total_return = response.get('total_return')
    if total_return is not None:
        # 前端代码：backtest.total_return >= -1 ? backtest.total_return : 0
        value_to_display = total_return if total_return >= -1 else 0
        frontend_result = value_to_display * 100
        print(f"  前端处理: {total_return} * 100 = {frontend_result:.2f}%")
        
        # 检查是否匹配用户看到的情况
        if 17.9 <= frontend_result <= 18.5:
            print(f"  ✅ 匹配用户看到的18%: {frontend_result:.2f}%")
        else:
            print(f"  ❌ 不匹配用户看到的18%: {frontend_result:.2f}%")
    else:
        print("  total_return字段不存在")

def main():
    print("=" * 60)
    print("测试API返回数据格式")
    print("=" * 60)
    
    # 模拟API响应
    response_percent, response_decimal, response_only_pct = simulate_api_response()
    
    # 测试各种情况
    test_frontend_display(response_percent, "情况1: total_return是百分比形式(1819.3614)")
    test_frontend_display(response_decimal, "情况2: total_return是小数形式(18.193614)")
    test_frontend_display(response_only_pct, "情况3: 只有total_return_pct字段")
    
    print("\n" + "=" * 60)
    print("分析:")
    print("=" * 60)
    
    print("用户反馈: 接口显示1819.3614%，界面显示18%")
    print("\n这意味着:")
    print("1. API返回的total_return字段是1819.3614（百分比形式）")
    print("2. 前端将1819.3614 * 100 = 181936.14%（错误！）")
    print("3. 但用户看到的是18%，不是181936.14%")
    print("\n矛盾点: 如果total_return是1819.3614，前端应该显示181936.14%，而不是18%")
    print("\n可能的情况:")
    print("1. 前端代码可能被修改了，没有乘以100")
    print("2. 前端可能使用了不同的字段（如enhanced_metrics.total_return）")
    print("3. 前端可能对值进行了其他处理（如除以100）")
    
    print("\n建议检查:")
    print("1. 前端实际接收到的total_return值是多少")
    print("2. 前端是否使用了enhanced_metrics.total_return字段")
    print("3. 前端是否有其他处理逻辑（如格式化函数）")

if __name__ == "__main__":
    main()

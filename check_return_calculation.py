# 验证总回报计算
initial_capital = 100000.0
final_equity = 1918866.0
total_return = (final_equity / initial_capital) - 1
total_return_pct = total_return * 100

print(f'初始资金: ${initial_capital:,.2f}')
print(f'最终权益: ${final_equity:,.2f}')
print(f'总回报: {total_return:.4f} ({total_return_pct:.2f}%)')
print(f'检查: 100,000 * (1 + {total_return:.4f}) = {initial_capital * (1 + total_return):,.2f}')

# 检查是否是显示格式问题
# 如果total_return_pct是1818.87%，但显示为18%，可能是除以100了
print(f'\n可能的显示问题:')
print(f'1818.87% 显示为 18% (可能是除以100后取整)')
print(f'1818.87 / 100 = {1818.87 / 100:.2f}%')
print(f'或者可能是格式化为整数: {int(1818.87):d}%')

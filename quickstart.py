#!/usr/bin/env python3
"""
快速开始：展示CollapseGPT算法的基本用法
"""
import sys
sys.path.append('src')

from utils import Item, PHI, THEORETICAL_BOUND
from collapse_gpt import collapse_knapsack
from dynamic_programming import dp_knapsack

def simple_example():
    """一个简单的例子"""
    print("🎒 背包问题 - Collapse理论演示\n")
    
    # 创建物品
    items = [
        Item(id=1, weight=10, value=60),
        Item(id=2, weight=20, value=100),
        Item(id=3, weight=30, value=120),
    ]
    
    capacity = 40
    
    print("物品:")
    for item in items:
        print(f"  物品{item.id}: 重量={item.weight}, 价值={item.value}")
    print(f"背包容量: {capacity}\n")
    
    # 使用CollapseGPT
    print("1. CollapseGPT算法:")
    collapse_items = [Item(id=i.id, weight=i.weight, value=i.value) for i in items]
    selected, time_taken = collapse_knapsack(collapse_items, capacity)
    
    print(f"  选中: {[item.id for item in selected]}")
    print(f"  总价值: {sum(item.value for item in selected)}")
    print(f"  时间: {time_taken*1000:.2f}ms\n")
    
    # 使用动态规划
    print("2. 动态规划算法:")
    dp_items = [Item(id=i.id, weight=i.weight, value=i.value) for i in items]
    dp_selected, dp_time = dp_knapsack(dp_items, capacity)
    
    print(f"  选中: {[item.id for item in dp_selected]}")
    print(f"  总价值: {sum(item.value for item in dp_selected)}")
    print(f"  时间: {dp_time*1000:.2f}ms\n")
    
    # 性能对比
    collapse_value = sum(item.value for item in selected)
    dp_value = sum(item.value for item in dp_selected)
    
    print("3. 性能对比:")
    print(f"  近似比: {collapse_value/dp_value:.2%}")
    print(f"  速度提升: {dp_time/time_taken:.1f}x")
    print(f"  理论保证: ≥ {THEORETICAL_BOUND:.1%}")

if __name__ == "__main__":
    simple_example()
    
    print("\n" + "-"*50)
    print("💡 想了解更多？")
    print("  - 阅读文档: docs/01-basic-concepts.md")
    print("  - 运行演示: python3 src/basic_knapsack_demo.py")
    print("  - 大规模实验: python3 src/collapse_vs_dp_experiment.py") 
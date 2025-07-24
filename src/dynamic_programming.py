# src/dynamic_programming.py
from typing import List, Tuple, Set
import time
from utils import Item

def dp_knapsack(items: List[Item], capacity: int) -> Tuple[List[Item], float]:
    """
    使用动态规划解决01背包问题
    
    参数:
        items: 物品列表
        capacity: 背包容量
    
    返回:
        (选中的物品列表, 运行时间)
    """
    start_time = time.time()
    n = len(items)
    
    # 创建DP表
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # 填充DP表
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if items[i-1].weight <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w - items[i-1].weight] + items[i-1].value
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # 回溯找出选中的物品
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(items[i-1])
            w -= items[i-1].weight
    
    elapsed_time = time.time() - start_time
    return selected, elapsed_time

def dp_knapsack_optimized(items: List[Item], capacity: int) -> Tuple[List[Item], float]:
    """
    空间优化的动态规划版本
    """
    start_time = time.time()
    n = len(items)
    
    # 使用一维数组
    dp = [0] * (capacity + 1)
    
    # 记录选择路径
    selected_at = [set() for _ in range(capacity + 1)]
    
    for i in range(n):
        # 从后往前遍历避免覆盖
        for w in range(capacity, items[i].weight - 1, -1):
            if dp[w - items[i].weight] + items[i].value > dp[w]:
                dp[w] = dp[w - items[i].weight] + items[i].value
                selected_at[w] = selected_at[w - items[i].weight].copy()
                selected_at[w].add(i)
    
    # 构建选中物品列表
    selected = [items[i] for i in selected_at[capacity]]
    
    elapsed_time = time.time() - start_time
    return selected, elapsed_time

def dp_knapsack_with_stats(items: List[Item], capacity: int) -> Tuple[List[Item], float, dict]:
    """
    带统计信息的动态规划
    """
    start_time = time.time()
    n = len(items)
    
    # 统计信息
    stats = {
        'table_size': (n + 1) * (capacity + 1),
        'iterations': 0,
        'max_value': 0,
        'fill_pattern': []
    }
    
    # 创建DP表
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # 填充DP表
    for i in range(1, n + 1):
        row_max = 0
        for w in range(capacity + 1):
            stats['iterations'] += 1
            
            if items[i-1].weight <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w - items[i-1].weight] + items[i-1].value
                )
            else:
                dp[i][w] = dp[i-1][w]
            
            row_max = max(row_max, dp[i][w])
        
        stats['fill_pattern'].append(row_max)
        stats['max_value'] = max(stats['max_value'], row_max)
    
    # 回溯找出选中的物品
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(items[i-1])
            w -= items[i-1].weight
    
    elapsed_time = time.time() - start_time
    stats['time'] = elapsed_time
    
    return selected, elapsed_time, stats

def compare_dp_versions(items: List[Item], capacity: int):
    """比较不同版本的动态规划实现"""
    print("\n=== 动态规划版本对比 ===")
    
    # 标准版本
    selected1, time1 = dp_knapsack(items.copy(), capacity)
    value1 = sum(item.value for item in selected1)
    
    # 优化版本
    selected2, time2 = dp_knapsack_optimized(items.copy(), capacity)
    value2 = sum(item.value for item in selected2)
    
    # 带统计版本
    selected3, time3, stats = dp_knapsack_with_stats(items.copy(), capacity)
    value3 = sum(item.value for item in selected3)
    
    print(f"标准版本: 价值={value1}, 时间={time1*1000:.3f}ms")
    print(f"优化版本: 价值={value2}, 时间={time2*1000:.3f}ms")
    print(f"统计版本: 价值={value3}, 时间={time3*1000:.3f}ms")
    print(f"\n统计信息:")
    print(f"  表格大小: {stats['table_size']}")
    print(f"  迭代次数: {stats['iterations']}")
    print(f"  空间节省: {1 - (capacity + 1) / stats['table_size']:.1%}")
    
    return {
        'standard': {'value': value1, 'time': time1},
        'optimized': {'value': value2, 'time': time2},
        'with_stats': {'value': value3, 'time': time3, 'stats': stats}
    } 
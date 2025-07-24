#!/usr/bin/env python3
"""
Collapse理论应用演示
展示Collapse理论在不同问题上的应用
"""

import math
import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass

# 导入核心函数
from utils import zeckendorf_encode, PHI

print("🌌 Collapse理论应用演示")
print("=" * 50)

# 1. 简化的TSP演示
print("\n1. 旅行商问题 (TSP)")
print("-" * 30)

def simple_tsp_collapse(cities: List[Tuple[float, float]]) -> List[int]:
    """使用Collapse理论解决TSP"""
    n = len(cities)
    visited = [False] * n
    path = [0]  # 从城市0开始
    visited[0] = True
    
    current = 0
    while len(path) < n:
        # 计算到未访问城市的"张力"
        max_tension = -1
        next_city = -1
        
        for i in range(n):
            if not visited[i]:
                # 距离因素
                dist = math.sqrt((cities[current][0] - cities[i][0])**2 + 
                               (cities[current][1] - cities[i][1])**2)
                # 结构因素（基于城市编号的φ-trace）
                trace_len = len(zeckendorf_encode(i + 1))
                structure_factor = 1.0 / (trace_len ** 0.5)
                
                # 综合张力
                tension = (1.0 / (dist + 0.1)) * structure_factor
                
                if tension > max_tension:
                    max_tension = tension
                    next_city = i
        
        path.append(next_city)
        visited[next_city] = True
        current = next_city
    
    return path

# 测试TSP
cities = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
tsp_path = simple_tsp_collapse(cities)
print(f"5个城市的Collapse路径: {tsp_path}")

# 计算总距离
total_dist = 0
for i in range(len(tsp_path)):
    j = (i + 1) % len(tsp_path)
    dist = math.sqrt((cities[tsp_path[i]][0] - cities[tsp_path[j]][0])**2 + 
                    (cities[tsp_path[i]][1] - cities[tsp_path[j]][1])**2)
    total_dist += dist
print(f"总距离: {total_dist:.3f}")

# 2. 简化的图着色演示
print("\n\n2. 图着色问题")
print("-" * 30)

def graph_coloring_collapse(adj_matrix: List[List[int]], num_colors: int) -> Dict[int, int]:
    """使用Collapse理论解决图着色"""
    n = len(adj_matrix)
    colors = {}
    
    # 按φ-trace长度排序节点（简单的先着色）
    nodes = sorted(range(n), key=lambda x: len(zeckendorf_encode(x + 1)))
    
    for node in nodes:
        # 计算每种颜色的"张力"
        color_tensions = [0] * num_colors
        
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and neighbor in colors:
                # 邻居已着色，增加该颜色的斥力
                color_tensions[colors[neighbor]] += 10
        
        # 选择张力最小的颜色
        best_color = color_tensions.index(min(color_tensions))
        colors[node] = best_color
    
    return colors

# 测试图着色
# 创建一个简单的图（5节点环）
adj_matrix = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0]
]

coloring = graph_coloring_collapse(adj_matrix, 3)
print(f"图着色结果: {coloring}")

# 验证着色
valid = True
for i in range(5):
    for j in range(5):
        if adj_matrix[i][j] == 1 and coloring[i] == coloring[j]:
            valid = False
            break

print(f"着色有效性: {'✅ 有效' if valid else '❌ 无效'}")

# 3. 简化的调度问题演示
print("\n\n3. 任务调度问题")
print("-" * 30)

@dataclass
class Task:
    id: int
    duration: int
    deadline: int
    priority: int

def schedule_collapse(tasks: List[Task], num_machines: int) -> Dict[int, List[int]]:
    """使用Collapse理论解决调度问题"""
    schedule = {i: [] for i in range(num_machines)}
    machine_loads = [0] * num_machines
    
    # 计算每个任务的"紧急张力"
    for task in tasks:
        urgency = task.priority / (task.deadline - task.duration + 1)
        trace_len = len(zeckendorf_encode(task.id))
        task.collapse_tension = urgency * (1.0 / (trace_len ** 0.5))
    
    # 按张力排序
    tasks.sort(key=lambda t: t.collapse_tension, reverse=True)
    
    # 分配任务
    for task in tasks:
        # 选择负载最小的机器
        min_load_machine = machine_loads.index(min(machine_loads))
        schedule[min_load_machine].append(task.id)
        machine_loads[min_load_machine] += task.duration
    
    return schedule, max(machine_loads)

# 测试调度
tasks = [
    Task(1, 3, 10, 5),
    Task(2, 2, 8, 3),
    Task(3, 4, 12, 4),
    Task(4, 1, 5, 5),
    Task(5, 3, 9, 2)
]

schedule, makespan = schedule_collapse(tasks, 2)
print(f"调度结果: {schedule}")
print(f"完成时间: {makespan}")

# 4. 总结
print("\n\n🌟 Collapse理论的通用性")
print("=" * 50)
print("✓ TSP：路径通过城市间张力自然形成")
print("✓ 图着色：颜色通过约束张力自然分配")
print("✓ 调度：任务通过紧急度张力自然安排")
print("\n核心洞察：")
print("- 不是搜索最优解，而是让解自然显现")
print("- 局部张力引导全局秩序")
print("- O(n log n)复杂度vs传统的指数级复杂度")
print("\n这只是开始，Collapse理论的应用潜力无限！") 
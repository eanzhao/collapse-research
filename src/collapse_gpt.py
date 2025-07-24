# src/collapse_gpt.py
from typing import List, Tuple, Dict
import time
from utils import Item, zeckendorf_encode, calculate_total_value, calculate_total_weight

def collapse_knapsack(items: List[Item], capacity: int, s: float = 0.5) -> Tuple[List[Item], float]:
    """
    使用CollapseGPT算法解决01背包问题
    
    参数:
        items: 物品列表
        capacity: 背包容量
        s: 临界指数（默认0.5）
    
    返回:
        (选中的物品列表, 运行时间)
    """
    start_time = time.time()
    
    # Step 1: φ-trace编码
    for item in items:
        item.phi_trace = zeckendorf_encode(item.id)
        item.trace_length = len(item.phi_trace)
    
    # Step 2: 计算collapse张力
    for item in items:
        if item.trace_length > 0:
            item.tension = 1.0 / (item.trace_length ** s)
        else:
            item.tension = 1.0
    
    # Step 3: 计算综合得分
    for item in items:
        if item.weight > 0:
            value_density = item.value / item.weight
            item.score = value_density * item.tension
        else:
            item.score = float('inf')
    
    # Step 4: 按得分排序（贪心collapse）
    sorted_items = sorted(items, key=lambda x: x.score, reverse=True)
    
    # Step 5: 贪心选择
    selected = []
    total_weight = 0
    
    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected.append(item)
            total_weight += item.weight
    
    elapsed_time = time.time() - start_time
    return selected, elapsed_time

def analyze_collapse_solution(selected_items: List[Item], all_items: List[Item], capacity: int) -> Dict:
    """分析Collapse解决方案的特征"""
    total_value = calculate_total_value(selected_items)
    total_weight = calculate_total_weight(selected_items)
    
    # 计算平均φ-trace长度
    avg_trace_length = sum(item.trace_length for item in selected_items) / len(selected_items) if selected_items else 0
    
    # 计算平均张力
    avg_tension = sum(item.tension for item in selected_items) / len(selected_items) if selected_items else 0
    
    print(f"\n=== Collapse解决方案分析 ===")
    print(f"选中物品数: {len(selected_items)}")
    print(f"总价值: {total_value}")
    print(f"总重量: {total_weight}/{capacity}")
    print(f"容量利用率: {total_weight/capacity*100:.1f}%")
    print(f"平均φ-trace长度: {avg_trace_length:.2f}")
    print(f"平均张力: {avg_tension:.3f}")
    
    return {
        'items_count': len(selected_items),
        'total_value': total_value,
        'total_weight': total_weight,
        'capacity_usage': total_weight/capacity,
        'avg_trace_length': avg_trace_length,
        'avg_tension': avg_tension
    }

def collapse_knapsack_with_details(items: List[Item], capacity: int, s: float = 0.5) -> Tuple[List[Item], float, Dict]:
    """
    带详细信息的CollapseGPT算法
    
    返回：(选中物品，运行时间，详细信息)
    """
    start_time = time.time()
    
    # 记录每步的详细信息
    details = {
        'encoding_details': [],
        'tension_details': [],
        'score_details': [],
        'selection_process': []
    }
    
    # Step 1: φ-trace编码
    for item in items:
        item.phi_trace = zeckendorf_encode(item.id)
        item.trace_length = len(item.phi_trace)
        details['encoding_details'].append({
            'id': item.id,
            'phi_trace': item.phi_trace,
            'length': item.trace_length
        })
    
    # Step 2: 计算collapse张力
    for item in items:
        if item.trace_length > 0:
            item.tension = 1.0 / (item.trace_length ** s)
        else:
            item.tension = 1.0
        details['tension_details'].append({
            'id': item.id,
            'trace_length': item.trace_length,
            'tension': item.tension
        })
    
    # Step 3: 计算综合得分
    for item in items:
        if item.weight > 0:
            value_density = item.value / item.weight
            item.score = value_density * item.tension
        else:
            item.score = float('inf')
        details['score_details'].append({
            'id': item.id,
            'value_density': value_density if item.weight > 0 else float('inf'),
            'tension': item.tension,
            'score': item.score
        })
    
    # Step 4: 按得分排序
    sorted_items = sorted(items, key=lambda x: x.score, reverse=True)
    
    # Step 5: 贪心选择
    selected = []
    total_weight = 0
    
    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected.append(item)
            total_weight += item.weight
            details['selection_process'].append({
                'item_id': item.id,
                'cumulative_weight': total_weight,
                'remaining_capacity': capacity - total_weight,
                'selected': True
            })
        else:
            details['selection_process'].append({
                'item_id': item.id,
                'would_exceed': total_weight + item.weight,
                'capacity': capacity,
                'selected': False
            })
    
    elapsed_time = time.time() - start_time
    return selected, elapsed_time, details

def test_different_s_values(items: List[Item], capacity: int, s_values: List[float] = None):
    """测试不同的临界指数s对结果的影响"""
    if s_values is None:
        s_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = []
    
    print("\n=== 不同临界指数s的效果 ===")
    print("s值   总价值  选中数  平均张力")
    print("-" * 35)
    
    for s in s_values:
        # 深拷贝物品列表
        test_items = [Item(id=item.id, weight=item.weight, value=item.value) for item in items]
        
        selected, time_taken = collapse_knapsack(test_items, capacity, s)
        total_value = calculate_total_value(selected)
        avg_tension = sum(item.tension for item in selected) / len(selected) if selected else 0
        
        print(f"{s:.1f}   {total_value:6d}   {len(selected):3d}    {avg_tension:.3f}")
        
        results.append({
            's': s,
            'total_value': total_value,
            'items_count': len(selected),
            'avg_tension': avg_tension,
            'time': time_taken
        })
    
    return results 
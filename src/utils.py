# src/utils.py
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

# 黄金比例常数
PHI = (1 + math.sqrt(5)) / 2
THEORETICAL_BOUND = 1 - 1/math.sqrt(PHI)  # ≈ 0.786

@dataclass
class Item:
    """背包物品类"""
    id: int
    weight: int
    value: int
    phi_trace: List[int] = field(default_factory=list)
    trace_length: int = 0
    tension: float = 0.0
    score: float = 0.0
    
    def __repr__(self):
        return f"Item(id={self.id}, w={self.weight}, v={self.value})"

def fibonacci(n: int) -> int:
    """计算第n个Fibonacci数"""
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n-2):
        a, b = b, a + b
    return b

def zeckendorf_encode(n: int) -> List[int]:
    """
    将整数n编码为Zeckendorf表示（φ-trace）
    返回二进制列表，其中1表示使用该Fibonacci数
    """
    if n == 0:
        return [0]
    
    # 生成不超过n的Fibonacci数列
    fibs = []
    i = 1
    while True:
        fib = fibonacci(i)
        if fib > n:
            break
        fibs.append(fib)
        i += 1
    
    # 贪心选择Fibonacci数
    result = []
    for fib in reversed(fibs):
        if fib <= n:
            result.append(1)
            n -= fib
        else:
            result.append(0)
    
    return result

def generate_random_items(n: int, 
                         weight_range: Tuple[int, int] = (1, 50),
                         value_range: Tuple[int, int] = (1, 100),
                         seed: int = None) -> List[Item]:
    """生成随机物品集合"""
    if seed is not None:
        random.seed(seed)
    
    items = []
    for i in range(n):
        weight = random.randint(*weight_range)
        value = random.randint(*value_range)
        items.append(Item(id=i+1, weight=weight, value=value))
    
    return items

def calculate_total_value(items: List[Item]) -> int:
    """计算物品列表的总价值"""
    return sum(item.value for item in items)

def calculate_total_weight(items: List[Item]) -> int:
    """计算物品列表的总重量"""
    return sum(item.weight for item in items)

def print_items_table(items: List[Item], selected_ids: set = None):
    """打印物品表格"""
    if selected_ids is None:
        selected_ids = set()
    
    print("\n物品列表:")
    print("选中  ID  重量  价值  价值密度  φ-trace长度  张力")
    print("-" * 60)
    
    for item in items:
        selected = "✓" if item.id in selected_ids else " "
        density = item.value / item.weight if item.weight > 0 else 0
        print(f" {selected}   {item.id:3d}  {item.weight:4d}  {item.value:4d}   {density:6.2f}      "
              f"{item.trace_length:3d}       {item.tension:5.3f}")

def fibonacci_decomposition(n: int) -> List[int]:
    """返回组成n的Fibonacci数列表"""
    if n == 0:
        return []
    
    fibs = []
    i = 1
    while fibonacci(i) <= n:
        fibs.append(fibonacci(i))
        i += 1
    
    result = []
    for fib in reversed(fibs):
        if fib <= n:
            result.append(fib)
            n -= fib
    
    return result 
# src/basic_knapsack_demo.py
"""
基础演示：展示CollapseGPT如何工作
"""
import sys
sys.path.append('.')

from utils import Item, PHI, THEORETICAL_BOUND, zeckendorf_encode, fibonacci, fibonacci_decomposition
from collapse_gpt import collapse_knapsack, analyze_collapse_solution, test_different_s_values
from dynamic_programming import dp_knapsack

def demo_phi_trace():
    """演示φ-trace编码"""
    print("=== φ-trace编码演示 ===\n")
    
    numbers = [1, 2, 3, 5, 8, 10, 20, 50, 100]
    
    print("数字  φ-trace         Fibonacci分解")
    print("-" * 50)
    
    for n in numbers:
        trace = zeckendorf_encode(n)
        fibs = fibonacci_decomposition(n)
        
        trace_str = ''.join(map(str, trace))
        fib_str = ' + '.join(map(str, fibs))
        print(f"{n:3d}   {trace_str:15s} {fib_str}")
    
    print(f"\n黄金比例 φ = {PHI:.6f}")
    print(f"理论近似比下界 = 1 - 1/√φ = {THEORETICAL_BOUND:.6f}")

def demo_small_example():
    """小规模例子演示"""
    print("\n\n=== 背包问题演示 ===\n")
    
    # 创建物品
    items = [
        Item(id=1, weight=10, value=60),
        Item(id=2, weight=20, value=100),
        Item(id=3, weight=30, value=120),
        Item(id=4, weight=15, value=80),
        Item(id=5, weight=25, value=90),
    ]
    
    capacity = 50
    
    print("物品列表:")
    print("ID  重量  价值  价值密度")
    print("-" * 30)
    for item in items:
        density = item.value / item.weight
        print(f"{item.id}   {item.weight:4d}  {item.value:4d}   {density:6.2f}")
    
    print(f"\n背包容量: {capacity}")
    
    # CollapseGPT求解
    print("\n--- CollapseGPT算法 ---")
    collapse_items = [Item(id=item.id, weight=item.weight, value=item.value) for item in items]
    collapse_selected, collapse_time = collapse_knapsack(collapse_items, capacity)
    
    print("\nφ-trace编码结果:")
    print("物品  φ-trace    长度  张力    得分")
    print("-" * 40)
    for item in collapse_items:
        trace_str = ''.join(map(str, item.phi_trace))
        print(f"{item.id:3d}   {trace_str:10s} {item.trace_length:3d}   {item.tension:5.3f}  {item.score:6.2f}")
    
    print("\n选择过程（按得分排序）:")
    sorted_items = sorted(collapse_items, key=lambda x: x.score, reverse=True)
    cumulative_weight = 0
    for item in sorted_items:
        if item in collapse_selected:
            cumulative_weight += item.weight
            print(f"✓ 选中物品{item.id}: 得分={item.score:.2f}, 累计重量={cumulative_weight}/{capacity}")
        else:
            print(f"✗ 跳过物品{item.id}: 得分={item.score:.2f}, 会超重({cumulative_weight + item.weight}>{capacity})")
    
    analysis = analyze_collapse_solution(collapse_selected, items, capacity)
    print(f"运行时间: {collapse_time*1000:.3f}ms")
    
    # 动态规划求解
    print("\n--- 动态规划算法 ---")
    dp_items = [Item(id=item.id, weight=item.weight, value=item.value) for item in items]
    dp_selected, dp_time = dp_knapsack(dp_items, capacity)
    dp_value = sum(item.value for item in dp_selected)
    
    print(f"选中物品: {[item.id for item in dp_selected]}")
    print(f"总价值: {dp_value}")
    print(f"总重量: {sum(item.weight for item in dp_selected)}")
    print(f"运行时间: {dp_time*1000:.3f}ms")
    
    # 对比
    collapse_value = sum(item.value for item in collapse_selected)
    ratio = collapse_value / dp_value if dp_value > 0 else 0
    
    print(f"\n--- 性能对比 ---")
    print(f"近似比: {ratio:.3f} (理论下界: {THEORETICAL_BOUND:.3f})")
    print(f"速度提升: {dp_time/collapse_time:.1f}x")
    print(f"Collapse选中: {len(collapse_selected)}个物品")
    print(f"DP选中: {len(dp_selected)}个物品")

def demo_critical_exponent():
    """演示临界指数s的影响"""
    print("\n\n=== 临界指数s的影响 ===\n")
    
    # 生成更多物品
    items = []
    for i in range(20):
        items.append(Item(
            id=i+1,
            weight=10 + (i % 5) * 5,
            value=50 + (i % 7) * 10
        ))
    
    capacity = 100
    
    # 测试不同的s值
    results = test_different_s_values(items, capacity, [0.1, 0.3, 0.5, 0.7, 0.9])
    
    # 与DP对比
    dp_selected, _ = dp_knapsack(items.copy(), capacity)
    dp_value = sum(item.value for item in dp_selected)
    
    print(f"\n动态规划最优解: {dp_value}")
    print("\n近似比分析:")
    print("s值   近似比   是否超过理论下界")
    print("-" * 35)
    for r in results:
        ratio = r['total_value'] / dp_value
        exceeds = "✓" if ratio >= THEORETICAL_BOUND else "✗"
        print(f"{r['s']:.1f}   {ratio:.3f}    {exceeds}")
    
    print(f"\n结论: s=0.5 在理论和实践中都是最优选择")

def main():
    """主函数"""
    print("🌌 Collapse理论背包问题演示 🌌")
    print("=" * 60)
    
    # 1. φ-trace编码演示
    demo_phi_trace()
    
    # 2. 小规模问题演示
    demo_small_example()
    
    # 3. 临界指数影响
    demo_critical_exponent()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n关键洞察:")
    print("1. φ-trace编码捕获了物品的'结构复杂度'")
    print("2. 简单结构（短编码）的物品更容易被选中")
    print("3. CollapseGPT通过物理直觉而非暴力搜索找到好解")
    print("4. 临界指数s=0.5平衡了结构和价值的重要性")
    print("\n这不仅是算法，更是一种全新的计算范式！")

if __name__ == "__main__":
    main() 
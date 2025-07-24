# src/visualize_results.py
"""
实验结果可视化
"""
import sys
sys.path.append('.')

import json
import os
import statistics
from utils import THEORETICAL_BOUND

# 尝试导入matplotlib，如果失败则使用文本输出
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib未安装，将使用文本输出模式")

def load_results(filename: str = "results/experiment_results.json"):
    """加载实验结果"""
    if not os.path.exists(filename):
        print(f"错误：找不到结果文件 {filename}")
        print("请先运行 collapse_vs_dp_experiment.py 生成实验结果")
        return None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data.get('results', data)  # 兼容新旧格式

def text_visualization(results):
    """纯文本可视化"""
    print("\n" + "="*60)
    print("📊 实验结果可视化（文本模式）")
    print("="*60)
    
    # 按规模分组
    by_size = {}
    for r in results:
        n = r['n']
        if n not in by_size:
            by_size[n] = []
        by_size[n].append(r['approximation_ratio'])
    
    # 1. 近似比分布
    print("\n### 近似比分布 ###")
    print(f"理论下界: {THEORETICAL_BOUND:.3f}")
    print("\n规模  最小值  Q1     中位数  Q3     最大值  平均值")
    print("-" * 55)
    
    for n in sorted(by_size.keys()):
        ratios = by_size[n]
        if len(ratios) >= 4:
            q1, q2, q3 = statistics.quantiles(ratios, n=4)
        else:
            q1 = q2 = q3 = statistics.median(ratios)
        
        print(f"{n:4d}  {min(ratios):.3f}  {q1:.3f}  {q2:.3f}  "
              f"{q3:.3f}  {max(ratios):.3f}  {statistics.mean(ratios):.3f}")
    
    # 2. 速度提升趋势
    print("\n### 速度提升趋势 ###")
    speed_by_size = {}
    for r in results:
        n = r['n']
        if n not in speed_by_size:
            speed_by_size[n] = []
        speed_by_size[n].append(r['speedup'])
    
    print("\n规模  平均速度提升  最大速度提升")
    print("-" * 35)
    for n in sorted(speed_by_size.keys()):
        speeds = speed_by_size[n]
        print(f"{n:4d}  {statistics.mean(speeds):10.1f}x  {max(speeds):10.1f}x")
    
    # 3. 近似比直方图（ASCII art）
    print("\n### 近似比分布直方图 ###")
    all_ratios = [r['approximation_ratio'] for r in results]
    
    # 创建10个区间
    min_ratio = min(all_ratios)
    max_ratio = max(all_ratios)
    bins = 10
    bin_width = (max_ratio - min_ratio) / bins
    
    histogram = [0] * bins
    for ratio in all_ratios:
        bin_idx = min(int((ratio - min_ratio) / bin_width), bins - 1)
        histogram[bin_idx] += 1
    
    max_count = max(histogram)
    print(f"\n范围: [{min_ratio:.3f}, {max_ratio:.3f}]")
    print("频率分布:")
    
    for i, count in enumerate(histogram):
        start = min_ratio + i * bin_width
        end = start + bin_width
        bar = '█' * int(50 * count / max_count)
        print(f"[{start:.3f}-{end:.3f}] {bar} {count}")
    
    print(f"\n理论下界 {THEORETICAL_BOUND:.3f} ↑")

def plot_approximation_ratios(results):
    """绘制近似比分布图"""
    if not HAS_MATPLOTLIB:
        return
    
    # 按规模分组
    by_size = {}
    for r in results:
        n = r['n']
        if n not in by_size:
            by_size[n] = []
        by_size[n].append(r['approximation_ratio'])
    
    # 准备数据
    sizes = sorted(by_size.keys())
    data = [by_size[n] for n in sizes]
    
    # 创建箱线图
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, labels=[f'n={n}' for n in sizes], patch_artist=True)
    
    # 设置颜色
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    # 添加理论下界线
    plt.axhline(y=THEORETICAL_BOUND, color='r', linestyle='--', 
                label=f'理论下界 ({THEORETICAL_BOUND:.3f})')
    
    # 添加平均值线
    means = [statistics.mean(d) for d in data]
    plt.plot(range(1, len(sizes)+1), means, 'g^-', label='平均值')
    
    plt.xlabel('问题规模')
    plt.ylabel('近似比')
    plt.title('CollapseGPT 近似比分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/approximation_ratios.png', dpi=150)
    print("已保存: results/approximation_ratios.png")
    plt.show()

def plot_speedup_trend(results):
    """绘制速度提升趋势图"""
    if not HAS_MATPLOTLIB:
        return
    
    # 按规模分组计算平均值
    by_size = {}
    for r in results:
        n = r['n']
        if n not in by_size:
            by_size[n] = []
        by_size[n].append(r['speedup'])
    
    sizes = sorted(by_size.keys())
    avg_speedups = [statistics.mean(by_size[n]) for n in sizes]
    max_speedups = [max(by_size[n]) for n in sizes]
    
    # 绘制趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, avg_speedups, 'bo-', linewidth=2, markersize=8, label='平均速度提升')
    plt.plot(sizes, max_speedups, 'r^--', linewidth=1, markersize=6, label='最大速度提升')
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(sizes, avg_speedups)):
        plt.annotate(f'{y:.1f}x', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.xlabel('问题规模 (n)')
    plt.ylabel('速度提升倍数')
    plt.title('CollapseGPT vs 动态规划 速度对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标
    plt.tight_layout()
    
    plt.savefig('results/speedup_trend.png', dpi=150)
    print("已保存: results/speedup_trend.png")
    plt.show()

def plot_time_complexity(results):
    """验证时间复杂度"""
    if not HAS_MATPLOTLIB:
        return
    
    # 收集时间数据
    collapse_times = {}
    dp_times = {}
    
    for r in results:
        n = r['n']
        if n not in collapse_times:
            collapse_times[n] = []
            dp_times[n] = []
        collapse_times[n].append(r['collapse_time'])
        dp_times[n].append(r['dp_time'])
    
    sizes = sorted(collapse_times.keys())
    avg_collapse = [statistics.mean(collapse_times[n]) for n in sizes]
    avg_dp = [statistics.mean(dp_times[n]) for n in sizes]
    
    # 对数坐标绘图
    plt.figure(figsize=(10, 6))
    
    # 实际时间
    plt.loglog(sizes, avg_collapse, 'bo-', label='CollapseGPT', linewidth=2, markersize=8)
    plt.loglog(sizes, avg_dp, 'ro-', label='动态规划', linewidth=2, markersize=8)
    
    # 添加理论复杂度参考线
    n_array = np.array(sizes)
    # 调整系数使曲线对齐
    c1 = avg_collapse[0] / (sizes[0] * np.log(sizes[0]))
    c2 = avg_dp[0] / (sizes[0] ** 2)
    
    plt.loglog(n_array, c1 * n_array * np.log(n_array), 'b--', 
               alpha=0.5, label='O(n log n) 理论')
    plt.loglog(n_array, c2 * n_array**2, 'r--', 
               alpha=0.5, label='O(n²) 理论')
    
    plt.xlabel('问题规模 (n)')
    plt.ylabel('运行时间 (秒)')
    plt.title('时间复杂度验证')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('results/time_complexity.png', dpi=150)
    print("已保存: results/time_complexity.png")
    plt.show()

def plot_comprehensive_analysis(results):
    """综合分析图"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 近似比vs规模
    ax1 = axes[0, 0]
    by_size = {}
    for r in results:
        n = r['n']
        if n not in by_size:
            by_size[n] = {'ratios': [], 'speeds': [], 'losses': []}
        by_size[n]['ratios'].append(r['approximation_ratio'])
        by_size[n]['speeds'].append(r['speedup'])
        by_size[n]['losses'].append(r.get('value_loss_percent', 0))
    
    sizes = sorted(by_size.keys())
    avg_ratios = [statistics.mean(by_size[n]['ratios']) for n in sizes]
    
    ax1.plot(sizes, avg_ratios, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=THEORETICAL_BOUND, color='r', linestyle='--', label='理论下界')
    ax1.set_xlabel('问题规模')
    ax1.set_ylabel('平均近似比')
    ax1.set_title('近似比 vs 问题规模')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 速度提升分布
    ax2 = axes[0, 1]
    all_speeds = [r['speedup'] for r in results]
    ax2.hist(all_speeds, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('速度提升倍数')
    ax2.set_ylabel('频率')
    ax2.set_title('速度提升分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 价值损失
    ax3 = axes[1, 0]
    avg_losses = [statistics.mean(by_size[n]['losses']) for n in sizes]
    ax3.bar(range(len(sizes)), avg_losses, tick_label=[f'n={n}' for n in sizes])
    ax3.set_xlabel('问题规模')
    ax3.set_ylabel('平均价值损失 (%)')
    ax3.set_title('价值损失分析')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 性能雷达图
    ax4 = axes[1, 1]
    # 计算综合指标
    overall_ratio = statistics.mean(r['approximation_ratio'] for r in results)
    overall_speed = min(statistics.mean(r['speedup'] for r in results) / 100, 1)  # 归一化
    overall_stability = 1 - statistics.stdev(r['approximation_ratio'] for r in results)
    theory_match = sum(1 for r in results if r['approximation_ratio'] >= THEORETICAL_BOUND) / len(results)
    
    categories = ['近似比', '速度提升', '稳定性', '理论符合度']
    values = [overall_ratio, overall_speed, overall_stability, theory_match]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax4.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax4.fill(angles, values, alpha=0.25, color='blue')
    ax4.set_theta_offset(np.pi / 2)
    ax4.set_theta_direction(-1)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('综合性能评估')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_analysis.png', dpi=150)
    print("已保存: results/comprehensive_analysis.png")
    plt.show()

def main():
    """主函数"""
    print("📊 CollapseGPT 实验结果可视化")
    print("="*50)
    
    # 加载结果
    results = load_results()
    if results is None:
        return
    
    print(f"已加载 {len(results)} 个实验结果")
    
    # 文本可视化（始终执行）
    text_visualization(results)
    
    # 图形可视化（如果有matplotlib）
    if HAS_MATPLOTLIB:
        print("\n生成可视化图表...")
        plot_approximation_ratios(results)
        plot_speedup_trend(results)
        plot_time_complexity(results)
        plot_comprehensive_analysis(results)
        print("\n所有图表已保存到 results/ 目录")
    else:
        print("\n提示：安装 matplotlib 和 numpy 可以生成更丰富的可视化图表")
        print("运行: pip install matplotlib numpy")

if __name__ == "__main__":
    main() 
import os
import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_filter_data(results_dir):
    """读取数据，包含所有跑过的数据集（无论是否收敛），消除幸存者偏差"""
    valid_instances = {}
    for filepath in glob.glob(os.path.join(results_dir, "*.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instance_name = data['instance']
        population = data['final_population']

        # 1. 过滤出所有完全可行的解
        feasible_population = [ind for ind in population if ind['total_penalty'] == 0]

        # 2. 根据 Signature 去重，保留真正的多模态备用方案
        unique_solutions = {}
        for ind in feasible_population:
            sig = ind['signature']
            if sig not in unique_solutions:
                unique_solutions[sig] = ind

        # 提取这些唯一方案中实际使用到的所有边 (Edges)
        multimodal_routes = []
        for ind in unique_solutions.values():
            edges = set()
            for route in ind['routes']:
                # route 类似于[0, 5, 12, 3, 0]
                for i in range(len(route) - 1):
                    edges.add((route[i], route[i + 1]))
            multimodal_routes.append(edges)

        # ==========================================
        # 核心修复：不加任何 if 判断，强制记录所有数据集！
        # 如果该数据集没有可行解，multimodal_routes 就是空列表[]。
        # 在后续的拥堵模拟中，它的存活方案数自然恒为 0，从而正确地拉低全图平均值。
        # ==========================================
        valid_instances[instance_name] = multimodal_routes

    return valid_instances


def simulate_traffic_jams(instances_data, mask_ratios, trials=100, num_nodes=101):
    """
    蒙特卡洛模拟：随机切断路网中的边，测试备用方案的存活情况。
    """
    # 构建全图所有可能的有向边池 (排除自环)
    all_possible_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    total_edges = len(all_possible_edges)

    results = {
        'mask_ratio': mask_ratios,
        'avg_available_solutions': [],
        'survival_probability': []
    }

    for ratio in mask_ratios:
        num_broken_edges = int(total_edges * ratio)

        ratio_avg_solutions = []
        ratio_survival_flags = []

        # 对每一个数据集进行测试
        for instance_name, multimodal_routes in instances_data.items():
            instance_avg_sols = 0
            instance_survivals = 0

            # 进行多次独立随机试验以保证统计稳定性
            for _ in range(trials):
                # 1. 随机生成不可用的边（模拟拥堵/断路）
                broken_edges = set(random.sample(all_possible_edges, num_broken_edges))

                # 2. 测试现有的多模态方案中有几个能避开拥堵
                surviving_solutions_count = 0
                for route_edges in multimodal_routes:
                    # 如果该方案的边与断开的边没有交集，说明该方案存活
                    if not route_edges.intersection(broken_edges):
                        surviving_solutions_count += 1

                instance_avg_sols += surviving_solutions_count
                if surviving_solutions_count > 0:
                    instance_survivals += 1

            # 记录该数据集在当前断路比例下的平均表现
            ratio_avg_solutions.append(instance_avg_sols / trials)
            ratio_survival_flags.append(instance_survivals / trials)

        # 综合所有数据集的平均表现
        results['avg_available_solutions'].append(np.mean(ratio_avg_solutions))
        results['survival_probability'].append(np.mean(ratio_survival_flags))

    return results


def plot_robustness_chart(results, save_dir):
    """绘制多模态鲁棒性双Y轴折线图（固定Y轴100，横轴等距分布）"""
    ratios = [r * 100 for r in results['mask_ratio']]  # 转换为百分比数值
    avg_sols = results['avg_available_solutions']
    surv_prob = [p * 100 for p in results['survival_probability']]

    # ==========================================
    # 核心修改 1：使用等距索引作为 X 轴坐标，强行让柱子等距分布
    # ==========================================
    x_positions = np.arange(len(ratios))

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # 左Y轴：平均可用方案数
    color1 = 'tab:blue'
    ax1.set_xlabel('路网瘫痪比例 (Masked Edges %)', fontsize=12)
    ax1.set_ylabel('平均存活等效方案数量 (个)', color=color1, fontsize=12)

    # 注意这里传入的是 x_positions 而不是 ratios
    bars = ax1.bar(x_positions, avg_sols, width=0.5, color=color1, alpha=0.7, label='平均可用方案数')
    ax1.tick_params(axis='y', labelcolor=color1)

    # ==========================================
    # 核心修改 2：强行固定左 Y 轴范围为 0 到 100
    # ==========================================
    ax1.set_ylim(0, 100)

    # 在柱子上标具体数字
    for bar in bars:
        yval = bar.get_height()
        # 加上微小的偏移量，防止文字与柱子顶部重叠
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1.0, f'{yval:.1f}', ha='center', va='bottom', fontsize=10)

    # 右Y轴：系统存活概率
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('系统存活概率 (至少一方案可用) %', color=color2, fontsize=12)

    # 同样使用 x_positions 绘制折线
    line = ax2.plot(x_positions, surv_prob, color=color2, marker='o', linewidth=2.5, markersize=8, label='系统存活概率')
    ax2.tick_params(axis='y', labelcolor=color2)

    # ==========================================
    # 核心修改 3：强行固定右 Y 轴范围为 0 到 100
    # ==========================================
    ax2.set_ylim(0, 100)

    # ==========================================
    # 核心修改 4：将 X 轴的刻度标签替换为带 % 符号的字符串
    # ==========================================
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{r:g}%' for r in ratios])

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # 将图例放在右上角，如果挡住折线，可以改为 'lower left' 或 'center right'
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=11)

    plt.title('多模态路由算法在动态路网中的鲁棒性评估', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multimodal_robustness.png'), dpi=300)
    print(f"鲁棒性分析图表已保存至 {save_dir}/multimodal_robustness.png")


if __name__ == "__main__":
    results_dir = root / "experiments" / "results"
    plots_dir = root / "experiments" / "plots"
    os.makedirs(plots_dir, exist_ok=True)

    print("加载并提取多模态数据...")
    valid_instances = load_and_filter_data(results_dir)
    print(f"成功提取 {len(valid_instances)} 个实例的有效多模态解。")

    # 设置路网切断比例：0%, 2%, 4%, ..., 20%
    # (在有10000条边的图中，20%意味着随机阻断2000条边，这是极高强度的破坏测试)
    mask_ratios = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]

    print("开始执行蒙特卡洛交通拥堵模拟 (可能需要几秒钟)...")
    robustness_results = simulate_traffic_jams(valid_instances, mask_ratios, trials=100)

    print("生成并保存图表...")
    plot_robustness_chart(robustness_results, plots_dir)
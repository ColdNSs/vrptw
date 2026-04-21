import os
import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 设置中文字体，防止图表中的中文显示为方块
# Windows通常为 'SimHei'，Mac通常为 'Arial Unicode MS' 或 'Heiti TC'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(results_dir):
    """读取目录下所有的 JSON 文件"""
    all_data = {}
    for filepath in glob.glob(os.path.join(results_dir, "*.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data[data['instance']] = data
    return all_data


def plot_convergence_curve(instance_data, save_dir):
    """绘制图1：双Y轴收敛曲线（惩罚值与距离优化）"""
    instance_name = instance_data['instance']
    history = instance_data['history']

    gen = list(range(1, len(history['min_penalty']) + 1))
    min_penalty = history['min_penalty']

    # 将 -1.0 (未找到可行解时的占位符) 替换为 NaN，使其在图上不显示断崖跌落
    avg_dist = [d if d != -1.0 else np.nan for d in history['avg_distance']]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 绘制惩罚值曲线 (左Y轴)
    color1 = 'tab:red'
    ax1.set_xlabel('进化代数 (Generation)')
    ax1.set_ylabel('种群最小总惩罚 (Total Penalty)', color=color1)
    ax1.plot(gen, min_penalty, color=color1, linewidth=2, label='最小惩罚值')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 创建共享X轴的右Y轴绘制距离曲线
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('前沿1平均行驶距离 (Avg Distance)', color=color2)
    ax2.plot(gen, avg_dist, color=color2, linewidth=2, label='平均行驶距离')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f'{instance_name} 算法收敛与惩罚下降曲线')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{instance_name}_convergence.png'), dpi=300)
    plt.close()


def plot_pareto_multimodal(instance_data, save_dir):
    """绘制图2与图3融合：帕累托前沿与多模态解密度图"""
    instance_name = instance_data['instance']
    population = instance_data['final_population']

    # 筛选出属于前沿1 (Rank 1) 且 惩罚为0 的完全可行解
    front_1 = [ind for ind in population if ind['rank'] == 1 and ind['total_penalty'] == 0]

    if not front_1:
        print(f"[{instance_name}] 最终种群无完全可行解，跳过帕累托图绘制。")
        return

    # 统计每个 (距离, 时间) 坐标点上有多少个 签名(Signature) 不同的个体
    # 这就是多模态的核心：相同的成本，不同的路线方案！
    point_signatures = {}
    for ind in front_1:
        # 为了防止浮点数精度导致的微小误差，将坐标保留两位小数进行分组
        coord = (round(ind['distance'], 2), round(ind['makespan'], 2))
        if coord not in point_signatures:
            point_signatures[coord] = set()
        point_signatures[coord].add(ind['signature'])

    distances = []
    makespans = []
    unique_counts = []

    for coord, signatures in point_signatures.items():
        distances.append(coord[0])
        makespans.append(coord[1])
        unique_counts.append(len(signatures))  # 该坐标点上的等效备用方案数量

    plt.figure(figsize=(8, 6))

    # 绘制气泡图，气泡越大/颜色越深，代表该成本下的等效备用方案越多
    scatter = plt.scatter(distances, makespans,
                          s=[count * 50 for count in unique_counts],  # 气泡大小
                          c=unique_counts, cmap='viridis', alpha=0.8, edgecolors='w')

    # plt.colorbar(scatter, label='等效调度方案数量 (多模态特性)')

    plt.xlabel('总行驶距离 (Total Distance)')
    plt.ylabel('最大完工时间 (Makespan)')
    plt.title(f'{instance_name} 帕累托前沿与多模态解分布')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{instance_name}_pareto_multimodal.png'), dpi=300)
    plt.close()


def plot_execution_time(all_data, save_dir):
    """绘制图4：不同数据集的算法运行时间对比"""
    instances = []
    times = []

    # 按实例名称排序 (如 C101, C102, R101...)
    for instance in sorted(all_data.keys()):
        instances.append(instance)
        times.append(all_data[instance]['execution_time_seconds'])

    plt.figure(figsize=(10, 5))
    bars = plt.bar(instances, times, color='skyblue', edgecolor='black')

    plt.xlabel('测试实例 (Solomon Instance)')
    plt.ylabel('运行耗时 (秒)')
    plt.title('算法在不同路网结构下的计算效率对比')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 在柱子上标明具体秒数
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_execution_times.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    results_directory = root / "experiments" / "results_pop200_gen300"
    plots_directory = root / "experiments" / "plots_20260418"
    os.makedirs(plots_directory, exist_ok=True)

    print("开始加载数据...")
    data_dict = load_data(results_directory)
    print(f"成功加载 {len(data_dict)} 个实例的结果数据。")

    print("开始生成图表...")
    for instance_name, data in data_dict.items():
        plot_convergence_curve(data, plots_directory)
        plot_pareto_multimodal(data, plots_directory)

    plot_execution_time(data_dict, plots_directory)
    print(f"所有图表已保存在 {plots_directory} 目录下！")
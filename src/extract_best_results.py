import os
import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

import json
import glob


def extract_best_distances(results_dir):
    filepaths = glob.glob(os.path.join(results_dir, "*.json"))
    # 按文件名排序，方便查看 (如 C101, C102...)
    filepaths.sort()

    print("-" * 75)
    print(
        f"{'数据集':<15} | {'最小行驶距离':<20} | {'目标2':<20} | {'使用车辆数':<15}")
    print("-" * 75)

    total_instances = 0
    feasible_instances = 0

    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instance_name = data['instance']
        population = data['final_population']
        total_instances += 1

        # 1. 过滤出完全可行的解 (惩罚为0)
        feasible_inds = [ind for ind in population if ind['total_penalty'] == 0]

        if not feasible_inds:
            print(f"{instance_name:<15} | {'未找到可行解 (Infeasible)':<20} | {'N/A':<20} | {'N/A':<15}")
            continue

        feasible_instances += 1

        # 2. 找出其中行驶距离最小的个体
        best_ind = min(feasible_inds, key=lambda x: x['distance'])

        distance = best_ind['distance']
        makespan = best_ind['makespan']
        vehicles = len(best_ind['routes'])

        print(f"{instance_name:<15} | {distance:<20.2f} | {makespan:<20.2f} | {vehicles:<15}")

    print("-" * 75)
    print(f"总计测试数据集: {total_instances}")
    print(f"成功找到可行解的数据集: {feasible_instances}")


if __name__ == "__main__":
    # results_directory = root / "experiments" / "results_pop200_gen500"
    results_directory = root / "experiments" / "results"
    if os.path.exists(results_directory):
        extract_best_distances(results_directory)
    else:
        print(f"找不到目录 {results_directory}，请检查路径。")
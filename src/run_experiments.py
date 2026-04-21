import os
import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

import time
import json
import glob
import torch

from src.parser import read_solomon_instance
from src.utils import calculate_euclidean_matrix, get_device
from models import GCNActorNetwork, NazariActorNetwork
from solvers import GCNSolver, GreedySolver, NazariSolver
from local_search import TwoOptLocalSearch, LNSLocalSearch
from evolution import MemeticEA, SplitEvaluator
from evolution.utils import fast_non_dominated_sort


def run_all_experiments(pop_size, max_gen):
    # 1. 路径与环境配置
    benchmark_dir = root / "data" / "benchmarks" / "solomon-100" / "r1*.txt"
    output_dir = root / "experiments" / f"results_pop{pop_size}_gen{max_gen}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")

    # 2. 提前加载训练好的深度学习模型（全局只需加载一次！）
    # actor_network = GCNActorNetwork().to(device)
    # checkpoint_path = root/ "checkpoints" / "gcn_actor_epoch_20.pt"  # 替换为你的最佳权重
    # if os.path.exists(checkpoint_path):
    #     actor_network.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    #     actor_network.eval()
    #     print("Successfully loaded GCN model.")
    # else:
    #     print("Warning: GCN checkpoint not found. Running with untrained weights.")

    actor_network = NazariActorNetwork().to(device)
    checkpoint_path = root/ "checkpoints" / "actor_epoch_60.pt"  # 替换为你的最佳权重
    if os.path.exists(checkpoint_path):
        actor_network.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        actor_network.eval()
        print("Successfully loaded Nazari model.")
    else:
        print("Warning: Nazari checkpoint not found. Running with untrained weights.")

    # 3. 获取所有实例文件并开始遍历
    instance_files = glob.glob(str(benchmark_dir))

    instance_files.sort()

    for filepath in instance_files:
        filename = os.path.basename(filepath)
        instance_name = filename.split('.')[0]

        # ==========================================
        # 新增：断点续传逻辑 (检查结果文件是否已存在)
        # ==========================================
        output_file = os.path.join(output_dir, f"{instance_name}_results.json")
        if os.path.exists(output_file):
            print(f"[跳过] {instance_name} 的结果文件已存在，直接进入下一个...")
            continue  # 核心：直接跳到下一个循环

        print(f"\n========== Starting Instance: {instance_name} ==========")

        # 解析数据与距离矩阵
        instance = read_solomon_instance(filepath)
        dist_matrix = calculate_euclidean_matrix(instance.nodes)

        # 初始化各个模块 (即插即用架构的优势在此体现)
        # 如果你想跑Greedy+2-opt做对比，只需在这里替换solver即可！
        # solver = GreedySolver(instance, dist_matrix)
        solver = NazariSolver(instance, dist_matrix, actor_network, device=device)
        # local_search = TwoOptLocalSearch(instance, dist_matrix)
        local_search = LNSLocalSearch(instance, dist_matrix)
        evaluator = SplitEvaluator(instance, dist_matrix, solver, local_search, w_fleet=2000.0, w_time=100.0)

        # 初始化进化算法 (根据需要调整参数，此处假设100代)
        ea = MemeticEA(instance, dist_matrix, evaluator, heuristic_init=0.1, pop_size=pop_size, max_gen=max_gen, F=0.13, log_history=True)

        # 开始计时与求解
        start_time = time.time()
        population = ea.solve()
        history = ea.get_history()
        execution_time = time.time() - start_time

        print(f"Finished {instance_name} in {execution_time:.2f} seconds.")

        # 4. 整理需要保存的数据结构
        result_data = {
            "instance": instance_name,
            "execution_time_seconds": execution_time,
            "history": history,
            "final_population": []  # 改为记录整个最终种群
        }

        final_fronts, rank, _ = fast_non_dominated_sort(population)

        # 遍历所有前沿层级
        for rank_idx, front in enumerate(final_fronts):
            for ind in front:
                # 提取具体的路径序列（节点ID），忽略头尾的车场(0)以便于存储
                routes_list = []
                for route in ind.routes:
                    routes_list.append([node.id for node in route.sequence])

                ind_data = {
                    "rank": rank_idx + 1,  # 记录该个体属于第几层前沿 (1代表最强)
                    "distance": ind.f1_distance,
                    "makespan": ind.f2_makespan,
                    "total_penalty": ind.total_penalty,
                    "fleet_penalty": ind.fleet_penalty,
                    "tw_penalty": ind.tw_penalty,
                    "signature": ind.signature,  # 用于证明多模态特性
                    "routes": routes_list
                }
                result_data["final_population"].append(ind_data)

        # 5. 保存为 JSON 文件
        output_file = os.path.join(output_dir, f"{instance_name}_results.json")
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=4)

        print(f"Data saved to {output_file}. Recorded {len(result_data['final_population'])} individuals.")


if __name__ == "__main__":
    run_all_experiments(pop_size=200, max_gen=299)
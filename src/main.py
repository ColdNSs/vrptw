import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

import numpy as np
import torch

from src.parser import read_solomon_instance
from src.utils import calculate_euclidean_matrix, apply_seed, get_device

from solvers import GreedySolver, GCNSolver
from local_search import TwoOptLocalSearch, LNSLocalSearch, NoLocalSearch
from evolution import MemeticEA, SplitEvaluator
from models import GCNActorNetwork
from evolution.utils import fast_non_dominated_sort

def run_memetic(instance, dist_matrix):
    print(f"--- Memetic Split Evolution ---")

    w_fleet = 200.0
    w_time = 1.0

    # USE CPU FOR INFERENCE!! You'll experience bottleneck translating things back and forth between cpu and gpu
    # device = get_device()
    device = torch.device("cpu")

    solver = GreedySolver(instance, dist_matrix)
    # print(f"Loading weights to device: {device}")
    # checkpoint_path = root / "checkpoints" / "gcn_actor_epoch_20.pt"
    # drl_actor = GCNActorNetwork().to(device)
    # drl_actor.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    # solver = GCNSolver(instance, dist_matrix, drl_actor, device)

    # local_search = NoLocalSearch(instance, dist_matrix)
    # local_search = LNSLocalSearch(instance, dist_matrix, max_iters=30, removal_fraction=0.3)
    local_search = TwoOptLocalSearch(instance, dist_matrix)

    evaluator = SplitEvaluator(instance, dist_matrix, solver, local_search, w_fleet, w_time)
    memetic = MemeticEA(instance, dist_matrix, evaluator, pop_size=100, max_gen=300, F=0.2)
    population = memetic.solve()
    fronts, _, _ = fast_non_dominated_sort(population)

    # Print top 10 fronts
    for i, front in enumerate(fronts):
        if i > 9:
            break
        print(f"Front {i + 1}:")
        for ind in front:
            print(ind)

def main():
    print("VRPTW environment ready")
    print("NumPy version:", np.__version__)
    print("PyTorch version:", torch.__version__)

    # Build path relative to repo root
    data_path = root / "data" / "benchmarks" / "solomon-100" / "c201.txt"

    instance = read_solomon_instance(data_path)
    dist_matrix = calculate_euclidean_matrix(instance.nodes)
    print(f"Loaded instance: {instance}")

    apply_seed()

    run_memetic(instance, dist_matrix)


if __name__ == "__main__":
    main()
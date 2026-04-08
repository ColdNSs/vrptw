import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

import numpy as np
import random
from src.parser import read_solomon_instance
from src.utils import calculate_euclidean_matrix
from src.solvers import GreedySolver
from src.local_search import TwoOptLocalSearch, LNSLocalSearch
from src.evolution import Evaluator, MMOEA_DL
from src.utils import calculate_penalty_weights
from copy import deepcopy, copy


def apply_seed(seed=None):
    # Seed Management
    seed = seed if seed is not None else random.randint(0, 2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")

def lower_level_test(instance, dist_matrix):
    print(f"--- Lower-level Test: Greedy and 2-opt / LNS ---")

    # Upper-level allocation

    # Expected result:
    # Greedy: [0, 27, 69, 1, 50, 77, 3, 0]
    # After 2-opt: [0, 27, 69, 1, 50, 3, 77, 0]
    test_nodes = [1, 3, 27, 50, 69, 77]

    unvisited = [instance.nodes[i] for i in test_nodes]

    # Lower-level: repeatedly run greedy solver on all unvisited nodes
    solver = GreedySolver(instance, dist_matrix)
    route = solver.solve(unvisited)
    route_copy = deepcopy(route)
    print(f"Greedy: {route}")

    # Lower-level: use 2-opt to optimize each route
    local_search = TwoOptLocalSearch(instance, dist_matrix)
    local_search.optimize(route)
    print(f"2-opt: {route}")

    # Lower-level: use LNS to optimize each route
    local_search = LNSLocalSearch(instance, dist_matrix)
    local_search.optimize(route_copy)
    print(f"LNS: {route_copy}")

def mmoea_dl_test(instance, dist_matrix):
    print(f"--- MMOEA-DL Test ---")

    instance.nodes = instance.nodes[:26]
    print(f"Cropped instance to the first 25 customers")

    w_load, w_time = calculate_penalty_weights(instance)

    solver = GreedySolver(instance, dist_matrix)
    local_search = TwoOptLocalSearch(instance, dist_matrix)
    # local_search = LNSLocalSearch(instance, dist_matrix, max_iters=30)
    evaluator = Evaluator(instance, dist_matrix, solver, local_search, w_load, w_time)
    mmoea_dl = MMOEA_DL(instance, dist_matrix, evaluator, max_gen=400)
    fronts = mmoea_dl.solve()

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

    # Build path relative to repo root
    data_path = root / "data" / "benchmarks" / "solomon-100" / "r102.txt"

    instance = read_solomon_instance(data_path)
    dist_matrix = calculate_euclidean_matrix(instance.nodes)
    print(f"Loaded instance: {instance}")

    apply_seed(3081261258)

    lower_level_test(instance, dist_matrix)

    mmoea_dl_test(instance, dist_matrix)


if __name__ == "__main__":
    main()

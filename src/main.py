import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

import numpy as np
from src.parser import read_solomon_instance
from src.solvers import GreedySolver
from src.local_search import TwoOptLocalSearch


def greedy_and_two_opt(instance):
    routes = []
    unvisited = instance.nodes[1:].copy()

    # This function has no upper-level allocation

    # Lower-level: repeatedly run greedy solver on all unvisited nodes
    while unvisited:
        solver = GreedySolver(instance)
        route = solver.solve(unvisited)
        routes.append(route)
    print(routes)

    # Lower-level: use 2-opt to optimize each route
    local_search = TwoOptLocalSearch()
    for route in routes:
        local_search.optimize(route)
    print(routes)
    print(f"Num of routes: {len(routes)}")

def main():
    print("VRPTW environment ready")
    print("NumPy version:", np.__version__)

    # Build path relative to repo root
    data_path = root / "data" / "benchmarks" / "solomon-100" / "r102.txt"

    instance = read_solomon_instance(data_path)
    print(f"Loaded instance: {instance}")

    greedy_and_two_opt(instance)


if __name__ == "__main__":
    main()

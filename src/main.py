import numpy as np
from parser import read_solomon_instance
from utils import calculate_euclidean_matrix
from solver import GreedySolver

def main():
    print("VRPTW environment ready")
    print("NumPy version:", np.__version__)
    instance = read_solomon_instance("../data/benchmarks/solomon-100/c101.txt")
    print(instance)
    distance = calculate_euclidean_matrix(instance.nodes)
    print(distance[0][1])

    greedy_solver = GreedySolver(instance)
    routes = greedy_solver.solve()
    print(routes)

if __name__ == "__main__":
    main()

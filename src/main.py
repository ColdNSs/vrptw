import numpy as np
from parser import read_solomon_instance

def main():
    print("VRPTW environment ready")
    print("NumPy version:", np.__version__)
    instance = read_solomon_instance("../data/benchmarks/solomon-100/c101.txt")
    print(instance)

if __name__ == "__main__":
    main()

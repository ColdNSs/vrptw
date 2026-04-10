from abc import ABC, abstractmethod
from typing import List


class Solver(ABC):
    """Abstract base for VRPTW solvers."""

    def __init__(self, instance, dist_matrix):
        self.instance = instance
        self.dist_matrix = dist_matrix

    @abstractmethod
    def solve(self, unvisited) -> List:
        """Return a list of Route objects."""
        ...

    def solve_batch(self, unvisited_lists):
        """
        Default fallback for heuristic solvers (like Greedy).
        DRL will override this to perform GPU batching.
        """
        return[self.solve(unvisited) for unvisited in unvisited_lists]
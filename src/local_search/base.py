from abc import ABC, abstractmethod


class BaseLocalSearch(ABC):
    """Abstract base for route-level local search operators."""

    def __init__(self, instance, dist_matrix):
        self.instance = instance
        self.dist_matrix = dist_matrix

    @abstractmethod
    def optimize(self, route):
        """Modify route in place. Return None."""
        ...

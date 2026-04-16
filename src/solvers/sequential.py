from .base import BaseSolver
from src.route import Route


class SequentialSolver(BaseSolver):
    """Generates a route with the given sequence."""

    def __init__(self, instance, dist_matrix):
        super().__init__(instance, dist_matrix)

    def solve(self, sequence):
        route = Route(self.instance, self.dist_matrix)

        for i in range(len(sequence)):
            best_node = sequence[i]
            route.add_node(best_node)

        route.close_route()
        route.update_state()

        return route

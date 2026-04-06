from .base import Solver
from src.route import Route
from src.utils import calculate_euclidean_matrix


class GreedySolver(Solver):
    """Nearest-neighbor heuristic with capacity + time window feasibility."""

    def __init__(self, instance):
        super().__init__(instance)
        self.dist_matrix = calculate_euclidean_matrix(self.instance.nodes)

    def solve(self, unvisited):
        route = Route(self.instance, self.dist_matrix)

        while unvisited:
            best_node = None
            min_dist = float('inf')
            current_node = route.last_node

            for candidate in unvisited:
                dist = self.dist_matrix[current_node][candidate]
                if dist < min_dist and route.is_feasible(candidate):
                    min_dist = dist
                    best_node = candidate

            if best_node:
                route.add_node(best_node)
                unvisited.remove(best_node)
            else:
                break

        route.close_route()
        route.update_state()

        return route

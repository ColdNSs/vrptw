from .base import Solver
from src.route import Route


class GreedySolver(Solver):
    """Nearest-neighbor heuristic with capacity + time window feasibility."""

    def __init__(self, instance, dist_matrix):
        super().__init__(instance, dist_matrix)

    def solve(self, unvisited):
        route = Route(self.instance, self.dist_matrix)

        while unvisited:
            best_feasible_node = None
            min_feasible_dist = float('inf')

            best_infeasible_node = None
            min_infeasible_dist = float('inf')

            current_node = route.last_node

            for candidate in unvisited:
                dist = self.dist_matrix[current_node][candidate]

                # Check if adding this candidate breaks constraints
                if route.is_feasible(candidate):
                    if dist < min_feasible_dist:
                        min_feasible_dist = dist
                        best_feasible_node = candidate
                else:
                    if dist < min_infeasible_dist:
                        min_infeasible_dist = dist
                        best_infeasible_node = candidate

            # CRITERIA: Always prioritize feasible nodes.
            # If none are feasible, fall back to the closest infeasible node.
            if best_feasible_node is not None:
                best_node = best_feasible_node
            else:
                best_node = best_infeasible_node

            # Add the node and remove from the pool
            route.add_node(best_node)
            unvisited.remove(best_node)

        route.close_route()
        route.update_state()

        return route

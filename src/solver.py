import numpy as np
from src.utils import calculate_euclidean_matrix


class Route:
    def __init__(self, instance, dist_matrix):
        self.capacity = instance.capacity
        self.depot = instance.nodes[0]
        self.dist_matrix = dist_matrix

        # Sequence of Node Objects
        self.sequence = [self.depot]
        self.load = 0.0
        self.time = 0.0
        self.cost = 0.0
        self.is_closed = False

    @property
    def last_node(self):
        return self.sequence[-1]

    def is_feasible(self, node):
        if self.is_closed: return False

        if self.load + node.demand > self.capacity:
            return False

        prev_node = self.last_node

        dist = self.dist_matrix[prev_node][node]
        departure_time = max(self.time, prev_node.ready_time) + prev_node.service_time
        arrival_time = departure_time + dist

        if arrival_time > node.due_date:
            return False

        return True

    def add_node(self, node):
        prev_node = self.last_node

        dist = self.dist_matrix[prev_node][node]

        self.load += node.demand

        departure_from_prev = max(self.time, prev_node.ready_time) + prev_node.service_time
        arrival_at_new = departure_from_prev + dist

        self.time = max(arrival_at_new, node.ready_time)
        self.cost += dist
        self.sequence.append(node)

    def close_route(self):
        if self.is_closed: return
        # Try to return to depot
        if self.is_feasible(self.depot):
            self.add_node(self.depot)
        else:
            print("Unable to reach depot when trying to close route!")
            self.add_node(self.depot)  # Force close
        self.is_closed = True

    def __repr__(self):
        path_ids = [n.id for n in self.sequence]
        return f"Route(Load={self.load}, Cost={self.cost:.2f}, Path={path_ids})"


class GreedySolver:
    def __init__(self, instance):
        self.instance = instance
        self.dist_matrix = calculate_euclidean_matrix(instance.nodes)
        self.routes = []

    def solve(self):
        unvisited = self.instance.nodes[1:].copy()

        while unvisited:
            current_route = Route(self.instance, self.dist_matrix)
            while True:
                best_node = None
                min_dist = float('inf')
                current_node = current_route.last_node

                for candidate in unvisited:
                    dist = self.dist_matrix[current_node][candidate]

                    if dist < min_dist:
                        if current_route.is_feasible(candidate):
                            min_dist = dist
                            best_node = candidate

                if best_node:
                    current_route.add_node(best_node)
                    unvisited.remove(best_node)
                else:
                    break

            current_route.close_route()
            self.routes.append(current_route)
        return self.routes
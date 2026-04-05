import numpy as np
from solvers import GreedySolver
from local_search import TwoOptLocalSearch


class Individual:
    """
    Represents a single Task Allocation scheme in the Upper Level.
    """

    def __init__(self, num_tasks, num_vehicles):
        self.num_tasks = num_tasks
        self.num_vehicles = num_vehicles

        # Continuous representation: Array of size num_tasks, values in[0, num_vehicles)
        self.chromosome = np.random.uniform(0, num_vehicles, size=num_tasks)

        # Objectives
        self.f1_distance = float('inf')
        self.f2_makespan = float('inf')

        # Store the actual routes for this individual
        self.routes = []

    def decode(self):
        """
        Converts the continuous chromosome into discrete vehicle assignments.
        Returns a dictionary mapping {vehicle_id: [list of customer ids]}
        """
        assignments = np.floor(self.chromosome).astype(int)

        allocation = {v: [] for v in range(self.num_vehicles)}
        # Note: Customer IDs usually start at 1 (0 is depot)
        for task_idx, vehicle_id in enumerate(assignments):
            customer_id = task_idx + 1
            allocation[vehicle_id].append(customer_id)

        return allocation


class Evaluator:
    """
    The 'Bridge' between Upper Level (EA) and Lower Level (Routing).
    """

    def __init__(self, instance, dist_matrix):
        self.instance = instance
        self.dist_matrix = dist_matrix
        self.solver = GreedySolver(instance)
        self.local_search = TwoOptLocalSearch()

    def evaluate(self, individual):
        """
        Calculates f1 and f2 for an individual.
        """
        allocation = individual.decode()
        individual.routes = []

        # Lower Level Optimization for each vehicle
        for vehicle_id, customer_ids in allocation.items():
            if not customer_ids:
                continue  # Skip empty vehicles

            # 1. Route Construction
            route = self._solve_lower_level(customer_ids)

            # 2. Route Improvement
            self.local_search.optimize(route)

            individual.routes.append(route)

        # Calculate Upper Level Objectives
        if not individual.routes:
            return

        individual.f1_distance = sum(r.cost for r in individual.routes)
        individual.f2_makespan = max(r.finish_time for r in individual.routes)

    def _solve_lower_level(self, customer_ids):
        unvisited = [self.instance.nodes[i] for i in customer_ids]
        route = self.solver.solve(unvisited)
        return route
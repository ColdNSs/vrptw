import numpy as np
from abc import ABC, abstractmethod


class Individual:
    """
    Represents a single Task Allocation scheme in the Upper Level.
    """

    def __init__(self, num_tasks, num_vehicles):
        self.num_tasks = num_tasks
        self.num_vehicles = num_vehicles

        # Continuous representation: Array of size num_tasks, values in[0, num_vehicles)
        self.chromosome = np.zeros(num_tasks)

        # Objectives
        self.f1_distance = float('inf')
        self.f2_makespan = float('inf')

        # Constraints
        self.total_penalty = float('inf')

        # Store the actual routes for this individual
        self.routes = []

        # Signature used for redundancy deletion
        self.signature = ""

    def set_chromosome(self, new_chromosome):
        """
        Safely assigns a new chromosome with strict dimension and boundary checks.
        """
        new_chromosome = np.array(new_chromosome, dtype=float)

        # 1. Dimension Check
        if len(new_chromosome) != self.num_tasks:
            raise ValueError(f"Dimension mismatch: Expected {self.num_tasks}, got {len(new_chromosome)}")

        # 2. Boundary Enforcement
        # We clip to (num_vehicles - 1e-5) to guarantee np.floor() never returns num_vehicles
        self.chromosome = np.clip(new_chromosome, 0.0, self.num_vehicles - 1e-5)

    def decode(self):
        """
        Converts the continuous chromosome into discrete vehicle assignments.
        Returns a list [[vehicle 0 list of customer ids], [vehicle 1 list of customer ids], ...]
        """
        assignments = np.floor(self.chromosome).astype(int)

        groups = {}
        for task_idx, v_id in enumerate(assignments):
            customer_id = task_idx + 1
            if v_id not in groups:
                groups[v_id] = []
            groups[v_id].append(customer_id)

        raw_clusters = list(groups.values())
        canonical_allocation = sorted([sorted(cluster) for cluster in raw_clusters if cluster])
        self.signature = str(canonical_allocation)

        return canonical_allocation

    def __repr__(self):
        return (f"Ind(Distance={self.f1_distance:.2f}, Makespan={self.f2_makespan:.2f}, Penalty={self.total_penalty:.2f}, "
                f"Routes={self.routes})")


class Evaluator:
    """
    The 'Bridge' between Upper Level (EA) and Lower Level (Routing).
    """

    def __init__(self, instance, dist_matrix, solver, local_search, w_load, w_time):
        self.instance = instance
        self.dist_matrix = dist_matrix
        self.solver = solver
        self.local_search = local_search
        self.w_load = w_load
        self.w_time = w_time

    def evaluate(self, individual):
        """
        Calculates f1 and f2 for an individual.
        """
        allocation = individual.decode()
        individual.routes = []

        individual.total_penalty = 0.0
        individual.f1_distance = 0.0
        individual.f2_makespan = 0.0

        # Lower Level Optimization for each vehicle
        for vehicle_id, customer_ids in enumerate(allocation):
            if not customer_ids:
                continue  # Skip empty vehicles

            # 1. Route Construction
            route = self._solve_lower_level(customer_ids)

            # 2. Route Improvement
            self.local_search.optimize(route)

            # 3. Aggregate Penalties using the weights
            route_penalty = (self.w_load * route.load_penalty) + (self.w_time * route.tw_penalties)
            individual.total_penalty += route_penalty

            individual.routes.append(route)

        # Calculate Upper Level Objectives
        if not individual.routes:
            individual.total_penalty = float('inf')
            return

        individual.f1_distance = sum(r.cost for r in individual.routes)
        individual.f2_makespan = max(r.finish_time for r in individual.routes)

    def _solve_lower_level(self, customer_ids):
        unvisited = [self.instance.nodes[i] for i in customer_ids]
        route = self.solver.solve(unvisited)
        return route


class Evolution(ABC):
    """Abstract base for Evolutionary Algorithms."""

    def __init__(self, instance, dist_matrix, evaluator):
        self.instance = instance
        self.dist_matrix = dist_matrix
        self.evaluator = evaluator

    @abstractmethod
    def solve(self) -> list[list[Individual]]:
        """Return a list of Pareto Fronts."""
        ...
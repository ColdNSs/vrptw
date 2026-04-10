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
        self.load_penalty = float('inf')
        self.tw_penalty = float('inf')
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
        return (f"Ind(Distance={self.f1_distance:.2f}, Makespan={self.f2_makespan:.2f}, "
                f"Load_Penalty={self.load_penalty}, TW_Penalty={self.tw_penalty:.2f}, Total_Penalty={self.total_penalty:.2f}, "
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
        Calculates f1 and f2 for an individual. (Legacy)
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
        """
        Calls lower-level. (Legacy)
        """
        unvisited = [self.instance.nodes[i] for i in customer_ids]
        route = self.solver.solve(unvisited)
        return route

    def evaluate_population(self, population):
        """
        Evaluates the entire population simultaneously to leverage GPU batching.
        """
        all_unvisited = []
        route_mapping = []  # Tracks which route belongs to which individual

        # 1. GATHER: Decode everyone and collect all tasks
        for ind_idx, individual in enumerate(population):
            allocation = individual.decode()
            individual.routes = []
            individual.load_penalty = 0.0
            individual.tw_penalty = 0.0
            individual.total_penalty = float('inf')
            individual.f1_distance = float('inf')
            individual.f2_makespan = float('inf')

            for vehicle_id, customer_ids in enumerate(allocation):
                if not customer_ids:
                    continue  # Skip empty vehicles

                # Fetch node objects
                unvisited = [self.instance.nodes[i] for i in customer_ids]
                all_unvisited.append(unvisited)
                route_mapping.append(ind_idx)

        # 2. BATCHED SOLVE: Hand all tasks to the solver at once
        if not all_unvisited:
            return

        # Greedy will loop this. DRL will run it in parallel
        solved_routes = self.solver.solve_batch(all_unvisited)

        # 3. SCATTER: Apply local search and map routes back to their individuals
        for route, ind_idx in zip(solved_routes, route_mapping):
            # Local search
            self.local_search.optimize(route)

            # Aggregate Penalties
            ind = population[ind_idx]
            ind.load_penalty += route.load_penalty
            ind.tw_penalty += route.tw_penalties
            ind.routes.append(route)

        # 4. Finalize Objectives
        # Optional: Adaptive penalty weights
        # self._update_penalty_weights(population)
        for individual in population:
            if individual.routes:
                individual.total_penalty = (self.w_load * individual.load_penalty) + \
                                           (self.w_time * individual.tw_penalty)
                individual.f1_distance = sum(r.cost for r in individual.routes)
                individual.f2_makespan = max(r.finish_time for r in individual.routes)

    def _update_penalty_weights(self, population):
        """
        Dynamically tilts the fitness landscape to prevent the EA from
        getting trapped in conflicting local optima.
        """
        # Calculate the proportion of individuals satisfying each constraint
        feasible_load_count = sum(1 for ind in population if ind.load_penalty == 0)
        feasible_tw_count = sum(1 for ind in population if ind.tw_penalty == 0)

        pop_size = len(population)

        # If almost everyone satisfies Load but fails TW, we are stuck in the Load Optimum.
        # We heavily increase the TW weight to force them to care about Time Windows.
        if feasible_load_count > pop_size * 0.8 and feasible_tw_count < pop_size * 0.2:
            self.w_time *= 1.2
            self.w_load *= 0.9

        # Vice versa: stuck in the TW optimum, ignoring capacity.
        elif feasible_tw_count > pop_size * 0.8 and feasible_load_count < pop_size * 0.2:
            self.w_load *= 1.2
            self.w_time *= 0.9

        # Prevent weights from collapsing to zero or exploding to infinity
        self.w_load = max(0.1, min(self.w_load, 1000.0))
        self.w_time = max(0.1, min(self.w_time, 1000.0))


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
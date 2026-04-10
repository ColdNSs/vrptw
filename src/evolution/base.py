import numpy as np
from abc import ABC, abstractmethod


class BaseIndividual(ABC):
    """
    Abstract base class for an MMOEA_DL_Individual in the Upper Level EA.
    """

    def __init__(self, num_tasks, num_vehicles):
        self.num_tasks = num_tasks
        self.num_vehicles = num_vehicles

        # Continuous representation (Genotype)
        self.chromosome = np.zeros(num_tasks)

        # Objectives
        self.f1_distance = float('inf')
        self.f2_makespan = float('inf')

        # Constraints
        self.load_penalty = float('inf')
        self.tw_penalty = float('inf')
        self.total_penalty = float('inf')

        # Store the actual routes for this individual (Phenotype)
        self.routes = []

        # Signature used for redundancy deletion
        self.signature = ""

    def set_chromosome(self, new_chromosome):
        """
        Safely assigns a new chromosome with strict dimension checks.
        (Boundary enforcement might differ between subclasses, but basic clipping is standard).
        """
        new_chromosome = np.array(new_chromosome, dtype=float)
        if len(new_chromosome) != self.num_tasks:
            raise ValueError(f"Dimension mismatch: Expected {self.num_tasks}, got {len(new_chromosome)}")

        self.chromosome = new_chromosome

    @abstractmethod
    def decode(self):
        """
        Converts the continuous chromosome into a routing representation.
        - Direct Allocation returns: list of vehicle clusters.
        - Random Key returns: a single Giant TSP Tour.
        """
        pass

    def __repr__(self):
        return (f"{self.__class__.__name__}(Dist={self.f1_distance:.2f}, Make={self.f2_makespan:.2f}, "
                f"LoadPen={self.load_penalty:.2f}, TWPen={self.tw_penalty:.2f}, "
                f"Routes={len(self.routes)})")


class BaseEvaluator(ABC):
    """
    Abstract base class for the 'Bridge' between EA and Lower Level.
    """

    def __init__(self, instance, dist_matrix, solver, local_search, w_load, w_time):
        self.instance = instance
        self.dist_matrix = dist_matrix
        self.solver = solver
        self.local_search = local_search
        self.w_load = w_load
        self.w_time = w_time

    @abstractmethod
    def evaluate_population(self, population):
        """
        Evaluates the entire population.
        - Direct MMOEA_DL_Evaluator: Solves distinct clusters.
        - Split MMOEA_DL_Evaluator: Runs Prins Split, routes the slices, and performs Lamarckian Write-back.
        """
        pass

    def update_penalty_weights(self, population):
        """
        Dynamically tilts the fitness landscape (Adaptive Penalty Weights).
        This logic is universal regardless of how individuals are decoded.
        """
        feasible_load_count = sum(1 for ind in population if ind.load_penalty == 0)
        feasible_tw_count = sum(1 for ind in population if ind.tw_penalty == 0)

        pop_size = len(population)

        if feasible_load_count > pop_size * 0.8 and feasible_tw_count < pop_size * 0.2:
            self.w_time *= 1.2
            self.w_load *= 0.9
        elif feasible_tw_count > pop_size * 0.8 and feasible_load_count < pop_size * 0.2:
            self.w_load *= 1.2
            self.w_time *= 0.9

        self.w_load = max(0.1, min(self.w_load, 1000.0))
        self.w_time = max(0.1, min(self.w_time, 1000.0))


class BaseEvolution(ABC):
    """Abstract base for Evolutionary Algorithms."""

    def __init__(self, instance, dist_matrix, evaluator):
        self.instance = instance
        self.dist_matrix = dist_matrix
        self.evaluator = evaluator

    @abstractmethod
    def solve(self) -> list[list[BaseIndividual]]:
        """Return a list of Pareto Fronts."""
        ...
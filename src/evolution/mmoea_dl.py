from .base import Evolution, Individual
import numpy as np
import random


class MMOEA_DL(Evolution):
    def __init__(self, evaluator, pop_size=100, max_gen=200, F=0.5, CR=0.9, seed=None):
        super().__init__(evaluator)
        self.pop_size = pop_size
        self.max_gen = max_gen

        # DE Parameters
        self.F = F  # Mutation scaling factor
        self.CR = CR  # Crossover probability

        # Seed Management
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        random.seed(self.seed)
        np.random.seed(self.seed)
        print(f"--> Initialized MMOEA_DL | Seed: {self.seed}")

    def solve(self):
        # 1. Initialization
        population = self._initialize_population()
        self._evaluate_population(population)

        for gen in range(self.max_gen):
            print(f"Generation {gen + 1}/{self.max_gen}")

            # 2. Reproduction (DE Mutation & Crossover)
            offspring_population = self._generate_offspring(population)
            self._evaluate_population(offspring_population)

            # 3. Combine
            combined_pop = population + offspring_population

            # 4. Sorting & Redundancy Deletion
            fronts = self._fast_non_dominated_sort(combined_pop)
            fronts = self._delete_redundant_solutions(fronts)

            # 5. Environmental Selection
            population = self._environmental_selection(fronts)

        return population  # Returns the final Pareto Fronts

    def _initialize_population(self):
        num_tasks = len(self.instance.nodes) - 1  # Number of client nodes
        num_vehicles = self.instance.num_vehicles  # Number of vehicles
        population = []
        for _ in range(self.pop_size):
            new_ind = Individual(num_tasks, num_vehicles)

            # Generate random array using seeded randomness
            rand_chrom = np.random.uniform(0, num_vehicles, size=num_tasks)
            new_ind.set_chromosome(rand_chrom)

            population.append(new_ind)
        return population

    def _generate_offspring(self, population):
        num_tasks = len(self.instance.nodes) - 1
        num_vehicles = self.instance.num_vehicles
        offspring = []

        # Sort population once for the simplified DBESM elite pool
        sorted_pop = sorted(population, key=lambda x: (x.total_penalty, x.f1_distance))
        elite_pool = sorted_pop[:max(1, self.pop_size // 10)]

        for i, parent in enumerate(population):
            # Select 2 random distinct individuals
            r1, r2 = random.sample([x for j, x in enumerate(population) if j != i], 2)

            # Simplified DBESM
            exemplar = random.choice(elite_pool)

            # DE Mutation
            v = parent.chromosome + self.F * (exemplar.chromosome - parent.chromosome) + self.F * (
                        r1.chromosome - r2.chromosome)

            # Binomial Crossover
            u = np.copy(parent.chromosome)

            # Create a boolean mask of where random() < CR
            crossover_mask = np.random.rand(num_tasks) < self.CR

            # DE RULE: At least ONE dimension must mutate to avoid an exact clone of the parent
            forced_mutation_idx = random.randint(0, num_tasks - 1)
            crossover_mask[forced_mutation_idx] = True

            # Apply mutation where the mask is True
            u[crossover_mask] = v[crossover_mask]

            # Create child safely using our new setter
            child = Individual(num_tasks, num_vehicles)
            child.set_chromosome(u)
            offspring.append(child)

        return offspring

    def _evaluate_population(self, population):
        for ind in population:
            self.evaluator.evaluate(ind)

    def _fast_non_dominated_sort(self, population):
        # Implement constrained dominance sorting here
        pass

    def _delete_redundant_solutions(self, fronts):
        # Use ind.signature to drop clones
        pass

    def _environmental_selection(self, fronts):
        # Pick top NP individuals using Front Rank and Crowding Distance
        pass
from .base import Evolution, Individual
from .utils import fast_non_dominated_sort, delete_redundant_solutions, calculate_crowding_distance, get_exemplar_dbesm
import numpy as np
import random


class MMOEA_DL(Evolution):
    def __init__(self, instance, dist_matrix, evaluator, pop_size=100, max_gen=200, F=0.5, CR=0.9):
        super().__init__(instance, dist_matrix, evaluator)
        self.pop_size = pop_size
        self.max_gen = max_gen

        # DE Parameters
        self.F = F  # Mutation scaling factor
        self.CR = CR  # Crossover probability

    def solve(self):
        # 1. Initialization
        population = self._initialize_population()
        self._evaluate_population(population)
        fronts = [[]]

        for gen in range(self.max_gen):
            assert len(population) == self.pop_size

            # 2. Reproduction (DE Mutation & Crossover)
            offspring_population = self._generate_offspring(population)
            self._evaluate_population(offspring_population)

            # 3. Combine
            combined_pop = population + offspring_population

            # 4. Sorting & Redundancy Deletion
            fronts, _, _ = fast_non_dominated_sort(combined_pop)
            fronts = delete_redundant_solutions(fronts)

            # 5. Environmental Selection
            population = self._environmental_selection(fronts)

            print(f"Generation {gen + 1}/{self.max_gen} complete")
            print(fronts[0][0])

        return fronts  # Returns the final Pareto Fronts

    def _initialize_population(self):
        size = self.pop_size
        population = self._generate_population(size)
        return population

    def _generate_population(self, size):
        num_tasks = len(self.instance.nodes) - 1  # Number of client nodes
        num_vehicles = self.instance.num_vehicles  # Number of vehicles
        population = []
        for _ in range(size):
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

        fronts, rank, _ = fast_non_dominated_sort(population)

        # Sort population once for the simplified DBESM elite pool
        # sorted_pop = sorted(population, key=lambda x: (x.total_penalty, x.f1_distance))
        # elite_pool = sorted_pop[:max(1, self.pop_size // 10)]

        for i, parent in enumerate(population):
            # Select 2 random distinct individuals
            r1, r2 = random.sample([x for j, x in enumerate(population) if j != i], 2)

            # Simplified DBESM
            # exemplar = random.choice(elite_pool)

            # Full DBESM
            exemplar = get_exemplar_dbesm(parent, population, fronts, rank)

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

    def _environmental_selection(self, fronts):
        """
        Selects exactly `pop_size` individuals for the next generation.
        """
        next_population = []
        remain = self.pop_size

        for front in fronts:
            if len(front) <= remain:
                # The whole front fits
                next_population.extend(front)
                remain -= len(front)
            elif remain > 0:
                # The front is too large, we must select the most diverse individuals
                # TODO: change to CSCD
                crowding_distance = calculate_crowding_distance(front)

                # Sort descending by crowding distance (larger distance = more isolated/diverse = better)
                front.sort(key=lambda x: crowding_distance[x], reverse=True)

                next_population.extend(front[:remain])
                remain = 0
                break

        # When too much redundant individuals are deleted, add random immigrants to prevent population shrinking
        if remain > 0:
            print(f"  [!] Injecting {remain} random individuals to maintain population size.")
            immigrants = self._generate_population(remain)
            self._evaluate_population(immigrants)
            next_population.extend(immigrants)

        return next_population
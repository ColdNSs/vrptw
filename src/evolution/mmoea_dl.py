from .base import Evolution, Individual
from .utils import fast_non_dominated_sort, delete_redundant_solutions, calculate_crowding_distance, get_exemplar_dbesm
import numpy as np
import random
from scipy.cluster.vq import kmeans2


class MMOEA_DL(Evolution):
    def __init__(self, instance, dist_matrix, evaluator, heuristic_init=0.1, pop_size=100, max_gen=200, F=0.5, CR=0.9):
        super().__init__(instance, dist_matrix, evaluator)
        self.heuristic_init = heuristic_init
        self.pop_size = pop_size
        self.max_gen = max_gen

        # DE Parameters
        self.F = F  # Mutation scaling factor
        self.CR = CR  # Crossover probability

    def solve(self):
        # 1. Initialization
        population = self._initialize_population(self.heuristic_init)
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

            sample_ind = fronts[0][0]
            f1_distance = sample_ind.f1_distance
            f2_makespan = sample_ind.f2_makespan
            load_penalty = sample_ind.load_penalty
            tw_penalty = sample_ind.tw_penalty
            total_penalty = sample_ind.total_penalty
            sample_str = (f"SampleInd(Distance={f1_distance:.2f}, Makespan={f2_makespan:.2f}, "
                          f"Load_Penalty={load_penalty}, TW_Penalty={tw_penalty:.2f}, Total_Penalty={total_penalty:.2f})")
            print(f"Generation {gen + 1}/{self.max_gen} completed. {sample_str}")

        return fronts  # Returns the final Pareto Fronts

    def _initialize_population(self, structured_pop):
        heuristic_size = max(1, int(self.pop_size * structured_pop))
        random_size = self.pop_size - heuristic_size

        pop_random = self._generate_population(random_size)
        pop_heuristic = self._generate_heuristic_population(heuristic_size)

        population = pop_random + pop_heuristic
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

    def _generate_heuristic_population(self, size):
        """
        Uses Spatial-Temporal K-Means clustering to create smart initial allocations.
        """
        num_tasks = len(self.instance.nodes) - 1
        num_vehicles = self.instance.num_vehicles
        population = []

        # 1. Extract and Normalize Spatial-Temporal Features[X, Y, Ready_Time]
        max_x = max(n.x for n in self.instance.nodes)
        max_y = max(n.y for n in self.instance.nodes)
        max_t = max(n.due_date for n in self.instance.nodes)

        base_features = np.zeros((num_tasks, 3))
        for i, n in enumerate(self.instance.nodes[1:]):
            base_features[i, 0] = n.x / max_x if max_x > 0 else 0
            base_features[i, 1] = n.y / max_y if max_y > 0 else 0
            base_features[i, 2] = n.ready_time / max_t if max_t > 0 else 0

        # Determine K (We can't have more clusters than tasks)
        k_clusters = min(num_vehicles, num_tasks)

        for _ in range(size):
            new_ind = Individual(num_tasks, num_vehicles)

            # 2. Inject Noise for Diversity
            # We want each heuristic individual to be slightly different.
            # Adding 5% Gaussian noise shifts the cluster boundaries just enough to create distinct groupings.
            noise = np.random.normal(0, 0.05, base_features.shape)
            noisy_features = base_features + noise

            # 3. K-Means Clustering
            try:
                # minit='points' randomly chooses initial centroids to further increase diversity
                centroids, labels = kmeans2(noisy_features, k_clusters, minit='points')
            except Exception:
                # Fallback to random if K-Means fails (extremely rare edge case)
                labels = np.random.randint(0, num_vehicles, size=num_tasks)

            # 4. Map Discrete Labels back to Continuous Chromosomes
            # If a task is assigned to Vehicle 2, its chromosome must be a float in[2.0, 3.0)
            rand_offsets = np.random.uniform(0.0, 1 - 1e-5, size=num_tasks)
            chrom = labels.astype(float) + rand_offsets

            # Use our safe setter
            new_ind.set_chromosome(chrom)
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
        self.evaluator.evaluate_population(population)

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
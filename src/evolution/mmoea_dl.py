from .base import BaseEvolution, BaseIndividual, BaseEvaluator
from .utils import fast_non_dominated_sort, delete_redundant_solutions, calculate_cscd, get_exemplar_dbesm
import numpy as np
import random
from scipy.cluster.vq import kmeans2


class MMOEA_DL(BaseEvolution):
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
            sample_str = (f"SampleInd(Dist={f1_distance:.2f}, MSpan={f2_makespan:.2f}, "
                          f"LoadPen={load_penalty:.2f}, TWPen={tw_penalty:.2f}, TotPen={total_penalty:.2f}")
            print(f"Generation {gen + 1}/{self.max_gen} completed. {sample_str}")

        return population

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
            new_ind = MMOEA_DL_Individual(num_tasks, num_vehicles)

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
            new_ind = MMOEA_DL_Individual(num_tasks, num_vehicles)

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
            child = MMOEA_DL_Individual(num_tasks, num_vehicles)
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
                crowding_distance = calculate_cscd(front)

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


class MMOEA_DL_Individual(BaseIndividual):
    """
    Represents a single Task Allocation scheme in the Upper Level.
    """

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
        return (f"{self.__class__.__name__}(Dist={self.f1_distance:.2f}, MSpan={self.f2_makespan:.2f}, "
                f"LoadPen={self.load_penalty:.2f}, TWPen={self.tw_penalty:.2f}, TotPen={self.total_penalty:.2f}, "
                f"Routes={self.routes})")


class MMOEA_DL_Evaluator(BaseEvaluator):
    def __init__(self, instance, dist_matrix, solver, local_search, w_load, w_time):
        super().__init__(instance, dist_matrix, solver, local_search)
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
        # self.update_penalty_weights(population)
        for individual in population:
            if individual.routes:
                individual.total_penalty = (self.w_load * individual.load_penalty) + \
                                           (self.w_time * individual.tw_penalty)
                individual.f1_distance = sum(r.cost for r in individual.routes)
                individual.f2_makespan = max(r.finish_time for r in individual.routes)

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
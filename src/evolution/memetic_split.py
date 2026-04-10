import numpy as np
from .base import BaseEvolution, BaseIndividual, BaseEvaluator
import random
from .utils import fast_non_dominated_sort, delete_redundant_solutions, calculate_crowding_distance, get_exemplar_dbesm


class MemeticEA(BaseEvolution):
    def __init__(self, instance, dist_matrix, evaluator, heuristic_init=0.1, pop_size=100, max_gen=200, F=0.2, CR=0.9):
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

            self.evaluator.update_penalty_weights(combined_pop)

            # 4. Sorting & Redundancy Deletion
            fronts, rank, _ = fast_non_dominated_sort(combined_pop)
            fronts = delete_redundant_solutions(fronts)

            # 5. Environmental Selection
            population = self._environmental_selection(fronts)

            sample_ind = fronts[0][0]
            f1_distance = sample_ind.f1_distance
            f2_makespan = sample_ind.f2_makespan
            fleet_penalty = sample_ind.fleet_penalty  # <-- Changed from load_penalty
            tw_penalty = sample_ind.tw_penalty
            total_penalty = sample_ind.total_penalty
            sample_str = (f"SampleInd(Dist={f1_distance:.2f}, MSpan={f2_makespan:.2f}, "
                          f"FleetPen={fleet_penalty:.2f}, TWPen={tw_penalty:.2f}, TotPen={total_penalty:.2f})")
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
            new_ind = RandomKeyIndividual(num_tasks, num_vehicles)

            # Generate random array using seeded randomness
            # For Random Keys, bounds [0, 1] are standard
            rand_chrom = np.random.uniform(0.0, 1.0, size=num_tasks)
            new_ind.set_chromosome(rand_chrom)

            population.append(new_ind)
        return population

    def _generate_heuristic_population(self, size):
        """
        Uses a Noisy Nearest-Neighbor heuristic to create smart initial Giant Tours.
        """
        num_tasks = len(self.instance.nodes) - 1
        num_vehicles = self.instance.num_vehicles
        population = []

        for _ in range(size):
            new_ind = RandomKeyIndividual(num_tasks, num_vehicles)

            unvisited = set(range(1, num_tasks + 1))
            current = 0  # Start at depot
            giant_tour = []

            # 1. Build a good geometric Giant Tour
            while unvisited:
                best_next = None
                best_dist = float('inf')
                for cand in unvisited:
                    # Inject 20% Gaussian noise into the distance to ensure diverse tours
                    # Without noise, every heuristic individual would be identical!
                    noise_factor = np.random.uniform(0.8, 1.2)
                    dist = self.dist_matrix[current][cand] * noise_factor

                    if dist < best_dist:
                        best_dist = dist
                        best_next = cand

                giant_tour.append(best_next)
                current = best_next
                unvisited.remove(best_next)

            # 2. Map the Giant Tour back to Continuous Chromosomes (Random Keys)
            sorted_keys = np.sort(np.random.uniform(0.0, 1.0, size=num_tasks))
            chrom = np.zeros(num_tasks)
            for seq_idx, cust_id in enumerate(giant_tour):
                chrom[cust_id - 1] = sorted_keys[seq_idx]

            new_ind.set_chromosome(chrom)
            population.append(new_ind)

        return population

    def _generate_offspring(self, population):
        num_tasks = len(self.instance.nodes) - 1
        num_vehicles = self.instance.num_vehicles
        offspring = []

        fronts, rank, _ = fast_non_dominated_sort(population)

        for i, parent in enumerate(population):
            r1, r2 = random.sample([x for j, x in enumerate(population) if j != i], 2)

            # Full DBESM
            exemplar = get_exemplar_dbesm(parent, population, fronts, rank)

            # DE Mutation
            v = parent.chromosome + self.F * (exemplar.chromosome - parent.chromosome) + self.F * (
                    r1.chromosome - r2.chromosome)

            # Binomial Crossover
            u = np.copy(parent.chromosome)

            crossover_mask = np.random.rand(num_tasks) < self.CR
            forced_mutation_idx = random.randint(0, num_tasks - 1)
            crossover_mask[forced_mutation_idx] = True

            u[crossover_mask] = v[crossover_mask]

            child = RandomKeyIndividual(num_tasks, num_vehicles)
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
                next_population.extend(front)
                remain -= len(front)
            elif remain > 0:
                crowding_distance = calculate_crowding_distance(front)
                front.sort(key=lambda x: crowding_distance[x], reverse=True)
                next_population.extend(front[:remain])
                remain = 0
                break

        # Prevent population shrinking by injecting random immigrants
        if remain > 0:
            print(f"  [!] Injecting {remain} random individuals to maintain population size.")
            immigrants = self._generate_population(remain)
            self._evaluate_population(immigrants)
            next_population.extend(immigrants)

        return next_population


class RandomKeyIndividual(BaseIndividual):
    """
    Uses Random Key (RK) encoding. The chromosome represents a single Giant TSP Tour.
    """

    def decode(self):
        """
        Sorts the continuous chromosome to extract the Giant Tour permutation.
        Returns a 1D list of customer IDs.
        """
        # argsort() returns the indices that would sort the array.
        # Example:[0.8, 0.1, 0.9] -> [1, 0, 2]
        # Since customers are 1-indexed (0 is depot), we add 1.
        giant_tour = np.argsort(self.chromosome) + 1

        # The signature is just the giant tour order
        self.signature = str(giant_tour.tolist())
        return giant_tour.tolist()

    def set_chromosome(self, new_chromosome):
        """
        Safely assigns a new chromosome with strict dimension and boundary checks.
        """
        new_chromosome = np.array(new_chromosome, dtype=float)

        # 1. Dimension Check
        if len(new_chromosome) != self.num_tasks:
            raise ValueError(f"Dimension mismatch: Expected {self.num_tasks}, got {len(new_chromosome)}")

        # 2. Boundary Enforcement
        # Clip to (1 - 1e-5)
        self.chromosome = np.clip(new_chromosome, 0.0, 1 - 1e-5)

    def encode_lamarckian(self, routes):
        """
        Lamarckian Evolution: The DRL and Local Search often find a better sequence
        than the EA's original Giant Tour. This method overwrites the EA's genes
        """
        # 1. Flatten the optimized routes back into a single giant sequence
        optimized_tour = []
        for route in routes:
            # Extract customer IDs, ignoring the start and end depots (id == 0)
            optimized_tour.extend([n.id for n in route.sequence if n.id != 0])

        # 2. Extract and sort the ORIGINAL continuous keys
        sorted_keys = np.sort(self.chromosome)

        noise = np.random.uniform(-0.05, 0.05, size=len(sorted_keys))
        sorted_keys = np.sort(sorted_keys + noise)

        # 3. Assign the smallest keys to the earliest nodes in the optimized tour
        for sequence_index, customer_id in enumerate(optimized_tour):
            self.chromosome[customer_id - 1] = sorted_keys[sequence_index]

        self.decode()

    def __repr__(self):
        return (f"{self.__class__.__name__}(Dist={self.f1_distance:.2f}, MSpan={self.f2_makespan:.2f}, "
                f"FleetPen={self.fleet_penalty:.0f}, TWPen={self.tw_penalty:.2f}, TotPen={self.total_penalty:.2f}, "
                f"Routes={self.routes})")


class SplitEvaluator(BaseEvaluator):
    """
    Evaluator that slices the Giant Tour using the Prins Split Algorithm,
    routes them via GPU DRL, and writes the knowledge back to the EA.
    """

    def __init__(self, instance, dist_matrix, solver, local_search, w_fleet, w_time):
        super().__init__(instance, dist_matrix, solver, local_search)
        self.w_fleet = w_fleet
        self.w_time = w_time

    def evaluate_population(self, population):
        all_unvisited = []
        route_mapping = []

        # 1. GATHER & SPLIT: Decode into Giant Tours and slice them optimally
        for ind_idx, ind in enumerate(population):
            giant_tour = ind.decode()

            # Run the Prins Split Algorithm
            slices = self._prins_split(giant_tour)

            ind.routes = []
            ind.tw_penalty = 0.0
            ind.fleet_penalty = 0.0
            ind.total_penalty = float('inf')
            ind.f1_distance = float('inf')
            ind.f2_makespan = float('inf')

            # Edge Case: If a single node is heavier than the truck, split fails.
            if not slices:
                continue

            for sub_tour in slices:
                unvisited = [self.instance.nodes[i] for i in sub_tour]
                all_unvisited.append(unvisited)
                route_mapping.append(ind_idx)

        # 2. BATCHED SOLVE: Hand all valid slices to the DRL / Greedy solver at once
        if not all_unvisited:
            return

        solved_routes = self.solver.solve_batch(all_unvisited)

        # 3. SCATTER: Apply local search and map routes back to their individuals
        for route, ind_idx in zip(solved_routes, route_mapping):
            self.local_search.optimize(route)

            ind = population[ind_idx]
            ind.tw_penalty += route.tw_penalties
            ind.routes.append(route)

        # 4. LAMARCKIAN WRITE-BACK & OBJECTIVES
        for ind in population:
            if ind.routes:
                # Write the DRL's intelligence directly into the EA's genes
                ind.encode_lamarckian(ind.routes)

                # Calculate Fleet Penalty
                fleet_overage = max(0, len(ind.routes) - ind.num_vehicles)
                ind.fleet_penalty = fleet_overage

                # Calculate Total Penalty
                ind.total_penalty = (self.w_fleet * ind.fleet_penalty) + (self.w_time * ind.tw_penalty)

                # Standard Objectives
                ind.f1_distance = sum(r.cost for r in ind.routes)
                ind.f2_makespan = max(r.finish_time for r in ind.routes)

    def _prins_split(self, giant_tour):
        """
        The Prins Split Algorithm (DAG Shortest Path).
        Strictly forbids edges that violate vehicle capacity.
        """
        n = len(giant_tour)

        # V[i] stores the minimum cost to route the first 'i' customers
        V = [float('inf')] * (n + 1)
        # P[i] stores the predecessor index to backtrack the shortest path
        P = [0] * (n + 1)
        V[0] = 0.0

        for i in range(n):
            if V[i] == float('inf'):
                continue

            load = 0.0
            cost = 0.0
            tw_pen = 0.0
            time = self.instance.nodes[0].ready_time

            for j in range(i + 1, n + 1):
                cust_idx = giant_tour[j - 1]
                cust_node = self.instance.nodes[cust_idx]

                # 1. Capacity Check: The absolute wall.
                load += cust_node.demand
                if load > self.instance.capacity:
                    break

                    # 2. Calculate Distance and Time incrementally
                if j == i + 1:
                    dist = self.dist_matrix[0][cust_idx]
                else:
                    prev_idx = giant_tour[j - 2]
                    dist = self.dist_matrix[prev_idx][cust_idx]

                cost += dist
                arrival = time + dist

                if arrival > cust_node.due_date:
                    tw_pen += (arrival - cust_node.due_date)

                time = max(arrival, cust_node.ready_time) + cust_node.service_time

                # 3. Simulate Return to Depot
                return_dist = self.dist_matrix[cust_idx][0]
                route_cost = cost + return_dist
                route_tw_pen = tw_pen

                arrival_depot = time + return_dist
                depot_due = self.instance.nodes[0].due_date
                if arrival_depot > depot_due:
                    route_tw_pen += (arrival_depot - depot_due)

                # 4. Total Edge Weight
                # We add a fixed "vehicle deployment cost" (e.g., 100) so the shortest
                # path naturally favors using fewer vehicles, preventing massive fleet overages.
                vehicle_fixed_cost = 100.0
                edge_weight = route_cost + (self.w_time * route_tw_pen) + vehicle_fixed_cost

                # 5. Bellman-Ford Shortest Path Update
                if V[i] + edge_weight < V[j]:
                    V[j] = V[i] + edge_weight
                    P[j] = i

        # 6. Safety Catch: Was a path found?
        if V[n] == float('inf'):
            return []

        # 7. Backtrack to extract the actual vehicle slices
        slices = []
        curr = n
        while curr > 0:
            prev = P[curr]
            slices.append(giant_tour[prev:curr])
            curr = prev

        slices.reverse()
        return slices

    def update_penalty_weights(self, population):
        """
        Dynamically tilts the fitness landscape (Adaptive Penalty Weights).
        This logic is universal regardless of how individuals are decoded.
        """
        feasible_fleet_count = sum(1 for ind in population if ind.fleet_penalty == 0)
        feasible_tw_count = sum(1 for ind in population if ind.tw_penalty == 0)

        pop_size = len(population)

        if feasible_fleet_count > pop_size * 0.8 and feasible_tw_count < pop_size * 0.2:
            self.w_time *= 1.2
            self.w_fleet *= 0.9
        elif feasible_tw_count > pop_size * 0.8 and feasible_fleet_count < pop_size * 0.2:
            self.w_fleet *= 1.2
            self.w_time *= 0.9

        self.w_fleet = max(0.1, min(self.w_fleet, 1000.0))
        self.w_time = max(0.1, min(self.w_time, 1000.0))
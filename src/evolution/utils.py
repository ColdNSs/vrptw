import random
import numpy as np

def dominates(ind_a, ind_b):
    """
    Returns True if ind_a strictly dominates ind_b using Constrained Dominance.
    """
    # Rule 1 & 2: Penalty comparisons
    if ind_a.total_penalty < ind_b.total_penalty:
        return True
    elif ind_a.total_penalty > ind_b.total_penalty:
        return False

    # Rule 3: If penalties are equal (e.g., both 0, or both equally infeasible),
    # use standard Pareto dominance on f1 (Distance) and f2 (Makespan)
    better_in_all = (ind_a.f1_distance <= ind_b.f1_distance) and (ind_a.f2_makespan <= ind_b.f2_makespan)
    strictly_better_in_one = (ind_a.f1_distance < ind_b.f1_distance) or (ind_a.f2_makespan < ind_b.f2_makespan)

    return better_in_all and strictly_better_in_one

def fast_non_dominated_sort(population):
    """
    Sorts the population into Pareto fronts based on Constrained Dominance.
    """
    fronts = [[]]
    domination_count = {ind: 0 for ind in population}
    rank = {ind: 0 for ind in population}
    dominated_solutions = {ind: [] for ind in population}

    for p in population:
        for q in population:
            if dominates(p, q):
                dominated_solutions[p].append(q)
            elif dominates(q, p):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while True:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
                    rank[q] = i + 1
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts, rank, dominated_solutions

def get_exemplar_dbesm(parent, population, fronts, rank):
    """
    Full DBESM: Finds the closest strictly better individual in the decision space.
    """
    better_inds = []

    # 1. Find all individuals that are "Better" (Using our Constrained Dominance)
    if rank[parent] > 0:
        for front in fronts[:rank[parent]]:
            better_inds.extend(front)
    # 2. Edge Case: If the parent is a top-tier elite (no one dominates it)
    elif rank[parent] == 0:
        # Gather all Rank 0 individuals (excluding the parent itself)
        front_1 = fronts[0].copy()
        front_1.remove(parent)
        if front_1:
            return random.choice(front_1)
        else:
            # Extreme fallback (e.g. pop size 1 or everyone is identical)
            return random.choice([p for p in population if p is not parent])

    # 3. Find the closest better individual (Euclidean distance on chromosomes)
    best_exemplar = None
    min_dist = float('inf')

    p_chrom = parent.chromosome
    for candidate in better_inds:
        # np.linalg.norm calculates the Euclidean distance between the arrays
        dist = np.linalg.norm(p_chrom - candidate.chromosome)
        if dist < min_dist:
            min_dist = dist
            best_exemplar = candidate

    return best_exemplar

def delete_redundant_solutions(fronts):
    """
    Removes symmetric clones from the fronts using the canonical signature.
    """
    clean_fronts = []
    seen_signatures = set()

    for front in fronts:
        clean_front = []
        for ind in front:
            if ind.signature not in seen_signatures:
                seen_signatures.add(ind.signature)
                clean_front.append(ind)
            else:
                print(f"  [!] Seen signature. Remove individual from the front.")

        if len(clean_front) > 0:
            clean_fronts.append(clean_front)

    return clean_fronts

def calculate_crowding_distance(front):
    """
    Standard NSGA-II Crowding Distance in Objective Space.
    """
    num_inds = len(front)
    crowding_distance = {ind: 0.0 for ind in front}

    if num_inds <= 2:
        for ind in front:
            crowding_distance[ind] = float('inf')
        return crowding_distance

    # Optimize over f1 (Distance)
    front.sort(key=lambda x: x.f1_distance)
    crowding_distance[front[0]] = float('inf')
    crowding_distance[front[-1]] = float('inf')

    f1_min, f1_max = front[0].f1_distance, front[-1].f1_distance
    f1_range = f1_max - f1_min if f1_max - f1_min > 0 else 1.0

    for i in range(1, num_inds - 1):
        crowding_distance[front[i]] += (front[i + 1].f1_distance - front[i - 1].f1_distance) / f1_range

    # Optimize over f2 (Makespan)
    front.sort(key=lambda x: x.f2_makespan)
    crowding_distance[front[0]] = float('inf')
    crowding_distance[front[-1]] = float('inf')

    f2_min, f2_max = front[0].f2_makespan, front[-1].f2_makespan
    f2_range = f2_max - f2_min if f2_max - f2_min > 0 else 1.0

    for i in range(1, num_inds - 1):
        crowding_distance[front[i]] += (front[i + 1].f2_makespan - front[i - 1].f2_makespan) / f2_range

    return crowding_distance
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

    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []

        for q in population:
            if dominates(p, q):
                p.dominated_solutions.append(q)
            elif dominates(q, p):
                p.domination_count += 1

        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while True:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts

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

# Being how much time late is as bad as being 1 unit overweight?
def calculate_penalty_weights(instance):
    weight_load = instance.nodes[0].due_date / instance.capacity
    weight_time = 1.0
    return weight_load, weight_time
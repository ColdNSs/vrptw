import numpy as np
import torch


def calculate_euclidean_matrix(nodes):
    n = len(nodes)
    matrix = np.zeros((n, n))

    # Access attributes with dot notation now
    coords = np.array([(node.x, node.y) for node in nodes])

    for i in range(n):
        diff = coords - coords[i]
        matrix[i] = np.sqrt(np.sum(diff ** 2, axis=1))

    return matrix

# Being how much time late is as bad as being 1 unit overweight?
def calculate_penalty_weights(instance):
    weight_load = instance.nodes[0].due_date / instance.capacity
    weight_time = 1.0
    return weight_load, weight_time

def assess_sequence(seq, dist_matrix, capacity):
    total_dist = 0.0
    total_load = 0.0
    time = seq[0].ready_time
    tw_penalties = 0.0

    for k in range(len(seq) - 1):
        prev_node = seq[k]
        next_node = seq[k + 1]

        # 1. Distance
        dist = dist_matrix[prev_node][next_node]
        total_dist += dist
        total_load += next_node.demand

        # 2. Time simulation
        departure_time = time + prev_node.service_time
        arrival_time = departure_time + dist

        # 3. Time Window Penalty (Late arrival)
        if arrival_time > next_node.due_date:
            # Penalty is proportional to how late we are
            tw_penalties += (arrival_time - next_node.due_date)

        # Update current time (we must wait if we arrive before ready_time)
        time = max(arrival_time, next_node.ready_time)

    return total_dist, time, max(0.0, total_load - capacity), tw_penalties

def get_device():
    """
    Returns the optimal PyTorch device (MPS for Apple Silicon, CUDA for Nvidia, CPU fallback).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
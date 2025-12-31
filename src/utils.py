import math
import numpy as np


def calculate_euclidean_matrix(nodes):
    n = len(nodes)
    matrix = np.zeros((n, n))

    # Access attributes with dot notation now
    coords = np.array([(node.x, node.y) for node in nodes])

    for i in range(n):
        diff = coords - coords[i]
        matrix[i] = np.sqrt(np.sum(diff ** 2, axis=1))

    return matrix
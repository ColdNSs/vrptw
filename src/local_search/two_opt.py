from .base import BaseLocalSearch
from .utils import is_new_seq_better


class TwoOptLocalSearch(BaseLocalSearch):
    """
    2-opt local search operator.
    Swaps edges within a single route to reduce distance.
    """

    def optimize(self, route):
        improved = True
        while improved:
            improved = False
            n = len(route.sequence)

            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    candidate_sequence = self._swap_sequence(route.sequence, i, j)

                    is_better = is_new_seq_better(route.sequence, candidate_sequence, self.dist_matrix,
                                            self.instance.capacity)

                    if is_better:
                        route.sequence = candidate_sequence
                        improved = True
                        break  # Break inner loop

                if improved:
                    break  # Break outer loop to restart the while loop

        route.update_state()

    def _swap_sequence(self, seq, i, j):
        sequence = seq.copy()
        sequence[i:j + 1] = reversed(sequence[i:j + 1])
        return sequence

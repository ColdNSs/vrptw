from .base import LocalSearch


class TwoOptLocalSearch(LocalSearch):
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
                    cost_before = route.cost
                    tw_penalties_before = route.tw_penalties

                    self._swap(route, i, j)

                    cost_after = route.cost
                    tw_penalties_after = route.tw_penalties

                    # ACCEPTANCE CRITERIA
                    # 1. If it strictly reduces penalties (moving towards feasibility), ACCEPT.
                    # 2. If penalties are the same (or both 0), but cost is reduced, ACCEPT.
                    is_better = False
                    if tw_penalties_after < tw_penalties_before:
                        is_better = True
                    elif tw_penalties_after == tw_penalties_before and cost_after < cost_before:
                        is_better = True

                    if is_better:
                        improved = True
                        break  # Break inner loop
                    else:
                        # Revert the swap
                        self._swap(route, i, j)

                if improved:
                    break  # Break outer loop to restart the while loop

    def _swap(self, route, i, j):
        sequence = route.sequence
        sequence[i:j + 1] = reversed(sequence[i:j + 1])
        route.update_state()

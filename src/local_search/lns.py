from .base import BaseLocalSearch
from .utils import is_new_seq_better
import random


class LNSLocalSearch(BaseLocalSearch):
    """
    Large Neighborhood Search for a Single Route.
    Uses Destroy (Removal) and Repair (Greedy Insertion) operators.
    """

    def __init__(self, instance, dist_matrix, max_iters=30, removal_fraction=0.3):
        super().__init__(instance, dist_matrix)
        self.max_iters = max_iters
        self.removal_fraction = removal_fraction

    def optimize(self, route):
        num_customers = len(route.sequence) - 2  # Exclude start and end depots
        if num_customers <= 2:
            route.update_state()
            return  # Too short to optimize

        # We keep track of the best sequence found so far
        best_sequence = route.sequence.copy()
        num_to_remove = max(1, int(num_customers * self.removal_fraction))

        for _ in range(self.max_iters):
            # 1. Destroy: Remove nodes
            partial_seq, removed_nodes = self._destroy(best_sequence, num_to_remove)

            # 2. Repair: Insert nodes back greedily
            candidate_sequence = self._repair(partial_seq, removed_nodes)

            # 3. Acceptance Criteria
            is_better = is_new_seq_better(best_sequence, candidate_sequence,
                                          self.dist_matrix, self.instance.capacity)

            if is_better:
                # Accept new best sequence
                best_sequence = candidate_sequence

        # 4. Finalize route with the absolute best sequence found
        assert len(route.sequence) == len(best_sequence)
        route.sequence = best_sequence
        route.update_state()

    def _destroy(self, sequence, num_to_remove):
        """
        Randomly removes `num_to_remove` customer nodes.
        Returns the partial sequence and the list of removed nodes.
        """
        customer_indices = list(range(1, len(sequence) - 1))
        indices_to_remove = set(random.sample(customer_indices, num_to_remove))

        partial_seq = []
        removed_nodes = []

        for i, node in enumerate(sequence):
            if i in indices_to_remove:
                removed_nodes.append(node)
            else:
                partial_seq.append(node)

        return partial_seq, removed_nodes

    def _repair(self, partial_seq, removed_nodes):
        """
        Greedy Insertion: For each removed node, find the best insertion index.
        """
        current_seq = partial_seq.copy()
        random.shuffle(removed_nodes)  # Shuffle to avoid deterministic loops

        for node_to_insert in removed_nodes:
            current_seq = self._repair_one_node(current_seq, node_to_insert)

        return current_seq

    def _repair_one_node(self, current_seq, node_to_insert):
        test_seq = current_seq.copy()
        test_seq.insert(1, node_to_insert)
        best_test_seq = test_seq

        for i in range(2, len(current_seq)):
            test_seq = current_seq.copy()
            test_seq.insert(i, node_to_insert)

            if is_new_seq_better(best_test_seq, test_seq, self.dist_matrix, self.instance.capacity):
                best_test_seq = test_seq

        return best_test_seq
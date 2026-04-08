from .base import LocalSearch
import random


class LNSLocalSearch(LocalSearch):
    """
    Large Neighborhood Search for a Single Route.
    Uses Destroy (Removal) and Repair (Greedy Insertion) operators.
    """

    def __init__(self, instance, dist_matrix, w_load, w_time, max_iters=50, removal_fraction=0.2):
        self.instance = instance
        self.dist_matrix = dist_matrix
        self.w_load = w_load
        self.w_time = w_time
        self.max_iters = max_iters
        self.removal_fraction = removal_fraction

    def optimize(self, route):
        # Initial evaluation
        route.update_state()
        best_cost = route.cost
        best_penalties = (self.w_load * route.load_penalty) + (self.w_time * route.tw_penalties)
        best_sequence = route.sequence.copy()

        num_customers = len(route.sequence) - 2  # Exclude start and end depots
        if num_customers <= 2:
            return  # Too short to optimize

        # Number of nodes to remove during Destroy
        num_to_remove = max(1, int(num_customers * self.removal_fraction))

        for iteration in range(self.max_iters):
            # 1. Destroy
            partial_seq, removed_nodes = self._destroy(best_sequence, num_to_remove)

            # 2. Repair
            candidate_seq = self._repair(partial_seq, removed_nodes, route)

            # 3. Evaluate Candidate
            route.sequence = candidate_seq
            route.update_state()
            candidate_cost = route.cost
            candidate_penalties = (self.w_load * route.load_penalty) + (self.w_time * route.tw_penalties)

            # 4. Acceptance Criteria (Strict Descent towards feasibility)
            is_better = False
            if candidate_penalties < best_penalties:
                is_better = True
            elif candidate_penalties == best_penalties and candidate_cost < best_cost:
                is_better = True

            if is_better:
                # Accept new best
                best_sequence = candidate_seq[:]
                best_cost = candidate_cost
                best_penalties = candidate_penalties
            else:
                # Revert route state for next loop iteration
                pass

                # Finalize route with the best sequence found
        route.sequence = best_sequence
        route.update_state()

    def _destroy(self, sequence, num_to_remove):
        """
        Randomly removes `num_to_remove` customer nodes.
        Returns the partial sequence and the list of removed nodes.
        """
        # Exclude index 0 (start depot) and -1 (end depot)
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

    def _repair(self, partial_seq, removed_nodes, route):
        """
        Greedy Insertion: For each removed node, find the position in the partial sequence
        that results in the lowest penalty/cost increase, and insert it there.
        """
        current_seq = partial_seq.copy()

        # Shuffle removed nodes to add some randomness to the greedy insertion
        random.shuffle(removed_nodes)

        for node_to_insert in removed_nodes:
            best_insert_idx = 1
            best_score = float('inf')

            # Try inserting the node at every possible position between depots
            for i in range(1, len(current_seq)):
                # Create a temporary sequence to test
                test_seq = current_seq.copy()
                test_seq.insert(i, node_to_insert)

                # Evaluate this temporary sequence using the Route's internal logic
                # To do this fast without modifying the actual route object too much:
                score = self._quick_evaluate(test_seq, route)

                if score < best_score:
                    best_score = score
                    best_insert_idx = i

            # Actually insert the node at the best found position
            current_seq.insert(best_insert_idx, node_to_insert)

        return current_seq

    def _quick_evaluate(self, seq, route):
        """
        A fast evaluator to score an insertion.
        Focuses heavily on Time Window penalties.
        """
        total_dist = 0.0
        current_time = seq[0].ready_time
        tw_penalties = 0.0

        for k in range(len(seq) - 1):
            prev_node = seq[k]
            next_node = seq[k + 1]

            dist = self.dist_matrix[prev_node][next_node]
            total_dist += dist

            arrival_time = current_time + prev_node.service_time + dist

            if arrival_time > next_node.due_date:
                tw_penalties += (arrival_time - next_node.due_date)

            current_time = max(arrival_time, next_node.ready_time)

        # Weighting: Heavily penalize TW violations during insertion
        score = total_dist + (tw_penalties * 100.0)
        return score
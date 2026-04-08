from src.utils import assess_sequence


# ACCEPTANCE CRITERIA
# 1. If it strictly reduces penalties (moving towards feasibility), ACCEPT.
# 2. If penalties are the same (or both 0), but cost is reduced, ACCEPT.
def is_new_seq_better(sequence_a, sequence_b, dist_matrix, capacity):
    cost_before, _, _, tw_penalties_before = assess_sequence(sequence_a, dist_matrix, capacity)
    cost_after, _, _, tw_penalties_after = assess_sequence(sequence_b, dist_matrix, capacity)

    is_better = False
    if tw_penalties_after < tw_penalties_before:
        is_better = True
    elif tw_penalties_after == tw_penalties_before and cost_after < cost_before:
        is_better = True
    return is_better
import torch
from .base import Solver
from src.route import Route


class DRLSolver(Solver):
    def __init__(self, instance, dist_matrix, actor_network, device=None):
        super().__init__(instance, dist_matrix)
        self.actor_network = actor_network
        self.device = device if device else torch.device("cpu")

        # Calculate normalization maximums once during initialization
        self.max_x = max(n.x for n in self.instance.nodes)
        self.max_y = max(n.y for n in self.instance.nodes)
        self.max_time = max(n.due_date for n in self.instance.nodes)
        self.capacity = self.instance.capacity

    def solve(self, assigned_nodes):
        route = Route(self.instance, self.dist_matrix)

        # Do not perform inference on 1-node tasks
        if len(assigned_nodes) == 1:
            route.add_node(assigned_nodes[0])
            route.close_route()
            route.update_state()
            return route

        # 1. Static features: shape (1, Num_Nodes, 5)
        static_features = self._extract_static_tensor(assigned_nodes)

        # 2. Initialize Mask: shape (1, Num_Nodes)
        mask = torch.zeros((1, len(assigned_nodes)), dtype=torch.bool, device=self.device)

        # Initialize hidden state for the GRU memory
        hidden_state = None

        while not mask.all():
            current_node = route.last_node

            # 3. Dynamic Context (Normalized!): shape (1, 1, 4)
            context_list = [
                current_node.x / self.max_x if self.max_x > 0 else 0,
                current_node.y / self.max_y if self.max_y > 0 else 0,
                route.time / self.max_time if self.max_time > 0 else 0,
                route.load / self.capacity if self.capacity > 0 else 0
            ]
            context = torch.tensor(context_list, dtype=torch.float32, device=self.device)
            context = context.unsqueeze(0).unsqueeze(0)  # Add Batch and Seq_Len dimensions

            # 4. Neural network inference
            with torch.no_grad():
                # Pass hidden_state back into the network to maintain memory
                probs, log_probs, attn_scores, hidden_state = self.actor_network(
                    static_features, context, mask, hidden_state
                )

            # Greedy Decoding: Get the index with the highest probability
            # probs shape is (1, Num_Nodes), argmax returns a 1D tensor, .item() gets the python int
            next_idx = torch.argmax(probs, dim=1).item()

            # 5. Map back to Python object
            best_node = assigned_nodes[next_idx]

            # 6. Update route and mask
            route.add_node(best_node)
            mask[0, next_idx] = True

        route.close_route()
        route.update_state()

        return route

    def _extract_static_tensor(self, node_ids):
        """
        Extracts and normalizes features for the assigned nodes.
        Returns a FloatTensor of shape (1, Num_Nodes, 5).
        """
        features = []
        for nid in node_ids:
            node = self.instance.nodes[nid]

            # Normalize each feature to [0.0, 1.0]
            norm_x = node.x / self.max_x if self.max_x > 0 else 0
            norm_y = node.y / self.max_y if self.max_y > 0 else 0
            norm_demand = node.demand / self.capacity if self.capacity > 0 else 0
            norm_ready = node.ready_time / self.max_time if self.max_time > 0 else 0
            norm_due = node.due_date / self.max_time if self.max_time > 0 else 0

            features.append([norm_x, norm_y, norm_demand, norm_ready, norm_due])

        # Convert to tensor, move to device, and add Batch dimension
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0)
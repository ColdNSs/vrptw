import torch
from .base import BaseSolver
from src.route import Route


class NazariDRLSolver(BaseSolver):
    def __init__(self, instance, dist_matrix, actor_network, device=None):
        super().__init__(instance, dist_matrix)
        self.actor_network = actor_network
        self.device = device if device else torch.device("cpu")

        # Calculate normalization maximums once during initialization
        self.max_x = max(n.x for n in self.instance.nodes)
        self.max_y = max(n.y for n in self.instance.nodes)
        self.max_time = max(n.due_date for n in self.instance.nodes)
        self.capacity = self.instance.capacity

        self._precompute_global_features()

    def _precompute_global_features(self):
        """
        Creates a single Tensor containing the normalized features for ALL nodes
        in the instance. Shape: (Total_Nodes, 6).
        Stored permanently on the GPU.
        """
        features = []
        # We assume self.instance.nodes is ordered by node.id (0 to N)
        for node in self.instance.nodes:
            norm_x = node.x / self.max_x if self.max_x > 0 else 0
            norm_y = node.y / self.max_y if self.max_y > 0 else 0
            norm_demand = node.demand / self.capacity if self.capacity > 0 else 0
            norm_ready = node.ready_time / self.max_time if self.max_time > 0 else 0
            norm_due = node.due_date / self.max_time if self.max_time > 0 else 0
            norm_service = node.service_time / self.max_time if self.max_time > 0 else 0

            features.append([norm_x, norm_y, norm_demand, norm_ready, norm_due, norm_service])

        self.global_features = torch.tensor(features, dtype=torch.float32, device=self.device)

    def solve(self, assigned_nodes):
        route = Route(self.instance, self.dist_matrix)

        if not assigned_nodes:
            return route

        # Do not perform inference on 1-node tasks
        if len(assigned_nodes) == 1:
            route.add_node(assigned_nodes[0])
            route.close_route()
            route.update_state()
            return route

        # 1. Static features: shape (1, Num_Nodes, 6)
        # static_features = self._extract_static_tensor(assigned_nodes)
        node_ids =[n.id for n in assigned_nodes]
        indices = torch.tensor(node_ids, dtype=torch.long, device=self.device)
        static_features = self.global_features[indices].unsqueeze(0)

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
            # We use attn_scores (logits) to pick the argmax for maximum precision
            next_idx = torch.argmax(attn_scores, dim=1).item()

            # 5. Map back to Python object
            best_node = assigned_nodes[next_idx]

            # 6. Update route and mask
            route.add_node(best_node)
            mask[0, next_idx] = True

        route.close_route()
        route.update_state()

        return route

    def _extract_static_tensor(self, nodes):
        """
        Extracts and normalizes features for the assigned nodes.
        Returns a FloatTensor of shape (1, Num_Nodes, 6).
        """
        # utilizing the DRY helper function to convert each node, then stacking them.
        tensor = torch.stack([self._get_node_features(node) for node in nodes])
        return tensor.unsqueeze(0)  # Add Batch dimension

    def solve_batch(self, unvisited_lists):
        """
        Massively parallel GPU inference for heterogeneous route lengths.
        """
        if not unvisited_lists:
            return []

        batch_size = len(unvisited_lists)

        # Find the longest sub-task in the batch
        max_customers = max(len(u) for u in unvisited_lists)
        max_nodes = max_customers + 1  # +1 for the Depot

        # 1. Initialize node indices
        node_indices = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=self.device)

        # Initialize mask as ALL TRUE (invalid). We will unmask the valid ones.
        mask = torch.ones((batch_size, max_nodes), dtype=torch.bool, device=self.device)

        # 2. Fill Tensors
        for b, unvisited in enumerate(unvisited_lists):
            mask[b, 0] = True  # Mask depot so it's not visited mid-route
            for i, node in enumerate(unvisited):
                node_indices[b, i + 1] = node.id
                mask[b, i + 1] = False  # Unmask valid customers

        static_features = self.global_features[node_indices]

        # 3. Setup Routing State
        context = static_features[:, 0, :4].unsqueeze(1)  # Start at Depot
        hidden_state = None
        sequences = []

        # 4. Parallel GPU Rollout
        for step in range(max_customers):
            # To prevent NaN crashes on routes that finish early:
            done = mask[:, 1:].all(dim=1)
            mask[done, 0] = False

            with torch.no_grad():
                probs, log_probs, attn_scores, hidden_state = self.actor_network(
                    static_features, context, mask, hidden_state
                )

            # Argmax for greedy deterministic inference
            next_idx = torch.argmax(attn_scores, dim=1)
            sequences.append(next_idx)

            # Update Mask
            mask[torch.arange(batch_size), next_idx] = True

            # Update Context (Approximation for the Attention mechanism)
            selected_features = static_features[torch.arange(batch_size), next_idx]
            new_x = selected_features[:, 0:1].unsqueeze(1)
            new_y = selected_features[:, 1:2].unsqueeze(1)

            dist = torch.sqrt((new_x - context[:, :, 0:1]) ** 2 + (new_y - context[:, :, 1:2]) ** 2)
            new_time = torch.max(context[:, :, 2:3] + dist, selected_features[:, 3:4].unsqueeze(1))
            new_load = context[:, :, 3:4] + selected_features[:, 2:3].unsqueeze(1)

            context = torch.cat([new_x, new_y, new_time, new_load], dim=2)

        # 5. Unpack GPU Tensors back into Python Route Objects
        routes = []
        for b in range(batch_size):
            route = Route(self.instance, self.dist_matrix)

            for step in range(max_customers):
                idx = sequences[step][b].item()
                if idx > 0:  # Ignore index 0 (Depot / Padding idle)
                    node = unvisited_lists[b][idx - 1]
                    route.add_node(node)

            route.close_route()
            route.update_state()
            routes.append(route)

        return routes

    def _get_node_features(self, node):
        """Helper to normalize a single node's features into a PyTorch tensor"""
        return torch.tensor([
            node.x / self.max_x if self.max_x > 0 else 0,
            node.y / self.max_y if self.max_y > 0 else 0,
            node.demand / self.capacity if self.capacity > 0 else 0,
            node.ready_time / self.max_time if self.max_time > 0 else 0,
            node.due_date / self.max_time if self.max_time > 0 else 0,
            node.service_time / self.max_time if self.max_time > 0 else 0
        ], dtype=torch.float32, device=self.device)
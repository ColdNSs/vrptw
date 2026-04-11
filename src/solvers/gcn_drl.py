import torch
import numpy as np
from .base import BaseSolver
from src.route import Route


class GCNSolver(BaseSolver):
    """
    Non-Autoregressive (NAR) Solver using Graph Convolutional Networks.
    Executes the neural network exactly ONCE per batch to generate an N x N Heatmap,
    then uses a lightning-fast Python/Tensor loop to decode the paths.
    """

    def __init__(self, instance, dist_matrix, actor_network, device=None):
        super().__init__(instance, dist_matrix)
        self.actor_network = actor_network
        self.device = device if device else torch.device("cpu")

        # Calculate normalization maximums once
        self.max_time = max(n.due_date for n in self.instance.nodes)
        self.capacity = self.instance.capacity
        self.max_dist = np.max(self.dist_matrix) if np.max(self.dist_matrix) > 0 else 1.0

        # --- PRECOMPUTE THE GLOBAL GRAPH ---
        self._precompute_global_features()

    def _precompute_global_features(self):
        """
        Creates global tensors for Node Features and the Distance Matrix.
        GCNs don't use X and Y; they only need (Demand, Ready, Due, Service).
        """
        features = []
        for node in self.instance.nodes:
            norm_demand = node.demand / self.capacity if self.capacity > 0 else 0
            norm_ready = node.ready_time / self.max_time if self.max_time > 0 else 0
            norm_due = node.due_date / self.max_time if self.max_time > 0 else 0
            norm_service = node.service_time / self.max_time if self.max_time > 0 else 0

            features.append([norm_demand, norm_ready, norm_due, norm_service])

        # Shape: (Total_Nodes, 4)
        self.global_node_features = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Shape: (Total_Nodes, Total_Nodes)
        # We also normalize the distance matrix to [0.0, 1.0] for the GCN Edge Encoder
        normalized_dist = np.array(self.dist_matrix) / self.max_dist
        self.global_dist_matrix = torch.tensor(normalized_dist, dtype=torch.float32, device=self.device)

    def solve(self, assigned_nodes):
        """Single route solver. Uses the global trick and Heatmap decoder."""
        route = Route(self.instance, self.dist_matrix)

        if not assigned_nodes:
            return route

        if len(assigned_nodes) == 1:
            route.add_node(assigned_nodes[0])
            route.close_route()
            route.update_state()
            return route

        # Extract specific nodes for this vehicle (Depot is index 0)
        node_ids = [0] + [n.id for n in assigned_nodes]
        indices = torch.tensor(node_ids, dtype=torch.long, device=self.device)

        # Build Sub-Graph Tensors
        static_features = self.global_node_features[indices].unsqueeze(0)  # (1, N, 4)
        sub_dist_matrix = self.global_dist_matrix[indices.unsqueeze(-1), indices.unsqueeze(0)].unsqueeze(0)  # (1, N, N)

        # ONE-SHOT NETWORK FORWARD PASS
        with torch.no_grad():
            heatmap_logits = self.actor_network(static_features, sub_dist_matrix, padding_mask=None)

        # DECODE THE HEATMAP
        num_customers = len(assigned_nodes)
        mask = torch.zeros((1, num_customers + 1), dtype=torch.bool, device=self.device)
        mask[0, 0] = True  # Mark depot as visited initially
        current_node = torch.tensor([0], dtype=torch.long, device=self.device)

        for _ in range(num_customers):
            # Slice the Heatmap for the current node's row
            current_row_logits = heatmap_logits[0, current_node, :].squeeze(0)
            current_row_logits = current_row_logits.masked_fill(mask[0], -1e8)

            next_idx = torch.argmax(current_row_logits).item()
            best_node = assigned_nodes[next_idx - 1]  # -1 because assigned_nodes excludes Depot

            route.add_node(best_node)
            mask[0, next_idx] = True
            current_node[0] = next_idx

        route.close_route()
        route.update_state()
        return route

    def solve_batch(self, unvisited_lists):
        """
        Massively parallel Heatmap extraction for heterogeneous route lengths.
        O(1) Neural Network pass.
        """
        if not unvisited_lists:
            return []

        batch_size = len(unvisited_lists)
        max_customers = max(len(u) for u in unvisited_lists)
        max_nodes = max_customers + 1  # +1 for the Depot

        # 1. Initialize Index Matrix
        node_indices = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=self.device)
        padding_mask = torch.ones((batch_size, max_nodes), dtype=torch.bool, device=self.device)

        # 2. Fill Index Matrix
        for b, unvisited in enumerate(unvisited_lists):
            padding_mask[b, 0] = False  # Unmask depot (valid node)
            for i, node in enumerate(unvisited):
                node_indices[b, i + 1] = node.id
                padding_mask[b, i + 1] = False  # Unmask valid customers

        # 3. Extract Sub-Graphs Instantly via Advanced Indexing
        static_features = self.global_node_features[node_indices]  # (Batch, max_nodes, 4)

        # Extract (Batch, max_nodes, max_nodes) distance matrices instantly!
        row_idx = node_indices.unsqueeze(-1)  # (Batch, max_nodes, 1)
        col_idx = node_indices.unsqueeze(1)  # (Batch, 1, max_nodes)
        sub_dist_matrix = self.global_dist_matrix[row_idx, col_idx]

        # 4. =======================================================
        # ONE-SHOT PARALLEL GPU NEURAL NETWORK PASS
        # =======================================================
        with torch.no_grad():
            heatmap_logits = self.actor_network(static_features, sub_dist_matrix, padding_mask)

        # 5. =======================================================
        # FAST DECODER LOOP (Zero Neural Networks)
        # =======================================================
        routing_mask = padding_mask.clone()  # Already True for padded/fake nodes
        routing_mask[:, 0] = True  # Mark depot as visited

        current_node = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        sequences = []

        for step in range(max_customers):
            # Idling Trick: If a route finishes early, unmask the depot so it safely idles there
            done = routing_mask[:, 1:].all(dim=1)
            routing_mask[done, 0] = False

            # Slice the heatmap using advanced indexing to get exactly the row we are currently standing on
            current_row_logits = heatmap_logits[torch.arange(batch_size), current_node, :]
            current_row_logits = current_row_logits.masked_fill(routing_mask, -1e8)

            # Argmax decoding
            next_idx = torch.argmax(current_row_logits, dim=1)
            sequences.append(next_idx)

            # Update routing state
            routing_mask[torch.arange(batch_size), next_idx] = True
            current_node = next_idx

        # 6. Unpack GPU Tensors back into Python Route Objects
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
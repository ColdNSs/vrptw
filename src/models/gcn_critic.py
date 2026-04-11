import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNCriticNetwork(nn.Module):
    def __init__(self, node_input_dim=4, hidden_dim=128):
        super(GCNCriticNetwork, self).__init__()

        # --- THE FIX: node_input_dim + 1 ---
        # We add 1 to accommodate the "Average Distance" feature!
        self.node_encoder = nn.Linear(node_input_dim + 1, hidden_dim)

        # Transformer gives global map awareness
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.context_layer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features, dist_matrix, padding_mask=None):
        """
        node_features: (Batch, N, 4)
        dist_matrix: (Batch, N, N)
        """
        # --- SPATIAL AWARENESS INJECTION ---
        # Calculate the average distance from each node to all other nodes.
        # Shape: (Batch, N, 1)
        mean_dists = dist_matrix.mean(dim=2, keepdim=True)

        # Concatenate this spatial metric to the raw node features
        # Shape becomes: (Batch, N, 5)
        enhanced_features = torch.cat([node_features, mean_dists], dim=-1)

        # 1. Translate
        h = self.node_encoder(enhanced_features)

        # 2. Town Hall Meeting
        h = self.context_layer(h, src_key_padding_mask=padding_mask)

        # 3. Graph Pooling (Get the single Global Summary)
        if padding_mask is not None:
            valid_nodes = (~padding_mask).float().unsqueeze(-1)
            h = h * valid_nodes
            graph_summary = h.sum(dim=1) / valid_nodes.sum(dim=1)
        else:
            graph_summary = h.mean(dim=1)

        # 4. Final Baseline Prediction
        baseline = self.value_head(graph_summary)
        return baseline
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNActorNetwork(nn.Module):
    def __init__(self, node_input_dim=4, hidden_dim=128):
        super(GCNActorNetwork, self).__init__()

        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(1, hidden_dim)

        # Transformer Context
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.context_layer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- THE OPTIMIZATION: Separate Projections ---
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)

        # A much smaller, faster layer to squash the 128D sum down to 1 scalar score
        self.score_combine = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, dist_matrix, padding_mask=None):
        N = node_features.size(1)

        # 1. Embed Nodes and Share Global Context
        h = self.node_encoder(node_features)
        h = self.context_layer(h, src_key_padding_mask=padding_mask)  # (Batch, N, 128)

        # 2. Project Queries and Keys
        q = self.query(h).unsqueeze(2)  # (Batch, N, 1, 128) - Column broadcast
        k = self.key(h).unsqueeze(1)  # (Batch, 1, N, 128) - Row broadcast

        # 3. Embed Edges
        e = self.edge_encoder(dist_matrix.unsqueeze(-1))  # (Batch, N, N, 128)

        # 4. --- THE MEMORY SAVIOR: Broadcasting Addition ---
        # Instead of concatenating to 384, we add them.
        # The shape stays (Batch, N, N, 128). PyTorch does this in C++ insanely fast.
        combined = q + k + e

        # 5. Predict the Heatmap
        # Apply Tanh to mix the added signals, then squash to 1 dimension
        heatmap_logits = self.score_combine(torch.tanh(combined)).squeeze(-1)  # (Batch, N, N)

        # 6. Safety Masking
        diag_mask = torch.eye(N, dtype=torch.bool, device=heatmap_logits.device).unsqueeze(0)
        heatmap_logits = heatmap_logits.masked_fill(diag_mask, -1e8)

        if padding_mask is not None:
            heatmap_logits = heatmap_logits.masked_fill(padding_mask.unsqueeze(1), -1e8)
            heatmap_logits = heatmap_logits.masked_fill(padding_mask.unsqueeze(2), -1e8)

        return heatmap_logits

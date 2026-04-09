import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    def __init__(self, static_input_dim=5, dynamic_input_dim=4, hidden_dim=128):
        super(CriticNetwork, self).__init__()

        # Encoders (Separate from the Actor so they learn independently)
        self.static_encoder = nn.Linear(static_input_dim, hidden_dim)
        self.dynamic_encoder = nn.Linear(dynamic_input_dim, hidden_dim)

        # Feed-Forward Neural Network to output the single baseline scalar
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, static_features, dynamic_context):
        """
        static_features: (Batch, Num_Nodes, 5)
        dynamic_context: (Batch, 1, 4)
        """
        static_hidden = self.static_encoder(static_features)  # (Batch, Num_Nodes, 128)

        # Remove the extra sequence dimension from the dynamic context
        dynamic_hidden = self.dynamic_encoder(dynamic_context).squeeze(1)  # (Batch, 128)

        # AGGREGATION (Mean Pooling)
        # We average the features of all unvisited nodes to get a "Global Map Summary"
        static_summary = static_hidden.mean(dim=1)  # (Batch, 128)

        # Concatenate the Map Summary with the Vehicle's Current Context
        combined = torch.cat([static_summary, dynamic_hidden], dim=1)  # (Batch, 256)

        # Pass through the MLP to get the baseline estimate
        out = torch.relu(self.fc1(combined))
        baseline = self.fc2(out)  # (Batch, 1)

        return baseline
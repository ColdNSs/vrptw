import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, static_input_dim=5, dynamic_input_dim=4, hidden_dim=128):
        super(ActorNetwork, self).__init__()

        # 1. Static Encoder (Reads the map properties)
        # Maps[x, y, demand, ready_time, due_date] -> 128-dim vector
        self.static_encoder = nn.Linear(static_input_dim, hidden_dim)

        # 2. Dynamic Encoder (Reads the current state)
        # Maps[curr_x, curr_y, time, load] -> 128-dim vector
        self.dynamic_encoder = nn.Linear(dynamic_input_dim, hidden_dim)

        # 3. The GRU (Memory of the sequence)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 4. Attention Mechanism layers
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1)

    def forward(self, static_features, dynamic_context, mask, hidden_state=None):
        """
        static_features: Tensor of shape (Batch, Num_Nodes, 5)
        dynamic_context: Tensor of shape (Batch, 1, 4)
        mask: Boolean Tensor of shape (Batch, Num_Nodes). True means VISITED (invalid).
        """

        # --- A. Encode Static Map (Only needs to happen once per route) ---
        # Shape: (Batch, Num_Nodes, 128)
        static_hidden = self.static_encoder(static_features)

        # --- B. Encode Dynamic Context & Update Memory ---
        # Shape: (Batch, 1, 128)
        dynamic_hidden = self.dynamic_encoder(dynamic_context)

        # GRU takes the dynamic context and the previous hidden state
        gru_out, hidden_state = self.decoder_gru(dynamic_hidden, hidden_state)

        # --- C. Attention Mechanism (Calculate Scores) ---
        # Query comes from the GRU (Where are we?)
        query = self.attn_query(gru_out)  # (Batch, 1, 128)

        # Keys come from the Static Map (Where can we go?)
        keys = self.attn_key(static_hidden)  # (Batch, Num_Nodes, 128)

        # Tanh attention scoring
        # Broadcasting query to match keys shape
        attn_scores = self.attn_v(torch.tanh(query + keys)).squeeze(-1)  # (Batch, Num_Nodes)

        # --- D. Apply Masking ---
        # We replace the scores of visited/invalid nodes with -infinity
        # So when Softmax runs, their probability becomes exactly 0
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # --- E. Output Probabilities ---
        probs = F.softmax(attn_scores, dim=1)
        log_probs = F.log_softmax(attn_scores, dim=1)

        return probs, log_probs, attn_scores, hidden_state
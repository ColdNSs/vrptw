import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import random

import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from models import GCNActorNetwork, GCNCriticNetwork
from src.utils import get_device


class GCNTrainer:
    def __init__(self, min_num_nodes=5, max_num_nodes=20, cluster=0.5, batch_size=256, lr=5e-4, epochs=20,
                 steps_per_epoch=1000):
        if min_num_nodes < 3:
            raise ValueError("min_num_nodes should be at least 3")
        self.device = get_device()
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.cluster = cluster
        self.batch_size = batch_size
        self.epoch = 0
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # 1. Initialize Networks (node_input_dim=4: demand, ready, due, service)
        self.actor = GCNActorNetwork(node_input_dim=4).to(self.device)
        self.critic = GCNCriticNetwork(node_input_dim=4).to(self.device)

        # 2. Adam Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def generate_batch_data(self, current_num_nodes, cluster):
        """
        Generates a batch of random synthetic VRPTW clusters for GCN.
        Generates x, y internally to create the distance matrix, but only outputs
        the explicitly required network features.
        """
        # 1. Generate Coordinates for the distance matrix
        coords = torch.rand((self.batch_size, current_num_nodes, 2), device=self.device)
        centers = torch.rand((self.batch_size, 3, 2), device=self.device)
        half_b = int(self.batch_size * cluster)

        if half_b:
            cluster_assignments = torch.randint(0, 3, (half_b, current_num_nodes), device=self.device)
            gathered_centers = torch.gather(centers[:half_b], 1, cluster_assignments.unsqueeze(-1).expand(-1, -1, 2))
            clustered_coords = gathered_centers + (
                        torch.randn((half_b, current_num_nodes, 2), device=self.device) * 0.1)
            coords[:half_b, :, :] = torch.clamp(clustered_coords, 0.0, 1.0)

        # 2. Compute Distance Matrix (Shape: Batch, N, N)
        # torch.cdist computes the pairwise Euclidean distance beautifully
        dist_matrix = torch.cdist(coords, coords, p=2)

        # 3. Generate Node Features [demand, ready, due_date, service_time]
        node_features = torch.rand((self.batch_size, current_num_nodes, 4), device=self.device)
        node_features[:, :, 3] = node_features[:, :, 3] * 0.05  # Service time reduction

        return node_features, dist_matrix

    def train(self):
        print(f"Starting GCN Training on {self.device}...")

        while self.epoch < self.epochs:
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            avg_reward = 0.0

            self.actor.train()
            self.critic.train()

            for step in range(self.steps_per_epoch):
                current_num_nodes = random.randint(self.min_num_nodes, self.max_num_nodes)
                node_features, dist_matrix = self.generate_batch_data(current_num_nodes, self.cluster)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # ========================================================
                # 1. ONE-SHOT NEURAL NETWORK PASS
                # ========================================================
                # No for loops! The entire map is solved into a heatmap instantly.
                heatmap_logits = self.actor(node_features, dist_matrix, padding_mask=None)
                baseline = self.critic(node_features, dist_matrix, padding_mask=None)

                # ========================================================
                # 2. FAST ROLLOUT DECODER (Zero Neural Networks Here)
                # ========================================================
                mask = torch.zeros((self.batch_size, current_num_nodes), dtype=torch.bool, device=self.device)
                mask[:, 0] = True  # Mask the depot

                current_node = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

                log_probs_list = []
                actions_list = [current_node]  # Start at depot

                for i in range(current_num_nodes - 1):
                    # Slice the heatmap for the row of the current node
                    # Shape: (Batch, N)
                    current_row_logits = heatmap_logits[torch.arange(self.batch_size), current_node, :]

                    # Mask visited nodes
                    current_row_logits = current_row_logits.masked_fill(mask, -1e8)

                    # Sample
                    m = Categorical(logits=current_row_logits)
                    action = m.sample()

                    actions_list.append(action)
                    log_probs_list.append(m.log_prob(action))

                    # Update mask & current node
                    mask = mask.clone()
                    mask[torch.arange(self.batch_size), action] = True
                    current_node = action

                # Force return to depot
                depot_ends = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
                actions_list.append(depot_ends)

                # ========================================================
                # 3. REWARD AND BACKPROPAGATION
                # ========================================================
                rewards = self._calculate_batched_rewards(node_features, dist_matrix, actions_list)

                advantage = rewards - baseline.detach()

                sum_log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1).unsqueeze(-1)

                actor_loss = -(advantage * sum_log_probs).mean()
                critic_loss = F.mse_loss(baseline, rewards)

                actor_loss.backward()
                critic_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                avg_reward += rewards.mean().item()

            print(f"Epoch {self.epoch + 1}/{self.epochs} | Actor Loss: {actor_loss_sum / self.steps_per_epoch:.4f} | "
                  f"Critic Loss: {critic_loss_sum / self.steps_per_epoch:.4f} | Avg Cost: {avg_reward / self.steps_per_epoch:.4f}")

            # Save Checkpoints
            checkpoints_path = root / "checkpoints"
            torch.save(self.actor.state_dict(), checkpoints_path / f"gcn_actor_epoch_{self.epoch + 1}.pt")
            torch.save(self.critic.state_dict(), checkpoints_path / f"gcn_critic_epoch_{self.epoch + 1}.pt")
            torch.save(self.actor_optimizer.state_dict(), checkpoints_path / f"gcn_actor_opt_epoch_{self.epoch + 1}.pt")
            torch.save(self.critic_optimizer.state_dict(),
                       checkpoints_path / f"gcn_critic_opt_epoch_{self.epoch + 1}.pt")

            self.epoch += 1

    def _calculate_batched_rewards(self, node_features, dist_matrix, actions_list):
        """
        Calculates score directly using the generated dist_matrix.
        node_features indices: 0: demand, 1: ready, 2: due, 3: service
        """
        batch_size = node_features.shape[0]
        num_nodes = len(actions_list)

        # Shape: (Batch, Num_Nodes)
        route_seq = torch.stack(actions_list, dim=1)
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)

        # 1. Distances (Incredibly fast direct lookup)
        origins = route_seq[:, :-1]
        destinations = route_seq[:, 1:]
        segment_distances = dist_matrix[batch_idx, origins, destinations]  # Shape: (Batch, Route_Length - 1)
        total_distances = segment_distances.sum(dim=1)

        # 2. Time Windows
        ordered_features = node_features[batch_idx, route_seq]
        current_times = ordered_features[:, 0, 1]  # Start at Node 0's ready_time
        tw_penalties = torch.zeros(batch_size, device=self.device)

        for k in range(num_nodes - 1):
            dist_step = segment_distances[:, k]
            service_times = ordered_features[:, k, 3]

            departure_times = current_times + service_times
            arrival_times = departure_times + dist_step

            next_due_dates = ordered_features[:, k + 1, 2]
            next_ready_times = ordered_features[:, k + 1, 1]

            lateness = arrival_times - next_due_dates
            tw_penalties += torch.clamp(lateness, min=0.0)

            current_times = torch.max(arrival_times, next_ready_times)

        penalty_weight = 100.0
        total_score = total_distances + (penalty_weight * tw_penalties)

        return -total_score.unsqueeze(1)

    def load_weights(self, actor_name, critic_name, actor_opt_name=None, critic_opt_name=None):
        actor_path = root / "checkpoints" / actor_name
        critic_path = root / "checkpoints" / critic_name
        actor_opt_path = root / "checkpoints" / actor_opt_name if actor_opt_name else None
        critic_opt_path = root / "checkpoints" / critic_opt_name if critic_opt_name else None

        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device, weights_only=True))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device, weights_only=True))
            print(f"--> Successfully loaded networks:\n  Actor: {actor_path}\n  Critic: {critic_path}")

        if actor_opt_path and os.path.exists(actor_opt_path):
            self.actor_optimizer.load_state_dict(
                torch.load(actor_opt_path, map_location=self.device, weights_only=False))
            print(f"--> Successfully loaded Actor Optimizer: {actor_opt_path}")

        if critic_opt_path and os.path.exists(critic_opt_path):
            self.critic_optimizer.load_state_dict(
                torch.load(critic_opt_path, map_location=self.device, weights_only=False))
            print(f"--> Successfully loaded Critic Optimizer: {critic_opt_path}")

    def load_epoch(self, epoch):
        if epoch < 1:
            raise ValueError("Invalid epoch number")
        actor_name = f"gcn_actor_epoch_{epoch}.pt"
        critic_name = f"gcn_critic_epoch_{epoch}.pt"
        actor_opt_name = f"gcn_actor_opt_epoch_{epoch}.pt"
        critic_opt_name = f"gcn_critic_opt_epoch_{epoch}.pt"

        self.load_weights(actor_name, critic_name, actor_opt_name, critic_opt_name)
        self.epoch = epoch


if __name__ == "__main__":
    os.makedirs(root / "checkpoints", exist_ok=True)
    trainer = GCNTrainer(epochs=20, max_num_nodes=50, batch_size=128, steps_per_epoch=1000)
    trainer.train()
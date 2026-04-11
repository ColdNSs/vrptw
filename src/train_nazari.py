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

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

from models.nazari_actor import NazariActorNetwork
from models.nazari_critic import NazariCriticNetwork
from src.utils import get_device


class DRLTrainer:
    def __init__(self, min_num_nodes=5, max_num_nodes=20, cluster=0.5, batch_size=256, lr=5e-4, epochs=20, steps_per_epoch=1000):
        if min_num_nodes < 3:
            raise ValueError("min_num_nodes should be at least 3")
        self.device = get_device()
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.cluster =cluster
        self.batch_size = batch_size
        self.epoch = 0
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # 1. Initialize Networks (Algorithm 1, Line 1)
        self.actor = NazariActorNetwork().to(self.device)
        self.critic = NazariCriticNetwork().to(self.device)

        # 2. Adam Optimizers for both networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def generate_batch_data(self, current_num_nodes, cluster):
        """
        Generates a batch of random synthetic VRPTW clusters.
        In reality, you generate random[x, y, demand, ready, due_date, service_time]
        For training, all values should be normalized between 0 and 1.
        """
        # Shape: (Batch_Size, Num_Nodes, 6 features)
        # Node 0 in dim=1 is always the depot.
        static_features = torch.rand((self.batch_size, current_num_nodes, 6), device=self.device)

        # We pick 3 random "cluster centers" for the batch
        centers = torch.rand((self.batch_size, 3, 2), device=self.device)

        # For half the batch, we overwrite the X,Y coords with clustered coords
        half_b = int(self.batch_size * cluster)

        if half_b:

            # Randomly assign each node to one of the 3 centers
            cluster_assignments = torch.randint(0, 3, (half_b, current_num_nodes), device=self.device)

            # Gather the center coords and add narrow Gaussian noise (0.1) to create tight clusters
            gathered_centers = torch.gather(centers[:half_b], 1, cluster_assignments.unsqueeze(-1).expand(-1, -1, 2))
            clustered_coords = gathered_centers + (torch.randn((half_b, current_num_nodes, 2), device=self.device) * 0.1)

            # Clip to [0,1] bounds and assign to the first half of the batch
            static_features[:half_b, :, :2] = torch.clamp(clustered_coords, 0.0, 1.0)

        # Force service times to be relatively small
        # so the vehicle doesn't spend its entire day at 1 node.
        static_features[:, :, 5] = static_features[:, :, 5] * 0.05

        return static_features

    def train(self):
        print(f"Starting DRL Training on {self.device}...")

        while self.epoch < self.epochs:
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            avg_reward = 0.0

            self.actor.train()
            self.critic.train()

            for step in range(self.steps_per_epoch):
                # 1. Generate N random problem instances (Algorithm 1, Line 4)
                current_num_nodes = random.randint(self.min_num_nodes, self.max_num_nodes)
                static_features = self.generate_batch_data(current_num_nodes, self.cluster)

                # 2. Reset gradients (Algorithm 1, Line 3)
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # 3. Initial State Setup
                # Context: [curr_x, curr_y, time, load]. Initially matches Depot features.
                dynamic_context = static_features[:, 0, :4].unsqueeze(1)  # Shape: (Batch, 1, 4)

                mask = torch.zeros((self.batch_size, current_num_nodes), dtype=torch.bool, device=self.device)
                # Mask the depot (Node 0)
                mask[:, 0] = True

                hidden_state = None

                # Variables to track the episode
                log_probs_list = []
                actions_list = []

                # Start at the depot
                depot_starts = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
                actions_list.append(depot_starts)

                # 4. Rollout Loop (Algorithm 1, Lines 6-9)
                for i in range(current_num_nodes - 1):
                    # Get probabilities from Actor
                    probs, log_probs, attn_scores, hidden_state = self.actor(static_features, dynamic_context, mask, hidden_state)

                    # SAMPLING: We don't use argmax in training! We roll the weighted dice.
                    m = Categorical(logits=attn_scores)
                    action = m.sample()  # Shape: (Batch,)
                    actions_list.append(action)

                    # Store the log probability of the chosen action for the Loss function
                    selected_log_probs = m.log_prob(action)
                    log_probs_list.append(selected_log_probs)

                    # Update Mask
                    mask = mask.clone()
                    mask[torch.arange(self.batch_size), action] = True

                    # Dynamic Context
                    selected_features = static_features[torch.arange(self.batch_size), action]
                    # We extract the components as individual (Batch, 1, 1) tensors
                    new_x = selected_features[:, 0:1].unsqueeze(1)
                    new_y = selected_features[:, 1:2].unsqueeze(1)
                    old_time = dynamic_context[:, :, 2:3]
                    old_load = dynamic_context[:, :, 3:4]

                    # We glue them back together into a fresh (Batch, 1, 4) tensor
                    dynamic_context = torch.cat([new_x, new_y, old_time, old_load], dim=2)

                actions_list.append(depot_starts)

                # 5. Compute the Reward 'L' (Algorithm 1, Line 10)
                # Note: You must calculate the total batched distance + penalties here.
                # For this skeleton, we pretend we have a function that returns the costs.
                rewards = self._calculate_batched_rewards(static_features, actions_list)  # Shape: (Batch, 1)

                # 6. Get Critic Baseline 'E'
                # Pass the initial map and depot context to the critic
                initial_context = static_features[:, 0, :4].unsqueeze(1)
                baseline = self.critic(static_features, initial_context)  # Shape: (Batch, 1)

                # 7. Calculate Advantage
                # Advantage = Actual Reward (Cost) - Baseline Expectation
                # We detach() the baseline so Actor gradients don't flow backward into the Critic!
                advantage = rewards - baseline.detach()

                # 8. Calculate Losses (Algorithm 1, Lines 12 & 13)
                sum_log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1).unsqueeze(-1)  # (Batch, 1)

                actor_loss = -(advantage * sum_log_probs).mean() # Negative because we maximize reward
                critic_loss = F.mse_loss(baseline, rewards)

                # 9. Backpropagation (Algorithm 1, Lines 14 & 15)
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
            torch.save(self.actor.state_dict(), checkpoints_path / f"actor_epoch_{self.epoch + 1}.pt")
            torch.save(self.critic.state_dict(), checkpoints_path / f"critic_epoch_{self.epoch + 1}.pt")
            torch.save(self.actor_optimizer.state_dict(), checkpoints_path / f"actor_opt_epoch_{self.epoch + 1}.pt")
            torch.save(self.critic_optimizer.state_dict(), checkpoints_path / f"critic_opt_epoch_{self.epoch + 1}.pt")

            self.epoch += 1

    def _calculate_batched_rewards(self, static_features, actions_list):
        """
        Translates routing logic to highly parallel GPU operations.
        static_features: (Batch, Num_Nodes, 6) ->[x, y, demand, ready_time, due_date, service_time]
        actions_list: A list of length Num_Nodes, where each element is a tensor of shape (Batch,)
        """
        batch_size = static_features.shape[0]
        num_nodes = len(actions_list)

        # 1. Stack the actions into a single sequence matrix
        # Shape: (Batch, Num_Nodes)
        route_seq = torch.stack(actions_list, dim=1)

        # 2. Gather the features for the nodes in the order they were visited
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)

        # ordered_features Shape: (Batch, Num_Nodes, 6)
        ordered_features = static_features[batch_idx, route_seq]

        # 3. Calculate Batched Distances
        # We extract the X, Y coordinates: Shape (Batch, Num_Nodes, 2)
        coords = ordered_features[:, :, :2]

        # Distance from Node i to Node i+1
        segment_diffs = coords[:, 1:] - coords[:, :-1]
        segment_distances = torch.norm(segment_diffs, p=2, dim=2)  # Euclidean distance

        # Total distance for each route (Shape: Batch)
        total_distances = segment_distances.sum(dim=1)

        # 4. Calculate Batched Time Windows (Sequential across the nodes)
        current_times = ordered_features[:, 0, 3]  # Start at Node 0's ready_time
        tw_penalties = torch.zeros(batch_size, device=self.device)

        for k in range(num_nodes - 1):
            # Distance for this specific step k across all 256 batches
            dist_step = segment_distances[:, k]

            # NEW: Extract the service time of the node we are LEAVING (Index 5)
            service_times = ordered_features[:, k, 5]

            # Departure = The time we started service + The duration of the service
            departure_times = current_times + service_times

            arrival_times = departure_times + dist_step

            next_due_dates = ordered_features[:, k + 1, 4]
            next_ready_times = ordered_features[:, k + 1, 3]

            # Penalty calculation using torch.clamp (only keeps positive differences)
            lateness = arrival_times - next_due_dates
            tw_penalties += torch.clamp(lateness, min=0.0)

            # Update current time: max(arrival_time, ready_time)
            current_times = torch.max(arrival_times, next_ready_times)

        # 5. Final Reward Calculation
        penalty_weight = 100.0
        total_score = total_distances + (penalty_weight * tw_penalties)

        # Reward is Negative Score
        rewards = -total_score.unsqueeze(1)

        return rewards

    def load_weights(self, actor_name, critic_name, actor_opt_name=None, critic_opt_name=None):
        """
        Loads pre-trained weights into the Actor and Critic networks,
        and optionally restores optimizer momentum states.
        """

        actor_path = root / "checkpoints" / actor_name
        critic_path = root / "checkpoints" / critic_name
        actor_opt_path = root / "checkpoints" / actor_opt_name if actor_opt_name else None
        critic_opt_path = root / "checkpoints" / critic_opt_name if critic_opt_name else None

        # 1. Load Model Weights
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            print(f"--> Successfully loaded networks:\n  Actor: {actor_path}\n  Critic: {critic_path}")

        # 2. Load Optimizer States (If provided)
        if actor_opt_path and os.path.exists(actor_opt_path):
            self.actor_optimizer.load_state_dict(torch.load(actor_opt_path, map_location=self.device))
            print(f"--> Successfully loaded Actor Optimizer: {actor_opt_path}")

        if critic_opt_path and os.path.exists(critic_opt_path):
            self.critic_optimizer.load_state_dict(torch.load(critic_opt_path, map_location=self.device))
            print(f"--> Successfully loaded Critic Optimizer: {critic_opt_path}")

    def load_epoch(self, epoch):
        if epoch < 1:
            raise ValueError("Invalid epoch number")
        actor_name = f"actor_epoch_{epoch}.pt"
        critic_name = f"critic_epoch_{epoch}.pt"
        actor_opt_name = f"actor_opt_epoch_{epoch}.pt"
        critic_opt_name = f"critic_opt_epoch_{epoch}.pt"

        self.load_weights(actor_name, critic_name, actor_opt_name, critic_opt_name)
        self.epoch = epoch


if __name__ == "__main__":
    os.makedirs(root / "checkpoints", exist_ok=True)
    trainer = DRLTrainer(epochs=60, max_num_nodes=50)
    trainer.load_epoch(39)
    trainer.train()
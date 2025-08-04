"""
State-of-the-art auto-scaling decision model using Deep Reinforcement Learning.
Combines forecasting predictions with system state to make optimal scaling decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import random
import math


class ScalingActionSpace:
    """Defines the scaling action space."""
    
    def __init__(self, 
                 min_jobs: int = 1,
                 max_jobs: int = 100,
                 scaling_steps: List[float] = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0]):
        
        self.min_jobs = min_jobs
        self.max_jobs = max_jobs
        self.scaling_steps = scaling_steps
        self.action_dim = len(scaling_steps)
        
        # Map actions to scaling multipliers
        self.action_to_multiplier = {i: step for i, step in enumerate(scaling_steps)}
        self.multiplier_to_action = {step: i for i, step in enumerate(scaling_steps)}
    
    def apply_action(self, current_jobs: int, action: int) -> int:
        """Apply scaling action to current job count."""
        multiplier = self.action_to_multiplier[action]
        new_jobs = int(current_jobs * multiplier)
        return max(self.min_jobs, min(self.max_jobs, new_jobs))
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        multiplier = self.action_to_multiplier[action]
        if multiplier < 1.0:
            return f"Scale Down ({multiplier}x)"
        elif multiplier > 1.0:
            return f"Scale Up ({multiplier}x)"
        else:
            return "No Change (1.0x)"


class DQNNetwork(nn.Module):
    """Deep Q-Network for scaling decisions."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.1,
                 use_dueling: bool = True,
                 use_noisy: bool = True):
        
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        
        # Feature extraction layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            if use_noisy:
                layers.append(NoisyLinear(prev_dim, hidden_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
            
            layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        if use_dueling:
            # Dueling DQN architecture
            self.value_head = nn.Sequential(
                NoisyLinear(prev_dim, hidden_dims[-1] // 2) if use_noisy else nn.Linear(prev_dim, hidden_dims[-1] // 2),
                nn.ReLU(),
                NoisyLinear(hidden_dims[-1] // 2, 1) if use_noisy else nn.Linear(hidden_dims[-1] // 2, 1)
            )
            
            self.advantage_head = nn.Sequential(
                NoisyLinear(prev_dim, hidden_dims[-1] // 2) if use_noisy else nn.Linear(prev_dim, hidden_dims[-1] // 2),
                nn.ReLU(),
                NoisyLinear(hidden_dims[-1] // 2, action_dim) if use_noisy else nn.Linear(hidden_dims[-1] // 2, action_dim)
            )
        else:
            # Standard DQN
            self.q_head = nn.Sequential(
                NoisyLinear(prev_dim, action_dim) if use_noisy else nn.Linear(prev_dim, action_dim)
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_extractor(state)
        
        if self.use_dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            
            # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise tensors (not parameters)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise tensors."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch with prioritized sampling."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = list(zip(*samples))
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)


class AutoScalingAgent(pl.LightningModule):
    """
    Deep RL agent for auto-scaling decisions.
    
    Uses Double DQN with Dueling Networks, Noisy Networks, and Prioritized Experience Replay.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_space: ScalingActionSpace,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 tau: float = 0.005,  # Soft update parameter
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 reward_scale: float = 1.0,
                 **network_kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.state_dim = state_dim
        self.action_space = action_space
        self.action_dim = action_space.action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.reward_scale = reward_scale
        
        # Networks
        self.q_network = DQNNetwork(state_dim, self.action_dim, **network_kwargs)
        self.target_network = DQNNetwork(state_dim, self.action_dim, **network_kwargs)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Metrics
        self.training_rewards = []
        self.episode_lengths = []
    
    def get_epsilon(self) -> float:
        """Get current epsilon for exploration."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 math.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using epsilon-greedy or deterministic policy."""
        if deterministic or random.random() > self.get_epsilon():
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                action = q_values.argmax().item()
        else:
            action = random.randrange(self.action_dim)
        
        if not deterministic:
            self.steps_done += 1
        
        return action
    
    def compute_td_error(self, states, actions, rewards, next_states, dones):
        """Compute TD errors for prioritized replay."""
        with torch.no_grad():
            # Double DQN
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        current_q_values = self.q_network(states).gather(1, actions)
        td_errors = target_q_values - current_q_values
        
        return td_errors.abs().squeeze()
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample from replay buffer
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(experiences[0]).to(self.device)
        actions = torch.LongTensor(experiences[1]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(experiences[2]).unsqueeze(1).to(self.device) * self.reward_scale
        next_states = torch.FloatTensor(experiences[3]).to(self.device)
        dones = torch.BoolTensor(experiences[4]).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute TD errors
        td_errors = self.compute_td_error(states, actions, rewards, next_states, dones)
        
        # Update priorities
        priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        # Compute loss with importance sampling weights
        current_q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            # Double DQN target
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Weighted MSE loss
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none').squeeze()).mean()
        
        # Soft update target network
        if self.global_step % self.target_update_freq == 0:
            self.soft_update_target_network()
        
        # Reset noise in noisy networks
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('epsilon', self.get_epsilon(), on_step=True)
        self.log('avg_q_value', current_q_values.mean(), on_step=True)
        
        return loss
    
    def soft_update_target_network(self):
        """Soft update of target network."""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.q_network.parameters(), lr=self.hparams.learning_rate)
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)


class ScalingEnvironment:
    """Environment simulator for training the scaling agent."""
    
    def __init__(self,
                 forecasting_model,
                 action_space: ScalingActionSpace,
                 cost_per_job: float = 1.0,
                 sla_penalty: float = 10.0,
                 target_queue_size: int = 100,
                 max_processing_time: float = 1000.0):
        
        self.forecasting_model = forecasting_model
        self.action_space = action_space
        self.cost_per_job = cost_per_job
        self.sla_penalty = sla_penalty
        self.target_queue_size = target_queue_size
        self.max_processing_time = max_processing_time
        
        # State variables
        self.current_jobs = 10
        self.current_queue_size = 0
        self.current_processing_time = 100.0
        self.step_count = 0
    
    def reset(self):
        """Reset environment."""
        self.current_jobs = random.randint(5, 20)
        self.current_queue_size = random.randint(0, 200)
        self.current_processing_time = random.uniform(50, 200)
        self.step_count = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Normalize state features
        state = np.array([
            self.current_jobs / self.action_space.max_jobs,
            self.current_queue_size / (self.target_queue_size * 2),
            self.current_processing_time / self.max_processing_time,
            # Add forecasting features here
        ])
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        # Apply scaling action
        new_jobs = self.action_space.apply_action(self.current_jobs, action)
        
        # Simulate system dynamics (simplified)
        # In practice, this would be based on real system behavior
        processing_capacity = new_jobs * 50  # events per time unit
        
        # Simulate new events arriving (would use forecasting model)
        new_events = random.randint(30, 150)
        
        # Update queue
        processed_events = min(self.current_queue_size + new_events, processing_capacity)
        self.current_queue_size = max(0, self.current_queue_size + new_events - processed_events)
        
        # Update processing time (inverse relationship with job count)
        self.current_processing_time = max(20, 200 - (new_jobs * 2))
        
        # Calculate reward
        reward = self.calculate_reward(action, new_jobs)
        
        # Update state
        self.current_jobs = new_jobs
        self.step_count += 1
        
        # Episode termination
        done = self.step_count >= 1000
        
        info = {
            'jobs': self.current_jobs,
            'queue_size': self.current_queue_size,
            'processing_time': self.current_processing_time,
            'action_name': self.action_space.get_action_name(action)
        }
        
        return self.get_state(), reward, done, info
    
    def calculate_reward(self, action: int, new_jobs: int) -> float:
        """Calculate reward for the action."""
        # Cost component (minimize resource usage)
        cost_penalty = -self.cost_per_job * new_jobs
        
        # SLA component (minimize queue size and processing time)
        queue_penalty = -self.sla_penalty * max(0, self.current_queue_size - self.target_queue_size) / self.target_queue_size
        
        processing_penalty = -self.sla_penalty * max(0, self.current_processing_time - 100) / 100
        
        # Efficiency bonus (reward optimal scaling)
        if self.current_queue_size < self.target_queue_size and new_jobs < self.current_jobs:
            efficiency_bonus = 5  # Reward scaling down when queue is small
        elif self.current_queue_size > self.target_queue_size and new_jobs > self.current_jobs:
            efficiency_bonus = 5  # Reward scaling up when queue is large
        else:
            efficiency_bonus = 0
        
        total_reward = cost_penalty + queue_penalty + processing_penalty + efficiency_bonus
        
        return total_reward


def train_scaling_agent(state_dim: int,
                       action_space: ScalingActionSpace,
                       forecasting_model=None,
                       num_episodes: int = 1000,
                       max_steps_per_episode: int = 1000,
                       **agent_kwargs) -> AutoScalingAgent:
    """Train the auto-scaling agent."""
    
    # Initialize agent and environment
    agent = AutoScalingAgent(state_dim, action_space, **agent_kwargs)
    env = ScalingEnvironment(forecasting_model, action_space)
    
    # Training loop
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(torch.FloatTensor(state))
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.add_experience(state, action, reward, next_state, done)
            
            # Update agent (if enough experiences)
            if len(agent.replay_buffer) > agent.batch_size:
                agent.training_step(None, None)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return agent


if __name__ == "__main__":
    # Example usage
    action_space = ScalingActionSpace()
    print(f"Action space: {action_space.action_dim} actions")
    for i in range(action_space.action_dim):
        print(f"Action {i}: {action_space.get_action_name(i)}")
    
    # Train agent (example)
    state_dim = 4  # Adjust based on actual state representation
    agent = train_scaling_agent(
        state_dim=state_dim,
        action_space=action_space,
        num_episodes=1000
    )
    
    print("Auto-scaling agent training completed!")

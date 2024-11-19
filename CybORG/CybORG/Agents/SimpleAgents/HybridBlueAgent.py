import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn
import os

from CybORG.Agents.SimpleAgents.PPOBlueAgent import PPOBlueAgent
from CybORG.Agents.SimpleAgents.DQNAgent import DQNAgent
from CybORG.Agents.SimpleAgents.PPO.ActorCritic import ActorCritic
from CybORG.Agents.SimpleAgents.PPO.Memory import Memory

class HybridBlueAgent:
    def __init__(self, input_dim=52, hidden_dim=64, output_dim=140):
        """Initialize Hybrid Blue Agent combining PPO with forced defensive sequences
        Args:
            input_dim: Size of observation space (default: 52)
            hidden_dim: Size of hidden layers (default: 64)
            output_dim: Size of action space (default: 140)
        """
        # Initialize device (GPU/CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize step counter for tracking episode progress
        self.step_counter = 0
        
        # Initialize loss function for value network
        self.MSE_loss = nn.MSELoss()
        
        # Minimum batch size for updates
        self.min_batch_size = 32
        
        # Track action statistics for adaptive behavior
        self.action_stats = {
            'success_rate': {},           # Success rate per action
            'reward_history': {},         # Historical rewards per action
            'usage_count': {},            # How often each action is used
            'last_success': {}            # When action was last successful
        }
        
        # Track system states and priorities
        self.system_states = {
            'enterprise0': {'compromised': False, 'success_rate': 0.0},
            'enterprise1': {'compromised': False, 'success_rate': 0.0},
            'enterprise2': {'compromised': False, 'success_rate': 0.0},
            'opserver0': {'compromised': False, 'success_rate': 0.0}
        }
        
        # System priority weights for decision making
        self.system_priorities = {
            'enterprise0': 0.5,  # High priority
            'enterprise1': 0.3,
            'enterprise2': 0.2,
            'opserver0': 0.6    # Highest priority
        }
        
        # Exploration parameters
        self.exploration_params = {
            'exploration_bonus': 0.2,     # Bonus for exploring new actions
            'history_window': 50,         # Window for tracking history
            'reward_threshold': -5.0      # Threshold for good performance
        }
        
        # Initialize PPO networks
        self.policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.old_policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # PPO hyperparameters
        self.gamma = 0.99           # Discount factor
        self.lr = 0.002            # Learning rate
        self.betas = [0.9, 0.990]  # Adam optimizer parameters
        self.K_epochs = 6          # Number of epochs per update
        self.eps_clip = 0.2        # PPO clipping parameter
        
        # Initialize experience memory
        self.memory = Memory()
        
        # Initialize action and reward history
        self.actions = []          # Track actions taken
        self.rewards = []          # Track rewards received
        self.rewards_history = []  # Track all historical rewards
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.lr,
            betas=self.betas
        )
    
    def get_action(self, observation, action_space):
        """Get action using PPO with forced defensive sequences
        Args:
            observation: Current environment state
            action_space: List of available actions
        Returns:
            int: Index of chosen action
        """
        state = torch.FloatTensor(observation).to(self.device)
        
        # Force exploration with 30% probability
        if np.random.random() < 0.3:
            return np.random.randint(len(action_space))
        
        # Force monitor periodically
        if len(self.actions) % 3 == 0 and 1 in action_space:
            return action_space.index(1)
        
        # Use policy network for action selection
        with torch.no_grad():
            action_probs = self.policy.actor(state)
            
            # Create action mask
            mask = torch.zeros_like(action_probs)
            for i, action in enumerate(action_space):
                mask[i] = 1.0
            
            # Apply mask and normalize
            action_probs = action_probs * mask
            action_probs = action_probs / (action_probs.sum() + 1e-8)
            
            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample().item()
            
            # Store experience
            self.memory.add(
                state=state.detach(),
                action=torch.tensor(action_idx),
                logprob=dist.log_prob(torch.tensor(action_idx)).detach(),
                reward=0,
                is_terminal=False
            )
            
            return action_idx
    
    def _shape_reward(self, reward):
        """Shape reward to encourage defensive sequences
        Args:
            reward: Original reward from environment
        Returns:
            float: Shaped reward
        """
        shaped_reward = reward
        
        # Stronger penalties for negative rewards
        if reward < 0:
            shaped_reward *= 3.0
        
        # Huge bonus for complete sequences
        if len(self.actions) >= 3:
            if self._is_valid_sequence(self.actions[-3:]):
                shaped_reward += 200.0
        
        # Progressive rewards for improvement
        if len(self.rewards) > 0:
            if shaped_reward > self.rewards[-1]:
                shaped_reward *= 1.5
        
        return shaped_reward
    
    def _is_valid_sequence(self, sequence):
        """Check if action sequence is valid (Monitor -> Analyze -> Restore)
        Args:
            sequence: List of last 3 actions
        Returns:
            bool: True if sequence is valid
        """
        if len(sequence) < 3:
            return False
        
        # Define valid sequences
        valid_sequences = {
            (1, 3, 133),  # Monitor -> Analyze E0 -> Restore E0
            (1, 4, 134),  # Monitor -> Analyze E1 -> Restore E1
            (1, 5, 135),  # Monitor -> Analyze E2 -> Restore E2
            (1, 9, 139),  # Monitor -> Analyze Op -> Restore Op
        }
        
        return tuple(sequence) in valid_sequences
    
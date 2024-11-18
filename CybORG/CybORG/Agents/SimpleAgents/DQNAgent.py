import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class DQNNetwork(nn.Module):
    """Neural network for Deep Q-Learning"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize network architecture
        Args:
            input_dim: Size of state space
            hidden_dim: Size of hidden layers
            output_dim: Size of action space
        """
        super(DQNNetwork, self).__init__()
        # Define network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # Input layer
            nn.ReLU(),                           # Activation
            nn.Linear(hidden_dim, hidden_dim),   # Hidden layer
            nn.ReLU(),                           # Activation
            nn.Linear(hidden_dim, output_dim)    # Output layer
        )
        
    def forward(self, x):
        """Forward pass through network
        Args:
            x: Input state
        Returns:
            Q-values for each action
        """
        # Convert input to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity):
        """Initialize buffer
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Store transition in buffer
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        # Convert states to numpy arrays
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
            
        # Ensure consistent shapes
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(next_state.shape) == 1:
            next_state = next_state.reshape(1, -1)
            
        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample batch of experiences
        Args:
            batch_size: Number of experiences to sample
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        # Random sample from buffer
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        
        # Stack with consistent shapes
        states = np.vstack([s.reshape(1, -1) if len(s.shape) == 1 else s for s in states])
        next_states = np.vstack([s.reshape(1, -1) if len(s.shape) == 1 else s for s in next_states])
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        """Get current buffer size"""
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network agent implementation"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize DQN agent
        Args:
            input_dim: Size of state space
            hidden_dim: Size of hidden layers
            output_dim: Size of action space
        """
        # Set device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize networks
        self.q_network = DQNNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network = DQNNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(1000)
        self.min_replay_size = 32
        self.batch_size = 64
        
        # Set hyperparameters
        self.gamma = 0.95          # Discount factor
        self.lr = 1e-4            # Learning rate
        self.target_update = 100   # Target network update frequency
        self.epsilon_start = 1.0   # Starting exploration rate
        self.epsilon_end = 0.01    # Minimum exploration rate
        self.epsilon_decay = 0.995 # Exploration decay rate
        self.epsilon = self.epsilon_start
        self.update_counter = 0
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
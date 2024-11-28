import torch
import torch.nn as nn
from torch.distributions import Categorical

# Use MPS if available (Apple Silicon), else CPU
device = (torch.device("mps") 
          if torch.backends.mps.is_available() 
          else torch.device("cpu"))
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Initialize Actor-Critic network architecture
        Args:
            input_dim: Dimension of state space (observation)
            output_dim: Dimension of action space
        """
        super(ActorCritic, self).__init__()
        
        # Adjust input dimension for scan state
        self.input_dim = input_dim  # Now includes observation + scan state
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, 64),  # Input now matches observation + scan state
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        ).to(device)  # Move to MPS
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 64),  # Input now matches observation + scan state
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)  # Move to MPS
        
        # Move entire model to MPS
        self.to(device)
        
    def forward(self, state):
        """Forward pass through both actor and critic networks
        Args:
            state: Current environment state
        Returns:
            tuple: (action_probs, state_value)
        """
        # Ensure state is on correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state = state.to(device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        return self.actor(state), self.critic(state)
        
    def evaluate(self, states, actions):
        """Evaluate actions given states for PPO update
        Args:
            states: Batch of states
            actions: Batch of actions taken
        Returns:
            tuple: (action_logprobs, state_values, entropy)
        """
        # Ensure states and actions are on correct device
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)
        states = states.to(device)
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(device)
        
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
            
        # Get action probabilities and state values
        action_probs = self.actor(states)
        state_values = self.critic(states)
        
        # Create categorical distribution from probabilities
        dist = Categorical(action_probs)
        
        # Get log probabilities of taken actions
        action_logprobs = dist.log_prob(actions.squeeze())
        
        # Calculate distribution entropy for exploration
        dist_entropy = dist.entropy()
        
        # Ensure consistent shapes for state values
        state_values = state_values.squeeze()
        if len(state_values.shape) == 0:
            state_values = state_values.unsqueeze(0)
            
        return action_logprobs, state_values, dist_entropy
        
    def act(self, state, memory, deterministic=False):
        """Action selection with explicit device handling"""
        # Ensure state is tensor on correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        else:
            state = state.to(device)
            
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Forward pass (already on MPS)
        action_probs = self.actor(state)
        
        # Create distribution
        dist = Categorical(action_probs)
        
        # Select action
        if deterministic:
            action = torch.argmax(action_probs)
        else:
            action = dist.sample()
            
        # Store action log probability
        if memory is not None:
            memory.logprobs.append(dist.log_prob(action))
            
        return action.item()
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Initialize Actor-Critic network architecture
        Args:
            input_dim: Dimension of state space (observation)
            output_dim: Dimension of action space
        """
        super(ActorCritic, self).__init__()
        
        # Actor network - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),     # First hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(64, 64),            # Second hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(64, output_dim),    # Output layer
            nn.Softmax(dim=-1)            # Convert to probabilities
        )
        
        # Critic network - outputs state value estimate
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),     # First hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(64, 64),            # Second hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(64, 1)              # Output single value
        )
        
    def forward(self, state):
        """Forward pass through both actor and critic networks
        Args:
            state: Current environment state
        Returns:
            tuple: (action_probs, state_value)
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        # Add batch dimension if needed
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
        # Ensure states have correct shape and type
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(device)
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
            
        # Ensure actions have correct shape and type
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions).to(device)
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
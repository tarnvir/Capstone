import torch
import torch.nn as nn
from torch.distributions import Categorical

# Get device from PPOAgent
device = (torch.device("mps") 
          if torch.backends.mps.is_available() 
          else torch.device("cpu"))

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Move to correct device immediately
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Move entire model to device
        self.to(device)

    def act(self, state, memory, deterministic=False, full=False):
        # Ensure state is on correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        else:
            state = state.to(device)
        
        # Get action probabilities
        action_probs = self.actor(state)
        
        # Return full probabilities if requested
        if full:
            return action_probs
            
        # Create distribution
        dist = Categorical(action_probs)
        
        # Select action
        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            action = dist.sample()
            
        # Store memory if provided
        if memory is not None and not deterministic:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            
        return action.item()

    def evaluate(self, state, action):
        # Ensure inputs are on correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor(action).to(device)
        
        # Get values and distribution
        state_value = self.critic(state)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        # Calculate log probs and entropy
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
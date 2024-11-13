import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def act(self, state, memory, deterministic=False):
        action_probs = self.actor(state)
        
        if deterministic:
            action = torch.argmax(action_probs).item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            memory.logprobs.append(dist.log_prob(torch.tensor(action)))
            
        memory.states.append(state)
        memory.actions.append(torch.tensor(action))
        
        return action
        
    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_logprobs, state_values, dist_entropy 
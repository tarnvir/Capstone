from CybORG.Agents.SimpleAgents.PPO.ActorCritic import ActorCritic
from CybORG.Agents.SimpleAgents.PPO.Memory import Memory
import torch
import torch.nn as nn
from CybORG.Agents import BaseAgent
import numpy as np

device = (torch.device("mps") 
          if torch.backends.mps.is_available() 
          else torch.device("cpu"))

class PPOAgent(BaseAgent):
    def __init__(self, input_dims=52, action_space=[i for i in range(158)], 
                 lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=6, 
                 eps_clip=0.2, restore=False, ckpt=None,
                 deterministic=False, training=True, start_actions=[]):
        
        # Initialize parameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.input_dims = input_dims + 10  # Add scan state dimension immediately
        self.restore = restore
        self.ckpt = ckpt
        self.deterministic = deterministic
        self.training = training
        self.start = start_actions
        
        # Initialize memory and networks
        self.memory = Memory()
        self.policy = ActorCritic(self.input_dims, len(action_space) + 9).to(device)
        self.old_policy = ActorCritic(self.input_dims, len(action_space) + 9).to(device)
        
        if restore:
            pretrained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretrained_model)
            
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.MSE_loss = nn.MSELoss()
        
        # Initialize decoy tracking
        self.decoy_ids = list(range(1000, 1009))
        self.action_space = action_space + self.decoy_ids
        self.end_episode()  # Initialize decoys and scan state

    def add_scan(self, observation):
        indices = [0, 4, 8, 12, 28, 32, 36, 40, 44, 48]
        for id, index in enumerate(indices):
            if observation[index] == 1 and observation[index+1] == 0:
                self.scan_state = [1 if x == 2 else x for x in self.scan_state]
                self.scan_state[id] = 2
                break

    def add_decoy(self, action, host):
        """Add decoy with improved tracking"""
        if action not in self.current_decoys[host]:
            self.current_decoys[host].append(action)
            return True
        return False

    def remove_decoy(self, action, host):
        """Remove decoy with validation"""
        if action in self.current_decoys[host]:
            self.current_decoys[host].remove(action)
            return True
        return False

    def select_decoy(self, host, observation):
        """Improved decoy selection strategy"""
        try:
            # Check if host has any decoys defined
            if not self.greedy_decoys[host]:  # If empty list
                # Fall back to analyze action for that host
                if host in range(1000, 1003):  # enterprise0-2
                    return 3 + (host - 1000)  # returns 3,4,5
                elif host == 1008:  # opserver0
                    return 9
                else:  # user hosts
                    return 11 + (host - 1003)  # returns 11-14 for users
            
            # Try to place critical decoy first
            available_decoys = [d for d in self.greedy_decoys[host] 
                              if d not in self.current_decoys[host]]
            if available_decoys:
                action = available_decoys[0]
                self.add_decoy(action, host)
                return action
            
            # If no decoys available, use policy for alternative action
            if not self.training:
                state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
                actions = self.old_policy.act(state, self.memory, full=True)
                max_actions = torch.sort(actions, dim=1, descending=True).indices[0]
                
                for action_ in max_actions:
                    a = self.action_space[action_]
                    # Check if action is valid decoy placement
                    if a in self.current_decoys:
                        if len(self.current_decoys[a]) < len(self.greedy_decoys[a]):
                            return self.select_decoy(a, observation)
                    # Or valid non-restore action
                    elif a not in self.restore_decoy_mapping:
                        return a
                    
            # Default to analyze action if no other options
            if host in range(1000, 1003):  # enterprise0-2
                return 3 + (host - 1000)
            elif host == 1008:  # opserver0
                return 9
            else:  # user hosts
                return 11 + (host - 1003)
            
        except Exception as e:
            print(f"Error in decoy selection: {e}")
            # Return safe default action (analyze)
            return 3  # analyze enterprise0

    def get_action(self, observation, action_space=None):
        # Process observation
        self.add_scan(observation)
        observation = np.concatenate((observation, self.scan_state))
        state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
        
        # Get action from policy
        action = self.old_policy.act(state, self.memory, deterministic=self.deterministic)
        action_ = self.action_space[action]
        
        # Force start actions if available
        if len(self.start_actions) > 0:
            action_ = self.start_actions[0]
            self.start_actions = self.start_actions[1:]
        
        # Handle decoy actions
        if action_ in self.decoy_ids:
            action_ = self.select_decoy(action_, observation)
            
        # Handle restore actions
        if action_ in self.restore_decoy_mapping.keys():
            for decoy in self.restore_decoy_mapping[action_]:
                for host in self.decoy_ids:
                    if decoy in self.current_decoys[host]:
                        self.remove_decoy(decoy, host)
                        
        return action_

    def store(self, reward, done):
        # Simple storage without reward shaping
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    def train(self):
        # Calculate returns
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        # Normalize returns
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Convert memory to tensor
        old_states = torch.squeeze(torch.stack(self.memory.states)).to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions)).to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs)).to(device)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate actions
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Calculate advantage
            advantages = returns - state_values.detach()
            
            # Calculate ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Calculate final loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MSE_loss(state_values, returns) - 0.01 * dist_entropy
            
            # Update policy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

    def end_episode(self):
        self.current_decoys = {id: [] for id in range(1000, 1009)}
        self.scan_state = np.zeros(10)
        self.start_actions = self.start.copy()

    def set_initial_values(self, action_space, observation=None):
        """Initialize agent with focused action space and decoy management"""
        self.memory = Memory()
        
        # Initialize greedy decoys focused on critical systems
        self.greedy_decoys = {
            1000: [55, 107],     # enterprise0 (critical) decoys
            1001: [43],          # enterprise1 (critical) decoys
            1002: [44],          # enterprise2 (critical) decoys
            1003: [],            # user1 (non-critical)
            1004: [51, 116],     # user2 (initial target)
            1005: [],            # user3 (non-critical)
            1006: [],            # user4 (non-critical)
            1007: [54, 106],     # defender (critical)
            1008: [61, 35]       # opserver0 (critical)
        }

        # Mapping of restore actions to decoys
        self.restore_decoy_mapping = dict()
        base_list = [28, 41, 54, 67, 80, 93, 106, 119]
        for i in range(13):
            self.restore_decoy_mapping[132 + i] = [x + i for x in base_list]

        # Add decoy actions (focused on critical systems)
        self.n_actions = len(action_space) + 9
        self.decoy_ids = list(range(1000, 1009))
        self.action_space = action_space + self.decoy_ids

        # Initialize networks with correct dimensions
        self.policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        if self.restore:
            pretrained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretrained_model)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.old_policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

    def clear_memory(self):
        """Clear the memory buffer"""
        self.memory.clear_memory()
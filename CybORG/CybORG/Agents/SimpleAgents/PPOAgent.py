# copied from https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
# only changes involve keeping track of decoys, adding scanning states, and reduction of action space

from CybORG.Agents.SimpleAgents.PPO.ActorCritic import ActorCritic
from CybORG.Agents.SimpleAgents.PPO.Memory import Memory
import torch
import torch.nn as nn
from CybORG.Agents import BaseAgent
import numpy as np
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent(BaseAgent):
    def __init__(self, input_dim, output_dim, lr=0.002, betas=[0.9, 0.999], 
                 gamma=0.99, K_epochs=4, eps_clip=0.2, start_actions=[], 
                 restore=False, ckpt=None, deterministic=False, training=True):
        """Initialize PPO Agent"""
        # Use MPS if available
        self.device = (torch.device("mps") 
                      if torch.backends.mps.is_available() 
                      else torch.device("cpu"))
        
        # Save dimensions
        self.input_dims = input_dim
        self.output_dim = output_dim
        
        # Save hyperparameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # Save training parameters
        self.restore = restore
        self.ckpt = ckpt
        self.deterministic = deterministic
        self.training = training
        
        # Initialize decoy IDs first
        self.decoy_ids = list(range(1000, 1009))
        
        # Initialize action space
        self.action_space = [
            133, 134, 135, 139,  # restore critical systems
            3, 4, 5, 9,         # analyse critical systems
            16, 17, 18, 22      # remove critical systems
        ]
        
        # Initialize decoy tracking
        self.current_decoys = {
            1000: [], # enterprise0
            1001: [], # enterprise1
            1002: [], # enterprise2
            1003: [], # user1
            1004: [], # user2
            1005: [], # user3
            1006: [], # user4
            1007: [], # defender
            1008: []  # opserver0
        }
        
        # Initialize networks and optimizer
        self.policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.old_policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        # Initialize memory
        self.memory = Memory()
        
        # Save start actions and create copy for resetting
        self.start_actions = start_actions
        self.start = start_actions.copy()
        
        # Initialize scan state as numpy arrays
        self.scan_state = np.zeros(10, dtype=np.float32)
        self.scan_state_old = np.zeros(10, dtype=np.float32)
        
        # Initialize MSE loss
        self.MSE_loss = nn.MSELoss()
        
        # Initialize last action
        self.last_action = None

    # add a decoy to the decoy list
    def add_decoy(self, id, host):
        # add to list of decoy actions
        if id not in self.current_decoys[host]:
            self.current_decoys[host].append(id)

    # remove a decoy from the decoy list
    def remove_decoy(self, id, host):
        # remove from decoy actions
        if id in self.current_decoys[host]:
            self.current_decoys[host].remove(id)

    # add scan information
    def add_scan(self, observation):
        indices = [0, 4, 8, 12, 28, 32, 36, 40, 44, 48]
        for id, index in enumerate(indices):
            # if scan seen on defender, enterprise 0-2, opserver0 or user 0-4
            if observation[index] == 1 and observation[index+1] == 0:
                # 1 if scanned before, 2 if is the latest scan
                self.scan_state = [1 if x == 2 else x for x in self.scan_state]
                self.scan_state[id] = 2
                break

    # concatenate the observation with the scan state
    def pad_observation(self, observation, old=False):
        """Pad observation with scan state"""
        # Convert observation to tensor if it isn't already
        if not isinstance(observation, torch.Tensor):
            observation = torch.FloatTensor(observation)
        
        # Move to device
        observation = observation.to(self.device)
        
        # Get appropriate scan state
        scan_state = self.scan_state_old if old else self.scan_state
        
        # Convert scan state to tensor and move to device
        scan_state = torch.FloatTensor(scan_state).to(self.device)
        
        # Concatenate on device
        return torch.cat((observation, scan_state))


    def get_action(self, observation, action_space=None):
        """Get action with proper memory storage"""
        # Process observation
        self.add_scan(observation)
        
        # Convert observation to tensor if it isn't already
        if not isinstance(observation, torch.Tensor):
            observation = torch.FloatTensor(observation)
        
        # Move to correct device and ensure consistent size
        observation = observation.to(self.device)
        
        # Ensure observation has correct size (52)
        if len(observation) > self.input_dims - 10:  # If larger than expected
            observation = observation[:self.input_dims - 10]  # Trim to expected size
        elif len(observation) < self.input_dims - 10:  # If smaller than expected
            # Pad with zeros to reach expected size
            padding = torch.zeros(self.input_dims - 10 - len(observation)).to(self.device)
            observation = torch.cat([observation, padding])
        
        # Convert scan_state to tensor and move to device
        scan_state = torch.FloatTensor(self.scan_state).to(self.device)
        
        # Concatenate on device
        padded_observation = torch.cat((observation, scan_state))
        
        # Double check final size
        if len(padded_observation) != self.input_dims:
            raise ValueError(f"Observation size mismatch. Expected {self.input_dims}, got {len(padded_observation)}")
        
        # Reshape for network
        padded_observation = padded_observation.reshape(1, -1)
        
        # Store state
        self.memory.states.append(padded_observation.squeeze())
        
        # Get action from policy
        action = self.policy.act(padded_observation, self.memory)
        
        # Ensure action is within bounds
        action = action % len(self.action_space)
        
        self.memory.actions.append(action)
        
        # Convert to environment action
        action_ = self.action_space[action]
        
        # Store last action for reward shaping
        self.last_action = action_
        
        return action_

    def store(self, reward, done):
        """Store experience in memory with more informative reward shaping"""
        # Base reward from environment
        shaped_reward = reward
        
        # Only apply defensive bonuses if there's activity
        scan_state_array = np.array(self.scan_state)
        threat_level = np.sum(scan_state_array)
        
        if threat_level > 0:  # If there's suspicious activity
            # Larger bonuses for appropriate responses to threats
            if self.last_action in [133, 134, 135, 139]:  # restore actions
                shaped_reward += 1.0  # Big bonus for restore when needed
            elif self.last_action in [3, 4, 5, 9]:  # analyse actions
                shaped_reward += 0.7  # Good bonus for analysis during threats
            elif self.last_action in [16, 17, 18, 22]:  # remove actions
                shaped_reward += 0.8  # Good bonus for remove during threats
                
            # Extra bonus for quick response
            if scan_state_array.max() == 2:  # If this is a new threat
                shaped_reward += 0.5  # Bonus for responding to new threats
        else:
            # Small bonuses for proactive monitoring
            if self.last_action in [3, 4, 5, 9]:  # analyse actions
                shaped_reward += 0.2  # Small bonus for staying vigilant
                
            # Small penalty for unnecessary aggressive actions
            if self.last_action in [133, 134, 135, 139]:  # restore actions
                shaped_reward -= 0.3  # Penalty for restore when not needed
            elif self.last_action in [16, 17, 18, 22]:  # remove actions
                shaped_reward -= 0.2  # Penalty for remove when not needed
        
        # Penalty for redundant actions
        if self.last_action in sum(self.current_decoys.values(), []):
            shaped_reward -= 0.4
        
        # Convert reward to tensor
        if not isinstance(shaped_reward, torch.Tensor):
            shaped_reward = torch.tensor(shaped_reward, dtype=torch.float32).to(self.device)
        
        # Store experience
        self.memory.rewards.append(shaped_reward)
        self.memory.is_terminals.append(done)

    def clear_memory(self):
        """Clear the memory buffer"""
        self.memory.clear_memory()

    def select_decoy(self, host, observation):
        try:
            # pick the top remaining decoy
            action = [a for a in self.greedy_decoys[host] if a not in self.current_decoys[host]][0]
            self.add_decoy(action, host)
        except:
            # # otherwise just use the remove action on that host
            # action = self.host_to_remove[host]

            # pick the top decoy again (a non-action)
            if self.training:
                action = self.greedy_decoys[host][0]

            # pick the next best available action (deterministic)
            else:
                state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
                actions = self.old_policy.act(state, self.memory, full=True)

                max_actions = torch.sort(actions, dim=1, descending=True)
                max_actions = max_actions.indices
                max_actions = max_actions.tolist()

                # don't need top action since already know it can't be used (hence could put [1:] here, left for clarity)
                for action_ in max_actions[0]:
                    a = self.action_space[action_]
                    # if next best action is decoy, check if its full also
                    if a in self.current_decoys.keys():
                        if len(self.current_decoys[a]) < len(self.greedy_decoys[a]):
                            action = self.select_decoy(a,observation)
                            self.add_decoy(action, a)
                            break
                    else:
                        # don't select a next best action if "restore", likely too aggressive for 30-50 episodes
                        if a not in self.restore_decoy_mapping.keys():
                            action = a
                            break
        return action

    def train(self):
        """Update policy using experiences in memory"""
        try:
            # Ensure all states are tensors
            states = [s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) 
                     for s in self.memory.states]
            
            # Stack and move to device
            old_states = torch.stack(states).to(self.device).detach()
            old_actions = torch.tensor(self.memory.actions, dtype=torch.long).to(self.device).detach()
            old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()
            
            # Compute discounted rewards
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            # Normalize rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # Rest of training code...
            for _ in range(self.K_epochs):
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                
                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = rewards - state_values.detach()
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = 0.5 * self.MSE_loss(state_values, rewards) - 0.01 * dist_entropy
                
                loss = actor_loss + critic_loss
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                
        except Exception as e:
            print(f"Error during training: {e}")
            print(f"States shape: {[s.shape for s in self.memory.states]}")
            self.clear_memory()
            return

    def end_episode(self):
        """Reset agent state at episode end"""
        # Reset decoys
        self.current_decoys = {
            1000: [], # enterprise0
            1001: [], # enterprise1
            1002: [], # enterprise2
            1003: [], # user1
            1004: [], # user2
            1005: [], # user3
            1006: [], # user4
            1007: [], # defender
            1008: []  # opserver0
        }
        
        # Reset scan states as numpy arrays
        self.scan_state = np.zeros(10, dtype=np.float32)
        self.scan_state_old = np.zeros(10, dtype=np.float32)
        
        # Reset start actions from original copy
        self.start_actions = self.start.copy()


    def set_initial_values(self, action_space, observation=None):
        """Set initial values for the agent"""
        self.memory = Memory()
        
        # Update input dimensions to include scan state
        self.input_dims += 10
        
        # Initialize greedy decoys
        self.greedy_decoys = {
            1000: [55, 107, 120, 29],  # enterprise0 decoy actions
            1001: [43],  # enterprise1 decoy actions
            1002: [44],  # enterprise2 decoy actions
            1003: [37, 115, 76, 102],  # user1 decoy actions
            1004: [51, 116, 38, 90],  # user2 decoy actions
            1005: [130, 91],  # user3 decoy actions
            1006: [131],  # user4 decoys
            1007: [54, 106, 28, 119], # defender decoys
            1008: [61, 35, 113, 126]  # opserver0 decoys
        }
        
        # Rest of the method...

        # make a mapping of restores to decoys
        self.restore_decoy_mapping = dict()
        # decoys for defender host
        base_list = [28, 41, 54, 67, 80, 93, 106, 119]
        # add for all hosts
        for i in range(13):
            self.restore_decoy_mapping[132 + i] = [x + i for x in base_list]

        # we statically add 9 decoy actions
        action_space_size = len(action_space)
        self.n_actions = action_space_size + 9
        self.decoy_ids = list(range(1000, 1009))

        # add decoys to action space (all except user0)
        self.action_space = action_space + self.decoy_ids

        # add 10 to input_dims for the scanning state
        self.input_dims += 10


        self.policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        if self.restore:
            pretained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.old_policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()
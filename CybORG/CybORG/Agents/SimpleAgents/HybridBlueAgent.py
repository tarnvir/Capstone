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
        
        # Initialize last observation
        self.last_observation = None
    
    def get_action(self, observation, action_space):
        """Get action using PPO with forced variety"""
        state = torch.FloatTensor(observation).to(self.device)
        action_space = list(action_space) if not isinstance(action_space, list) else action_space
        
        # Force action variety using a cycle-based approach
        step_in_cycle = len(self.actions) % 3
        
        # First action in cycle: Monitor (but not always)
        if step_in_cycle == 0:
            if np.random.random() < 0.7:  # 70% chance to monitor
                if 1 in action_space:
                    return action_space.index(1)
            else:  # 30% chance to do something else
                available_actions = [a for a in action_space if a != 1]
                if available_actions:
                    return action_space.index(np.random.choice(available_actions))
        
        # Second action in cycle: Analyze
        elif step_in_cycle == 1 and len(self.actions) > 0:
            if self.actions[-1] == 1:  # If last action was Monitor
                analyze_actions = [3, 4, 5, 9]
                available_analyze = [a for a in analyze_actions if a in action_space]
                if available_analyze:
                    # Choose analyze target based on threat assessment
                    threats = {
                        3: sum(observation[0:4]),    # enterprise0
                        4: sum(observation[4:8]),    # enterprise1
                        5: sum(observation[8:12]),   # enterprise2
                        9: sum(observation[28:32])   # opserver0
                    }
                    # Filter to available actions and add noise
                    valid_threats = {a: threats[a] + np.random.normal(0, 0.1) 
                                   for a in available_analyze}
                    return action_space.index(max(valid_threats.items(), key=lambda x: x[1])[0])
        
        # Third action in cycle: Restore or Decoy
        elif step_in_cycle == 2 and len(self.actions) > 1:
            if self.actions[-2] == 1 and self.actions[-1] in [3,4,5,9]:
                # Map analyze actions to corresponding restore/decoy actions
                action_map = {
                    3: [133, 69],  # Restore or Decoy for enterprise0
                    4: [134, 70],  # Restore or Decoy for enterprise1
                    5: [135, 71],  # Restore or Decoy for enterprise2
                    9: [139, 72]   # Restore or Decoy for opserver0
                }
                possible_actions = action_map[self.actions[-1]]
                available_actions = [a for a in possible_actions if a in action_space]
                if available_actions:
                    return action_space.index(np.random.choice(available_actions))
        
        # Use policy network with exploration
        with torch.no_grad():
            action_probs = self.policy.actor(state)
            
            # Create action mask
            mask = torch.ones_like(action_probs)
            for i, action in enumerate(action_space):
                # Reduce probability of recent actions
                if len(self.actions) > 0 and action in self.actions[-3:]:
                    mask[i] *= 0.3
                
                # Boost important actions
                if action == 1:  # Monitor
                    mask[i] *= 1.5
                elif action in [3,4,5,9]:  # Analyze
                    mask[i] *= 1.3
                elif action in [133,134,135,139]:  # Restore
                    mask[i] *= 1.3
                elif action in range(69, 78):  # Decoys
                    mask[i] *= 1.2
            
            # Apply mask and add exploration noise
            action_probs = action_probs * mask
            noise = torch.randn_like(action_probs) * 0.2
            action_probs = torch.softmax(action_probs + noise, dim=-1)
            
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
    
    def _assess_threat(self, observation):
        """Enhanced threat assessment"""
        # System-specific threat assessment
        threats = {
            'enterprise0': sum(observation[0:4]),
            'enterprise1': sum(observation[4:8]),
            'enterprise2': sum(observation[8:12]),
            'opserver0': sum(observation[28:32])
        }
        
        # Weight threats by system priority
        weighted_threats = {
            system: threat * self.system_priorities[system]
            for system, threat in threats.items()
        }
        
        # Calculate overall threat metrics
        max_threat = max(weighted_threats.values())
        avg_threat = sum(weighted_threats.values()) / len(weighted_threats)
        
        # Combine metrics with emphasis on maximum threat
        return 0.7 * max_threat + 0.3 * avg_threat
    
    def _get_priority_target(self, observation):
        """Determine highest priority target based on threats
        Args:
            observation: Current state observation
        Returns:
            str: Name of highest priority target
        """
        threats = {
            'enterprise0': sum(observation[0:4]),
            'enterprise1': sum(observation[4:8]),
            'enterprise2': sum(observation[8:12]),
            'opserver0': sum(observation[28:32])
        }
        
        # Weight threats by system priority
        weighted_threats = {
            system: threat * self.system_priorities[system]
            for system, threat in threats.items()
        }
        
        return max(weighted_threats.items(), key=lambda x: x[1])[0]
    
    def _shape_reward(self, reward):
        """Enhanced reward shaping with better incentives and penalties"""
        shaped_reward = reward
        
        # Base reward scaling
        if reward < 0:
            shaped_reward *= 2.0  # Reduced penalty multiplier to avoid too negative rewards
        
        # Get current state assessment
        threat_level = self._assess_threat(self.last_observation)
        
        # Get last action (if any)
        last_action = None
        if len(self.actions) > 0:
            last_action = self.actions[-1]
            
            # Reward for successful defensive actions
            if last_action == 1 and threat_level > 0.3:
                shaped_reward += 50.0  # Bonus for detecting threats
                
            # Reward for Analyze actions that follow Monitor
            if last_action in [3,4,5,9] and len(self.actions) > 1 and self.actions[-2] == 1:
                shaped_reward += 75.0  # Higher bonus for proper sequence
                
            # Reward for Restore/Remove actions that follow Analyze
            if last_action in [133,134,135,139] and len(self.actions) > 1 and self.actions[-2] in [3,4,5,9]:
                shaped_reward += 100.0  # Highest bonus for completing sequence
                
            # Reward for decoy actions
            if last_action in range(69, 78):  # Decoy actions
                if threat_level > 0.5:  # Higher threat level
                    shaped_reward += 150.0  # Large bonus for deploying decoys when needed
                else:
                    shaped_reward += 25.0  # Smaller bonus for proactive decoys
            
            # Penalty for resource wastage
            if len(self.actions) >= 3:
                recent_actions = self.actions[-3:]
                if len(set(recent_actions)) == 1:  # Same action repeated 3 times
                    shaped_reward -= 50.0  # Penalty for wasteful repetition
            
            # Track success/failure for adaptive learning
            if last_action not in self.action_stats['reward_history']:
                self.action_stats['reward_history'][last_action] = []
            self.action_stats['reward_history'][last_action].append(shaped_reward)
            
            # Update success rate
            recent_rewards = self.action_stats['reward_history'][last_action][-100:]
            success_rate = sum(r > 0 for r in recent_rewards) / len(recent_rewards)
            self.action_stats['success_rate'][last_action] = success_rate
        
        return shaped_reward
    
    def _is_valid_sequence(self, sequence):
        """Check if action sequence is valid"""
        if len(sequence) < 3:
            return False
        
        # Define valid sequences including decoys
        valid_sequences = {
            (1, 3, 133),  # Monitor -> Analyze E0 -> Restore E0
            (1, 4, 134),  # Monitor -> Analyze E1 -> Restore E1
            (1, 5, 135),  # Monitor -> Analyze E2 -> Restore E2
            (1, 9, 139),  # Monitor -> Analyze Op -> Restore Op
            # Add decoy sequences
            (1, 3, 69),   # Monitor -> Analyze E0 -> Deploy Decoy E0
            (1, 4, 70),   # Monitor -> Analyze E1 -> Deploy Decoy E1
            (1, 5, 71),   # Monitor -> Analyze E2 -> Deploy Decoy E2
            (1, 9, 72),   # Monitor -> Analyze Op -> Deploy Decoy Op
        }
        
        return tuple(sequence) in valid_sequences
    
    def save(self, path):
        """Save model and training state
        Args:
            path: Path to save checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save only essential model components
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'old_policy_state_dict': self.old_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_counter': self.step_counter,
            'hyperparameters': {
                'gamma': self.gamma,
                'lr': self.lr,
                'eps_clip': self.eps_clip,
                'K_epochs': self.K_epochs
            }
        }, path)

    def load(self, path):
        """Load model and training state
        Args:
            path: Path to checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load network states
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.old_policy.load_state_dict(checkpoint['old_policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load step counter if exists
        if 'step_counter' in checkpoint:
            self.step_counter = checkpoint['step_counter']
        
        # Load hyperparameters if exist
        if 'hyperparameters' in checkpoint:
            hyperparameters = checkpoint['hyperparameters']
            self.gamma = hyperparameters.get('gamma', self.gamma)
            self.lr = hyperparameters.get('lr', self.lr)
            self.eps_clip = hyperparameters.get('eps_clip', self.eps_clip)
            self.K_epochs = hyperparameters.get('K_epochs', self.K_epochs)
    
    def store_reward(self, reward, observation=None):
        """Store and shape reward for learning
        Args:
            reward: Raw reward from environment
            observation: Current observation (optional)
        """
        # Update last observation if provided
        if observation is not None:
            self.last_observation = observation
        
        # Shape the reward
        shaped_reward = self._shape_reward(reward)
        
        # Store reward
        self.rewards.append(shaped_reward)
        
        # Update action statistics
        if len(self.actions) > 0:
            last_action = self.actions[-1]
            
            # Update reward history for this action
            if last_action not in self.action_stats['reward_history']:
                self.action_stats['reward_history'][last_action] = []
            self.action_stats['reward_history'][last_action].append(shaped_reward)
            
            # Update usage count
            if last_action not in self.action_stats['usage_count']:
                self.action_stats['usage_count'][last_action] = 0
            self.action_stats['usage_count'][last_action] += 1
            
            # Update success rate
            rewards = self.action_stats['reward_history'][last_action][-100:]  # Last 100 rewards
            success_rate = sum(r > 0 for r in rewards) / len(rewards)
            self.action_stats['success_rate'][last_action] = success_rate
            
            # Update last success
            if shaped_reward > 0:
                self.action_stats['last_success'][last_action] = self.step_counter
    
    def update(self, next_state, done):
        """Update policy using PPO
        Args:
            next_state: Next environment state
            done: Whether episode is done
        """
        # Only update if we have enough experiences
        if len(self.memory.states) < self.min_batch_size:
            return
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        discounted_reward = 0
        
        # Get value estimate for next state
        with torch.no_grad():
            next_value = self.policy.critic(
                torch.FloatTensor(next_state).to(self.device)
            ).detach()
        
        # Calculate GAE and returns
        for reward, is_terminal in zip(reversed(self.rewards), 
                                     reversed([done] + [False] * (len(self.rewards)-1))):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        # Convert to tensors
        old_states = torch.stack(self.memory.states)
        old_actions = torch.stack(self.memory.actions)
        old_logprobs = torch.stack(self.memory.logprobs)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Update policy for K epochs
        for _ in range(self.K_epochs):
            # Get current policy outputs
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Calculate advantages
            advantages = returns - state_values.detach()
            
            # Calculate ratios and surrogate losses
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Calculate final loss
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.MSE_loss(state_values, returns)
            entropy_loss = -0.01 * dist_entropy.mean()
            
            total_loss = policy_loss + value_loss + entropy_loss
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Clear memory buffer
        self.memory.clear_memory()
        
        # Reset episode-specific variables if done
        if done:
            self.actions = []
            self.rewards = []
    
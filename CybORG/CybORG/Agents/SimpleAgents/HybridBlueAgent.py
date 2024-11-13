import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn

from CybORG.Agents.SimpleAgents.PPOBlueAgent import PPOBlueAgent
from CybORG.Agents.SimpleAgents.DQNAgent import DQNAgent
from CybORG.Agents.SimpleAgents.PPO.ActorCritic import ActorCritic
from CybORG.Agents.SimpleAgents.PPO.Memory import Memory

class HybridBlueAgent:
    def __init__(self, input_dim=52, hidden_dim=64, output_dim=140):
        # Initialize device first
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize MSE Loss
        self.MSE_loss = nn.MSELoss()
        
        # Initialize optimizer
        self.optimizer = None  # Will be initialized after policy creation
        
        # Initialize min_batch_size
        self.min_batch_size = 32
        
        # System priorities must be defined first
        self.system_priorities = {
            'enterprise0': 0.5,
            'enterprise1': 0.3,
            'enterprise2': 0.2,
            'opserver0': 0.0
        }
        
        # Base priorities copy can now be created
        self.base_priorities = self.system_priorities.copy()
        
        # Initialize PPO and DQN agents
        self.ppo_agent = PPOBlueAgent(input_dim, hidden_dim, output_dim)
        self.dqn_agent = DQNAgent(input_dim, hidden_dim, output_dim)
        
        # Add dynamic action sequences
        self.action_sequences = {
            'monitor_analyze': {
                'enterprise0': [1, 3],
                'enterprise1': [1, 4],
                'enterprise2': [1, 5],
                'opserver0': [1, 9]
            },
            'analyze_restore': {
                'enterprise0': [3, 133],
                'enterprise1': [4, 134],
                'enterprise2': [5, 135],
                'opserver0': [9, 139]
            }
        }
        
        # Add success tracking per sequence
        self.sequence_success = defaultdict(lambda: {'successes': 0, 'attempts': 0})
        
        # Add system state tracking
        self.system_states = {
            'enterprise0': {'compromised': False, 'last_action': None, 'success_rate': 0},
            'enterprise1': {'compromised': False, 'last_action': None, 'success_rate': 0},
            'enterprise2': {'compromised': False, 'last_action': None, 'success_rate': 0},
            'opserver0': {'compromised': False, 'last_action': None, 'success_rate': 0}
        }
        
        # Track current sequence
        self.current_sequence = None
        self.sequence_position = 0
        
        # Store history
        self.last_observation = None
        self.current_observation = None
        self.actions = []
        self.rewards = []
        
        # Mixing weights
        self.ppo_weight = 0.7  # 70% PPO, 30% DQN
        
        # Initialize valid actions first
        self.valid_actions = {
            0,                          # Sleep
            1,                          # Monitor
            3, 4, 5, 9,                # Analyze enterprise and opserver
            133, 134, 135, 139,        # Restore enterprise and opserver
            16, 17, 18, 22             # Remove enterprise and opserver
        }
        
        # Initialize success memory for all possible actions
        self.success_memory = {}
        for action in range(output_dim):
            self.success_memory[action] = 0.0
        
        # Make sure all valid actions are in success memory
        for action in self.valid_actions:
            if action not in self.success_memory:
                self.success_memory[action] = 0.0
        
        # Enhanced exploration parameters
        self.exploration_params = {
            'initial_epsilon': 0.9,
            'final_epsilon': 0.05,
            'epsilon_decay': 0.995,
            'current_epsilon': 0.9,
            'exploration_bonus': 0.1  # Bonus for trying new actions
        }
        
        # Reward shaping parameters
        self.reward_shaping = {
            'monitor_bonus': 2.0,      # Increased from 1.0
            'analyze_bonus': 3.0,      # Increased from 2.0
            'restore_bonus': 5.0,      # Increased from 3.0
            'sequence_bonus': 10.0,    # Increased from 5.0
            'enterprise_bonus': 4.0,   # Increased from 2.0
            'speed_bonus': 1.0         # Increased from 0.5
        }
        
        # Action diversity tracking
        self.action_history = []
        self.action_frequencies = defaultdict(int)
        self.min_action_frequency = 0.1  # Minimum desired frequency per action
        
        # Add adaptive exploration
        self.adaptive_params = {
            'decay': 1.0,
            'min_decay': 0.8,
            'max_decay': 1.0,
            'reward_threshold': -5.0,
            'history_window': 50
        }
        
        # Track rewards for adaptation
        self.rewards_history = []
        
        # Add forced start sequence
        self.start_sequence = [1, 3, 133]  # Monitor -> Analyze -> Restore
        self.start_position = 0
        
        # Add pattern recognition
        self.action_patterns = {
            'defensive': [1, 3, 133],
            'aggressive': [3, 133, 1],
            'cautious': [1, 0, 1]
        }
        
        # Replace our simple networks with their ActorCritic
        self.policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.old_policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Replace our memory with their Memory class
        self.memory = Memory()
        
        # Remove our separate actor/critic networks since we're using their ActorCritic
        # self.actor = ...  # Remove
        # self.critic = ... # Remove
        
        # Initialize with better parameters
        self.gamma = 0.99
        self.lr = 0.002
        self.betas = [0.9, 0.990]
        self.K_epochs = 6
        self.eps_clip = 0.2

        # Add their start actions sequence
        self.start_actions = [1004, 1004, 1000]  # user 2 decoy * 2, ent0 decoy
        
        # Reduce sleep probability
        self.sleep_threshold = 0.1  # Reduced from 0.3
        
        # Force more complete sequences
        self.min_actions_before_sleep = 3  # Must try at least 3 actions before sleeping
        
        # After policy initialization, create optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
    
    def get_action(self, observation, action_space):
        """Get action using their approach"""
        # Convert observation to tensor
        state = torch.FloatTensor(observation).to(self.device)
        
        # Convert valid_actions set to list for indexing
        valid_actions_list = list(self.valid_actions)
        
        # Use their ActorCritic to get action
        action_idx = self.old_policy.act(state, self.memory)
        
        # Convert action index to valid action
        action = valid_actions_list[action_idx % len(valid_actions_list)]  # Use modulo to ensure valid index
        
        # Store action
        self.actions.append(action)
        return action
    
    def _get_best_action(self, observation):
        """Get the best action based on past success"""
        # Calculate success rates for each action
        action_scores = {}
        for action in self.valid_actions:
            # Base score on success memory
            base_score = self.success_memory[action]
            
            # Bonus for actions that were part of successful sequences
            if len(self.actions) >= 2:
                last_two = self.actions[-2:]
                sequence = tuple(last_two + [action])
                if sequence in self.action_patterns.values():
                    base_score += 2.0
                    
            # Penalty for repeated actions
            if action in self.actions[-3:]:
                base_score -= 1.0
                
            action_scores[action] = base_score
        
        # Return action with highest score
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def _log_action(self, action):
        """Log the action taken for monitoring and debugging"""
        print(f"Action taken: {action}")
    
    def _calculate_action_diversity(self):
        """Calculate action frequency scores"""
        total_actions = len(self.action_history)
        if total_actions == 0:
            return defaultdict(float)
                
        frequencies = {}
        for action in set(self.action_history):
            count = self.action_frequencies[action]
            frequencies[action] = count / total_actions
        return frequencies
    
    def _weight_risks_by_history(self, risks):
        """Weight risks based on action history"""
        weighted_risks = risks.copy()
        
        for system, risk in weighted_risks.items():
            # Add diversity bonus
            action_score = self._calculate_action_diversity().get(system, 0)
            exploration_bonus = (1 - action_score) * self.exploration_params['exploration_bonus']
            
            # Add success history bonus
            success_rate = self.system_states[system]['success_rate']
            success_bonus = success_rate * self.reward_shaping['sequence_bonus']
            
            weighted_risks[system] = risk * (1 + exploration_bonus + success_bonus)
        
        return weighted_risks
    
    def _update_system_states(self, observation):
        """Update system states based on observation"""
        # Define action mappings
        analyze_restore_actions = {
            'enterprise0': [3, 133],
            'enterprise1': [4, 134],
            'enterprise2': [5, 135],
            'opserver0': [9, 139]
        }
        
        for system in self.system_states:
            # Check if system is compromised
            risk = self._get_system_risk(observation, system)
            self.system_states[system]['compromised'] = risk > 0.5
            
            # Update success rate
            if len(self.actions) > 0:
                last_action = self.actions[-1]
                # Check if last action was analyze or restore for this system
                if last_action in analyze_restore_actions[system]:
                    success = len(self.rewards) > 0 and self.rewards[-1] > -5
                    stats = self.system_states[system]
                    stats['success_rate'] = (stats['success_rate'] * 0.95 + 
                                          (0.05 if success else 0))
    
    def _adjust_risks_by_success(self):
        """Adjust risk levels based on success rates"""
        risks = self._get_system_risks(self.current_observation)
        for system, risk in risks.items():
            success_rate = self.system_states[system]['success_rate']
            # Increase priority for systems we're successful with
            risks[system] = risk * (1 + success_rate)
        return risks
    
    def _select_target_system(self, risks):
        """Select target system using softmax distribution"""
        # Convert risks to probabilities
        risk_values = np.array(list(risks.values()))
        risk_probs = np.exp(risk_values) / np.sum(np.exp(risk_values))
        
        # Sample system based on probabilities
        systems = list(risks.keys())
        return np.random.choice(systems, p=risk_probs)
    
    def _adjust_ppo_weight(self):
        """Dynamically adjust PPO weight based on performance"""
        if len(self.rewards) < 10:
            return self.ppo_weight
            
        # Calculate recent performance
        recent_rewards = self.rewards[-10:]
        avg_reward = np.mean(recent_rewards)
        
        # Adjust weight based on performance
        if avg_reward > -5:  # Good performance
            return min(0.9, self.ppo_weight + 0.05)
        else:  # Poor performance
            return max(0.5, self.ppo_weight - 0.05)
    
    def _get_system_risks(self, observation):
        """Calculate risk levels for all systems"""
        return {
            system: self._get_system_risk(observation, system)
            for system in self.system_priorities.keys()
        }
    
    def _get_system_risk(self, observation, system):
        """Calculate risk level for a specific system"""
        system_indices = {
            'enterprise0': (0, 4),
            'enterprise1': (4, 8),
            'enterprise2': (8, 12),
            'opserver0': (28, 32)
        }
        
        if system in system_indices:
            start, end = system_indices[system]
            # Get system observation slice
            system_obs = observation[start:end]
            
            # Calculate risk score based on observation values and system priority
            base_risk = sum(system_obs)  # Sum of observation values
            priority_multiplier = self.system_priorities[system]
            
            # Add success rate factor
            success_factor = self.system_states[system]['success_rate']
            
            # Combine factors
            risk_score = base_risk * priority_multiplier * (1 + success_factor)
            
            return risk_score
        return 0.0
    
    def store_reward(self, reward):
        """Store reward using their Memory"""
        shaped_reward = self._shape_reward(reward)  # Keep our reward shaping
        
        # Store in their memory format
        self.memory.rewards.append(shaped_reward)
        self.memory.is_terminals.append(False)  # Set to True in update() when episode ends
        
        # Keep our reward tracking
        self.rewards.append(shaped_reward)
        self.rewards_history.append(reward)
    
    def update(self, next_state, done):
        """Update using their PPO implementation"""
        # Set terminal flag for episode end
        if done:
            self.memory.is_terminals[-1] = True
        
        # Only update if we have enough experience
        if len(self.memory.rewards) >= self.min_batch_size:
            # Calculate returns and advantages
            returns = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.memory.rewards), 
                                         reversed(self.memory.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
            
            # Normalize returns
            returns = torch.tensor(returns).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
            # Get old states, actions, logprobs
            old_states = torch.stack(self.memory.states).to(self.device).detach()
            old_actions = torch.stack(self.memory.actions).to(self.device).detach()
            old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()
            
            # Update policy for K epochs
            for _ in range(self.K_epochs):
                # Evaluate actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    old_states, old_actions)
                
                # Calculate advantages
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
            
            # Clear memory
            self.memory.clear_memory()
        
        # Reset if episode done
        if done:
            self.current_sequence = None
            self.sequence_position = 0
            self.actions = []
            self.rewards = []
            self.last_observation = None
            self.current_observation = None
    
    def save(self, path):
        # Save both agents
        self.ppo_agent.save(path + '_ppo')
        self.dqn_agent.save(path + '_dqn')
    
    def load(self, path):
        # Load both agents
        self.ppo_agent.load(path + '_ppo')
        self.dqn_agent.load(path + '_dqn')
    
    def _update_adaptive_decay(self):
        """Update decay based on recent performance"""
        if len(self.rewards_history) > self.adaptive_params['history_window']:
            recent_performance = sum(self.rewards_history[-50:]) / 50
            if recent_performance < self.adaptive_params['reward_threshold']:
                # Decrease decay if performance is poor
                self.adaptive_params['decay'] = max(
                    self.adaptive_params['min_decay'],
                    self.adaptive_params['decay'] - 0.01
                )
            else:
                # Increase decay if performing well
                self.adaptive_params['decay'] = min(
                    self.adaptive_params['max_decay'],
                    self.adaptive_params['decay'] + 0.01
                )
                
    def _is_system_at_risk(self, observation, system):
        """Check if a system is at risk based on observation"""
        if observation is None:
            return False
            
        system_indices = {
            'enterprise0': (0, 4),
            'enterprise1': (4, 8),
            'enterprise2': (8, 12),
            'opserver0': (28, 32)
        }
        
        if system in system_indices:
            start, end = system_indices[system]
            system_obs = observation[start:end]
            return any(val > 0.5 for val in system_obs)
        return False
    
    def _shape_reward(self, reward):
        """Shape the reward to encourage desired behavior"""
        shaped_reward = reward
        
        # Detect and penalize loops
        if len(self.actions) >= 4:
            if len(set(self.actions[-4:])) == 1:  # Same action repeated 4 times
                shaped_reward -= 5.0
                print("Penalty applied for looping actions.")
        
        # Reward successful sequences
        if len(self.actions) >= 3:
            last_three = self.actions[-3:]
            if (last_three[0] == 1 and  # Monitor
                last_three[1] in [3,4,5,9] and  # Analyze
                last_three[2] in [133,134,135,139]):  # Restore
                
                shaped_reward += self.reward_shaping['sequence_bonus']
                
                # Extra bonus for enterprise systems
                if last_three[1] in [3,4,5]:  # Enterprise system
                    shaped_reward += self.reward_shaping['enterprise_bonus']
                    
                # Speed bonus
                steps_taken = len(self.actions)
                if steps_taken < 10:
                    shaped_reward += self.reward_shaping['speed_bonus'] * (10 - steps_taken)
                    
                # Success bonus
                if reward > 0:
                    shaped_reward *= 1.5  # 50% bonus for successful sequences
        
        # Penalize sleeping too early
        if len(self.actions) > 0 and self.actions[-1] == 0:
            if len(self.actions) < self.min_actions_before_sleep:
                shaped_reward -= 3.0
        
        # Reward for monitoring when needed
        if len(self.actions) > 0 and self.actions[-1] == 1:
            if any(self._is_system_at_risk(self.current_observation, sys) 
                   for sys in ['enterprise0', 'enterprise1', 'enterprise2', 'opserver0']):
                shaped_reward += self.reward_shaping['monitor_bonus']
        
        # Reward for analyzing after monitoring
        if len(self.actions) >= 2:
            if self.actions[-2] == 1 and self.actions[-1] in [3,4,5,9]:
                shaped_reward += self.reward_shaping['analyze_bonus']
        
        # Reward for restoring after analyzing
        if len(self.actions) >= 2:
            last_two = self.actions[-2:]
            analyze_to_restore = {3: 133, 4: 134, 5: 135, 9: 139}
            if last_two[0] in analyze_to_restore and last_two[1] == analyze_to_restore[last_two[0]]:
                shaped_reward += self.reward_shaping['restore_bonus']
        
        return shaped_reward
    
import torch
import numpy as np
from collections import defaultdict

from CybORG.Agents.SimpleAgents.PPOBlueAgent import PPOBlueAgent
from CybORG.Agents.SimpleAgents.DQNAgent import DQNAgent

class HybridBlueAgent:
    def __init__(self, input_dim=52, hidden_dim=512, output_dim=145):
        # System priorities must be defined first
        self.system_priorities = {
            'enterprise0': 0.4,
            'enterprise1': 0.3,
            'enterprise2': 0.2,
            'opserver0': 0.1
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
        
        # Success tracking
        self.success_memory = defaultdict(lambda: {'successes': 0, 'attempts': 0})
        
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
            'monitor_bonus': 1.0,      # Bonus for monitoring when needed
            'analyze_bonus': 2.0,      # Bonus for analyzing after monitor
            'restore_bonus': 3.0,      # Bonus for restoring after analyze
            'sequence_bonus': 5.0,     # Bonus for completing M->A->R sequence
            'enterprise_bonus': 2.0,   # Extra bonus for enterprise systems
            'speed_bonus': 0.5         # Bonus for quick responses
        }
        
        # Action diversity tracking
        self.action_history = []
        self.action_frequencies = defaultdict(int)
        self.min_action_frequency = 0.1  # Minimum desired frequency per action
    
    def get_action(self, observation, action_space):
        """Get action with strict sequence enforcement"""
        self.last_observation = self.current_observation
        self.current_observation = observation
        
        # Define valid sequences
        SEQUENCES = {
            'enterprise0': {'monitor': 1, 'analyze': 3, 'restore': 133},
            'enterprise1': {'monitor': 1, 'analyze': 4, 'restore': 134},
            'enterprise2': {'monitor': 1, 'analyze': 5, 'restore': 135},
            'opserver0':  {'monitor': 1, 'analyze': 9, 'restore': 139}
        }
        
        # Force sequence completion
        if len(self.actions) > 0:
            last_action = self.actions[-1]
            
            # After monitor (1), must analyze
            if last_action == 1:
                # Get system risks
                risks = self._get_system_risks(observation)
                # Weight by priorities and success rates
                for system in risks:
                    risks[system] *= (self.system_priorities[system] * 
                                   (1 + self.system_states[system]['success_rate']))
                
                # Select highest risk system
                target_system = max(risks.items(), key=lambda x: x[1])[0]
                analyze_action = SEQUENCES[target_system]['analyze']
                
                if analyze_action in action_space:
                    self.actions.append(analyze_action)
                    return analyze_action
            
            # After analyze, must restore same system
            for system, actions in SEQUENCES.items():
                if last_action == actions['analyze']:
                    restore_action = actions['restore']
                    if restore_action in action_space:
                        self.actions.append(restore_action)
                        return restore_action
            
            # After restore, evaluate next action
            if last_action in [s['restore'] for s in SEQUENCES.values()]:
                # Check if any system needs attention
                risks = self._get_system_risks(observation)
                if max(risks.values()) > 0.3:
                    self.actions.append(1)  # Monitor if risks detected
                    return 1
                else:
                    self.actions.append(0)  # Sleep if safe
                    return 0
        
        # Default to monitor
        self.actions.append(1)
        return 1
    
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
        for system in self.system_states:
            # Check if system is compromised
            risk = self._get_system_risk(observation, system)
            self.system_states[system]['compromised'] = risk > 0.5
            
            # Update success rate
            if len(self.actions) > 0:
                last_action = self.actions[-1]
                if last_action in self.action_sequences['analyze_restore'][system]:
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
        """Store reward with enhanced shaping"""
        shaped_reward = reward
        
        # Sequence completion bonus
        if len(self.actions) >= 3:
            last_three = self.actions[-3:]
            
            if (last_three[0] == 1 and  # Monitor
                last_three[1] in [3,4,5,9] and  # Analyze
                last_three[2] in [133,134,135,139]):  # Restore
                
                shaped_reward += self.reward_shaping['sequence_bonus']
                
                # Enterprise bonus
                if last_three[1] in [3,4,5]:
                    shaped_reward += self.reward_shaping['enterprise_bonus']
                
                # Speed bonus
                steps_taken = len(self.actions)
                if steps_taken < 10:
                    shaped_reward += self.reward_shaping['speed_bonus'] * (10 - steps_taken)
        
        # Individual action bonuses
        if len(self.actions) > 0:
            last_action = self.actions[-1]
            
            if last_action == 1:  # Monitor
                shaped_reward += self.reward_shaping['monitor_bonus']
            elif last_action in [3,4,5,9]:  # Analyze
                shaped_reward += self.reward_shaping['analyze_bonus']
            elif last_action in [133,134,135,139]:  # Restore
                shaped_reward += self.reward_shaping['restore_bonus']
        
        # Store rewards
        self.rewards.append(shaped_reward)
        self.ppo_agent.store_reward(shaped_reward)
        
        if len(self.actions) > 0:
            self.dqn_agent.store_transition(
                self.last_observation,
                self.actions[-1],
                shaped_reward,
                self.current_observation,
                False
            )
        
        # Update exploration
        self.exploration_params['current_epsilon'] = max(
            self.exploration_params['final_epsilon'],
            self.exploration_params['current_epsilon'] * self.exploration_params['epsilon_decay']
        )
    
    def update(self, next_state, done):
        """Update both agents"""
        self.ppo_agent.update(next_state, done)
        if len(self.dqn_agent.replay_buffer) > self.dqn_agent.min_replay_size:
            self.dqn_agent.update()
            
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
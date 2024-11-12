import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import Counter, defaultdict

class PPONetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PPONetwork, self).__init__()
        # Increase network capacity
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        shared_features = self.shared(x)
        action_probs = F.softmax(self.actor_head(shared_features), dim=-1)
        value = self.critic_head(shared_features)
        return action_probs, value

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQNNetwork, self).__init__()
        # Increase network capacity
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        shared_features = self.shared(x)
        action_probs = F.softmax(self.actor_head(shared_features), dim=-1)
        value = self.critic_head(shared_features)
        return action_probs, value

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class PPOBlueAgent:
    def __init__(self, input_dim=52, hidden_dim=512, output_dim=None):
        # Force CUDA if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("WARNING: GPU not available, using CPU")
        
        # Use default output_dim if none provided
        if output_dim is None:
            output_dim = 145
        
        # Initialize network with better architecture
        self.network = PPONetwork(input_dim, hidden_dim, output_dim).to(self.device)
        
        # Define action types based on john-cardiff's implementation
        self.action_types = {
            'sleep': [0],              # Sleep action (do nothing)
            'monitor': [1],            # Monitor system
            'analyse': [3, 4, 5, 9],   # Analyze different systems
            'restore': [133, 134, 135, 139]  # Restore systems
        }
        
        # Action sequences with sleep
        self.action_sequences = {
            'enterprise0': [1, 3, 133, 0],  # Monitor -> Analyze -> Restore -> Sleep
            'enterprise1': [1, 4, 134, 0],
            'enterprise2': [1, 5, 135, 0],
            'opserver0': [1, 9, 139, 0]
        }
        
        # Sleep strategy parameters
        self.sleep_threshold = 0.3     # Probability to sleep when no threats
        self.min_sleep_steps = 2       # Minimum steps to sleep
        self.max_sleep_steps = 5       # Maximum steps to sleep
        self.sleeping = False
        self.sleep_steps_left = 0
        
        # Track current sequence and position
        self.current_sequence = None
        self.sequence_position = 0
        
        # Track system states
        self.system_states = {
            'enterprise0': {'compromised': False, 'last_action': None},
            'enterprise1': {'compromised': False, 'last_action': None},
            'enterprise2': {'compromised': False, 'last_action': None},
            'opserver0': {'compromised': False, 'last_action': None}
        }
        
        # Action priorities
        self.action_priorities = {
            'restore': [133, 134, 135, 139],  # Highest priority
            'analyze': [3, 4, 5, 9],          # Second priority
            'remove': [16, 17, 18, 22],       # Third priority
            'monitor': [0]                     # Baseline action
        }
        
        # Success thresholds
        self.high_success_threshold = 0.85
        self.low_success_threshold = 0.4
        
        # Exploration parameters
        self.initial_explore_rate = 0.2
        self.final_explore_rate = 0.02
        self.explore_decay = 0.997
        
        # Action sequence learning - store as numpy arrays
        self.successful_sequences = np.array([])  # Initialize as empty array
        self.sequence_length = 3
        self.min_sequence_reward = -5
        
        # Modified hyperparameters based on successful implementation
        self.gamma = 0.97  # Shorter horizon
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.2  # Start with higher exploration
        self.learning_rate = 5e-5
        
        # Action selection parameters
        self.exploit_threshold = 0.7  # Use successful actions above this success rate
        self.explore_rate = 0.2       # Random exploration rate
        self.sequence_use_rate = 0.4  # Rate to use successful sequences
        
        # Add action masking
        self.action_mask_threshold = 0.1
        self.success_threshold = 0.3
        
        # Add exploration decay
        self.min_entropy_coef = 0.01
        self.entropy_decay = 0.995
        
        # Increase batch size
        self.min_batch_size = 32
        self.update_epochs = 10
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.learning_rate,
            eps=1e-5,
            weight_decay=1e-6
        )
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # Tracking metrics
        self.training_step = 0
        self.episode_rewards = []
        self.total_actions = Counter()
        self.running_reward = 0
        
        # Action tracking
        self.action_stats = {
            'success_rate': {},
            'reward_history': defaultdict(list),
            'usage_count': defaultdict(int),
            'last_success': defaultdict(int)
        }
        
        # Action grouping for better strategy
        self.action_groups = {
            'restore': [133, 134, 135, 139],
            'analyse': [3, 4, 5, 9],
            'remove': [16, 17, 18, 22],
            'monitor': [0]
        }
        
        # Adjust weights to prioritize enterprise systems
        self.group_weights = {
            'restore': 0.7,     # Highest priority
            'analyse': 0.2,     # Second priority
            'remove': 0.08,     # Lower priority
            'monitor': 0.02,    # Lowest priority
        }
        
        # System priorities
        self.system_priorities = {
            'enterprise0': 0.4,  # Highest
            'enterprise1': 0.3,
            'enterprise2': 0.2,
            'opserver0': 0.1    # Lowest
        }
        
        # Success tracking per group
        self.group_success = {group: [] for group in self.action_groups.keys()}
        
        # Add success streak tracking
        self.success_streaks = defaultdict(int)
        self.max_streak_bonus = 2.0
        
        # Define core action sequences that work
        self.core_sequences = [
            [0, 3],    # Monitor -> Analyze
            [3, 133],  # Analyze -> Restore
            [4, 134], 
            [5, 135],
            [9, 139]
        ]
        
        # Blacklist for failing actions
        self.action_blacklist = set()
        self.blacklist_threshold = 5  # Number of failures before blacklisting
        self.blacklist_cooldown = 50  # Steps before removing from blacklist
        self.last_blacklist = defaultdict(int)
        
        # Success tracking
        self.consecutive_successes = defaultdict(int)
        self.success_bonus = 2.0
        
        # Adjust weights more aggressively
        self.group_weights = {
            'restore': 0.7,    # Highest priority
            'analyse': 0.2,    # Second priority
            'remove': 0.08,    # Lower priority
            'monitor': 0.02    # Lowest priority
        }
        
        # Add DQN components for hybrid learning
        self.dqn_network = DQNNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network = DQNNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.dqn_network.state_dict())
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(50000)
        
        # Hybrid learning parameters
        self.dqn_update_freq = 4
        self.target_update_freq = 1000
        self.min_replay_size = 1000
        self.batch_size = 32
        self.dqn_learning_rate = 1e-4
        
        # Strategy selection
        self.strategies = {
            'monitor_analyze': {'actions': [1, 3], 'reward_threshold': -2.0},
            'analyze_restore': {'actions': [3, 133], 'reward_threshold': -3.0},
            'sleep_monitor': {'actions': [0, 1], 'reward_threshold': -1.0}
        }
        
        # Initialize success memory for ALL possible actions
        self.success_memory = {}
        for action in range(145):  # Initialize for all possible actions
            self.success_memory[action] = {
                'successes': 0,
                'attempts': 0,
                'decay': 0.99
            }
        
        # Explicitly add restore actions
        restore_actions = [133, 134, 135, 139]
        for action in restore_actions:
            if action not in self.success_memory:
                self.success_memory[action] = {
                    'successes': 0,
                    'attempts': 0,
                    'decay': 0.99
                }

    def get_action(self, observation, action_space):
        # Force complete sequences with restore
        if len(self.actions) > 0:
            last_action = self.actions[-1].item()
            
            # After monitor (1), must analyze and restore immediately
            if last_action == 1:
                system_risks = self._get_system_risks(observation)
                riskiest_system = max(system_risks.items(), key=lambda x: x[1])[0]
                
                analyze_restore = {
                    'enterprise0': ([3, 133], 0.4),   # [analyze, restore] pairs with priority
                    'enterprise1': ([4, 134], 0.3),
                    'enterprise2': ([5, 135], 0.2),
                    'opserver0': ([9, 139], 0.1)
                }
                
                # Weight system selection by priority
                if np.random.random() < analyze_restore[riskiest_system][1]:
                    return analyze_restore[riskiest_system][0][0]  # Use priority system
                else:
                    # Fallback to next highest priority system
                    sorted_systems = sorted(analyze_restore.items(), key=lambda x: x[1][1], reverse=True)
                    for sys, (actions, _) in sorted_systems:
                        if sys != riskiest_system and self._is_system_at_risk(observation, sys):
                            return actions[0]
                
                return analyze_restore[riskiest_system][0][0]  # Default to riskiest
                
            # After analyze, immediately restore
            analyze_to_restore = {3: 133, 4: 134, 5: 135, 9: 139}
            if last_action in analyze_to_restore:
                return analyze_to_restore[last_action]
                
            # After restore, only sleep if truly safe
            if last_action in analyze_to_restore.values():
                if not any(self._is_system_at_risk(observation, sys) 
                          for sys in ['enterprise0', 'enterprise1', 'enterprise2', 'opserver0']):
                    if len(self.actions) > 5:  # Only sleep after sufficient actions
                        return 0
                return 1  # Otherwise keep monitoring
        
        return 1  # Start with monitor

    def _should_sleep(self, observation):
        """Determine if agent should sleep based on system state"""
        system_risks = self._get_system_risks(observation)
        if any(risk > 0.3 for risk in system_risks.values()):
            return False
        return np.random.random() < self.sleep_threshold

    def _get_defensive_action(self, observation):
        # Choose defensive action based on risk assessment
        system_risks = self._get_system_risks(observation)
        if any(risk > 0.7 for risk in system_risks.values()):
            # High risk - immediate restore
            return self._get_restore_action(system_risks)
        elif any(risk > 0.4 for risk in system_risks.values()):
            # Medium risk - analyze
            return self._get_analyze_action(system_risks)
        else:
            # Low risk - monitor or sleep
            return 0 if np.random.random() < 0.3 else 1

    def _is_system_at_risk(self, observation, system):
        """Check if system shows signs of compromise"""
        system_indices = {
            'enterprise0': [0, 4],   # Start indices for each system
            'enterprise1': [4, 8],
            'enterprise2': [8, 12],
            'opserver0': [28, 32]
        }
        
        if system in system_indices:
            start, end = system_indices[system]
            # Check relevant observation bits
            system_obs = observation[start:end]
            return any(x > 0.5 for x in system_obs)  # System shows activity
        return False

    def store_reward(self, reward):
        # Bigger bonus for quick responses
        if len(self.actions) >= 3:
            last_three = [a.item() for a in self.actions[-3:]]
            if (last_three[0] == 1 and  # Monitor
                last_three[1] in [3,4,5,9] and  # Analyze
                last_three[2] in [133,134,135,139]):  # Restore
                steps_taken = len(self.actions)
                quick_response_bonus = max(0, 10.0 - steps_taken)  # More bonus for speed
                reward += quick_response_bonus
                
                # Extra bonus for enterprise systems
                if last_three[1] in [3,4,5]:  # Enterprise system analysis
                    reward += 2.0  # Double bonus for critical systems
        
        # Add sleep efficiency bonus
        if len(self.actions) > 0 and self.actions[-1].item() == 0:
            if not any(self._is_system_at_risk(self.last_observation, sys) 
                      for sys in ['enterprise0', 'enterprise1', 'enterprise2', 'opserver0']):
                reward += 0.5
        
        self.rewards.append(reward)
        
        if len(self.actions) > 0:
            last_action = self.actions[-1].item()
            
            # Initialize success rate tracking
            if last_action not in self.action_stats['success_rate']:
                self.action_stats['success_rate'][last_action] = {
                    'success': 0,
                    'total': 0,
                    'last_rewards': []
                }
            
            stats = self.action_stats['success_rate'][last_action]
            stats['total'] += 1
            stats['last_rewards'].append(reward)
            
            # Keep only last 100 rewards
            if len(stats['last_rewards']) > 100:
                stats['last_rewards'].pop(0)
            
            # Update success count based on reward
            if reward > -5:  # Success threshold
                stats['success'] += 1
                
            # Calculate running success rate
            success_rate = stats['success'] / stats['total']
            print(f"Action {last_action} success rate updated: {success_rate:.2%}")

    def update(self, next_state, done):
        # Update both PPO and DQN
        self._update_ppo(next_state, done)
        if len(self.replay_buffer) > self.min_replay_size:
            self._update_dqn()
            
        # Update success tracking
        self._update_success_rates()
        
        # Adjust strategy thresholds
        self._adjust_strategies()

    def _update_ppo(self, next_state, done):
        if len(self.states) < self.min_batch_size:
            return
            
        # Decay entropy coefficient
        self.entropy_coef = max(0.01, self.entropy_coef * self.entropy_decay)

        
        # Compute returns and advantages
        returns = []
        advantages = []
        next_value = 0 if done else self.network(torch.FloatTensor(next_state).to(self.device))[1].item()
        
        # GAE calculation
        gae = 0
        for reward, value in zip(reversed(self.rewards), reversed(self.values)):
            delta = reward + self.gamma * next_value - value.item()
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + value.item())
            advantages.insert(0, gae)
            next_value = value.item()
            
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Add importance sampling for better learning
        with torch.no_grad():
            importance_weights = torch.ones(len(self.rewards)).to(self.device)
            
            # Give higher weight to successful episodes
            for i, r in enumerate(self.rewards):
                if r > -5:
                    importance_weights[i] = 2.0
                    
            importance_weights = importance_weights / importance_weights.mean()
        
        # PPO update
        for _ in range(self.update_epochs):  # Multiple epochs
            # Get current policy outputs
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratios and surrogate losses
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Store loss values for metrics
            self.actor_loss = actor_loss.item()
            self.critic_loss = critic_loss.item()
            self.total_loss = total_loss.item()
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def get_metrics(self):
        """Get current training metrics"""
        return {
            'action_distribution': dict(self.total_actions),
            'mean_value': np.mean([v.item() for v in self.values]) if self.values else 0,
            'mean_reward': np.mean(self.rewards) if self.rewards else 0,
            'losses': {
                'actor': self.actor_loss if hasattr(self, 'actor_loss') else 0,
                'critic': self.critic_loss if hasattr(self, 'critic_loss') else 0,
                'total': self.total_loss if hasattr(self, 'total_loss') else 0
            },
            'action_success_rates': {
                action: stats['success']/stats['total'] 
                for action, stats in self.action_stats['success_rate'].items()
                if stats['total'] > 0
            },
            'action_usage': dict(self.action_stats['usage_count']),
            'mean_rewards_per_action': {
                action: np.mean(rewards) 
                for action, rewards in self.action_stats['reward_history'].items()
                if rewards
            }
        }

    def save(self, path):
        """Save model and training state"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'running_reward': self.running_reward,
            'total_actions': self.total_actions,
            'action_stats': self.action_stats,
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'learning_rate': self.learning_rate
            }
        }, path)

    def load(self, path):
        """Load model and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.running_reward = checkpoint['running_reward']
        self.total_actions = checkpoint['total_actions']
        self.action_stats = checkpoint['action_stats']
        
        # Load hyperparameters
        for key, value in checkpoint['hyperparameters'].items():
            setattr(self, key, value)

    def _update_success_rates(self):
        """Update success rates for actions with decay"""
        for action, memory in self.success_memory.items():
            # Apply decay to past successes
            memory['successes'] *= memory['decay']
            memory['attempts'] *= memory['decay']
            
            # Update from recent actions
            if len(self.actions) > 0 and self.actions[-1].item() == action:
                memory['attempts'] += 1
                if len(self.rewards) > 0 and self.rewards[-1] > -5:  # Success threshold
                    memory['successes'] += 1
                    
        # Update strategy thresholds based on success rates
        for strategy in self.strategies.values():
            success_rates = []
            for action in strategy['actions']:
                if self.success_memory[action]['attempts'] > 0:
                    rate = (self.success_memory[action]['successes'] / 
                           self.success_memory[action]['attempts'])
                    success_rates.append(rate)
            
            if success_rates:
                # Adjust reward threshold based on success rate
                avg_success = np.mean(success_rates)
                if avg_success > 0.7:
                    strategy['reward_threshold'] *= 0.95  # Make threshold stricter
                elif avg_success < 0.3:
                    strategy['reward_threshold'] *= 1.05  # Make threshold more lenient

    def _adjust_strategies(self):
        """Adjust strategy parameters based on performance"""
        # Calculate overall success rate
        total_success = sum(memory['successes'] for memory in self.success_memory.values())
        total_attempts = sum(memory['attempts'] for memory in self.success_memory.values())
        
        if total_attempts > 0:
            overall_success = total_success / total_attempts
            
            # Adjust sleep threshold based on success
            if overall_success > 0.7:
                self.sleep_threshold = min(0.4, self.sleep_threshold + 0.01)
            elif overall_success < 0.3:
                self.sleep_threshold = max(0.1, self.sleep_threshold - 0.01)
                
            # Adjust sequence use rate
            if overall_success > 0.8:
                self.sequence_use_rate = min(0.8, self.sequence_use_rate + 0.02)
            elif overall_success < 0.4:
                self.sequence_use_rate = max(0.2, self.sequence_use_rate - 0.02)
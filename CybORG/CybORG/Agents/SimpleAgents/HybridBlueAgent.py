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
    def __init__(self, input_dim=52, hidden_dim=64, output_dim=41, lr=0.0005):
        # Original PPO hyperparameters
        self.eps = 0.9
        self.eps_decay = 0.003
        self.eps_min = 0.1
        self.gamma = 0.99
        self.K_epochs = 4
        self.eps_clip = 0.2
        self.action_std = 0.5
        
        # Add GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks and move to device (only once)
        self.policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.policy_old = ActorCritic(input_dim, output_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Original tracking variables
        self.actions = []
        self.action_stats = defaultdict(int)
        
        # Original decoy actions
        self.decoy_actions = {
            'enterprise0': 69,
            'enterprise1': 70,
            'enterprise2': 71,
            'opserver0': 72,
            'user_hosts': [73, 74, 75, 76],
            'defender': 77
        }
        
        # Initialize DQN for decoy decisions
        self.decoy_dqn = DQNAgent(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=len(self.decoy_actions)
        )
        
        # PPO hyperparameters
        self.gamma = 0.97
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.K_epochs = 10
        self.entropy_coef = 0.3
        self.lr = 5e-5
        self.min_batch_size = 32
        self.MSE_loss = nn.MSELoss()
        
        # Initialize memory and optimizer
        self.memory = Memory(buffer_size=2048)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.lr,
            eps=1e-5,
            weight_decay=1e-6
        )
        
        # System priorities
        self.system_priorities = {
            'enterprise0': 0.4,
            'enterprise1': 0.3,
            'enterprise2': 0.2,
            'opserver0': 0.1
        }
        
        # Original tracking variables
        self.step_counter = 0
        self.rewards = []
        self.last_observation = None
        
        # Original action statistics
        self.action_stats = {
            'reward_history': {},
            'usage_count': {},
            'success_rate': {},
            'last_success': {}
        }
    
    def get_action(self, observation, action_space):
        """Enhanced action selection with specific counter-strategies"""
        state = torch.FloatTensor(observation).to(self.device)
        action_space = list(action_space) if not isinstance(action_space, list) else action_space
        
        # First check if we should place decoy
        if self._should_place_decoy(observation):
            # Use DQN for decoy placement
            with torch.no_grad():
                decoy_values = self.decoy_dqn.q_network(observation)
                # Convert tensor to numpy and get max index
                decoy_values_np = decoy_values.cpu().numpy()
                max_value_idx = np.argmax(decoy_values_np)
                
                # Map index to decoy action
                decoy_systems = ['enterprise0', 'enterprise1', 'enterprise2', 'opserver0', 'user_hosts', 'defender']
                selected_system = decoy_systems[max_value_idx]
                
                if selected_system == 'user_hosts':
                    # Randomly select one of the user host decoys
                    decoy_action = np.random.choice(self.decoy_actions[selected_system])
                else:
                    decoy_action = self.decoy_actions[selected_system]
                    
                if decoy_action in action_space:
                    return action_space.index(decoy_action)
        
        # Otherwise use PPO for normal actions
        return self._get_ppo_action(observation, action_space)
    
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
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'old_policy_state_dict': self.policy_old.state_dict(),  # Keep both names for compatibility
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model checkpoint with backwards compatibility"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # Try different possible keys for old policy state dict
        if 'policy_old_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        elif 'old_policy_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['old_policy_state_dict'])
        else:
            print("Warning: No old policy state found in checkpoint, copying from current policy")
            self.policy_old.load_state_dict(self.policy.state_dict())
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
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
    
    def update(self, next_observation, done):
        """Update agent's policy"""
        # Store transition if we have a previous action
        if len(self.actions) > 0 and len(self.rewards) > 0:
            # Convert observations to float tensors
            if isinstance(self.last_observation, np.ndarray):
                state = self.last_observation
            else:
                state = np.array(self.last_observation, dtype=np.float32)
                
            if isinstance(next_observation, np.ndarray):
                next_state = next_observation
            else:
                next_state = np.array(next_observation, dtype=np.float32)
                
            # Store experience
            self.memory.states.append(state)
            self.memory.actions.append(self.actions[-1])
            self.memory.rewards.append(self.rewards[-1])
            self.memory.next_states.append(next_state)
            self.memory.dones.append(done)

        # Update policy if enough samples
        if len(self.memory.states) >= self.min_batch_size:
            try:
                # Convert to tensors
                states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
                actions = torch.LongTensor(self.memory.actions).to(self.device)
                rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
                next_states = torch.FloatTensor(np.array(self.memory.next_states)).to(self.device)
                dones = torch.FloatTensor(self.memory.dones).to(self.device)

                # Compute returns with GAE
                advantages = []
                gae = 0
                with torch.no_grad():
                    _, next_values = self.policy(next_states)
                    _, values = self.policy(states)
                    next_values = next_values.squeeze()
                    values = values.squeeze()
                    
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_value = next_values[t] * (1 - dones[t])
                    else:
                        next_value = values[t + 1]
                        
                    delta = rewards[t] + self.gamma * next_value - values[t]
                    gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                    advantages.insert(0, gae)
                    
                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = advantages + values
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get old action probabilities
                old_probs, _ = self.policy_old(states)
                old_probs = old_probs.detach()
                
                # Update policy for K epochs
                for _ in range(self.K_epochs):
                    # Get current action probabilities and state values
                    probs, state_values = self.policy(states)
                    dist = torch.distributions.Categorical(probs)
                    
                    # Compute ratio and surrogate loss
                    new_probs = dist.log_prob(actions)
                    old_probs_log = torch.log(old_probs.gather(1, actions.unsqueeze(1)).squeeze())
                    ratio = torch.exp(new_probs - old_probs_log)
                    
                    # Compute actor and critic losses
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = self.MSE_loss(state_values.squeeze(), returns)
                    
                    # Add entropy bonus for exploration
                    entropy_loss = -self.entropy_coef * dist.entropy().mean()
                    
                    # Total loss
                    loss = actor_loss + 0.5 * critic_loss + entropy_loss
                    
                    # Update policy
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Copy new weights into old policy
                self.policy_old.load_state_dict(self.policy.state_dict())
                
                # Clear memory
                self.memory.clear()
                
            except Exception as e:
                print(f"Error in policy update: {str(e)}")
                self.memory.clear()
        
        # Update last observation
        self.last_observation = next_observation
        self.step_counter += 1
    
    def _detect_bline_pattern(self, observation):
        """Detect if B_lineAgent pattern is being used"""
        if len(self.actions) < 3:
            return False
        
        # B_line uses fixed sequence: Discover -> Exploit -> Privilege Escalate
        # Check for typical B_line targets
        targets = {
            'User0': sum(observation[0:4]),
            'Enterprise0': sum(observation[4:8]),
            'Enterprise2': sum(observation[8:12]),
            'Op_Server0': sum(observation[28:32])
        }
        
        # High activity on these systems suggests B_line
        return any(v > 0.5 for v in targets.values())

    def _counter_bline(self, observation, action_space):
        """Enhanced B_line counter strategy"""
        # B_line's exact attack sequence
        attack_sequence = [
            ('User0', 3, 133, 69),       # system, analyze, restore, decoy
            ('Enterprise0', 4, 134, 70),  # B_line's second target
            ('Enterprise2', 5, 135, 71),  # B_line's third target
            ('Op_Server0', 9, 139, 72)    # Final target
        ]
        
        # Always monitor if we haven't in last 2 steps
        if len(self.actions) >= 2 and 1 not in self.actions[-2:]:
            if 1 in action_space:
                return action_space.index(1)
        
        # Get current position in B_line's attack path
        current_position = -1
        max_threat = 0
        for i, (system, _, _, _) in enumerate(attack_sequence):
            threat = self._get_system_threat(observation, system)
            if threat > max_threat:
                max_threat = threat
                current_position = i
                
        # If we detect activity, defend aggressively
        if current_position != -1:
            # Immediately restore current system if compromised
            _, analyze, restore, decoy = attack_sequence[current_position]
            if restore in action_space:
                return action_space.index(restore)
                
            # Place decoy on next system in path
            next_pos = (current_position + 1) % len(attack_sequence)
            _, _, _, next_decoy = attack_sequence[next_pos]
            if next_decoy in action_space:
                return action_space.index(next_decoy)
                
            # Analyze current system
            if analyze in action_space:
                return action_space.index(analyze)
        
        # Proactively defend the path
        for i in range(len(attack_sequence)):
            # Check two steps ahead of last known activity
            defend_pos = (current_position + i + 1) % len(attack_sequence)
            _, analyze, restore, decoy = attack_sequence[defend_pos]
            
            # Proactive defense sequence
            if analyze in action_space:
                return action_space.index(analyze)
            if decoy in action_space:
                return action_space.index(decoy)
            if restore in action_space:
                return action_space.index(restore)
        
        return None

    def _detect_meander_pattern(self, observation):
        """Detect if RedMeanderAgent pattern is being used"""
        if len(self.actions) < 5:
            return False
        
        # Meander uses random exploration and opportunistic attacks
        # Check for scattered activity
        system_activity = {
            'User0': sum(observation[0:4]),
            'Enterprise0': sum(observation[4:8]),
            'Enterprise1': sum(observation[8:12]),
            'Enterprise2': sum(observation[12:16]),
            'Op_Server0': sum(observation[28:32])
        }
        
        # Multiple systems with medium activity suggests Meander
        active_systems = sum(v > 0.3 for v in system_activity.values())
        return active_systems >= 2

    def _counter_meander_strategy(self, observation):
        """Counter RedMeanderAgent's random behavior"""
        # Deploy decoys more aggressively
        if np.random.random() < 0.4:  # 40% chance to use decoys
            active_systems = self._get_active_systems(observation)
            if active_systems:
                # Deploy decoy near most active system
                system = max(active_systems.items(), key=lambda x: x[1])[0]
                decoy_actions = {
                    'User0': 69,
                    'Enterprise0': 70,
                    'Enterprise1': 71,
                    'Enterprise2': 72,
                    'Op_Server0': 73
                }
                return decoy_actions[system]
        
        # Regular defensive sequence
        if len(self.actions) % 3 == 0:
            return 1  # Monitor
        elif len(self.actions) > 0 and self.actions[-1] == 1:
            # Choose analyze based on threat
            threats = {
                3: sum(observation[0:4]),    # User0
                4: sum(observation[4:8]),    # Enterprise0
                5: sum(observation[8:12]),   # Enterprise1
                9: sum(observation[28:32])   # Op_Server0
            }
            return max(threats.items(), key=lambda x: x[1])[0]
        
        return None

    def _get_system_threat(self, observation, system):
        """Get threat level for specific system"""
        # System-specific observation indices
        system_indices = {
            'User0': slice(0, 4),
            'Enterprise0': slice(4, 8),
            'Enterprise1': slice(8, 12),
            'Enterprise2': slice(12, 16),
            'Op_Server0': slice(28, 32)
        }
        
        if system not in system_indices:
            return 0.0
        
        # Get relevant observation values
        values = observation[system_indices[system]]
        
        # Weight different indicators
        anomaly_weight = 2.0  # Increased weight for anomalies
        access_weight = 1.5   # Weight for access indicators
        
        threat = 0.0
        for i, value in enumerate(values):
            if i == 0:  # Anomaly indicator
                threat += value * anomaly_weight
            else:  # Access indicators
                threat += value * access_weight
                
        return threat / (anomaly_weight + 2 * access_weight)  # Normalize

    def _get_active_systems(self, observation):
        """Get activity levels for all systems"""
        return {
            'User0': sum(observation[0:4]),
            'Enterprise0': sum(observation[4:8]),
            'Enterprise1': sum(observation[8:12]),
            'Enterprise2': sum(observation[12:16]),
            'Op_Server0': sum(observation[28:32])
        }
    
    def _should_place_decoy(self, observation):
        """Determine if we should place a decoy based on:
        1. Threat level
        2. Existing decoys
        3. System vulnerabilities
        """
        threat_level = self._assess_threat(observation)
        return threat_level > 0.5 or np.random.random() < 0.2  # 20% random decoy placement
    
    def store_decoy_experience(self, state, action, reward, next_state, done):
        """Store decoy-related experiences separately"""
        if action in self.decoy_actions.values():
            self.decoy_dqn.replay_buffer.push(
                state, action, reward, next_state, done
            )
    
    def _get_ppo_action(self, observation, action_space):
        """Get action using PPO policy
        Args:
            observation: Current state observation
            action_space: List of available actions
        Returns:
            int: Index of chosen action
        """
        state = torch.FloatTensor(observation).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy.actor(state)
            
            # Create action mask
            mask = torch.ones_like(action_probs)
            for i, action in enumerate(action_space):
                # Boost important actions based on context
                if action == 1:  # monitor
                    mask[i] *= 2.0
                elif action in [3,4,5,9]:  # analyze
                    mask[i] *= 1.5
                elif action in [133,134,135,139]:  # restore
                    mask[i] *= 1.5
                elif action in range(69, 78):  # decoys
                    mask[i] *= 1.2
                
                # Reduce probability of recent actions
                if len(self.actions) > 0 and action in self.actions[-3:]:
                    mask[i] *= 0.5
            
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
    
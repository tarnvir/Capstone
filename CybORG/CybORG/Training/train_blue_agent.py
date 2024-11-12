import inspect
import os
import json
import time
import numpy as np
import torch
from collections import defaultdict, Counter

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.PPOBlueAgent import PPOBlueAgent
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper

# Import RedAgent from SimpleAgents
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

# Import HybridBlueAgent from SimpleAgents
from CybORG.Agents.SimpleAgents.HybridBlueAgent import HybridBlueAgent

# Import DQNAgent and ReplayBuffer from SimpleAgents
from CybORG.Agents.SimpleAgents.DQNAgent import DQNAgent, ReplayBuffer

class RedAgent(BaseAgent):
    """Active Red Agent that performs systematic attacks"""
    def __init__(self):
        super().__init__()
        self.action_space = None
        self.step_counter = 0
        self.last_observation = None
        self.last_action = None
        self.consecutive_attacks = 0
        
        # More aggressive parameters
        self.attack_frequency = 0.8  # Increased from 0.6
        self.attack_intensity = 3.0  # Increased from 2.5
        
        # B_lineAgent-like sequences
        self.attack_sequences = [
            ('DiscoverRemoteSystems', 'ExploitRemoteService', 'PrivilegeEscalate', 'Impact', 'Impact', 'Impact'),
            ('Impact', 'Impact', 'Impact', 'PrivilegeEscalate', 'Impact', 'Impact'),
            ('ExploitRemoteService', 'PrivilegeEscalate', 'Impact', 'Impact', 'Impact', 'Impact')
        ]
        
        # Focus more on critical systems
        self.target_systems = {
            'enterprise0': {'priority': 0.5, 'last_attack': 0},  # Increased priority
            'enterprise1': {'priority': 0.3, 'last_attack': 0},
            'enterprise2': {'priority': 0.2, 'last_attack': 0},
            'opserver0': {'priority': 0.0, 'last_attack': 0}  # Reduced priority
        }
        
        self.current_sequence = None
        self.sequence_position = 0
        self.current_target = 0
    
    def get_action(self, observation, action_space=None):
        """Choose next attack action"""
        if action_space is not None:
            self.action_space = action_space
            
        self.step_counter += 1
        self.last_observation = observation
        
        if np.random.random() < self.attack_frequency:
            self.consecutive_attacks += 1
            # Increase attack intensity with consecutive attacks
            intensity_bonus = min(self.consecutive_attacks * 0.1, 0.5)
            
            # Choose more aggressive action sequence
            sequence_idx = np.random.randint(len(self.attack_sequences))
            sequence = self.attack_sequences[sequence_idx]
            
            # Target selection based on success history
            target = self._select_vulnerable_target(observation)
            
            return {
                'action': sequence[self.sequence_position],
                'session': 0,
                'agent': 'Red',
                'hostname': target,
                'intensity': self.attack_intensity + intensity_bonus
            }
        else:
            self.consecutive_attacks = 0
            return None
    
    def end_episode(self):
        """Reset agent's state at end of episode"""
        self.step_counter = 0
        self.current_sequence = None
        self.sequence_position = 0
        self.last_observation = None
        self.last_action = None
    
    def set_initial_values(self, action_space, observation):
        """Set initial values for the agent"""
        self.action_space = action_space
        self.last_observation = observation
        return observation

    def train(self, results):
        """Implement training method (required by BaseAgent)"""
        # Store last observation and action
        self.last_observation = results.observation
        if hasattr(results, 'action'):
            self.last_action = results.action
        return results.observation

    def get_observation(self, observation):
        """Implement get_observation method (required by BaseAgent)"""
        return observation

    def _select_vulnerable_target(self, observation):
        """Select most vulnerable target based on observation"""
        # Convert observation to dict if it's not already
        if not isinstance(observation, dict):
            observation = dict(observation)
        
        # Calculate vulnerability scores
        vulnerability_scores = {}
        for system, info in self.target_systems.items():
            # Calculate base vulnerability score from priority
            vulnerability_score = info['priority']
            
            # Add time factor (prefer systems not recently attacked)
            steps_since_attack = self.step_counter - info['last_attack']
            time_factor = min(steps_since_attack / 10.0, 1.0)  # Cap at 1.0
            
            # Combine scores
            vulnerability_scores[system] = vulnerability_score * time_factor
        
        # Select most vulnerable system
        target = max(vulnerability_scores.items(), key=lambda x: x[1])[0]
        
        # Update last attack time
        self.target_systems[target]['last_attack'] = self.step_counter
        
        return target

def _update_scan_state(observation, scan_state):
    """Update scan state based on observation"""
    # Define indices for each system
    indices = {
        'defender': 0,
        'enterprise0': 1,
        'enterprise1': 2, 
        'enterprise2': 3,
        'user0': 4,
        'user1': 5,
        'user2': 6,
        'user3': 7,
        'user4': 8,
        'opserver0': 9
    }
    
    # Check each system's observation indices
    for system, idx in indices.items():
        if system == 'defender':
            obs_idx = 0
        elif 'enterprise' in system:
            obs_idx = int(system[-1]) * 4
        elif 'user' in system:
            obs_idx = 12 + int(system[-1]) * 4
        else:  # opserver0
            obs_idx = 28
            
        # Check if system is scanned
        if observation[obs_idx] == 1 and observation[obs_idx+1] == 0:
            # Set all previous scans to 1
            scan_state = np.where(scan_state == 2, 1, scan_state)
            # Set current scan to 2
            scan_state[idx] = 2
            
    return scan_state

def _is_meander(observation):
    """Detect if red agent is using meander pattern"""
    # Meander pattern shows as multiple scans on different hosts
    scan_count = 0
    last_scan_idx = -1
    
    # Check enterprise systems
    for i in range(0, 12, 4):
        if observation[i] == 1 and observation[i+1] == 0:
            scan_count += 1
            if last_scan_idx >= 0 and i - last_scan_idx > 4:
                # Non-consecutive scans indicate meander
                return True
            last_scan_idx = i
            
    # Check user systems
    for i in range(12, 32, 4):
        if observation[i] == 1 and observation[i+1] == 0:
            scan_count += 1
            if last_scan_idx >= 0 and i - last_scan_idx > 4:
                # Non-consecutive scans indicate meander
                return True
            last_scan_idx = i
            
    # Multiple scans indicate meander behavior
    return scan_count >= 2

def train_blue_agent(episodes=10000, steps_per_episode=30):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    print(f"Loading scenario from: {path}")
    
    # Define complete action sequences with restore
    action_sequences = {
        'enterprise0': [1, 3, 133],  # Monitor -> Analyze -> Restore (no sleep)
        'enterprise1': [1, 4, 134],
        'enterprise2': [1, 5, 135],
        'opserver0': [1, 9, 139]
    }
    
    # Define host-specific decoy actions
    decoy_actions = {
        'enterprise0': [55, 107, 120, 29],
        'enterprise1': [43],
        'enterprise2': [44],
        'user1': [37, 115, 76, 102],
        'user2': [51, 116, 38, 90],
        'user3': [130, 91],
        'user4': [131],
        'defender': [54, 106, 28, 119],
        'opserver0': [61, 35, 113, 126]
    }
    
    # Define initial sequence based on insights
    initial_sequence = {
        'first_actions': [51, 116],  # Fermitter User2, Tomcat User2
        'third_actions': {
            'enterprise0': 55,  # SMTP Ent0
            'enterprise1': 43   # Fermitter Ent1
        }
    }
    
    try:
        # Create environment
        cyborg = CybORG(path, 'sim', agents={'Red': RedAgent})
        env = ReduceActionSpaceWrapper(cyborg)
        env = ChallengeWrapper(env=env, agent_name='Blue')
        
        # Initialize hybrid agent with reduced action space
        action_space = [
            0,                          # Sleep
            1,                          # Monitor
            3, 4, 5, 9,                # Analyze
            133, 134, 135, 139,        # Restore
            16, 17, 18, 22             # Remove
        ]
        
        hybrid_agent = HybridBlueAgent(
            input_dim=52,
            hidden_dim=512,
            output_dim=len(action_space)
        )
        
        # Initialize metrics
        metrics = {
            'episode_rewards': [],
            'action_counts': defaultdict(int),
            'best_reward': float('-inf'),
            'best_episode': 0
        }
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(os.path.dirname(path), "../../../Checkpoints/")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Training loop
        for episode in range(episodes):
            observation = env.reset()
            total_reward = 0
            episode_actions = []
            
            # Episode loop
            for step in range(steps_per_episode):
                # Get action from agent
                action = hybrid_agent.get_action(observation, action_space)
                
                # Take step
                next_observation, reward, done, info = env.step(action)
                
                # Store transition
                hybrid_agent.store_reward(reward)
                episode_actions.append(action)
                total_reward += reward
                
                if done:
                    break
                    
                observation = next_observation
            
            # Update agent
            hybrid_agent.update(next_observation, done)
            
            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Step Reward: {total_reward/len(episode_actions):.2f}")
            print(f"Actions Used: {Counter(episode_actions).most_common(3)}")
            
            # Update metrics
            metrics['episode_rewards'].append(total_reward)
            metrics['action_counts'].update(Counter(episode_actions))
            
            # Track best performance
            if total_reward > metrics['best_reward']:
                metrics['best_reward'] = total_reward
                metrics['best_episode'] = episode
                hybrid_agent.ppo_agent.save(os.path.join(checkpoint_dir, 'best_model.pt'))
            
            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Step Reward: {total_reward/len(episode_actions):.2f}")
            print(f"Actions Used: {Counter(episode_actions).most_common(3)}")
            
            # Save checkpoints
            if (episode + 1) % 100 == 0:
                print(f"\nCheckpoint {episode + 1}")
                print(f"Average Reward (last 100): {np.mean(metrics['episode_rewards'][-100:]):.2f}")
                print(f"Best Reward: {metrics['best_reward']:.2f} (Episode {metrics['best_episode']})")
                print(f"Training Time: {time.time() - start_time:.2f}s")
                
                # Save model and metrics
                hybrid_agent.ppo_agent.save(os.path.join(checkpoint_dir, f'checkpoint_{episode+1}.pt'))
                
                with open(os.path.join(checkpoint_dir, f'metrics_{episode+1}.json'), 'w') as f:
                    json_metrics = {
                        'episode_rewards': [float(r) for r in metrics['episode_rewards']],
                        'action_counts': {str(k): int(v) for k, v in metrics['action_counts'].items()},
                        'best_reward': float(metrics['best_reward']),
                        'best_episode': int(metrics['best_episode'])
                    }
                    json.dump(json_metrics, f)
        
        return hybrid_agent, metrics
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Increase episodes and modify training parameters
    agent, metrics = train_blue_agent(
        episodes=1000,  # Increased from 2000
        steps_per_episode=30  # Keep this constant based on evaluation
    )
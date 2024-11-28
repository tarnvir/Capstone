import inspect
import os
import time
import numpy as np
import mlx

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.HybridBlueAgent import HybridBlueAgent
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper

def train_blue_agent(episodes=20000, steps_per_episode=30):
    try:
        # Check if MLX is available
        device = mlx.get_device()  # Get the MLX device
        print(f"Using device: {device}")

        # Get the path to CybORG installation
        path = str(inspect.getfile(CybORG))
        # Set path to scenario file
        path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
        
        # Initialize CybORG environment
        cyborg = CybORG(path, 'sim')
        # Wrap environment to reduce action space
        env = ReduceActionSpaceWrapper(cyborg)
        # Add challenge wrapper for blue agent
        env = ChallengeWrapper(env=env, agent_name='Blue')
        
        # Define all possible actions for blue agent
        action_space = [
            1,                    # monitor
            133, 134, 135, 139,  # restore actions for enterprise and opserver
            3, 4, 5, 9,          # analyse actions for enterprise and opserver
            16, 17, 18, 22,      # remove malware from enterprise and opserver
            11, 12, 13, 14,      # analyse actions for user hosts
            141, 142, 143, 144,  # restore actions for user hosts
            132,                 # restore defender system
            2,                   # analyse defender system
            15, 24, 25, 26, 27,  # remove malware from defender and user hosts
            # Add decoy actions
            69, 70, 71, 72,      # Deploy decoy on enterprise0-2 and opserver0
            73, 74, 75, 76,      # Deploy decoy on user hosts
            77                   # Deploy decoy on defender
        ]
        
        # Initialize hybrid blue agent with proper dimensions
        hybrid_agent = HybridBlueAgent(
            input_dim=52,         # Size of observation space
            hidden_dim=256,       # Size of hidden layers
            output_dim=len(action_space)  # Number of possible actions
        ).to(device)
        
        # Initialize training metrics
        best_reward = float('-inf')  # Track best average reward
        recent_rewards = []          # Store recent rewards for averaging
        timestep = 0                 # Track total steps taken
        
        # Main training loop
        for episode in range(episodes):
            # Reset environment at start of episode
            observation = env.reset()
            # Convert observation to tensor and send to device
            observation = mlx.array(observation, dtype=mlx.float32).to(device)
            episode_reward = 0
            actions_taken = []
            
            # Episode loop
            for step in range(steps_per_episode):
                timestep += 1
                
                # Get action from agent
                action_idx = hybrid_agent.get_action(observation, action_space)
                # Convert action index to actual action
                action = action_space[action_idx]
                actions_taken.append(action)
                
                # Take step in environment
                next_observation, reward, done, _ = env.step(action)
                # Convert next observation to tensor and send to device
                next_observation = mlx.array(next_observation, dtype=mlx.float32).to(device)
                # Store reward with current observation
                hybrid_agent.store_reward(reward, observation)
                episode_reward += reward
                
                # Break if episode is done
                if done:
                    break
                    
                observation = next_observation
            
            # Update agent's policy
            hybrid_agent.update(next_observation, done)
            
            # Track performance
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:  # Keep only last 100 rewards
                recent_rewards.pop(0)
            avg_reward = np.mean(recent_rewards)
            
            # Save if performance improved
            if avg_reward > best_reward:
                best_reward = avg_reward
                hybrid_agent.save('CybORG/Checkpoints/best_model.pt')
                print(f"\nNew best model saved! Average reward: {avg_reward:.2f}")
            
            # Print progress every 50 episodes
            if (episode + 1) % 50 == 0:
                print(f"\nEpisode {episode + 1}/{episodes}")
                print(f"Running reward: {avg_reward:.2f}")
                print(f"Episode reward: {episode_reward:.2f}")
                print(f"Action distribution: {dict(zip(actions_taken, np.bincount(actions_taken)))}")
        
        return hybrid_agent
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    # Set random seeds for reproducibility
    mlx.manual_seed(0)
    np.random.seed(0)
    # Start training
    agent = train_blue_agent(episodes=20000, steps_per_episode=30)

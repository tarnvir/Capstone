import inspect
import os
import time
import numpy as np
import torch

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.HybridBlueAgent import HybridBlueAgent
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper

def train_blue_agent(episodes=10000, steps_per_episode=30):
    try:
        # Setup environment
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
        
        cyborg = CybORG(path, 'sim')
        env = ReduceActionSpaceWrapper(cyborg)
        env = ChallengeWrapper(env=env, agent_name='Blue')
        
        # Define action space like john-cardiff's implementation
        action_space = [
            133, 134, 135, 139,  # restore enterprise and opserver
            3, 4, 5, 9,          # analyse enterprise and opserver
            16, 17, 18, 22,      # remove enterprise and opserver
            11, 12, 13, 14,      # analyse user hosts
            141, 142, 143, 144,  # restore user hosts
            132,                 # restore defender
            2,                   # analyse defender
            15, 24, 25, 26, 27   # remove defender and user hosts
        ]
        
        # Initialize agent with proper action space
        hybrid_agent = HybridBlueAgent(input_dim=52, hidden_dim=512, output_dim=len(action_space))
        
        # Training parameters from their implementation
        update_timestep = 20000  # Update every 20000 steps
        K_epochs = 6
        eps_clip = 0.2
        gamma = 0.99
        lr = 0.002
        betas = [0.9, 0.990]
        
        # Training tracking
        best_reward = float('-inf')
        recent_rewards = []
        time_step = 0  # Track total steps for buffer updates
        
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            observation = env.reset()
            episode_reward = 0
            actions_taken = []
            
            for step in range(steps_per_episode):
                time_step += 1
                action = hybrid_agent.get_action(observation, action_space)
                actions_taken.append(action)
                
                next_observation, reward, done, _ = env.step(action)
                hybrid_agent.store_reward(reward)
                episode_reward += reward
                
                # Update every update_timestep steps
                if time_step % update_timestep == 0:
                    hybrid_agent.update(next_observation, done)
                    time_step = 0
                
                if done:
                    break
                    
                observation = next_observation
            
            # Track performance
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            avg_reward = np.mean(recent_rewards)
            
            # Print progress every 50 episodes (like their print_interval)
            if (episode + 1) % 50 == 0:
                action_counts = np.bincount(actions_taken)
                most_used = action_counts.argmax()
                print(f"\nEpisode {episode + 1}/{episodes}")
                print(f"Running reward: {avg_reward:.2f}")
                print(f"Episode reward: {episode_reward:.2f}")
                print(f"Action distribution: {dict(zip(np.nonzero(action_counts)[0], action_counts[action_counts > 0]))}")
            
            # Save if performance improved
            if avg_reward > best_reward:
                best_reward = avg_reward
                hybrid_agent.save('CybORG/Checkpoints/best_model.pt')
                print(f"\nNew best model saved! Average reward: {avg_reward:.2f}")
        
        return hybrid_agent
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    agent = train_blue_agent(
        episodes=20000,  # Double the episodes
        steps_per_episode=30
    )
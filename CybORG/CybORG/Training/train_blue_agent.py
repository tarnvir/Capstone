import torch
import numpy as np
import os
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper
import inspect
from CybORG.Agents.SimpleAgents.PPOAgent import PPOAgent
import random

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = (torch.device("mps") 
          if torch.backends.mps.is_available() 
          else torch.device("cpu"))
print(f"Training device: {device}")


def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100, start_actions=[]):

    agent = PPOAgent(
        input_dim=input_dims,
        output_dim=len(action_space),
        lr=lr,
        betas=betas,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        start_actions=start_actions
    )
    
    # Set initial values including action space
    agent.set_initial_values(action_space)
    agent.policy.to(device)

    # Initialize tracking
    best_reward = float('-inf')
    running_reward = 0
    
    # Training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        # Episode loop
        for t in range(max_timesteps):
            # Get action index from agent
            action_idx = agent.get_action(state)
            
            # Ensure action index is valid
            action_idx = action_idx % len(action_space)  # Safety check
            action = action_space[action_idx]
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Store experience
            agent.store(reward, done)
            
            state = next_state
            
            # Update if memory buffer is full
            if len(agent.memory.states) >= agent.memory.buffer_size:
                agent.train()
                agent.clear_memory()
        
        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        
        # Save best model
        if running_reward > best_reward:
            best_reward = running_reward
            torch.save(agent.policy.state_dict(), 
                      os.path.join(ckpt_folder, 'best_model.pth'))

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            print('Checkpoint saved')

        if i_episode % print_interval == 0:
            avg_reward = running_reward / print_interval
            print(f'Episode {i_episode} \t' 
                  f'Avg reward: {avg_reward:.3f} \t'
                  f'Last action: {agent.last_action} \t'
                  f'Threat level: {np.sum(agent.scan_state)}')
            running_reward = 0


if __name__ == '__main__':

    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Adjusted hyperparameters for better learning
    max_episodes = 20000
    max_timesteps = 30
    update_timestep = 256  # More frequent updates
    K_epochs = 12         # More policy updates
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.00005         # Lower learning rate for stability
    betas = [0.9, 0.999]

    # Add entropy coefficient for exploration
    entropy_coef = 0.01  # Encourage exploration

    # Simplified action space focusing on critical actions
    action_space = [
        133, 134, 135, 139,  # restore critical systems
        3, 4, 5, 9,         # analyse critical systems
        16, 17, 18, 22      # remove critical systems
    ]

    # Start with analysis actions
    start_actions = [3, 4, 5, 9]  # Start with system analysis

    # Training setup
    folder = 'bline'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    # Environment setup
    CYBORG = CybORG(PATH, 'sim', agents={'Red': B_lineAgent})
    env = ChallengeWrapper(env=CYBORG, agent_name="Blue")
    input_dims = env.observation_space.shape[0] + 10  # Add 10 for scan state

    # Print training info
    print(f"Input dimensions: {input_dims}")
    print(f"Action space size: {len(action_space)}")
    print(f"Starting training...")

    train(env, input_dims, action_space,
          max_episodes=max_episodes, 
          max_timesteps=max_timesteps,
          update_timestep=update_timestep, 
          K_epochs=K_epochs,
          eps_clip=eps_clip, 
          gamma=gamma, 
          lr=lr,
          betas=betas, 
          ckpt_folder=ckpt_folder,
          print_interval=50, 
          save_interval=200, 
          start_actions=start_actions)
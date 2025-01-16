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

def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100, start_actions=[]):

    # Initialize agent with optimized parameters
    agent = PPOAgent(
        input_dims=input_dims,
        action_space=action_space,
        lr=lr,
        betas=betas,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        start_actions=start_actions
    )
    
    # Initialize agent's values including greedy_decoys
    agent.set_initial_values(action_space)

    # Training loop variables
    running_reward = 0
    time_step = 0

    # Main training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        # Episode loop
        for t in range(max_timesteps):
            time_step += 1
            
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in environment
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Store experience
            agent.store(reward, done)

            # Update if enough steps have been taken
            if time_step % update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0

        # End episode
        agent.end_episode()
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Save checkpoint
        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, f'{i_episode}.pth')
            torch.save(agent.policy.state_dict(), ckpt)
            print('Checkpoint saved')

        # Print metrics
        if i_episode % print_interval == 0:
            print(f'Episode {i_episode} \t' 
                  f'Avg reward: {running_reward:.3f}')

if __name__ == '__main__':
    # Set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Optimized hyperparameters from successful implementation
    max_episodes = 100000  # Much longer training
    max_timesteps = 100    # Full episode length
    update_timestep = 20000  # Much larger buffer
    K_epochs = 6           # Fewer policy updates
    eps_clip = 0.2        # Standard PPO clip
    gamma = 0.99         # Standard discount
    lr = 0.002          # Higher learning rate
    betas = [0.9, 0.990]

    # Focused action space
    action_space = [
        133, 134, 135, 139,  # restore enterprise and opserver
        3, 4, 5, 9,          # analyse enterprise and opserver
        16, 17, 18, 22       # remove enterprise and opserver
    ]

    # Start with critical decoys
    start_actions = [1004, 1004, 1000]  # user2 decoy x2, ent0 decoy

    # Setup training folder
    folder = 'bline'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    # Setup environment
    CYBORG = CybORG(PATH, 'sim', agents={'Red': B_lineAgent})
    env = ChallengeWrapper(env=CYBORG, agent_name="Blue")
    input_dims = env.observation_space.shape[0]

    # Print training info
    print(f"Input dimensions: {input_dims}")
    print(f"Action space size: {len(action_space)}")
    print(f"Starting training...")

    # Start training
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
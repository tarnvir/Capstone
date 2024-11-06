# train_blue_agent.py

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor as SB3Monitor
from gymnasium import spaces

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Simulator.Actions import Monitor as CybORGMonitor
from CybORG.Simulator.Actions import Sleep, Analyse, Remove, Restore

from my_blue_agent import MyBlueAgent, BlueAction

class BlueCybORGEnv(gym.Env):
    def __init__(self):
        super(BlueCybORGEnv, self).__init__()

        # Initialize with much larger step limit to avoid the error
        max_steps = 500  # Episode length
        scenario_steps = 10000  # Much larger than episode length
        sg = EnterpriseScenarioGenerator(steps=scenario_steps)  # Increased steps
        self.cyborg = CybORG(scenario_generator=sg)
        self.agent_name = 'blue_agent_0'

        # Create action list FIRST
        self.action_list = self.create_action_list()
        
        # THEN define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(100,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.get_action_space_size())

        # Initialize agent AFTER spaces are defined
        self.blue_agent = MyBlueAgent(action_space=self.action_space, 
                                    observation_space=self.observation_space)

        self.current_step = 0
        self.max_steps = max_steps

    def reset(self, seed=None, options=None):
        """
        Reset the environment
        
        Args:
            seed: The seed for random number generation
            options: Additional options for reset
            
        Returns:
            observation: The initial observation
            info: A dict containing additional information
        """
        # Handle the seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset CybORG environment
        result = self.cyborg.reset()
        observation = result.observation
        self.current_step = 0
        
        # Process observation
        processed_obs = self.blue_agent.process_observation(observation)
        
        # Return both observation and info dict as per Gymnasium API
        info = {}
        return processed_obs, info

    def step(self, action_idx):
        """
        Take a step in the environment
        
        Returns:
            observation: The current observation
            reward: The reward obtained
            terminated: Whether the episode has ended
            truncated: Whether the episode was artificially terminated
            info: Additional information
        """
        # Map action index to CybORG action
        action = self.map_action(action_idx)

        # Take a step in the CybORG environment
        result = self.cyborg.step(agent=self.agent_name, action=action)
        observation = result.observation
        reward = self.calculate_reward(result)
        
        # Split done into terminated and truncated as per Gymnasium API
        terminated = False  # Episode ended naturally
        truncated = self.current_step >= self.max_steps  # Episode truncated due to max steps
        
        info = {}
        processed_obs = self.blue_agent.process_observation(observation)
        
        self.current_step += 1

        return processed_obs, reward, terminated, truncated, info

    def process_observation(self, observation):
        return self.blue_agent.process_observation(observation)

    def map_action(self, action_idx):
        # Retrieve the action from the action list
        action_entry = self.action_list[action_idx]
        action = action_entry.get_action()
        return action

    def create_action_list(self):
        action_list = []

        # Add Sleep action without parameters
        action_list.append(BlueAction(Sleep))
        
        # Basic actions with parameters
        default_params = {
            'session': 0,
            'agent': self.agent_name
        }

        # Add Monitor action
        action_list.append(BlueAction(CybORGMonitor, default_params.copy()))

        # Actions with hostnames
        H = 16  # Number of hosts
        for i in range(H):
            hostname = f'Host_{i}'
            host_params = default_params.copy()
            host_params['hostname'] = hostname
            
            action_list.append(BlueAction(Analyse, host_params.copy()))
            action_list.append(BlueAction(Remove, host_params.copy()))
            action_list.append(BlueAction(Restore, host_params.copy()))

        return action_list

    def get_action_space_size(self):
        return len(self.action_list)

    def calculate_reward(self, result):
        """Improved reward function"""
        reward = result.reward
        observation = result.observation

        # Base reward for taking action
        reward += 0.1  # Small positive reward to encourage exploration
        
        # Reward for maintaining system health
        compromised_hosts = 0
        for i in range(16):
            host_key = f'host_{i}'
            if host_key in observation and 'Compromised' in observation[host_key]:
                compromised_hosts += 1
        
        # Penalize based on number of compromised hosts
        if compromised_hosts == 0:
            reward += 1.0  # Bonus for keeping all hosts safe
        else:
            reward -= compromised_hosts * 0.5  # Smaller penalty per compromised host

        return reward

    def render(self, mode='human'):
        pass

# Instantiate the environment
env = BlueCybORGEnv()
env = SB3Monitor(env)

# Instantiate the RL model
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1,
    ent_coef=0.05,  # Increased for more exploration
    learning_rate=0.0001,  # Reduced for more stable learning
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
)

# Train the model
total_timesteps = 5000000  # More timesteps for better learning
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save('ppo_blue_agent_model')

# Evaluate the model
episodes = 10
for episode in range(episodes):
    # Fix: Only take first element (observation) from reset() tuple
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Added deterministic=True for evaluation
        # Fix: Handle both terminated and truncated conditions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f'Episode {episode + 1}: Total Reward = {total_reward}')

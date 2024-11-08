import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor as SB3Monitor
from gymnasium import spaces
import torch

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Simulator.Actions import Monitor as CybORGMonitor
from CybORG.Simulator.Actions import Sleep, Analyse, Remove, Restore

from my_blue_agent import MyBlueAgent, BlueAction

# Define our own learning rate schedule function
def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
        final_value: Final learning rate
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end)
        """
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule

class BlueCybORGEnv(gym.Env):
    def __init__(self):
        super(BlueCybORGEnv, self).__init__()

        # Initialize with a larger step limit in order avoid the error
        max_steps = 500  # Episode length
        scenario_steps = 10000  # Much larger than episode length
        sg = EnterpriseScenarioGenerator(steps=scenario_steps) 
        self.cyborg = CybORG(scenario_generator=sg)
        self.agent_name = 'blue_agent_0'

        # Create action list
        self.action_list = self.create_action_list()
        
        # define observations and action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(100,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.get_action_space_size())

        # Initialize agent AFTER spaces are defined
        self.blue_agent = MyBlueAgent(action_space=self.action_space, 
                                    observation_space=self.observation_space)

        self.current_step = 0
        self.max_steps = max_steps
        self.episode_rewards = []  # Track rewards for curriculum learning

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

        # Add Slerep action without thge parameters
        action_list.append(BlueAction(Sleep))
        
        # Basic actions with parameters
        default_params = {
            'session': 0,
            'agent': self.agent_name
        }

        # Add Monitoring
        action_list.append(BlueAction(CybORGMonitor, default_params.copy()))

        # Actions need  hostnames
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
        reward = result.reward
        observation = result.observation

        # Base reward with random noise for exploration
        reward += 0.1 + np.random.normal(0, 0.01)  # Small random noise

        # System health assessment
        compromised_hosts = 0
        total_hosts = 16
        for i in range(total_hosts):
            host_key = f'host_{i}'
            if host_key in observation and 'Compromised' in observation[host_key]:
                compromised_hosts += 1

        # Non-linear reward scaling
        health_ratio = (total_hosts - compromised_hosts) / total_hosts
        reward += np.power(health_ratio, 2) * 3.0  # Quadratic scaling

        # Dynamic reward system
        if compromised_hosts == 0:
            reward += 5.0 * (1.0 + np.random.uniform(0, 0.1))  # Variable perfect defense bonus
        elif compromised_hosts < 3:
            reward += 2.0
        elif compromised_hosts < total_hosts / 2:
            reward += 0.5

        # Progressive penalties with randomness
        if compromised_hosts > total_hosts / 2:
            penalty = compromised_hosts * compromised_hosts / total_hosts
            penalty *= (1.0 + np.random.uniform(0, 0.2))  # Variable penalty
            reward -= penalty

        # Dynamic time pressure
        time_factor = np.sin(np.pi * self.current_step / self.max_steps)  # Sinusoidal time pressure
        reward *= (1.0 + 0.5 * time_factor)

        # Scale rewards
        reward = reward / 30.0  # Adjusted scaling factor

        return reward

    def render(self, mode='human'):
        pass

# Instantiate the environment
env = BlueCybORGEnv()
env = SB3Monitor(env)

# Enhanced PPO configuration with aggressive exploration
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1,
    ent_coef=0.5,          # Much higher entropy for forced exploration
    learning_rate=linear_schedule(0.003, 0.0001),  # Higher initial learning rate
    n_steps=1024,          # Shorter rollouts
    batch_size=32,         # Even smaller batch size
    n_epochs=5,            # Fewer epochs but more frequent updates
    gamma=0.99,            # Standard discount
    clip_range=0.5,        # Even larger clip range
    vf_coef=0.8,          # Adjusted value function coefficient
    normalize_advantage=True,
    target_kl=0.02,        # Target KL divergence
    max_grad_norm=1.0,     # Keep gradient clipping
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Deeper policy network
            vf=[512, 256, 128]   # Much larger value network
        ),
        activation_fn=torch.nn.ReLU,
        ortho_init=True
    )
)

# Increased training time
total_timesteps = 300000  # 50% more training time

# Training with evaluation callbacks
from stable_baselines3.common.callbacks import EvalCallback

# Create evaluation environment
eval_env = BlueCybORGEnv()
eval_env = SB3Monitor(eval_env)

# Set up evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./logs/',
    log_path='./logs/',
    eval_freq=1000,        # Even more frequent evaluation
    n_eval_episodes=15,    # More evaluation episodes
    deterministic=True,
    render=False
)

# Add training stability callback
from stable_baselines3.common.callbacks import CheckpointCallback

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=2000,        # More frequent checkpoints
    save_path="./logs/",
    name_prefix="ppo_blue_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# Combine callbacks
callbacks = [eval_callback, checkpoint_callback]

# Train with both callbacks
model.learn(
    total_timesteps=total_timesteps,
    callback=callbacks
)

# Enhanced evaluation
episodes = 20  # Increased from 10 for better evaluation
eval_rewards = []

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
    eval_rewards.append(total_reward)
    print(f'Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}')

# Print evaluation statistics
print(f'\nEvaluation Statistics:')
print(f'Average Reward: {np.mean(eval_rewards):.2f}')
print(f'Std Dev Reward: {np.std(eval_rewards):.2f}')
print(f'Min Reward: {np.min(eval_rewards):.2f}')
print(f'Max Reward: {np.max(eval_rewards):.2f}')

# Save the model
model.save('ppo_blue_agent_model')

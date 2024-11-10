# my_blue_agent.py

import numpy as np
from gymnasium import spaces
from CybORG.Shared.AgentInterface import AgentInterface
from CybORG.Simulator.Actions import *
from CybORG.Shared.Observation import Observation

class BlueAction:
    def __init__(self, action_class, params=None):
        self.action_class = action_class
        # Set default parameters required for all actions
        self.params = {
            'session': 0,
            'agent': 'blue_agent_0'
        }
        # Update with any additional parameters
        if params:
            self.params.update(params)

    def get_action(self):
        if self.action_class == Sleep:
            # Sleep action doesn't need any parameters
            return self.action_class()
        else:
            # Other actions use all parameters
            return self.action_class(**self.params)

class MyBlueAgent(AgentInterface):
    def __init__(self, action_space, observation_space):
        # Create a simple agent object with required methods
        class AgentObj:
            def set_initial_values(self, *args, **kwargs):
                pass
            
            def get_action(self, *args, **kwargs):
                return Sleep()

        # Initialize with required arguments for AgentInterface
        agent_obj = AgentObj()
        agent_name = 'blue_agent_0'
        actions = [Sleep, Monitor, Analyse, Remove, Restore]  # List of available actions
        allowed_subnets = ['User', 'Enterprise']  # Default allowed subnets
        scenario = None  # Will be set by environment
        
        super().__init__(agent_obj=agent_obj,
                        agent_name=agent_name,
                        actions=actions,
                        allowed_subnets=allowed_subnets,
                        scenario=scenario)
        
        # Store RL-specific spaces
        self.action_space = action_space
        self.observation_space = observation_space

    def get_action(self, observation: Observation):
        # This method is not used during training with Stable Baselines3
        pass

    def process_observation(self, observation):
        # Process the observation dictionary into a numpy array
        obs_vector = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Example processing (you'll need to adjust based on actual observation structure)
        # Index 0: Mission Phase
        mission_phase = observation.get('mission_phase', 0)
        obs_vector[0] = mission_phase / 2  # Normalize if mission phase ranges from 0 to 2

        # Indices 1 to H: Malicious activity detected on hosts
        # Assuming H is the number of hosts (e.g., 16)
        H = 16
        for i in range(H):
            host_key = f'host_{i}'
            if host_key in observation and 'Compromised' in observation[host_key]:
                obs_vector[i + 1] = 1.0  # Mark as compromised

        return obs_vector

    def end_episode(self):
        # Reset any necessary variables at the end of an episode
        pass

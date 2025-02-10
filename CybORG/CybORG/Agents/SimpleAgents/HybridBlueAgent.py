import copy

from CybORG.Agents.SimpleAgents.PPOAgent import PPOAgent
from CybORG.Agents.SimpleAgents.SleepAgent import SleepAgent
import numpy as np
import os
import torch

class HybridBlueAgent(PPOAgent):
    def __init__(self):
        self.action_space = [
            133, 134, 135, 139,  # restore enterprise and opserver
            3, 4, 5, 9,          # analyse enterprise and opserver
            16, 17, 18, 22       # remove enterprise and opserver
        ]
        self.agent = None
        self.start_actions = [1004, 1004, 1000]  # Initial decoy placements
        self.end_episode()

    def get_action(self, observation, action_space=None):
        if self.agent is None:
            self.agent = self.load_bline()
        return self.agent.get_action(observation, action_space)

    def end_episode(self):
        if self.agent is not None:
            self.agent.end_episode()

    def set_initial_values(self, action_space, observation):
        if self.agent is not None:
            self.agent.set_initial_values(self.action_space, observation)

    def load_sleep(self):
        return SleepBlueAgent()

    def load_bline(self):
        """Load or create PPO agent with proper initialization"""
        # Define model paths
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        models_path = os.path.join(base_path, "Models", "bline")
        ckpt = os.path.join(models_path, "model.pth")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        
        # Check if model exists
        if os.path.exists(ckpt):
            print(f"Loading model from {ckpt}")
            return PPOAgent(52, self.action_space, restore=True, ckpt=ckpt,
                          deterministic=True, training=False, start_actions=self.start_actions)
        else:
            print(f"No model found at {ckpt}, creating new agent")
            # Create new agent
            agent = PPOAgent(52, self.action_space, restore=False,
                           deterministic=True, training=False, start_actions=self.start_actions)
            
            # Initialize networks
            agent.set_initial_values(self.action_space)
            
            # Save initial model
            torch.save(agent.policy.state_dict(), ckpt)
            print(f"Saved initial model to {ckpt}")
            
            return agent

    def load_meander(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ckpt = os.path.join(base_dir, "Models", "meander", "model.pth")
        return PPOAgent(52, self.action_space, restore=True, ckpt=ckpt,
                       deterministic=True, training=False)

    def fingerprint_meander(self):
        return np.sum(self.scan_state) == 3

    def fingerprint_bline(self):
        return np.sum(self.scan_state) == 2

    def end_episode(self):
        self.scan_state = np.zeros(10)
        self.start_actions = [51, 116, 55]
        self.agent_loaded = False




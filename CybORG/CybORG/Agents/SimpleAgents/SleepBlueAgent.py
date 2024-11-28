from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

class SleepBlueAgent(BaseAgent):
    """A simple agent that does nothing (sleeps)"""
    
    def __init__(self):
        self.action_space = None
        
    def get_action(self, observation, action_space=None):
        """Always return the 'Sleep' action"""
        if action_space is None:
            return 0  # Sleep action
        
        # Find sleep action in action space
        sleep_action = 0  # Default to first action if Sleep not found
        for action_id, action in enumerate(action_space):
            if 'Sleep' in str(action):
                sleep_action = action_id
                break
                
        return sleep_action
        
    def end_episode(self):
        """Reset agent state at episode end"""
        pass
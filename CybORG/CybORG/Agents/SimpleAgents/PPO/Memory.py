class Memory:
    """Efficient memory management for PPO"""
    def __init__(self, buffer_size=20000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []
        self.buffer_size = buffer_size
        
    def clear_memory(self):
        """Efficiently clear memory"""
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]
        
    def add(self, state, action, reward, done, logprob):
        """Add experience with buffer management"""
        if len(self.states) >= self.buffer_size:
            self.clear_memory()
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.logprobs.append(logprob)
    
    def get_buffer(self):
        """Get current buffer state"""
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'logprobs': self.logprobs,
            'is_terminals': self.is_terminals
        }
        
    def load_buffer(self, buffer_data):
        """Load buffer state"""
        self.states = buffer_data.get('states', [])
        self.actions = buffer_data.get('actions', [])
        self.rewards = buffer_data.get('rewards', [])
        self.logprobs = buffer_data.get('logprobs', [])
        self.is_terminals = buffer_data.get('is_terminals', [])
    
    def is_full(self):
        """Check if buffer is full"""
        return len(self.states) >= self.buffer_size
    
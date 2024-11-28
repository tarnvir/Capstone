class Memory:
    def __init__(self, buffer_size=2048):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminals = []
        self.buffer_size = buffer_size

    def clear_memory(self):
        """Clear all memory buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminals = []
    
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
    
class Memory:
    def __init__(self, buffer_size=2048):
        """Initialize memory buffer for storing experiences
        Args:
            buffer_size: Maximum number of experiences to store (default: 2048)
        """
        # Fixed size buffer for experience replay
        self.buffer_size = buffer_size
        
        # Lists to store different components of experiences
        self.states = []          # Environment states
        self.actions = []         # Actions taken
        self.logprobs = []        # Log probabilities of actions
        self.rewards = []         # Rewards received
        self.is_terminals = []    # Episode termination flags
        self.next_states = []     # Next states
        self.dones = []          # Done flags
        self.values = []          # Values
        
    def add(self, state, action, logprob, reward, is_terminal):
        """Add new experience to memory
        Args:
            state: Environment state
            action: Action taken
            logprob: Log probability of action
            reward: Reward received
            is_terminal: Whether this is a terminal state
        """
        # Add new experience
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.next_states.append(state)
        self.dones.append(is_terminal)
        self.values.append(reward)
        
        # Maintain fixed buffer size by removing oldest experiences
        if len(self.states) > self.buffer_size:
            self.states = self.states[-self.buffer_size:]
            self.actions = self.actions[-self.buffer_size:]
            self.logprobs = self.logprobs[-self.buffer_size:]
            self.rewards = self.rewards[-self.buffer_size:]
            self.is_terminals = self.is_terminals[-self.buffer_size:]
            self.next_states = self.next_states[-self.buffer_size:]
            self.dones = self.dones[-self.buffer_size:]
            self.values = self.values[-self.buffer_size:]
    
    def clear_memory(self):
        """Clear all experiences from memory"""
        # Reset all experience lists
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_states[:]
        del self.dones[:]
        del self.values[:]
    
    def clear(self):
        """Clear all memory buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.logprobs = []
        self.values = []
    
    def __len__(self):
        return len(self.states)
    
    def is_full(self):
        """Check if memory buffer is full"""
        return len(self) >= self.buffer_size
    
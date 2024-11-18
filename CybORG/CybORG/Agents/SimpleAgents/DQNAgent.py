import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Store transition with proper state conversion"""
        # Convert states to numpy arrays if they aren't already
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
            
        # Ensure states have consistent shape
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(next_state.shape) == 1:
            next_state = next_state.reshape(1, -1)
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample batch with consistent state shapes"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        
        # Stack states with consistent shape
        states = np.vstack([s.reshape(1, -1) if len(s.shape) == 1 else s for s in states])
        next_states = np.vstack([s.reshape(1, -1) if len(s.shape) == 1 else s for s in next_states])
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = 140  # Increase to handle all possible actions (including restore actions)
        
        # Networks with proper output dimension
        self.q_network = DQNNetwork(input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_network = DQNNetwork(input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Reduced buffer size and batch size
        self.replay_buffer = ReplayBuffer(1000)
        self.min_replay_size = 32
        self.batch_size = 64
        
        # Other parameters
        self.gamma = 0.95
        self.lr = 1e-4
        self.target_update = 100
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        self.update_counter = 0
        
        # Full action mapping
        self.action_groups = {
            'restore': [133, 134, 135, 139],
            'analyze': [3, 4, 5, 9],
            'remove': [16, 17, 18, 22],
            'monitor': [0, 1]
        }
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
    
    def store_transition(self, state, action, reward, next_state, done):
        # Shape rewards based on action sequences
        shaped_reward = reward
        
        # Track action history for sequences
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(action)
        
        # Reward shaping for sequences
        if len(self.action_history) >= 3:
            last_three = self.action_history[-3:]
            
            # Monitor -> Analyze -> Restore sequence
            if (last_three[0] == 1 and  # Monitor
                last_three[1] in [3,4,5,9] and  # Analyze
                last_three[2] in [133,134,135,139]):  # Restore
                shaped_reward += 5.0  # Bonus for complete sequence
                
                # Extra bonus for enterprise systems
                if last_three[1] in [3,4,5]:
                    shaped_reward += 2.0
        
        # Clear history if done
        if done:
            self.action_history = []
        
        # Store transition with shaped reward
        self.replay_buffer.push(state, action, shaped_reward, next_state, done)
    
    def update(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return
            
        try:
            # Sample and prepare batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            # Convert to tensors with proper shapes
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Get current Q values
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Get next Q values
            with torch.no_grad():
                next_q = self.target_network(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q
            
            # Update network
            loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network
            self.update_counter += 1
            if self.update_counter % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        except Exception as e:
            print(f"DQN update error: {str(e)}")
            return
    
    def get_action(self, observation, action_space=None):
        """Select action using epsilon-greedy policy with forced sequences"""
        if hasattr(self, 'action_history') and len(self.action_history) > 0:
            last_action = self.action_history[-1]
            
            # Force analyze after monitor
            if last_action == 1:
                analyze_actions = [3,4,5,9]
                valid_analyzes = [a for a in analyze_actions if a in action_space]
                if valid_analyzes:
                    return np.random.choice(valid_analyzes)
                    
            # Force restore after analyze
            analyze_to_restore = {3:133, 4:134, 5:135, 9:139}
            if last_action in analyze_to_restore and analyze_to_restore[last_action] in action_space:
                return analyze_to_restore[last_action]
        
        # Regular epsilon-greedy selection
        if np.random.random() < self.epsilon:
            return np.random.choice(action_space) if action_space else np.random.randint(self.output_dim)
        
        # Q-value based selection
        state = torch.FloatTensor(observation).reshape(1, -1)[:, :self.input_dim].to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
            if action_space is not None:
                mask = torch.zeros(self.output_dim, device=self.device)
                mask[list(action_space)] = 1
                q_values = q_values.masked_fill(mask.eq(0), float('-inf'))
            return q_values.argmax(1).item()
    
    def save(self, path):
        """Save the DQN model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load the DQN model"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
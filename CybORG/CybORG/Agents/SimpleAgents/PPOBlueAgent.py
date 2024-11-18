import torch
import torch.nn as nn
import numpy as np
from collections import Counter, defaultdict

class PPOBlueAgent:
    def __init__(self, input_dim=52, hidden_dim=512, output_dim=None):
        """Initialize PPO Blue Agent
        Args:
            input_dim: Size of observation space (default: 52)
            hidden_dim: Size of hidden layers (default: 512)
            output_dim: Size of action space (optional)
        """
        # Set device to GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Network architecture parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # PPO hyperparameters - tuned for better performance
        self.gamma = 0.97          # Discount factor - shorter horizon
        self.gae_lambda = 0.95     # GAE parameter
        self.clip_range = 0.2      # PPO clipping parameter
        self.value_coef = 0.5      # Value loss coefficient
        self.entropy_coef = 0.3    # Start with higher exploration
        self.learning_rate = 5e-5  # Lower learning rate for stability
        
        # Action selection parameters
        self.exploit_threshold = 0.7  # Use successful actions above this rate
        self.explore_rate = 0.2       # Random exploration rate
        self.sequence_use_rate = 0.4  # Rate to use successful sequences
        
        # Action masking parameters
        self.action_mask_threshold = 0.1
        self.success_threshold = 0.3
        
        # Exploration decay parameters
        self.min_entropy_coef = 0.01
        self.entropy_decay = 0.995
        
        # Batch parameters
        self.min_batch_size = 32
        self.update_epochs = 10
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.learning_rate,
            eps=1e-5,
            weight_decay=1e-6
        )
        
        # Memory buffers for experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # Performance tracking metrics
        self.training_step = 0
        self.episode_rewards = []
        self.total_actions = Counter()
        self.running_reward = 0
        
        # Action statistics tracking
        self.action_stats = {
            'success_rate': {},           # Track success rate per action
            'reward_history': defaultdict(list),  # Track rewards per action
            'usage_count': defaultdict(int),      # Track action usage
            'last_success': defaultdict(int)      # Track last successful use
        }
        
        # Group actions by type for better strategy
        self.action_groups = {
            'restore': [133, 134, 135, 139],  # Restore actions
            'analyse': [3, 4, 5, 9],          # Analysis actions
            'remove': [16, 17, 18, 22],       # Remove malware actions
            'monitor': [0]                    # Basic monitoring
        }
        
        # Prioritize action groups
        self.group_weights = {
            'restore': 0.7,     # Highest priority
            'analyse': 0.2,     # Second priority
            'remove': 0.08,     # Lower priority
            'monitor': 0.02,    # Lowest priority
        }
        
        # System priority weights
        self.system_priorities = {
            'enterprise0': 0.4,  # Highest priority
            'enterprise1': 0.3,
            'enterprise2': 0.2,
            'opserver0': 0.1    # Lowest priority
        }
        
        # Track success per action group
        self.group_success = {group: [] for group in self.action_groups.keys()}
        
        # Track success streaks for bonus rewards
        self.success_streaks = defaultdict(int)
        self.max_streak_bonus = 2.0
"""
Deep Q-Network (DQN) Agent
==========================

A DQN agent that learns to solve CartPole using:
- Neural network for Q-value approximation
- Experience replay buffer
- Target network for stability

Author: Educational RL Project
"""

import numpy as np
import random
import os

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. Run: pip install torch")


# =============================================================================
# NEURAL NETWORK
# =============================================================================

if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """
        Neural network that approximates Q-values.
        
        Input: State (4 numbers for CartPole)
        Output: Q-values for each action (2 for CartPole)
        """
        
        def __init__(self, state_size=4, action_size=2, hidden_size=24):
            """
            Initialize the Q-network.
            
            Architecture:
                Input (4) → Hidden1 (24) → ReLU → Hidden2 (24) → ReLU → Output (2)
            
            Args:
                state_size: Dimension of state space
                action_size: Number of actions
                hidden_size: Size of hidden layers
            """
            super(QNetwork, self).__init__()
            
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
        
        def forward(self, state):
            """
            Forward pass through the network.
            
            Args:
                state: Input state (batch_size, state_size)
                
            Returns:
                Q-values for each action (batch_size, action_size)
            """
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)


# =============================================================================
# EXPERIENCE REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer to store and sample experiences.
    
    Why we need this:
    - Breaks correlation between consecutive experiences
    - Allows reusing experiences multiple times
    - More efficient learning
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


# =============================================================================
# DQN AGENT
# =============================================================================

class DQNAgent:
    """
    DQN Agent with experience replay and target network.
    
    Key components:
    1. Q-Network: Predicts Q-values
    2. Target Network: Provides stable targets
    3. Replay Buffer: Stores experiences for learning
    """
    
    def __init__(
        self,
        state_size=4,
        action_size=2,
        hidden_size=24,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state
            action_size: Number of actions
            hidden_size: Hidden layer size
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update: How often to update target network
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Run: pip install torch")
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training counter
        self.steps_done = 0
    
    def choose_action(self, state, training=True):
        """
        Choose an action using ε-greedy policy.
        
        Args:
            state: Current state
            training: If True, use exploration
            
        Returns:
            action: Selected action
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)
        
        # Exploit: best action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def learn(self):
        """
        Learn from a batch of experiences.
        
        This is the core DQN training step.
        """
        # Don't learn if buffer is too small
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath):
        """Save the model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model."""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
        print(f"✅ Model loaded from {filepath}")
        return True


# =============================================================================
# TEST
# =============================================================================

def test_agent():
    """Test DQN agent components."""
    print("Testing DQN Agent")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not installed!")
        return
    
    # Create agent
    agent = DQNAgent()
    
    print(f"\nDevice: {agent.device}")
    print(f"Network architecture:\n{agent.q_network}")
    
    # Test Q-network
    state = np.array([0.1, 0.2, -0.05, 0.3])
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = agent.q_network(state_tensor)
    
    print(f"\nTest state: {state}")
    print(f"Q-values: {q_values.detach().numpy()}")
    print(f"Best action: {q_values.argmax().item()}")
    
    # Test replay buffer
    print("\nTesting replay buffer...")
    for _ in range(10):
        agent.memory.push(state, 0, 1.0, state, False)
    
    print(f"Buffer size: {len(agent.memory)}")


if __name__ == '__main__':
    test_agent()
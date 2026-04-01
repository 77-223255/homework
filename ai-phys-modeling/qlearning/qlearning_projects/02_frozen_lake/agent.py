"""
Q-Learning Agent for Frozen Lake
=================================

A simple Q-learning agent that learns to cross the frozen lake.

Key difference from Tic-Tac-Toe:
- States are simple numbers (0-15) instead of board strings
- Environment is stochastic (actions might slip)

Author: Educational RL Project
"""

import numpy as np
import pickle
import os


class QLearningAgent:
    """
    A Q-learning agent for Frozen Lake.
    
    Attributes:
        n_states: Number of states (16 for 4x4 grid)
        n_actions: Number of actions (4: left, down, right, up)
        learning_rate: How fast to learn
        discount_factor: How much future rewards matter
        exploration_rate: Probability of random exploration
        q_table: Dictionary mapping state to action Q-values
    """
    
    def __init__(
        self, 
        n_states=16, 
        n_actions=4,
        learning_rate=0.1, 
        discount_factor=0.99, 
        exploration_rate=1.0,
        exploration_decay=0.9995,
        min_exploration=0.01
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            n_states: Number of states in environment
            n_actions: Number of possible actions
            learning_rate: Learning rate (α)
            discount_factor: Discount factor (γ)
            exploration_rate: Initial exploration rate (ε)
            exploration_decay: How much ε decreases each episode
            min_exploration: Minimum exploration rate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Initialize Q-table with zeros
        # q_table[state] = {action: q_value}
        self.q_table = {}
        for state in range(n_states):
            self.q_table[state] = {a: 0.0 for a in range(n_actions)}
        
        # Statistics
        self.total_steps = 0
        self.wins = 0
        self.losses = 0
    
    def choose_action(self, state, training=True):
        """
        Choose an action using ε-greedy policy.
        
        Args:
            state: Current state (0-15)
            training: If True, use exploration
            
        Returns:
            action: Chosen action (0-3)
        """
        # Exploration: random action
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(self.n_actions)
        
        # Exploitation: best action
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        
        # Get all actions with max Q-value (might be ties)
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        # Random tie-break
        return np.random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table based on experience.
        
        Q-Learning Formula:
            Q(s,a) = Q(s,a) + α × [r + γ × max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: reward + discounted future
            max_next_q = max(self.q_table[next_state].values())
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_exploration(self):
        """
        Decrease exploration rate.
        
        This is important for Frozen Lake because:
        - Early training: need lots of exploration to find goal
        - Later training: exploit the learned path
        """
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
    
    def record_result(self, won):
        """Record game result."""
        if won:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_stats(self):
        """Get learning statistics."""
        total = self.wins + self.losses
        win_rate = self.wins / total * 100 if total > 0 else 0
        
        return {
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'exploration_rate': self.exploration_rate
        }
    
    def print_stats(self):
        """Print current statistics."""
        stats = self.get_stats()
        print(f"\n📊 Agent Statistics:")
        print(f"   Wins: {stats['wins']}")
        print(f"   Losses: {stats['losses']}")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
        print(f"   Exploration Rate: {stats['exploration_rate']:.3f}")
    
    def save(self, filepath):
        """Save the Q-table to a file."""
        data = {
            'q_table': self.q_table,
            'exploration_rate': self.exploration_rate,
            'stats': {'wins': self.wins, 'losses': self.losses}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load a Q-table from a file."""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.exploration_rate = data['exploration_rate']
        self.wins = data['stats']['wins']
        self.losses = data['stats']['losses']
        
        print(f"✅ Agent loaded from {filepath}")
        return True


# =============================================================================
# TEST
# =============================================================================

def test_agent():
    """Test the Q-learning agent."""
    print("Testing Frozen Lake Q-Learning Agent")
    print("=" * 50)
    
    agent = QLearningAgent()
    
    # Simulate learning
    print("\nSimulating Q-learning updates...")
    
    # State 14 (one below goal), take action RIGHT (2) to reach goal
    print("\nBefore learning:")
    print(f"  Q(14, RIGHT) = {agent.q_table[14][2]:.3f}")
    
    # Learn: from state 14, go right, reach goal, get reward 1
    agent.learn(14, 2, 1.0, 15, done=True)
    
    print("After learning (reached goal from state 14 with RIGHT):")
    print(f"  Q(14, RIGHT) = {agent.q_table[14][2]:.3f}")
    
    # Learn backward propagation
    print("\nPropagating reward backward...")
    
    # State 13, go RIGHT to state 14 (which now has high Q)
    agent.learn(13, 2, 0.0, 14, done=False)
    print(f"  Q(13, RIGHT) = {agent.q_table[13][2]:.3f}")
    
    # State 10, go RIGHT to state 11, then can reach 14
    agent.learn(10, 2, 0.0, 11, done=False)
    print(f"  Q(10, RIGHT) = {agent.q_table[10][2]:.3f}")


if __name__ == '__main__':
    test_agent()
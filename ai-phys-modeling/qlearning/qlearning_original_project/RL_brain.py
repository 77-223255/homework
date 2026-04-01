"""
Q-Learning Brain for Reinforcement Learning Tutorial
======================================================

This implements a Q-learning agent that learns to make decisions
by building a Q-table of state-action values.

Key Concepts:
- Q(s, a): The expected reward for taking action 'a' in state 's'
- Learning: Update Q-values based on experience
- Policy: Choose the action with highest Q-value (usually)

Author: Improved version for teaching
"""

import numpy as np
import pandas as pd


# =============================================================================
# Q-LEARNING AGENT CLASS
# =============================================================================

class QLearningAgent:
    """
    A Q-learning agent that learns from experience.
    
    The agent maintains a Q-table that maps (state, action) pairs
    to expected rewards. It learns by trying actions and updating
    the Q-table based on results.
    
    Attributes:
        actions: List of possible actions
        learning_rate: How fast to learn (0 to 1)
        discount_factor: How much to value future rewards (0 to 1)
        exploration_rate: Probability of random exploration (0 to 1)
        q_table: The table storing Q-values
    """
    
    def __init__(
        self, 
        actions, 
        learning_rate=0.01, 
        discount_factor=0.9, 
        exploration_rate=0.9
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            actions: List of possible actions (e.g., [0, 1, 2, 3])
            learning_rate: How much to trust new experiences (α)
                          - High (0.5+): Learn fast, but may be unstable
                          - Low (0.01): Learn slowly, but more stable
            discount_factor: How much to care about future rewards (γ)
                            - High (0.9+): Plan ahead, value future rewards
                            - Low (0.1): Focus on immediate rewards
            exploration_rate: How often to try random actions (ε)
                             - High (0.9): Exploit (use best known action)
                             - Low (0.1): Explore (try random actions)
        """
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Create empty Q-table
        # Rows = states, Columns = actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
        # Statistics for tracking progress
        self.total_updates = 0
        self.states_visited = 0
    
    # =========================================================================
    # CORE METHODS
    # =========================================================================
    
    def choose_action(self, state):
        """
        Choose an action based on the current state.
        
        Uses ε-greedy strategy:
        - With probability ε: Choose the best known action (exploit)
        - Otherwise: Choose a random action (explore)
        
        Args:
            state: Current state (any hashable type, usually string)
            
        Returns:
            action: The chosen action
        """
        # Make sure this state exists in Q-table
        self._ensure_state_exists(state)
        
        # ε-greedy action selection
        if np.random.uniform() < self.exploration_rate:
            # EXPLOIT: Choose best action
            action = self._get_best_action(state)
        else:
            # EXPLORE: Choose random action
            action = np.random.choice(self.actions)
        
        return action
    
    def learn(self, state, action, reward, next_state):
        """
        Learn from one experience (update Q-table).
        
        Q-Learning Formula:
            Q(s, a) = Q(s, a) + α × [r + γ × max(Q(s', a')) - Q(s, a)]
        
        This means:
        1. Predict: What we currently think Q(s,a) is worth
        2. Target: What we actually got (reward + discounted future)
        3. Update: Adjust Q(s,a) toward target
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received (-1, 0, or +1)
            next_state: State after taking action
        """
        # Make sure both states exist in Q-table
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        
        # Current Q-value (our prediction)
        current_q = self.q_table.loc[state, action]
        
        # Calculate target Q-value
        if next_state == 'terminal':
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: reward + best future Q-value
            best_future_q = self.q_table.loc[next_state].max()
            target_q = reward + self.discount_factor * best_future_q
        
        # Update Q-value (move toward target)
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table.loc[state, action] = new_q
        
        # Track statistics
        self.total_updates += 1
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _ensure_state_exists(self, state):
        """
        Add a state to Q-table if it doesn't exist.
        
        New states are initialized with Q-value = 0 for all actions.
        
        Args:
            state: State to check/add
        """
        if state not in self.q_table.index:
            # Add new row with zeros
            self.q_table.loc[state] = [0.0] * len(self.actions)
            self.states_visited += 1
    
    def _get_best_action(self, state):
        """
        Get the action with highest Q-value for a state.
        
        If multiple actions have the same Q-value, choose randomly
        among them (to encourage exploration).
        
        Args:
            state: The state to check
            
        Returns:
            action: The best action for this state
        """
        # Get Q-values for this state
        state_q_values = self.q_table.loc[state]
        
        # Shuffle to break ties randomly
        shuffled = state_q_values.sample(frac=1)
        
        # Return action with highest Q-value
        return shuffled.idxmax()
    
    # =========================================================================
    # VISUALIZATION METHODS (For Teaching)
    # =========================================================================
    
    def print_q_table(self, title="Current Q-Table"):
        """
        Print the Q-table in a readable format.
        
        Args:
            title: Title to display above the table
        """
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print(f"{'=' * 50}")
        
        if self.q_table.empty:
            print("  (No states visited yet)")
            return
        
        # Print header
        header = "State      |"
        for action in self.actions:
            action_name = self._action_name(action)
            header += f" {action_name:>7} |"
        print(header)
        print("-" * len(header))
        
        # Print each state
        for state in self.q_table.index:
            row = f"{state:>10} |"
            for action in self.actions:
                q_value = self.q_table.loc[state, action]
                row += f" {q_value:>7.3f} |"
            print(row)
        
        print(f"{'=' * 50}\n")
    
    def print_best_actions(self):
        """Print the best action for each known state."""
        print("\n🎯 Best Actions for Each State:")
        print("-" * 40)
        
        if self.q_table.empty:
            print("  (No states visited yet)")
            return
        
        for state in self.q_table.index:
            best_action = self._get_best_action(state)
            best_q = self.q_table.loc[state, best_action]
            action_name = self._action_name(best_action)
            print(f"  {state}: {action_name} (Q = {best_q:.3f})")
        
        print("-" * 40)
    
    def get_summary(self):
        """
        Get a summary of the agent's learning.
        
        Returns:
            dict: Statistics about the agent
        """
        return {
            'states_discovered': len(self.q_table.index),
            'total_updates': self.total_updates,
            'table_size': f"{len(self.q_table.index)} × {len(self.actions)}",
            'has_learned': len(self.q_table.index) > 0
        }
    
    def _action_name(self, action):
        """Convert action index to readable name."""
        names = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
        return names.get(action, str(action))
    
    # =========================================================================
    # SERIALIZATION METHODS (Save/Load)
    # =========================================================================
    
    def save_q_table(self, filepath):
        """
        Save Q-table to a CSV file.
        
        Args:
            filepath: Path to save file
        """
        self.q_table.to_csv(filepath)
        print(f"✅ Q-table saved to {filepath}")
    
    def load_q_table(self, filepath):
        """
        Load Q-table from a CSV file.
        
        Args:
            filepath: Path to load file
        """
        self.q_table = pd.read_csv(filepath, index_col=0)
        print(f"✅ Q-table loaded from {filepath}")


# =============================================================================
# BACKWARD COMPATIBILITY - Old class name still works
# =============================================================================

class QLearningTable(QLearningAgent):
    """
    Backward compatible wrapper.
    
    This allows old code using 'QLearningTable' to work
    with the improved QLearningAgent class.
    """
    
    def check_state_exist(self, state):
        """Old method name - now calls _ensure_state_exists."""
        self._ensure_state_exists(state)


# =============================================================================
# TEST FUNCTION - Run this file directly to test
# =============================================================================

def test_q_learning():
    """Test the Q-learning agent with a simple example."""
    
    print("Testing Q-Learning Agent")
    print("=" * 50)
    
    # Create agent
    agent = QLearningAgent(
        actions=[0, 1, 2, 3],  # up, down, right, left
        learning_rate=0.5,     # Fast learning for demo
        discount_factor=0.9,
        exploration_rate=0.8
    )
    
    print("\nInitial Q-table:")
    agent.print_q_table("Initial State (Empty)")
    
    # Simulate some experiences
    experiences = [
        # (state, action, reward, next_state)
        ("(0,0)", 2, -1, "terminal"),  # Move right into hell
        ("(0,0)", 1, 0, "(1,0)"),      # Move down safely
        ("(1,0)", 2, 1, "terminal"),   # Move right to goal
        ("(0,0)", 1, 0, "(1,0)"),      # Move down again
        ("(1,0)", 2, 1, "terminal"),   # Move right to goal
    ]
    
    print("\nSimulating learning experiences...")
    for i, (state, action, reward, next_state) in enumerate(experiences):
        action_name = agent._action_name(action)
        print(f"\nExperience {i + 1}:")
        print(f"  State: {state}, Action: {action_name}, Reward: {reward}")
        
        agent.learn(state, action, reward, next_state)
        
        print(f"  Updated Q({state}, {action_name}) = {agent.q_table.loc[state, action]:.3f}")
    
    print("\nFinal Q-table:")
    agent.print_q_table("After Learning")
    
    print("\nBest actions:")
    agent.print_best_actions()
    
    print("\nSummary:")
    summary = agent.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    test_q_learning()
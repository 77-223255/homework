"""
Q-Learning Agent for Tic-Tac-Toe
=================================

This agent learns to play Tic-Tac-Toe by building a Q-table
that maps (board_state, action) pairs to expected rewards.

Author: Educational RL Project
"""

import numpy as np
import pandas as pd
import pickle
import os


class QLearningAgent:
    """
    A Q-learning agent for Tic-Tac-Toe.
    
    The agent learns by playing games and updating its Q-table
    based on the outcomes.
    
    Attributes:
        actions: List of possible actions (0-8 for board positions)
        learning_rate: How fast to update Q-values
        discount_factor: How much to care about future rewards
        exploration_rate: Probability of random exploration
        q_table: Dictionary mapping states to Q-values
    """
    
    def __init__(
        self, 
        learning_rate=0.1, 
        discount_factor=0.95, 
        exploration_rate=0.3
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            learning_rate: How much to trust new experiences
            discount_factor: How much future rewards matter
            exploration_rate: How often to try random moves
        """
        # Actions are board positions 0-8
        self.actions = list(range(9))
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table: dict of {state: {action: q_value}}
        self.q_table = {}
        
        # Statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def get_q_values(self, state):
        """
        Get Q-values for all actions in a state.
        
        Args:
            state: Current board state as string
            
        Returns:
            Dictionary of {action: q_value}
        """
        if state not in self.q_table:
            # Initialize new state with zeros
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state]
    
    def choose_action(self, state, available_actions, training=True):
        """
        Choose an action using ε-greedy strategy.
        
        Args:
            state: Current board state
            available_actions: Valid moves (empty positions)
            training: If True, use exploration; if False, always exploit
            
        Returns:
            The chosen action (position 0-8)
        """
        # Get Q-values for this state
        q_values = self.get_q_values(state)
        
        # Filter to only available actions
        available_q = {a: q_values[a] for a in available_actions}
        
        # Decide: explore or exploit?
        if training and np.random.random() < self.exploration_rate:
            # EXPLORE: Random move
            return np.random.choice(available_actions)
        else:
            # EXPLOIT: Best known move
            # Add small random noise to break ties
            best_value = max(available_q.values())
            best_actions = [a for a, v in available_q.items() if v == best_value]
            return np.random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done, available_next_actions=None):
        """
        Update Q-table based on experience.
        
        Q-Learning Update:
            Q(s,a) = Q(s,a) + α × [r + γ × max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether the game ended
            available_next_actions: Valid moves in next state
        """
        # Get current Q-value
        current_q = self.get_q_values(state)[action]
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: reward + discounted future
            next_q_values = self.get_q_values(next_state)
            
            # Only consider available next actions
            if available_next_actions:
                max_next_q = max(next_q_values[a] for a in available_next_actions)
            else:
                max_next_q = max(next_q_values.values())
            
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
    
    def record_game(self, result):
        """
        Record game result for statistics.
        
        Args:
            result: 'win', 'loss', or 'draw'
        """
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1
    
    def get_stats(self):
        """Get learning statistics."""
        total = self.games_played
        if total == 0:
            return {
                'games': 0,
                'win_rate': 0,
                'states_learned': len(self.q_table)
            }
        
        return {
            'games': total,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.wins / total * 100,
            'states_learned': len(self.q_table)
        }
    
    def print_stats(self):
        """Print learning statistics."""
        stats = self.get_stats()
        print(f"\n📊 Agent Statistics:")
        print(f"   Games played: {stats['games']}")
        print(f"   Wins: {stats['wins']} ({stats['win_rate']:.1f}%)")
        print(f"   Losses: {stats['losses']}")
        print(f"   Draws: {stats['draws']}")
        print(f"   States learned: {stats['states_learned']}")
    
    def print_best_moves(self, states=None):
        """
        Print the best move for specific states.
        
        Args:
            states: List of states to show, or None for random sample
        """
        print("\n🎯 Best Moves for Sample States:")
        print("-" * 50)
        
        if not self.q_table:
            print("  (No states learned yet)")
            return
        
        if states is None:
            # Sample up to 5 states
            all_states = list(self.q_table.keys())
            states = np.random.choice(all_states, min(5, len(all_states)), replace=False)
        
        for state in states:
            q_values = self.get_q_values(state)
            best_action = max(q_values, key=q_values.get)
            best_q = q_values[best_action]
            
            # Format board
            board = list(state)
            print(f"\n  Board: {board[0]}|{board[1]}|{board[2]}")
            print(f"         {board[3]}|{board[4]}|{board[5]}")
            print(f"         {board[6]}|{board[7]}|{board[8]}")
            print(f"  Best move: Position {best_action} (Q = {best_q:.3f})")
        
        print("-" * 50)
    
    def save(self, filepath):
        """Save the Q-table to a file."""
        data = {
            'q_table': self.q_table,
            'stats': {
                'games_played': self.games_played,
                'wins': self.wins,
                'losses': self.losses,
                'draws': self.draws
            }
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
        stats = data['stats']
        self.games_played = stats['games_played']
        self.wins = stats['wins']
        self.losses = stats['losses']
        self.draws = stats['draws']
        
        print(f"✅ Agent loaded from {filepath}")
        return True


# =============================================================================
# TEST
# =============================================================================

def test_agent():
    """Test the Q-learning agent."""
    print("Testing Q-Learning Agent")
    print("=" * 50)
    
    agent = QLearningAgent()
    
    # Simulate some experiences
    state = "         "  # Empty board
    
    # Agent should prefer center (position 4) from empty board
    action = agent.choose_action(state, list(range(9)), training=False)
    print(f"\nEmpty board: Agent chose position {action}")
    
    # After some learning...
    agent.learn(state, 4, 0.0, "X   X    ", False, [0, 1, 2, 3, 5, 6, 7, 8])
    agent.learn(state, 4, 1.0, "terminal", True)
    
    print("\nAfter learning that center leads to win:")
    q_values = agent.get_q_values(state)
    print(f"  Q(4) = {q_values[4]:.3f}")
    
    action = agent.choose_action(state, list(range(9)), training=False)
    print(f"  Agent now chooses: position {action}")


if __name__ == '__main__':
    test_agent()
"""
Q-Learning Agent for Blackjack
==============================

Learn optimal Blackjack strategy using Q-learning.

State: (player_sum, dealer_showing, usable_ace)
Actions: STAND (0) or HIT (1)

Author: Educational RL Project
"""

import numpy as np
import pickle
import os


class BlackjackAgent:
    """
    Q-learning agent for Blackjack.
    
    Learns when to hit and when to stand based on:
    - Player's current sum
    - Dealer's visible card
    - Whether player has a usable Ace
    
    Attributes:
        learning_rate: How fast to learn
        discount_factor: How much future rewards matter
        exploration_rate: Probability of random exploration
        q_table: Dictionary mapping states to Q-values
    """
    
    def __init__(
        self, 
        learning_rate=0.1, 
        discount_factor=1.0, 
        exploration_rate=1.0,
        exploration_decay=0.9999,
        min_exploration=0.05
    ):
        """
        Initialize the Blackjack agent.
        
        Args:
            learning_rate: Learning rate (α)
            discount_factor: Discount factor (γ) - use 1.0 for episodic
            exploration_rate: Initial exploration rate (ε)
            exploration_decay: ε decay per episode
            min_exploration: Minimum exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Q-table: state -> {0: Q_stand, 1: Q_hit}
        self.q_table = {}
        
        # Initialize Q-table for all possible states
        self._init_q_table()
        
        # Statistics
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def _init_q_table(self):
        """Initialize Q-table with zeros for all possible states."""
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                for usable_ace in [False, True]:
                    state = (player_sum, dealer_showing, usable_ace)
                    self.q_table[state] = {0: 0.0, 1: 0.0}  # STAND, HIT
    
    def get_state_key(self, state):
        """
        Convert state to Q-table key.
        
        Args:
            state: (player_sum, dealer_showing, usable_ace)
            
        Returns:
            State tuple for Q-table lookup
        """
        player_sum, dealer_showing, usable_ace = state
        
        # If player sum is below 12, always hit (not a decision state)
        if player_sum < 12:
            return None
        
        # Cap at 21
        if player_sum > 21:
            return None
        
        return (player_sum, dealer_showing, usable_ace)
    
    def choose_action(self, state, training=True):
        """
        Choose an action using ε-greedy policy.
        
        Args:
            state: Current state
            training: If True, use exploration
            
        Returns:
            action: 0 (STAND) or 1 (HIT)
        """
        state_key = self.get_state_key(state)
        
        if state_key is None:
            # Below 12, always hit
            return 1
        
        # Exploration
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(2)
        
        # Exploitation
        q_values = self.q_table[state_key]
        
        if q_values[0] >= q_values[1]:
            return 0  # STAND
        else:
            return 1  # HIT
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether game ended
        """
        state_key = self.get_state_key(state)
        
        if state_key is None:
            return  # Not a decision state
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        if done:
            # Terminal state
            target_q = reward
        else:
            # Non-terminal: reward + best future Q
            next_key = self.get_state_key(next_state)
            if next_key is None:
                # Next state is below 12 (always hit)
                max_next_q = 0
            else:
                max_next_q = max(self.q_table[next_key].values())
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def decay_exploration(self):
        """Decrease exploration rate."""
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
    
    def record_result(self, reward):
        """Record game result."""
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1
    
    def get_stats(self):
        """Get learning statistics."""
        total = self.wins + self.losses + self.draws
        win_rate = self.wins / total * 100 if total > 0 else 0
        
        return {
            'games': total,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': win_rate,
            'exploration_rate': self.exploration_rate
        }
    
    def print_stats(self):
        """Print statistics."""
        stats = self.get_stats()
        print(f"\n📊 Agent Statistics:")
        print(f"   Games: {stats['games']}")
        print(f"   Wins: {stats['wins']} ({stats['win_rate']:.1f}%)")
        print(f"   Losses: {stats['losses']}")
        print(f"   Draws: {stats['draws']}")
        print(f"   Exploration Rate: {stats['exploration_rate']:.3f}")
    
    def save(self, filepath):
        """Save Q-table to file."""
        data = {
            'q_table': self.q_table,
            'exploration_rate': self.exploration_rate,
            'stats': {
                'wins': self.wins,
                'losses': self.losses,
                'draws': self.draws
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table from file."""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.exploration_rate = data['exploration_rate']
        self.wins = data['stats']['wins']
        self.losses = data['stats']['losses']
        self.draws = data['stats']['draws']
        
        print(f"✅ Agent loaded from {filepath}")
        return True
    
    def print_q_values(self, state):
        """
        Print Q-values for a specific state.
        
        Args:
            state: State tuple
        """
        state_key = self.get_state_key(state)
        if state_key and state_key in self.q_table:
            q = self.q_table[state_key]
            print(f"\nState: Player={state_key[0]}, Dealer={state_key[1]}, Ace={state_key[2]}")
            print(f"  Q(STAND): {q[0]:.3f}")
            print(f"  Q(HIT):   {q[1]:.3f}")
            print(f"  Best: {'STAND' if q[0] >= q[1] else 'HIT'}")


# =============================================================================
# BASIC STRATEGY (for comparison)
# =============================================================================

def basic_strategy(state):
    """
    Return the mathematically optimal action for a given state.
    
    This is "Basic Strategy" from Blackjack theory.
    
    Args:
        state: (player_sum, dealer_showing, usable_ace)
        
    Returns:
        action: 0 (STAND) or 1 (HIT)
    """
    player_sum, dealer_showing, usable_ace = state
    
    if usable_ace:
        # Soft hand (Ace counting as 11)
        if player_sum >= 19:
            return 0  # STAND
        elif player_sum == 18:
            if dealer_showing in [9, 10, 1]:
                return 1  # HIT
            else:
                return 0  # STAND
        else:
            return 1  # HIT
    else:
        # Hard hand (no usable Ace)
        if player_sum >= 17:
            return 0  # STAND
        elif player_sum >= 13:
            if dealer_showing in [2, 3]:
                return 0  # STAND
            else:
                return 1  # HIT
        elif player_sum == 12:
            if dealer_showing in [4, 5, 6]:
                return 0  # STAND
            else:
                return 1  # HIT
        else:
            return 1  # HIT


# =============================================================================
# TEST
# =============================================================================

def test_agent():
    """Test the Blackjack agent."""
    print("Testing Blackjack Q-Learning Agent")
    print("=" * 50)
    
    agent = BlackjackAgent()
    
    # Test some states
    states = [
        (16, 10, False),  # Hard 16 vs 10
        (16, 6, False),   # Hard 16 vs 6
        (18, 10, True),   # Soft 18 vs 10
    ]
    
    print("\nInitial Q-values (all zeros):")
    for state in states:
        agent.print_q_values(state)
    
    # Simulate learning
    print("\n" + "=" * 50)
    print("After some learning experiences...")
    
    # Learn that standing on 16 vs 10 is often bad
    for _ in range(100):
        agent.learn((16, 10, False), 0, -0.8, None, True)  # STAND -> lose
        agent.learn((16, 10, False), 1, -0.6, None, True)  # HIT -> lose less
    
    for state in states:
        agent.print_q_values(state)


if __name__ == '__main__':
    test_agent()
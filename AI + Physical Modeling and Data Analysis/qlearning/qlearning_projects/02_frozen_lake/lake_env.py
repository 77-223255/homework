"""
Frozen Lake Environment
=======================

A grid world where an agent must cross a frozen lake without falling into holes.
The ice is slippery, so actions might not always go as planned!

This teaches:
- Stochastic (random) transitions
- Sparse rewards
- Exploration in uncertain environments

Author: Educational RL Project
"""

import numpy as np


# =============================================================================
# FROZEN LAKE ENVIRONMENT
# =============================================================================

class FrozenLake:
    """
    A Frozen Lake grid world environment.
    
    The agent starts at position (0,0) and must reach the goal at (3,3)
    without falling into holes.
    
    Grid Layout:
        S F F F
        F H F H
        F F F H
        H F F G
    
    Legend:
        S = Start (safe)
        F = Frozen (safe)
        H = Hole (fall = game over)
        G = Goal (win!)
    
    The ice is slippery:
        - Intended direction: 33% chance
        - Perpendicular directions: 33% each
    
    Attributes:
        size: Grid size (4x4 by default)
        is_slippery: If True, actions are stochastic
    """
    
    # Action definitions
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    
    ACTION_NAMES = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    
    def __init__(self, size=4, is_slippery=True):
        """
        Initialize the Frozen Lake environment.
        
        Args:
            size: Size of the grid (size x size)
            is_slippery: If True, actions have random outcomes
        """
        self.size = size
        self.is_slippery = is_slippery
        
        # Define the map
        # S = Start, F = Frozen, H = Hole, G = Goal
        self.map = [
            'SFFF',
            'FHFH',
            'FFFH',
            'HFFG'
        ]
        
        # Number of states and actions
        self.n_states = size * size
        self.n_actions = 4
        
        # Current state (position as single number: row*size + col)
        self.state = 0
        
        # Hole positions
        self.holes = {5, 7, 11, 12}
        
        # Goal position
        self.goal = 15
        
        # Start position
        self.start = 0
    
    def reset(self):
        """
        Reset the environment to the starting state.
        
        Returns:
            state: The starting state (0)
        """
        self.state = 0
        return self.state
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: The action to take (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)
            
        Returns:
            next_state: The new state
            reward: The reward received
            done: Whether the episode is over
        """
        # Get possible transitions
        if self.is_slippery:
            # Stochastic: might go in different directions
            transitions = self._get_slippery_transitions(action)
        else:
            # Deterministic: always go intended direction
            transitions = [(self._get_next_state(self.state, action), 1.0)]
        
        # Sample from transitions
        probs = [t[1] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        next_state = transitions[idx][0]
        
        # Update state
        self.state = next_state
        
        # Calculate reward and done
        if next_state == self.goal:
            return next_state, 1.0, True  # Reached goal!
        elif next_state in self.holes:
            return next_state, -1.0, True  # Fell in hole (penalty!)
        else:
            return next_state, 0.0, False  # Normal move (including hitting walls)
    
    def _get_slippery_transitions(self, action):
        """
        Get possible transitions for a slippery action.
        
        Args:
            action: Intended action
            
        Returns:
            List of (next_state, probability) tuples
        """
        # Intended direction: 33%
        # Left perpendicular: 33%
        # Right perpendicular: 33%
        
        if action == self.UP:
            directions = [self.UP, self.LEFT, self.RIGHT]
        elif action == self.DOWN:
            directions = [self.DOWN, self.RIGHT, self.LEFT]
        elif action == self.LEFT:
            directions = [self.LEFT, self.DOWN, self.UP]
        else:  # RIGHT
            directions = [self.RIGHT, self.UP, self.DOWN]
        
        transitions = []
        for dir in directions:
            next_state = self._get_next_state(self.state, dir)
            transitions.append((next_state, 1/3))
        
        return transitions
    
    def _get_next_state(self, state, action):
        """
        Get the next state after taking an action (ignoring slipperiness).
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state (might be same if hitting wall)
        """
        row = state // self.size
        col = state % self.size
        
        if action == self.UP:
            row = max(0, row - 1)
        elif action == self.DOWN:
            row = min(self.size - 1, row + 1)
        elif action == self.LEFT:
            col = max(0, col - 1)
        elif action == self.RIGHT:
            col = min(self.size - 1, col + 1)
        
        return row * self.size + col
    
    def render(self):
        """Print the current state of the lake."""
        print("\n  Frozen Lake:")
        
        for row in range(self.size):
            line = "  "
            for col in range(self.size):
                state = row * self.size + col
                
                if state == self.state:
                    # Agent position
                    char = 'A'  # Agent
                elif state in self.holes:
                    char = 'H'  # Hole
                elif state == self.goal:
                    char = 'G'  # Goal
                elif state == self.start:
                    char = 'S'  # Start
                else:
                    char = 'F'  # Frozen
                
                line += char + ' '
            print(line)
        print()
    
    def state_to_pos(self, state):
        """Convert state number to (row, col) position."""
        return state // self.size, state % self.size
    
    def pos_to_state(self, row, col):
        """Convert (row, col) position to state number."""
        return row * self.size + col


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def print_policy(agent, size=4):
    """
    Print the learned policy (best action for each state).
    
    Args:
        agent: Trained Q-learning agent
        size: Grid size
    """
    action_symbols = ['←', '↓', '→', '↑']
    
    print("\n🗺️ Learned Policy (Best Direction):")
    print("-" * (size * 4 + 1))
    
    for row in range(size):
        line = "| "
        for col in range(size):
            state = row * size + col
            
            # Special positions
            if state == 15:  # Goal
                line += "G | "
            elif state in {5, 7, 11, 12}:  # Holes
                line += "H | "
            else:
                # Get best action
                if state in agent.q_table:
                    q_values = agent.q_table[state]
                    best_action = max(range(4), key=lambda a: q_values[a])
                    line += action_symbols[best_action] + " | "
                else:
                    line += "? | "
        print(line)
    print("-" * (size * 4 + 1))


def print_policy_with_qmax(agent, size=4):
    """
    Print the learned policy with Q-max values for each state.
    
    Args:
        agent: Trained Q-learning agent
        size: Grid size
    """
    action_symbols = ['←', '↓', '→', '↑']
    
    print("\n" + "=" * 60)
    print("   🎯 Q-MAX DIRECTION & VALUE FOR EACH STATE")
    print("=" * 60)
    
    # Print policy with Q-max values
    print("\n🗺️ Best Direction (arrow) with Q-max value:")
    print("-" * (size * 20 + 1))
    
    for row in range(size):
        line = "| "
        for col in range(size):
            state = row * size + col
            
            # Special positions
            if state == 15:  # Goal
                line += "  🏁 GOAL  | "
            elif state in {5, 7, 11, 12}:  # Holes
                line += "  💀 HOLE  | "
            else:
                if state in agent.q_table:
                    q_values = agent.q_table[state]
                    best_action = max(range(4), key=lambda a: q_values[a])
                    best_q = q_values[best_action]
                    symbol = action_symbols[best_action]
                    line += f" {symbol} ({best_q:5.2f}) | "
                else:
                    line += "    ?     | "
        print(line)
    print("-" * (size * 20 + 1))
    
    # Print detailed Q-values for each state
    print("\n📊 Detailed Q-Values for Each State:")
    print("-" * 60)
    
    for row in range(size):
        for col in range(size):
            state = row * size + col
            
            if state == 15:  # Goal
                print(f"State {state:2d} ({row},{col}): 🏁 GOAL")
            elif state in {5, 7, 11, 12}:  # Holes
                print(f"State {state:2d} ({row},{col}): 💀 HOLE")
            else:
                if state in agent.q_table:
                    q = agent.q_table[state]
                    best_action = max(range(4), key=lambda a: q[a])
                    best_symbol = action_symbols[best_action]
                    
                    q_str = f"LEFT:{q[0]:6.3f}  DOWN:{q[1]:6.3f}  RIGHT:{q[2]:6.3f}  UP:{q[3]:6.3f}"
                    print(f"State {state:2d} ({row},{col}): Best={best_symbol}  | {q_str}")
    print("-" * 60)


def print_value_function(agent, size=4):
    """
    Print the value function (max Q-value for each state).
    
    Args:
        agent: Trained Q-learning agent
        size: Grid size
    """
    print("\n📊 State Values (Max Q):")
    print("-" * (size * 8 + 1))
    
    for row in range(size):
        line = "| "
        for col in range(size):
            state = row * size + col
            
            if state == 15:  # Goal
                line += " GOAL  | "
            elif state in {5, 7, 11, 12}:  # Holes
                line += " HOLE  | "
            else:
                if state in agent.q_table:
                    value = max(agent.q_table[state].values())
                    line += f"{value:6.3f} | "
                else:
                    line += " 0.000 | "
        print(line)
    print("-" * (size * 8 + 1))


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_environment():
    """Test the Frozen Lake environment."""
    print("=" * 50)
    print("   FROZEN LAKE TEST")
    print("=" * 50)
    
    env = FrozenLake(is_slippery=True)
    env.render()
    
    print("Taking random actions...")
    print("-" * 50)
    
    state = env.reset()
    total_reward = 0
    
    for step in range(20):
        action = np.random.randint(0, 4)
        action_name = env.ACTION_NAMES[action]
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: {action_name:6s} -> State {next_state:2d}, Reward: {reward}")
        env.render()
        
        if done:
            if reward > 0:
                print("🎉 Reached the goal!")
            else:
                print("💀 Fell in a hole!")
            break
        
        state = next_state
    
    print(f"\nTotal reward: {total_reward}")


if __name__ == '__main__':
    test_environment()
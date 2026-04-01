"""
Maze Environment for Q-Learning Tutorial
==========================================

This creates a simple 4x4 grid maze where an agent learns to navigate.

Elements:
- 🔴 Red Rectangle: Agent (Explorer) - starts at top-left
- ⬛ Black Rectangles: Hells (Danger) - reward = -1
- 🟡 Yellow Oval: Goal (Paradise) - reward = +1
- ⬜ White Cells: Ground (Safe) - reward = 0

The agent must learn to reach the goal while avoiding hells.

Author: Improved version for teaching
"""

import numpy as np
import time
import sys

# Handle Python 2 vs 3 tkinter import
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


# =============================================================================
# CONSTANTS - Easy to understand and modify
# =============================================================================

CELL_SIZE = 40        # Size of each cell in pixels
GRID_WIDTH = 4        # Number of columns
GRID_HEIGHT = 4       # Number of rows

# Window size in pixels
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Half of cell size (used for centering elements)
HALF_CELL = CELL_SIZE // 2

# Agent size (half-width of rectangles/ovals)
AGENT_SIZE = 15

# Starting position (center of top-left cell)
START_X = HALF_CELL
START_Y = HALF_CELL

# Hell positions (column, row) - 0-indexed
HELL_1_POSITION = (2, 1)  # Column 2, Row 1
HELL_2_POSITION = (1, 2)  # Column 1, Row 2

# Goal position (column, row)
GOAL_POSITION = (2, 2)    # Column 2, Row 2


# =============================================================================
# MAZE CLASS - The Game Environment
# =============================================================================

class Maze(tk.Tk):
    """
    A maze environment for reinforcement learning.
    
    The agent (red square) must learn to reach the goal (yellow circle)
    while avoiding hells (black squares).
    
    Attributes:
        action_space: List of possible actions ['up', 'down', 'left', 'right']
        n_actions: Number of possible actions (4)
    """
    
    def __init__(self, render_delay=0.1, reset_delay=0.5):
        """
        Initialize the maze environment.
        
        Args:
            render_delay: Seconds to wait when rendering (for visualization)
            reset_delay: Seconds to wait when resetting (for visualization)
        """
        super(Maze, self).__init__()
        
        # Define possible actions
        self.action_space = ['up', 'down', 'right', 'left']
        self.n_actions = len(self.action_space)
        
        # Action indices (for easy reference)
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_RIGHT = 2
        self.ACTION_LEFT = 3
        
        # Timing settings
        self.render_delay = render_delay
        self.reset_delay = reset_delay
        
        # Set up the window
        self.title('Q-Learning Maze')
        self.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')
        
        # Build the maze
        self._build_maze()
    
    def _grid_to_pixel(self, grid_x, grid_y):
        """
        Convert grid position to pixel coordinates.
        
        Args:
            grid_x: Column (0 to GRID_WIDTH-1)
            grid_y: Row (0 to GRID_HEIGHT-1)
            
        Returns:
            (x, y) pixel coordinates for the center of that cell
        """
        pixel_x = HALF_CELL + grid_x * CELL_SIZE
        pixel_y = HALF_CELL + grid_y * CELL_SIZE
        return np.array([pixel_x, pixel_y])
    
    def _build_maze(self):
        """Build the maze with grid, hells, goal, and agent."""
        
        # Create the canvas (white background)
        self.canvas = tk.Canvas(
            self, 
            bg='white',
            height=WINDOW_HEIGHT,
            width=WINDOW_WIDTH
        )
        
        # Draw grid lines
        self._draw_grid()
        
        # Create the obstacles and goal
        self._create_hells()
        self._create_goal()
        self._create_agent()
        
        # Display everything
        self.canvas.pack()
    
    def _draw_grid(self):
        """Draw the grid lines."""
        
        # Vertical lines
        for col in range(0, WINDOW_WIDTH + 1, CELL_SIZE):
            self.canvas.create_line(col, 0, col, WINDOW_HEIGHT)
        
        # Horizontal lines
        for row in range(0, WINDOW_HEIGHT + 1, CELL_SIZE):
            self.canvas.create_line(0, row, WINDOW_WIDTH, row)
    
    def _create_hells(self):
        """Create the hell obstacles (black rectangles)."""
        
        # Hell 1
        hell1_pos = self._grid_to_pixel(*HELL_1_POSITION)
        self.hell1 = self.canvas.create_rectangle(
            hell1_pos[0] - AGENT_SIZE, hell1_pos[1] - AGENT_SIZE,
            hell1_pos[0] + AGENT_SIZE, hell1_pos[1] + AGENT_SIZE,
            fill='black'
        )
        
        # Hell 2
        hell2_pos = self._grid_to_pixel(*HELL_2_POSITION)
        self.hell2 = self.canvas.create_rectangle(
            hell2_pos[0] - AGENT_SIZE, hell2_pos[1] - AGENT_SIZE,
            hell2_pos[0] + AGENT_SIZE, hell2_pos[1] + AGENT_SIZE,
            fill='black'
        )
    
    def _create_goal(self):
        """Create the goal (yellow oval)."""
        
        goal_pos = self._grid_to_pixel(*GOAL_POSITION)
        self.goal = self.canvas.create_oval(
            goal_pos[0] - AGENT_SIZE, goal_pos[1] - AGENT_SIZE,
            goal_pos[0] + AGENT_SIZE, goal_pos[1] + AGENT_SIZE,
            fill='yellow'
        )
    
    def _create_agent(self):
        """Create the agent (red rectangle) at starting position."""
        
        start_pos = np.array([START_X, START_Y])
        self.agent = self.canvas.create_rectangle(
            start_pos[0] - AGENT_SIZE, start_pos[1] - AGENT_SIZE,
            start_pos[0] + AGENT_SIZE, start_pos[1] + AGENT_SIZE,
            fill='red'
        )
    
    def reset(self):
        """
        Reset the environment - put agent back to start.
        
        Returns:
            The starting position coordinates
        """
        self.update()
        time.sleep(self.reset_delay)
        
        # Remove old agent
        self.canvas.delete(self.agent)
        
        # Create new agent at start
        start_pos = np.array([START_X, START_Y])
        self.agent = self.canvas.create_rectangle(
            start_pos[0] - AGENT_SIZE, start_pos[1] - AGENT_SIZE,
            start_pos[0] + AGENT_SIZE, start_pos[1] + AGENT_SIZE,
            fill='red'
        )
        
        return self.canvas.coords(self.agent)
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take (0=up, 1=down, 2=right, 3=left)
            
        Returns:
            next_state: The new position after taking action
            reward: The reward received (-1, 0, or +1)
            done: Whether the episode is finished
        """
        # Get current position
        current_pos = self.canvas.coords(self.agent)
        
        # Calculate movement
        movement = np.array([0, 0])
        
        if action == self.ACTION_UP:
            if current_pos[1] > CELL_SIZE:  # Not at top edge
                movement[1] -= CELL_SIZE
                
        elif action == self.ACTION_DOWN:
            if current_pos[1] < (GRID_HEIGHT - 1) * CELL_SIZE:  # Not at bottom
                movement[1] += CELL_SIZE
                
        elif action == self.ACTION_RIGHT:
            if current_pos[0] < (GRID_WIDTH - 1) * CELL_SIZE:  # Not at right
                movement[0] += CELL_SIZE
                
        elif action == self.ACTION_LEFT:
            if current_pos[0] > CELL_SIZE:  # Not at left edge
                movement[0] -= CELL_SIZE
        
        # Move the agent
        self.canvas.move(self.agent, movement[0], movement[1])
        
        # Get new position
        next_state = self.canvas.coords(self.agent)
        
        # Calculate reward
        reward, done = self._calculate_reward(next_state)
        
        return next_state, reward, done
    
    def _calculate_reward(self, position):
        """
        Calculate reward based on position.
        
        Args:
            position: Current position coordinates
            
        Returns:
            reward: -1 (hell), +1 (goal), or 0 (ground)
            done: True if reached hell or goal
        """
        # Check if reached goal
        if position == self.canvas.coords(self.goal):
            return 1, True
        
        # Check if hit hell
        if position in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            return -1, True
        
        # Still on ground
        return 0, False
    
    def render(self):
        """Update the display (for visualization)."""
        time.sleep(self.render_delay)
        self.update()
    
    def get_position_name(self, coords):
        """
        Convert coordinates to a readable grid position.
        
        Args:
            coords: Pixel coordinates [x, y, x, y]
            
        Returns:
            String like "(0, 0)" representing grid position
        """
        x = int((coords[0] - HALF_CELL + AGENT_SIZE) // CELL_SIZE)
        y = int((coords[1] - HALF_CELL + AGENT_SIZE) // CELL_SIZE)
        return f"({x}, {y})"


# =============================================================================
# TEST FUNCTION - Run this file directly to test
# =============================================================================

def test_maze():
    """Test the maze by moving randomly."""
    
    print("Testing Maze Environment")
    print("=" * 40)
    print("The agent will move randomly.")
    print("Close the window to exit.\n")
    
    env = Maze(render_delay=0.2)
    
    def random_walk():
        for episode in range(5):
            state = env.reset()
            print(f"Episode {episode + 1}: Starting at {env.get_position_name(state)}")
            
            steps = 0
            while True:
                env.render()
                
                # Random action
                action = np.random.randint(0, env.n_actions)
                action_name = env.action_space[action]
                
                next_state, reward, done = env.step(action)
                steps += 1
                
                print(f"  Step {steps}: {action_name} -> {env.get_position_name(next_state)}, reward={reward}")
                
                if done:
                    if reward == 1:
                        print(f"  🎉 Reached goal in {steps} steps!")
                    else:
                        print(f"  💀 Hit hell after {steps} steps")
                    break
                
                state = next_state
            
            print()
        
        print("Test complete!")
        env.destroy()
    
    env.after(100, random_walk)
    env.mainloop()


if __name__ == '__main__':
    test_maze()
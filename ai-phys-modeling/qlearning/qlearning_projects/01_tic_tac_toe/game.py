"""
Tic-Tac-Toe Game Environment
============================

A simple Tic-Tac-Toe game that can be played by humans or AI agents.

Board positions are numbered 0-8:
    
     0 | 1 | 2
    -----------
     3 | 4 | 5
    -----------
     6 | 7 | 8

Author: Educational RL Project
"""

import numpy as np


class TicTacToe:
    """
    A Tic-Tac-Toe game environment.
    
    The agent plays as 'X' (goes first).
    The opponent plays as 'O'.
    
    Attributes:
        board: Current state of the board (length-9 array)
        current_player: 'X' or 'O'
        winner: 'X', 'O', 'Draw', or None (if game ongoing)
    """
    
    # Possible positions on the board
    POSITIONS = list(range(9))
    
    # Winning combinations (rows, columns, diagonals)
    WIN_LINES = [
        [0, 1, 2],  # Top row
        [3, 4, 5],  # Middle row
        [6, 7, 8],  # Bottom row
        [0, 3, 6],  # Left column
        [1, 4, 7],  # Middle column
        [2, 5, 8],  # Right column
        [0, 4, 8],  # Diagonal top-left to bottom-right
        [2, 4, 6],  # Diagonal top-right to bottom-left
    ]
    
    def __init__(self):
        """Initialize an empty board."""
        self.reset()
    
    def reset(self):
        """
        Reset the game to the starting state.
        
        Returns:
            state: The initial empty board state
        """
        # Board is represented as a list of 9 elements
        # ' ' = empty, 'X' = agent, 'O' = opponent
        self.board = [' '] * 9
        self.current_player = 'X'  # X always goes first
        self.winner = None
        return self.get_state()
    
    def get_state(self):
        """
        Get the current state as a string.
        
        Returns:
            A string like "X O  X O " representing the board
        """
        return ''.join(self.board)
    
    def get_available_actions(self):
        """
        Get all valid moves (empty positions).
        
        Returns:
            List of position indices that are empty
        """
        return [i for i in range(9) if self.board[i] == ' ']
    
    def step(self, action):
        """
        Make a move and return the result.
        
        Args:
            action: Position to place mark (0-8)
            
        Returns:
            next_state: The new board state
            reward: Reward for the move
            done: Whether the game is over
        """
        # Validate move
        if action not in self.get_available_actions():
            raise ValueError(f"Invalid move: position {action} is not empty")
        
        # Make the move
        self.board[action] = self.current_player
        
        # Check for winner
        self._check_winner()
        
        # Determine reward and done
        if self.winner == 'X':
            reward = 1.0
            done = True
        elif self.winner == 'O':
            reward = -1.0
            done = True
        elif self.winner == 'Draw':
            reward = 0.5  # Small reward for draw (better than losing)
            done = True
        else:
            reward = 0.0
            done = False
        
        # Switch player
        if not done:
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return self.get_state(), reward, done
    
    def _check_winner(self):
        """Check if someone has won or if it's a draw."""
        # Check each winning line
        for line in self.WIN_LINES:
            marks = [self.board[i] for i in line]
            if marks[0] != ' ' and marks[0] == marks[1] == marks[2]:
                self.winner = marks[0]
                return
        
        # Check for draw (no empty spaces)
        if ' ' not in self.board:
            self.winner = 'Draw'
    
    def render(self):
        """Print the current board state."""
        print("\n  Current Board:")
        print(f"   {self.board[0]} | {self.board[1]} | {self.board[2]} ")
        print("  -----------")
        print(f"   {self.board[3]} | {self.board[4]} | {self.board[5]} ")
        print("  -----------")
        print(f"   {self.board[6]} | {self.board[7]} | {self.board[8]} \n")
    
    def copy(self):
        """Create a copy of the current game state."""
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        return new_game
    
    @staticmethod
    def state_to_board(state):
        """
        Convert a state string to a board list.
        
        Args:
            state: String like "X O  X O "
            
        Returns:
            List of 9 characters
        """
        return list(state)


# =============================================================================
# OPPONENT FUNCTIONS - For training
# =============================================================================

def random_opponent(game):
    """
    An opponent that makes random valid moves.
    
    Args:
        game: The current TicTacToe game
        
    Returns:
        A random valid position
    """
    return np.random.choice(game.get_available_actions())


def smart_opponent(game):
    """
    An opponent that makes semi-intelligent moves.
    
    Priority:
    1. Win if possible
    2. Block if opponent can win
    3. Take center if available
    4. Take a corner if available
    5. Random move
    
    Args:
        game: The current TicTacToe game
        
    Returns:
        A good position choice
    """
    available = game.get_available_actions()
    
    # Try to win
    for pos in available:
        test_game = game.copy()
        test_game.board[pos] = 'O'
        test_game._check_winner()
        if test_game.winner == 'O':
            return pos
    
    # Block opponent from winning
    for pos in available:
        test_game = game.copy()
        test_game.board[pos] = 'X'
        test_game._check_winner()
        if test_game.winner == 'X':
            return pos
    
    # Take center
    if 4 in available:
        return 4
    
    # Take a corner
    corners = [0, 2, 6, 8]
    available_corners = [c for c in corners if c in available]
    if available_corners:
        return np.random.choice(available_corners)
    
    # Random move
    return np.random.choice(available)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_game():
    """Test the Tic-Tac-Toe game with human input."""
    print("=" * 50)
    print("   TIC-TAC-TOE TEST GAME")
    print("=" * 50)
    print("\nBoard positions:")
    print("  0 | 1 | 2")
    print("  ---------")
    print("  3 | 4 | 5")
    print("  ---------")
    print("  6 | 7 | 8\n")
    
    game = TicTacToe()
    game.render()
    
    while game.winner is None:
        # Human (X) moves
        print("Your turn (X)!")
        try:
            pos = int(input("Enter position (0-8): "))
            if pos not in game.get_available_actions():
                print("Invalid move! Try again.")
                continue
        except ValueError:
            print("Please enter a number 0-8.")
            continue
        
        game.step(pos)
        game.render()
        
        if game.winner:
            break
        
        # Random opponent (O) moves
        print("Opponent (O) is thinking...")
        opp_move = random_opponent(game)
        game.step(opp_move)
        print(f"Opponent chose position {opp_move}")
        game.render()
    
    # Game over
    if game.winner == 'Draw':
        print("It's a draw!")
    else:
        print(f"{game.winner} wins!")


if __name__ == '__main__':
    test_game()
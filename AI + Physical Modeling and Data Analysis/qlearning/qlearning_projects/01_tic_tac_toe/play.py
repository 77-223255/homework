"""
Play Against Trained Tic-Tac-Toe Agent
======================================

Play a game of Tic-Tac-Toe against the trained AI!

Author: Educational RL Project
"""

import os
from game import TicTacToe
from agent import QLearningAgent


# =============================================================================
# CONFIGURATION
# =============================================================================

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'trained_agent.pkl')


# =============================================================================
# GAME FUNCTIONS
# =============================================================================

def play_game(agent):
    """
    Play a single game against the trained agent.
    
    Args:
        agent: The trained Q-learning agent
    """
    game = TicTacToe()
    
    print("\n" + "=" * 50)
    print("       🎮 TIC-TAC-TOE 🎮")
    print("=" * 50)
    print("\nBoard positions:")
    print("  0 | 1 | 2")
    print("  ---------")
    print("  3 | 4 | 5")
    print("  ---------")
    print("  6 | 7 | 8")
    print("\nYou are O. Agent is X.")
    print("Agent goes first.\n")
    
    # Agent goes first (X)
    state = game.reset()
    done = False
    
    # Agent's first move
    print("Agent's turn...")
    action = agent.choose_action(state, game.get_available_actions(), training=False)
    state, _, done = game.step(action)
    print(f"Agent chose position {action}")
    game.render()
    
    # Game loop
    while not done:
        # Human's turn
        print("Your turn (O)!")
        
        try:
            pos = int(input("Enter position (0-8): "))
        except ValueError:
            print("Please enter a number 0-8.")
            continue
        
        if pos not in game.get_available_actions():
            print("Invalid move! That position is taken.")
            continue
        
        # Make human's move
        state, _, done = game.step(pos)
        game.render()
        
        if done:
            break
        
        # Agent's turn
        print("Agent's turn...")
        action = agent.choose_action(state, game.get_available_actions(), training=False)
        state, _, done = game.step(action)
        print(f"Agent chose position {action}")
        game.render()
    
    # Game result
    print("=" * 50)
    if game.winner == 'X':
        print("   🤖 Agent wins! Better luck next time!")
    elif game.winner == 'O':
        print("   🎉 You win! Congratulations!")
    else:
        print("   🤝 It's a draw! Well played!")
    print("=" * 50)


def main():
    """Main function to play against the agent."""
    
    print("\n" + "=" * 60)
    print("   TIC-TAC-TOE: Play Against Trained Agent")
    print("=" * 60)
    
    # Load trained agent
    agent = QLearningAgent()
    
    if not os.path.exists(SAVE_PATH):
        print("\n⚠️ No trained agent found!")
        print("Please run 'python train.py' first to train the agent.")
        return
    
    agent.load(SAVE_PATH)
    agent.print_stats()
    
    # Play loop
    while True:
        play_game(agent)
        
        # Ask to play again
        print("\nWant to play again?")
        response = input("Play again? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("\n👋 Thanks for playing! See you next time!")
            break


if __name__ == '__main__':
    main()
"""
Training Script for Tic-Tac-Toe Q-Learning Agent
=================================================

Train the agent by having it play against itself.

Author: Educational RL Project
"""

import numpy as np
from game import TicTacToe, random_opponent, smart_opponent
from agent import QLearningAgent
import os


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Number of training games
NUM_EPISODES = 10000

# Print progress every N games
PROGRESS_INTERVAL = 1000

# Q-Learning hyperparameters
LEARNING_RATE = 0.1       # How fast to learn
DISCOUNT_FACTOR = 0.95    # How much future matters
EXPLORATION_RATE = 0.3    # How often to explore

# Opponent type: 'random' or 'smart'
OPPONENT_TYPE = 'smart'

# Save path
SAVE_PATH = os.path.join(os.path.dirname(__file__), 'trained_agent.pkl')


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_agent(
    num_episodes=NUM_EPISODES,
    learning_rate=LEARNING_RATE,
    discount_factor=DISCOUNT_FACTOR,
    exploration_rate=EXPLORATION_RATE,
    opponent_type=OPPONENT_TYPE,
    save_path=SAVE_PATH,
    verbose=True
):
    """
    Train a Q-learning agent to play Tic-Tac-Toe.
    
    Args:
        num_episodes: Number of games to play
        learning_rate: Learning rate (α)
        discount_factor: Discount factor (γ)
        exploration_rate: Exploration rate (ε)
        opponent_type: 'random' or 'smart'
        save_path: Where to save the trained agent
        verbose: Print progress
        
    Returns:
        agent: The trained agent
        history: Training history
    """
    print("=" * 60)
    print("       🎮 TIC-TAC-TOE Q-LEARNING TRAINING 🎮")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Discount Factor: {discount_factor}")
    print(f"  Exploration Rate: {exploration_rate}")
    print(f"  Opponent: {opponent_type}")
    print("-" * 60)
    
    # Create agent and game
    agent = QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate
    )
    game = TicTacToe()
    
    # Choose opponent
    if opponent_type == 'smart':
        opponent = smart_opponent
    else:
        opponent = random_opponent
    
    # Training history
    history = {
        'wins': [],
        'losses': [],
        'draws': [],
        'win_rates': []
    }
    
    # Recent results for progress tracking
    recent_results = []
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    for episode in range(1, num_episodes + 1):
        # Reset game
        state = game.reset()
        done = False
        
        # Store the episode trajectory for learning
        trajectory = []
        
        while not done:
            # Agent's turn (X)
            available = game.get_available_actions()
            action = agent.choose_action(state, available, training=True)
            
            # Store experience
            trajectory.append({
                'state': state,
                'action': action,
                'available': available.copy()
            })
            
            # Make move
            next_state, reward, done = game.step(action)
            
            if done:
                # Game ended on agent's move
                result = 'win' if reward > 0 else 'draw' if reward > 0.4 else 'loss'
                trajectory[-1]['next_state'] = 'terminal'
                trajectory[-1]['reward'] = reward
                trajectory[-1]['done'] = True
            else:
                # Opponent's turn (O)
                opp_action = opponent(game)
                opp_state, opp_reward, done = game.step(opp_action)
                
                if done:
                    # Opponent won or draw
                    agent_reward = -1 if game.winner == 'O' else 0.5
                    result = 'loss' if game.winner == 'O' else 'draw'
                    trajectory[-1]['next_state'] = 'terminal'
                    trajectory[-1]['reward'] = agent_reward
                    trajectory[-1]['done'] = True
                else:
                    # Game continues
                    trajectory[-1]['next_state'] = opp_state
                    trajectory[-1]['reward'] = 0
                    trajectory[-1]['done'] = False
                    state = opp_state
        
        # Learn from the episode (backwards for efficiency)
        for exp in reversed(trajectory):
            agent.learn(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state'],
                exp['done']
            )
        
        # Record result
        agent.record_game(result)
        recent_results.append(result)
        
        # Progress report
        if verbose and episode % PROGRESS_INTERVAL == 0:
            recent = recent_results[-PROGRESS_INTERVAL:]
            wins = recent.count('win')
            losses = recent.count('loss')
            draws = recent.count('draw')
            win_rate = wins / len(recent) * 100
            
            print(f"Episode {episode:6d}/{num_episodes} | "
                  f"Win: {wins:4d} ({win_rate:5.1f}%) | "
                  f"Loss: {losses:4d} | "
                  f"Draw: {draws:4d} | "
                  f"States: {len(agent.q_table)}")
            
            history['wins'].append(wins)
            history['losses'].append(losses)
            history['draws'].append(draws)
            history['win_rates'].append(win_rate)
    
    # ==========================================================================
    # TRAINING COMPLETE
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("       🎉 TRAINING COMPLETE! 🎉")
    print("=" * 60)
    
    agent.print_stats()
    
    # Save agent
    if save_path:
        agent.save(save_path)
    
    return agent, history


def evaluate_agent(agent, num_games=100):
    """
    Evaluate a trained agent against random and smart opponents.
    
    Args:
        agent: Trained Q-learning agent
        num_games: Number of games per opponent
    """
    print("\n" + "=" * 60)
    print("       📊 AGENT EVALUATION 📊")
    print("=" * 60)
    
    game = TicTacToe()
    
    # Temporarily disable exploration
    old_epsilon = agent.exploration_rate
    agent.exploration_rate = 0.0
    
    # Test against random opponent
    print(f"\nTesting against RANDOM opponent ({num_games} games)...")
    wins, losses, draws = 0, 0, 0
    
    for _ in range(num_games):
        state = game.reset()
        done = False
        
        while not done:
            # Agent's turn
            action = agent.choose_action(state, game.get_available_actions(), training=False)
            state, reward, done = game.step(action)
            
            if not done:
                # Opponent's turn
                opp_action = random_opponent(game)
                state, _, done = game.step(opp_action)
        
        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1
    
    print(f"  Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    # Test against smart opponent
    print(f"\nTesting against SMART opponent ({num_games} games)...")
    wins, losses, draws = 0, 0, 0
    
    for _ in range(num_games):
        state = game.reset()
        done = False
        
        while not done:
            # Agent's turn
            action = agent.choose_action(state, game.get_available_actions(), training=False)
            state, reward, done = game.step(action)
            
            if not done:
                # Opponent's turn
                opp_action = smart_opponent(game)
                state, _, done = game.step(opp_action)
        
        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1
    
    print(f"  Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    # Restore exploration rate
    agent.exploration_rate = old_epsilon
    
    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training function."""
    
    # Train the agent
    agent, history = train_agent()
    
    # Show some best moves
    agent.print_best_moves()
    
    # Evaluate
    evaluate_agent(agent)
    
    print("\n✅ Training complete!")
    print(f"Run 'python play.py' to play against the trained agent!")


if __name__ == '__main__':
    main()
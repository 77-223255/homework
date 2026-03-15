"""
Training Script for Blackjack Q-Learning Agent
===============================================

Train the agent to play optimal Blackjack!

Key differences from previous projects:
- Episodic task (discount factor = 1.0)
- Probability-based outcomes
- Comparing against "Basic Strategy" (mathematically optimal)

Author: Educational RL Project
"""

import numpy as np
from blackjack_env import Blackjack, print_strategy_table
from agent import BlackjackAgent, basic_strategy
import os


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

NUM_EPISODES = 100000

# Print progress every N episodes
PROGRESS_INTERVAL = 10000

# Q-Learning hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1.0      # 1.0 for episodic tasks
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.9999
MIN_EXPLORATION = 0.05

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
    exploration_decay=EXPLORATION_DECAY,
    min_exploration=MIN_EXPLORATION,
    save_path=SAVE_PATH,
    verbose=True
):
    """
    Train a Blackjack agent.
    
    Args:
        num_episodes: Number of games to play
        learning_rate: Learning rate (α)
        discount_factor: Discount factor (γ)
        exploration_rate: Initial exploration rate (ε)
        exploration_decay: ε decay per episode
        min_exploration: Minimum exploration rate
        save_path: Where to save trained agent
        verbose: Print progress
        
    Returns:
        agent: Trained agent
        history: Training history
    """
    print("=" * 60)
    print("       🃏 BLACKJACK Q-LEARNING TRAINING 🃏")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Discount Factor: {discount_factor}")
    print(f"  Initial Exploration: {exploration_rate}")
    print(f"  Exploration Decay: {exploration_decay}")
    print("-" * 60)
    
    # Create environment and agent
    env = Blackjack()
    agent = BlackjackAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay,
        min_exploration=min_exploration
    )
    
    # Training history
    history = {
        'win_rates': [],
        'exploration_rates': []
    }
    
    # Track recent results
    recent_results = []
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    for episode in range(1, num_episodes + 1):
        # Start new game
        state = env.reset()
        done = False
        
        # Check for natural blackjack (21 from start)
        if state[0] == 21:
            # Player has blackjack! Dealer plays
            dealer_sum = env._play_dealer()
            if dealer_sum == 21:
                reward = 0.0  # Draw (both have 21)
            else:
                reward = 1.0  # Blackjack win!
            
            agent.record_result(reward)
            recent_results.append(reward)
            agent.decay_exploration()
            continue
        
        # Episode loop
        while not done:
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
        
        # Record result
        agent.record_result(reward)
        recent_results.append(reward)
        
        # Decay exploration
        agent.decay_exploration()
        
        # Progress report
        if verbose and episode % PROGRESS_INTERVAL == 0:
            recent = recent_results[-PROGRESS_INTERVAL:]
            wins = sum(1 for r in recent if r > 0)
            losses = sum(1 for r in recent if r < 0)
            draws = sum(1 for r in recent if r == 0)
            win_rate = wins / len(recent) * 100
            
            print(f"Episode {episode:7,}/{num_episodes:,} | "
                  f"Win: {wins:5d} ({win_rate:5.1f}%) | "
                  f"Loss: {losses:5d} | "
                  f"Draw: {draws:4d} | "
                  f"ε: {agent.exploration_rate:.3f}")
            
            history['win_rates'].append(win_rate)
            history['exploration_rates'].append(agent.exploration_rate)
    
    # ==========================================================================
    # TRAINING COMPLETE
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("       🎉 TRAINING COMPLETE! 🎉")
    print("=" * 60)
    
    agent.print_stats()
    
    # Show learned strategy
    print_strategy_table(agent)
    
    # Compare with basic strategy
    compare_with_basic_strategy(agent)
    
    # Save agent
    if save_path:
        agent.save(save_path)
    
    return agent, history


def compare_with_basic_strategy(agent, num_games=10000):
    """
    Compare learned strategy with mathematically optimal basic strategy.
    
    Args:
        agent: Trained agent
        num_games: Number of games to test
    """
    print("\n" + "=" * 60)
    print("   📊 COMPARISON: Learned vs Basic Strategy")
    print("=" * 60)
    
    env = Blackjack()
    
    # Test learned strategy
    print(f"\nTesting Learned Strategy ({num_games:,} games)...")
    learned_wins = 0
    old_epsilon = agent.exploration_rate
    agent.exploration_rate = 0.0  # No exploration
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state, training=False)
            state, _, done = env.step(action)
        
        if env._sum_hand(env.player_hand) <= 21:
            if env._sum_hand(env.dealer_hand) > 21 or \
               env._sum_hand(env.player_hand) > env._sum_hand(env.dealer_hand):
                learned_wins += 1
    
    agent.exploration_rate = old_epsilon
    learned_win_rate = learned_wins / num_games * 100
    
    # Test basic strategy
    print(f"Testing Basic Strategy ({num_games:,} games)...")
    basic_wins = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            action = basic_strategy(state)
            state, _, done = env.step(action)
        
        if env._sum_hand(env.player_hand) <= 21:
            if env._sum_hand(env.dealer_hand) > 21 or \
               env._sum_hand(env.player_hand) > env._sum_hand(env.dealer_hand):
                basic_wins += 1
    
    basic_win_rate = basic_wins / num_games * 100
    
    # Results
    print("\n" + "-" * 40)
    print(f"  Learned Strategy: {learned_win_rate:.2f}% win rate")
    print(f"  Basic Strategy:   {basic_win_rate:.2f}% win rate")
    print(f"  Difference:       {learned_win_rate - basic_win_rate:.2f}%")
    print("-" * 40)
    
    if abs(learned_win_rate - basic_win_rate) < 2:
        print("\n✅ Agent learned near-optimal strategy!")
    else:
        print("\n⚠️  Agent strategy differs from basic strategy.")
        print("    This could be due to:")
        print("    - Insufficient training")
        print("    - Different state representation")
        print("    - Exploration rate too high")


def evaluate_agent(agent, num_games=10000):
    """
    Evaluate trained agent.
    
    Args:
        agent: Trained agent
        num_games: Number of games to play
    """
    print("\n" + "=" * 60)
    print("       📊 AGENT EVALUATION 📊")
    print("=" * 60)
    
    env = Blackjack()
    
    # Disable exploration
    old_epsilon = agent.exploration_rate
    agent.exploration_rate = 0.0
    
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, done = env.step(action)
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    
    # Restore exploration
    agent.exploration_rate = old_epsilon
    
    print(f"\nResults over {num_games:,} games:")
    print(f"  Wins:   {wins:6d} ({wins/num_games*100:5.1f}%)")
    print(f"  Losses: {losses:6d} ({losses/num_games*100:5.1f}%)")
    print(f"  Draws:  {draws:6d} ({draws/num_games*100:5.1f}%)")
    
    print("\n💡 Note: In Blackjack, the dealer has a natural advantage.")
    print("   A ~42-43% win rate is near-optimal!")
    print("   (This includes wins from blackjack naturals)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training function."""
    
    # Train the agent
    agent, history = train_agent()
    
    # Evaluate
    evaluate_agent(agent)
    
    print("\n✅ Training complete!")
    print(f"Run 'python play.py' to play against the trained agent!")


if __name__ == '__main__':
    main()
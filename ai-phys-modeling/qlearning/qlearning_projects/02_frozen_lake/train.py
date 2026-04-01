"""
Training Script for Frozen Lake Q-Learning Agent
=================================================

Train the agent to cross the frozen lake!

Key differences from Tic-Tac-Toe:
- Exploration rate DECAYS over time (starts high, ends low)
- Environment is stochastic (same action might have different results)
- Sparse rewards (only get reward at goal)

Author: Educational RL Project
"""

import numpy as np
from lake_env import FrozenLake, print_policy, print_value_function, print_policy_with_qmax
from agent import QLearningAgent
import os


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Number of training episodes
NUM_EPISODES = 10000

# Maximum steps per episode (prevent infinite loops)
MAX_STEPS = 100

# Print progress every N episodes
PROGRESS_INTERVAL = 1000

# Q-Learning hyperparameters
LEARNING_RATE = 0.1          # How fast to learn
DISCOUNT_FACTOR = 0.99       # How much future matters (high for sparse rewards)
EXPLORATION_RATE = 1.0       # Start with full exploration
EXPLORATION_DECAY = 0.9995   # Decay rate
MIN_EXPLORATION = 0.01       # Minimum exploration

# Environment settings
IS_SLIPPERY = True           # Set to False for deterministic version

# Save path
SAVE_PATH = os.path.join(os.path.dirname(__file__), 'trained_agent.pkl')


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_agent(
    num_episodes=NUM_EPISODES,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    discount_factor=DISCOUNT_FACTOR,
    exploration_rate=EXPLORATION_RATE,
    exploration_decay=EXPLORATION_DECAY,
    min_exploration=MIN_EXPLORATION,
    is_slippery=IS_SLIPPERY,
    save_path=SAVE_PATH,
    verbose=True
):
    """
    Train a Q-learning agent to cross the frozen lake.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Learning rate (α)
        discount_factor: Discount factor (γ)
        exploration_rate: Initial exploration rate (ε)
        exploration_decay: ε decay per episode
        min_exploration: Minimum ε
        is_slippery: If True, ice is slippery (stochastic)
        save_path: Where to save trained agent
        verbose: Print progress
        
    Returns:
        agent: Trained agent
        history: Training history
    """
    print("=" * 60)
    print("       🧊 FROZEN LAKE Q-LEARNING TRAINING 🧊")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Discount Factor: {discount_factor}")
    print(f"  Initial Exploration: {exploration_rate}")
    print(f"  Exploration Decay: {exploration_decay}")
    print(f"  Slippery Ice: {is_slippery}")
    print("-" * 60)
    
    # Create environment and agent
    env = FrozenLake(is_slippery=is_slippery)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay,
        min_exploration=min_exploration
    )
    
    # Training history
    history = {
        'rewards': [],
        'steps': [],
        'win_rates': [],
        'exploration_rates': []
    }
    
    # Track recent results
    recent_results = []
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Episode loop
        for step in range(max_steps):
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update tracking
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Record result
        won = total_reward > 0
        agent.record_result(won)
        recent_results.append(won)
        
        # Decay exploration
        agent.decay_exploration()
        
        # Progress report
        if verbose and episode % PROGRESS_INTERVAL == 0:
            recent = recent_results[-PROGRESS_INTERVAL:]
            win_rate = sum(recent) / len(recent) * 100
            
            print(f"Episode {episode:6d}/{num_episodes} | "
                  f"Win Rate: {win_rate:5.1f}% | "
                  f"ε: {agent.exploration_rate:.3f} | "
                  f"Avg Steps: {steps:.0f}")
            
            history['win_rates'].append(win_rate)
            history['exploration_rates'].append(agent.exploration_rate)
    
    # ==========================================================================
    # TRAINING COMPLETE
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("       🎉 TRAINING COMPLETE! 🎉")
    print("=" * 60)
    
    agent.print_stats()
    
    # Show learned policy with Q-max values
    print_policy_with_qmax(agent)
    
    # Save agent
    if save_path:
        agent.save(save_path)
    
    return agent, history


def evaluate_agent(agent, num_episodes=100, is_slippery=True):
    """
    Evaluate the trained agent.
    
    Args:
        agent: Trained agent
        num_episodes: Number of test episodes
        is_slippery: Environment slipperiness
    """
    print("\n" + "=" * 60)
    print("       📊 AGENT EVALUATION 📊")
    print("=" * 60)
    
    env = FrozenLake(is_slippery=is_slippery)
    
    # Disable exploration for evaluation
    old_epsilon = agent.exploration_rate
    agent.exploration_rate = 0.0
    
    wins = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        steps = 0
        
        for _ in range(100):
            action = agent.choose_action(state, training=False)
            state, reward, done = env.step(action)
            steps += 1
            
            if done:
                if reward > 0:
                    wins += 1
                break
        
        total_steps += steps
    
    win_rate = wins / num_episodes * 100
    avg_steps = total_steps / num_episodes
    
    print(f"\nResults over {num_episodes} episodes:")
    print(f"  Wins: {wins} ({win_rate:.1f}%)")
    print(f"  Losses: {num_episodes - wins}")
    print(f"  Average Steps: {avg_steps:.1f}")
    
    # Theoretical maximum for slippery ice: ~75%
    if is_slippery:
        print(f"\n💡 Note: Due to slippery ice, even optimal play can't reach 100%!")
        print(f"   Best possible is around 74-75% win rate.")
    
    # Restore exploration rate
    agent.exploration_rate = old_epsilon
    
    print("\n" + "=" * 60)


# =============================================================================
# COMPARISON: SLIPPERY vs NON-SLIPPERY
# =============================================================================

def compare_slippery():
    """Compare performance on slippery vs non-slippery ice."""
    
    print("\n" + "=" * 60)
    print("   COMPARISON: Slippery vs Non-Slippery Ice")
    print("=" * 60)
    
    print("\n🧊 Training on SLIPPERY ice...")
    agent_slippery, _ = train_agent(
        num_episodes=10000,
        is_slippery=True,
        verbose=False
    )
    evaluate_agent(agent_slippery, is_slippery=True)
    
    print("\n❄️ Training on NON-SLIPPERY ice...")
    agent_normal, _ = train_agent(
        num_episodes=10000,
        is_slippery=False,
        verbose=False
    )
    evaluate_agent(agent_normal, is_slippery=False)


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
    print(f"Run 'python play.py' to watch the trained agent!")


if __name__ == '__main__':
    main()
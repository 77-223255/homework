"""
Q-Learning Training Script
==========================

This script trains a Q-learning agent to navigate a maze.

The agent learns by:
1. Exploring the maze (trying different actions)
2. Getting rewards (+1 for goal, -1 for hell, 0 for ground)
3. Updating its Q-table based on experiences

After training, the agent will know the optimal path!

Author: Improved version for teaching
"""

from maze_env import Maze
from RL_brain import QLearningAgent
import time


# =============================================================================
# TRAINING CONFIGURATION - Adjust these to experiment!
# =============================================================================

# Number of training episodes (one episode = one attempt to reach goal/hell)
NUM_EPISODES = 100

# How often to print progress (every N episodes)
PROGRESS_INTERVAL = 10

# How often to show Q-table (every N episodes)
QTABLE_INTERVAL = 25

# Q-Learning hyperparameters
LEARNING_RATE = 0.01      # How fast to learn (0.01 is slow but stable)
DISCOUNT_FACTOR = 0.9     # How much to care about future (0.9 = plan ahead)
EXPLORATION_RATE = 0.9    # How often to use best action vs random (0.9 = mostly exploit)

# Speed settings
RENDER_DELAY = 0.05       # Delay between renders (seconds) - lower = faster
RESET_DELAY = 0.1         # Delay on reset (seconds)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train():
    """
    Train the Q-learning agent to navigate the maze.
    
    Returns:
        agent: The trained Q-learning agent
        stats: Training statistics
    """
    print("=" * 60)
    print("       🎮 Q-LEARNING MAZE TRAINING 🎮")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Learning Rate (α): {LEARNING_RATE}")
    print(f"  Discount Factor (γ): {DISCOUNT_FACTOR}")
    print(f"  Exploration Rate (ε): {EXPLORATION_RATE}")
    print("\n" + "-" * 60)
    
    # Create environment and agent
    env = Maze(render_delay=RENDER_DELAY, reset_delay=RESET_DELAY)
    agent = QLearningAgent(
        actions=list(range(env.n_actions)),
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        exploration_rate=EXPLORATION_RATE
    )
    
    # Statistics tracking
    stats = {
        'wins': 0,
        'losses': 0,
        'total_steps': 0,
        'episode_rewards': [],
        'episode_steps': []
    }
    
    # Training loop
    for episode in range(1, NUM_EPISODES + 1):
        # Reset environment
        state = env.reset()
        state_str = str(state)
        
        episode_reward = 0
        episode_steps = 0
        
        # Episode loop
        while True:
            # Render (visualize)
            env.render()
            
            # Agent chooses action
            action = agent.choose_action(state_str)
            
            # Execute action
            next_state, reward, done = env.step(action)
            next_state_str = str(next_state)
            
            # Convert terminal state
            if done:
                next_state_str = 'terminal'
            
            # Agent learns from experience
            agent.learn(state_str, action, reward, next_state_str)
            
            # Update tracking
            episode_reward += reward
            episode_steps += 1
            
            # Move to next state
            state_str = next_state_str
            
            # Check if episode ended
            if done:
                if reward == 1:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
                
                stats['total_steps'] += episode_steps
                stats['episode_rewards'].append(episode_reward)
                stats['episode_steps'].append(episode_steps)
                break
        
        # Progress report
        if episode % PROGRESS_INTERVAL == 0:
            win_rate = stats['wins'] / episode * 100
            avg_steps = stats['total_steps'] / episode
            print(f"Episode {episode:3d}/{NUM_EPISODES} | "
                  f"Wins: {stats['wins']:3d} ({win_rate:5.1f}%) | "
                  f"Avg Steps: {avg_steps:.1f}")
        
        # Show Q-table periodically
        if episode % QTABLE_INTERVAL == 0:
            agent.print_q_table(f"Q-Table after Episode {episode}")
    
    # Training complete
    print("\n" + "=" * 60)
    print("       🎉 TRAINING COMPLETE! 🎉")
    print("=" * 60)
    
    # Final summary
    print_training_summary(stats, agent)
    
    # Show final Q-table
    agent.print_q_table("Final Q-Table")
    agent.print_best_actions()
    
    env.destroy()
    
    return agent, stats


def print_training_summary(stats, agent):
    """Print a summary of the training results."""
    
    print("\n📊 Training Summary:")
    print("-" * 40)
    
    total = stats['wins'] + stats['losses']
    win_rate = stats['wins'] / total * 100 if total > 0 else 0
    avg_steps = stats['total_steps'] / total if total > 0 else 0
    avg_reward = sum(stats['episode_rewards']) / total if total > 0 else 0
    
    print(f"  Total Episodes: {total}")
    print(f"  Wins: {stats['wins']} ({win_rate:.1f}%)")
    print(f"  Losses: {stats['losses']}")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"  Average Reward: {avg_reward:.3f}")
    
    print("\n🧠 Agent Info:")
    summary = agent.get_summary()
    print(f"  States Discovered: {summary['states_discovered']}")
    print(f"  Total Q-Updates: {summary['total_updates']}")
    print(f"  Q-Table Size: {summary['table_size']}")
    
    print("-" * 40)


def test_trained_agent(agent):
    """
    Test a trained agent.
    
    Args:
        agent: A trained Q-learning agent
    """
    print("\n" + "=" * 60)
    print("       🧪 TESTING TRAINED AGENT 🧪")
    print("=" * 60)
    print("\nThe agent will use its learned Q-table to navigate.")
    print("Watch how it reaches the goal!\n")
    
    env = Maze(render_delay=0.1, reset_delay=0.3)
    
    for test_episode in range(3):
        state = env.reset()
        state_str = str(state)
        
        print(f"Test Episode {test_episode + 1}:")
        print(f"  Starting at: {env.get_position_name(state)}")
        
        steps = 0
        while True:
            env.render()
            
            # Always use best action (no exploration)
            action = agent.choose_action(state_str)
            action_name = env.action_space[action]
            
            next_state, reward, done = env.step(action)
            steps += 1
            
            print(f"  Step {steps}: {action_name} -> {env.get_position_name(next_state)}")
            
            if done:
                if reward == 1:
                    print(f"  ✅ Reached goal in {steps} steps!")
                else:
                    print(f"  ❌ Hit hell after {steps} steps")
                break
            
            state_str = str(next_state)
        
        print()
    
    env.destroy()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main function - run training and testing."""
    
    # Train the agent
    agent, stats = train()
    
    # Ask user if they want to test
    print("\n" + "=" * 60)
    print("Want to see the trained agent in action?")
    print("=" * 60)
    
    while True:
        response = input("\nTest the trained agent? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            test_trained_agent(agent)
            break
        elif response in ['no', 'n']:
            print("\n👍 Done! Thanks for training!")
            break
        else:
            print("Please type 'yes' or 'no'")


if __name__ == "__main__":
    main()
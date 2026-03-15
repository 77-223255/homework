"""
DQN Training Script
===================

Train a DQN agent to solve CartPole!

Key differences from Q-table methods:
- Learn from batches of experiences
- Use target network for stable targets
- Handle continuous state space

Author: Educational RL Project
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole_env import CartPole

# Check for PyTorch
try:
    from dqn_agent import DQNAgent, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. Run: pip install torch")


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

NUM_EPISODES = 500
MAX_STEPS = 500

# Print progress every N episodes
PROGRESS_INTERVAL = 10

# DQN hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TARGET_UPDATE = 5

# Solved threshold (average score over 100 episodes)
SOLVED_THRESHOLD = 195

# Save path
SAVE_PATH = os.path.join(os.path.dirname(__file__), 'dqn_model.pt')


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_agent(
    num_episodes=NUM_EPISODES,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    target_update=TARGET_UPDATE,
    save_path=SAVE_PATH,
    verbose=True
):
    """
    Train a DQN agent to solve CartPole.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Epsilon decay per episode
        buffer_size: Replay buffer size
        batch_size: Training batch size
        target_update: Target network update frequency
        save_path: Model save path
        verbose: Print progress
        
    Returns:
        agent: Trained agent
        history: Training history
    """
    if not TORCH_AVAILABLE:
        print("❌ PyTorch is required for DQN!")
        print("Run: pip install torch")
        return None, None
    
    print("=" * 60)
    print("       🧠 DQN TRAINING: CartPole 🧠")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Gamma (γ): {gamma}")
    print(f"  Epsilon: {epsilon_start} → {epsilon_end}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Target Update: every {target_update} steps")
    print(f"  Solved Threshold: {SOLVED_THRESHOLD} avg over 100 episodes")
    print("-" * 60)
    
    # Create environment and agent
    env = CartPole()
    agent = DQNAgent(
        state_size=4,
        action_size=2,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update=target_update
    )
    
    # Training history
    history = {
        'scores': [],
        'avg_scores': [],
        'losses': [],
        'epsilons': []
    }
    
    # Recent scores for averaging
    recent_scores = []
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        
        for step in range(max_steps):
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Learn
            loss = agent.learn()
            if loss is not None:
                history['losses'].append(loss)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Record score
        history['scores'].append(score)
        history['epsilons'].append(agent.epsilon)
        recent_scores.append(score)
        
        # Keep only last 100 scores
        if len(recent_scores) > 100:
            recent_scores.pop(0)
        
        avg_score = np.mean(recent_scores)
        history['avg_scores'].append(avg_score)
        
        # Progress report
        if verbose and episode % PROGRESS_INTERVAL == 0:
            print(f"Episode {episode:4d}/{num_episodes} | "
                  f"Score: {score:6.0f} | "
                  f"Avg(100): {avg_score:6.1f} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # Check if solved
        if len(recent_scores) >= 100 and avg_score >= SOLVED_THRESHOLD:
            print("\n" + "=" * 60)
            print(f"   🎉 SOLVED! Episode {episode}")
            print(f"   Average score: {avg_score:.1f}")
            print("=" * 60)
            break
    
    # ==========================================================================
    # TRAINING COMPLETE
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("       🎉 TRAINING COMPLETE! 🎉")
    print("=" * 60)
    
    # Final stats
    final_avg = np.mean(history['scores'][-100:])
    print(f"\n📊 Final Statistics:")
    print(f"   Episodes: {len(history['scores'])}")
    print(f"   Final Average Score (100): {final_avg:.1f}")
    print(f"   Max Score: {max(history['scores']):.0f}")
    print(f"   Final Epsilon: {agent.epsilon:.3f}")
    
    if final_avg >= SOLVED_THRESHOLD:
        print(f"\n✅ CartPole SOLVED!")
    else:
        print(f"\n⚠️  Not yet solved. Try more episodes.")
    
    # Save model
    if save_path:
        agent.save(save_path)
    
    return agent, history


def plot_training(history):
    """
    Plot training progress.
    
    Args:
        history: Training history dictionary
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skip plotting.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Scores
    axes[0, 0].plot(history['scores'], alpha=0.6, label='Score')
    axes[0, 0].plot(history['avg_scores'], label='Avg (100)')
    axes[0, 0].axhline(y=SOLVED_THRESHOLD, color='r', linestyle='--', label='Solved')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Epsilon
    axes[0, 1].plot(history['epsilons'])
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_title('Exploration Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss
    if history['losses']:
        axes[1, 0].plot(history['losses'])
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Score distribution
    axes[1, 1].hist(history['scores'], bins=30, alpha=0.7)
    axes[1, 1].axvline(x=np.mean(history['scores']), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(history["scores"]):.1f}')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), 'training_plot.png')
    plt.savefig(plot_path)
    print(f"📊 Training plot saved to {plot_path}")
    
    plt.show()


def evaluate_agent(agent, num_episodes=10):
    """
    Evaluate trained agent.
    
    Args:
        agent: Trained DQN agent
        num_episodes: Number of test episodes
    """
    print("\n" + "=" * 60)
    print("       📊 AGENT EVALUATION 📊")
    print("=" * 60)
    
    env = CartPole()
    
    # Disable exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        score = 0
        
        for _ in range(500):
            action = agent.choose_action(state, training=False)
            state, reward, done = env.step(action)
            score += reward
            
            if done:
                break
        
        scores.append(score)
        print(f"  Episode {episode + 1}: Score = {score:.0f}")
    
    # Restore epsilon
    agent.epsilon = old_epsilon
    
    print(f"\n📊 Average Score: {np.mean(scores):.1f}")
    print(f"📊 Max Score: {max(scores):.0f}")
    print(f"📊 Min Score: {min(scores):.0f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training function."""
    
    # Train the agent
    agent, history = train_agent()
    
    if agent is None:
        return
    
    # Plot training
    try:
        plot_training(history)
    except:
        pass
    
    # Evaluate
    evaluate_agent(agent)
    
    print("\n✅ Training complete!")
    print(f"Run 'python play.py' to watch the trained agent!")


if __name__ == '__main__':
    main()